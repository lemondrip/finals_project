"""
House Price Prediction Page
===========================
Streamlit prediction/model page for the group project.

Goal: predict house prices from property characteristics using Scikit-Learn regression models. 
The page trains the models, compares their performance, lets the user switch between models,
and provides a what-if house price predictor.



"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Price"
NUMERIC_BASE = ["Area_SqFt", "Rooms", "Build_Year"]
CATEGORICAL_BASE = [
    "Location",
    "Street_Type",
    "Furnishing",
    "Property_Type",
    "Has_Pool",
]
ENGINEERED_NUMERIC = ["House_Age", "Area_Per_Room", "Pool_Flag"]
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Small UI helpers
# -----------------------------------------------------------------------------
def _money(value: float) -> str:
    """Format a number as dollars for the dashboard."""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def _section(title: str, description: str | None = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------
def _candidate_data_paths() -> List[Path]:
    """Search common locations for dataset_2.csv so the file is easy to move."""
    candidates: List[Path] = []

    try:
        here = Path(__file__).resolve().parent
        candidates.extend(
            [
                here / "dataset_2.csv",
                here.parent / "dataset_2.csv",
                here / "data" / "dataset_2.csv",
                here.parent / "data" / "dataset_2.csv",
                here.parent.parent / "dataset_2.csv",
            ]
        )
    except NameError:
        pass

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "dataset_2.csv",
            cwd / "data" / "dataset_2.csv",
            cwd / "src" / "dataset_2.csv",
        ]
    )

    # Remove duplicates while preserving order.
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _find_data_path() -> Path | None:
    for path in _candidate_data_paths():
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    """Load and lightly clean the house price dataset."""
    df = pd.read_csv(path)

    required = NUMERIC_BASE + CATEGORICAL_BASE + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Clean numeric columns.
    for col in NUMERIC_BASE + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no target. Keep feature missing values because the pipeline
    # imputes them during training.
    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET] > 0].copy()

    # Standardize categorical missing values.
    for col in CATEGORICAL_BASE:
        df[col] = df[col].astype("object").fillna("Unknown")

    return _add_engineered_features(df)


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple features that are useful for price prediction."""
    out = df.copy()

    current_year = pd.Timestamp.today().year
    out["House_Age"] = current_year - pd.to_numeric(out["Build_Year"], errors="coerce")
    out["Area_Per_Room"] = (
        pd.to_numeric(out["Area_SqFt"], errors="coerce")
        / pd.to_numeric(out["Rooms"], errors="coerce").replace(0, np.nan)
    )
    out["Pool_Flag"] = (
        out["Has_Pool"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    )

    return out


def _split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    numeric_features = [c for c in NUMERIC_BASE + ENGINEERED_NUMERIC if c in df.columns]
    categorical_features = [c for c in CATEGORICAL_BASE if c in df.columns]
    X = df[numeric_features + categorical_features].copy()
    y = df[TARGET].copy()
    return X, y, numeric_features, categorical_features


# -----------------------------------------------------------------------------
# Model training
# -----------------------------------------------------------------------------
def _make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder in a way that works on old and new sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def _available_models() -> Dict[str, object]:
    """Models shown on the page. Keeping four gives users real comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def _build_model(estimator: object, numeric_features: List[str], categorical_features: List[str]) -> TransformedTargetRegressor:
    """
    Build a full model pipeline.

    The model learns log(price) internally, then returns predictions back in
    dollars. This usually improves price models because prices are right-skewed.
    """
    regressor = Pipeline(
        steps=[
            ("preprocessor", _preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


@st.cache_resource(show_spinner=False)
def train_models(
    csv_path: str,
    selected_models: Tuple[str, ...],
    test_size: float,
    cv_folds: int,
) -> dict:
    """Train selected models and return leaderboard, fitted models, and test data."""
    df = load_data_from_path(csv_path)
    X, y, numeric_features, categorical_features = _split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    all_models = _available_models()
    fitted_models: Dict[str, TransformedTargetRegressor] = {}
    rows = []

    for model_name in selected_models:
        model = _build_model(all_models[model_name], numeric_features, categorical_features)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # price should never be negative

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="r2",
            cv=cv_folds,
        )

        rows.append(
            {
                "Model": model_name,
                "R²": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAPE (%)": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
                "CV R²": float(np.mean(cv_scores)),
            }
        )
        fitted_models[model_name] = model

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("R²", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "df": df,
        "X_test": X_test,
        "y_test": y_test,
        "leaderboard": leaderboard,
        "models": fitted_models,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


# -----------------------------------------------------------------------------
# Explainability helpers for this prediction page
# -----------------------------------------------------------------------------
def _clean_feature_name(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("_", " ")
    return name


def _feature_importance(model: TransformedTargetRegressor) -> pd.DataFrame:
    """Return top feature importances or absolute coefficients from a fitted model."""
    pipe = model.regressor_
    preprocessor = pipe.named_steps["preprocessor"]
    estimator = pipe.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    importance = pd.DataFrame(
        {
            "Feature": [_clean_feature_name(name) for name in feature_names],
            "Importance": values,
        }
    )
    return importance.sort_values("Importance", ascending=False).head(15)


# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------
def render() -> None:
    st.title("🏠 House Price Prediction")
    st.write(
        "This page trains multiple regression models to estimate house prices. "
        "Users can compare model performance, switch between models, and enter "
        "property details to generate a predicted fair market price."
    )

    data_path = _find_data_path()
    if data_path is None:
        st.error(
            "Could not find `dataset_2.csv`. Put it in the project root, `src/`, "
            "or a `data/` folder, then refresh the app."
        )
        return

    try:
        df = load_data_from_path(str(data_path))
    except Exception as exc:
        st.error(f"The dataset could not be loaded: {exc}")
        return

    _section(
        "1. Dataset used for prediction",
        "The target variable is `Price`. The model uses size, rooms, build year, "
        "location, street type, furnishing, property type, and pool information.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Average price", _money(df[TARGET].mean()))
    c3.metric("Median price", _money(df[TARGET].median()))
    c4.metric("Price range", f"{_money(df[TARGET].min())} - {_money(df[TARGET].max())}")

    with st.expander("Preview dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    _section(
        "2. Train at least two models",
        "This satisfies the project requirement that users can switch between two or more models.",
    )

    all_model_names = list(_available_models().keys())
    left, right = st.columns([2, 1])
    chosen_models = left.multiselect(
        "Choose models to train",
        all_model_names,
        default=["Linear Regression", "Ridge Regression", "Random Forest"],
    )
    test_size = right.slider("Test set size", 0.10, 0.40, 0.20, step=0.05)
    cv_folds = right.slider("Cross-validation folds", 3, 5, 5, step=1)

    train_clicked = st.button("🚀 Train selected models", type="primary")

    if train_clicked:
        if len(chosen_models) < 2:
            st.warning("Please select at least two models to match the project requirement.")
        else:
            with st.spinner("Training models and building leaderboard..."):
                st.session_state["house_prediction_state"] = train_models(
                    str(data_path),
                    tuple(chosen_models),
                    test_size,
                    cv_folds,
                )
            st.success("Models trained successfully.")

    state = st.session_state.get("house_prediction_state")
    if state is None:
        st.info("Click **Train selected models** to create the leaderboard and unlock the predictor.")
        return

    leaderboard = state["leaderboard"]
    models = state["models"]
    best_model_name = leaderboard.iloc[0]["Model"]

    _section("3. Model leaderboard", "Models are ranked by test-set R². Higher R² is better; lower error is better.")

    best = leaderboard.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Champion model", best_model_name)
    m2.metric("Test R²", f"{best['R²']:.3f}")
    m3.metric("MAE", _money(best["MAE"]))
    m4.metric("RMSE", _money(best["RMSE"]))

    display_board = leaderboard.copy()
    display_board["R²"] = display_board["R²"].map(lambda x: f"{x:.3f}")
    display_board["CV R²"] = display_board["CV R²"].map(lambda x: f"{x:.3f}")
    display_board["MAPE (%)"] = display_board["MAPE (%)"].map(lambda x: f"{x:.1f}%")
    for col in ["MAE", "RMSE"]:
        display_board[col] = display_board[col].map(_money)
    st.dataframe(display_board, use_container_width=True, hide_index=True)

    # Bar chart comparing MAE.
    fig_mae, ax_mae = plt.subplots(figsize=(8, 4))
    ordered = leaderboard.sort_values("MAE", ascending=True)
    ax_mae.barh(ordered["Model"], ordered["MAE"])
    ax_mae.set_xlabel("Mean Absolute Error ($)")
    ax_mae.set_title("Model comparison: lower MAE is better")
    ax_mae.invert_yaxis()
    st.pyplot(fig_mae)

    _section("4. Choose a model and predict a house price", "Change the inputs below to test different house scenarios.")

    model_names = list(models.keys())
    default_index = model_names.index(best_model_name) if best_model_name in model_names else 0
    selected_model_name = st.selectbox(
        "Model used for the main prediction",
        model_names,
        index=default_index,
        help="The user can switch between trained models here.",
    )

    input_left, input_right = st.columns(2)

    with input_left:
        area = st.slider(
            "Area / size (square feet)",
            min_value=int(max(200, df["Area_SqFt"].min())),
            max_value=int(min(12000, df["Area_SqFt"].max())),
            value=int(df["Area_SqFt"].median()),
            step=50,
        )
        rooms = st.slider(
            "Number of rooms",
            min_value=int(max(1, df["Rooms"].min())),
            max_value=int(max(df["Rooms"].max(), 8)),
            value=int(df["Rooms"].median()),
            step=1,
        )
        build_year = st.slider(
            "Build year",
            min_value=int(df["Build_Year"].min()),
            max_value=int(df["Build_Year"].max()),
            value=int(df["Build_Year"].median()),
            step=1,
        )
        has_pool = st.radio("Has pool?", sorted(df["Has_Pool"].dropna().unique()), horizontal=True)

    with input_right:
        location = st.selectbox("Location", sorted(df["Location"].dropna().unique()))
        street_type = st.selectbox("Street type", sorted(df["Street_Type"].dropna().unique()))
        furnishing = st.selectbox("Furnishing", sorted(df["Furnishing"].dropna().unique()))
        property_type = st.selectbox("Property type", sorted(df["Property_Type"].dropna().unique()))

    input_row = pd.DataFrame(
        [
            {
                "Area_SqFt": float(area),
                "Rooms": float(rooms),
                "Build_Year": int(build_year),
                "Location": location,
                "Street_Type": street_type,
                "Furnishing": furnishing,
                "Property_Type": property_type,
                "Has_Pool": has_pool,
            }
        ]
    )
    input_row = _add_engineered_features(input_row)

    selected_model = models[selected_model_name]
    predicted_price = float(selected_model.predict(input_row)[0])
    predicted_price = max(predicted_price, 0)

    st.metric(
        f"Predicted price using {selected_model_name}",
        _money(predicted_price),
        delta=f"{_money(predicted_price - df[TARGET].median())} vs dataset median",
    )

    # Show predictions from every trained model for the same house.
    comparison_rows = []
    for name, fitted_model in models.items():
        pred = max(float(fitted_model.predict(input_row)[0]), 0)
        comparison_rows.append({"Model": name, "Predicted Price": pred})
    prediction_comparison = pd.DataFrame(comparison_rows).sort_values("Predicted Price")
    prediction_display = prediction_comparison.copy()
    prediction_display["Predicted Price"] = prediction_display["Predicted Price"].map(_money)
    st.markdown("#### Predictions from all trained models")
    st.dataframe(prediction_display, use_container_width=True, hide_index=True)

    fig_pred, ax_pred = plt.subplots(figsize=(8, 4))
    ax_pred.barh(prediction_comparison["Model"], prediction_comparison["Predicted Price"])
    ax_pred.set_xlabel("Predicted price ($)")
    ax_pred.set_title("Same house, different model predictions")
    st.pyplot(fig_pred)

    _section("5. Champion diagnostics", "Check whether the best model predicts close to the real prices on the test set.")

    X_test = state["X_test"]
    y_test = state["y_test"]
    champion = models[best_model_name]
    y_pred = np.maximum(champion.predict(X_test), 0)
    residuals = y_pred - y_test

    diag_left, diag_right = st.columns(2)
    with diag_left:
        fig_actual, ax_actual = plt.subplots(figsize=(5, 5))
        ax_actual.scatter(y_test, y_pred, alpha=0.6)
        min_val = float(min(y_test.min(), y_pred.min()))
        max_val = float(max(y_test.max(), y_pred.max()))
        ax_actual.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        ax_actual.set_xlabel("Actual price ($)")
        ax_actual.set_ylabel("Predicted price ($)")
        ax_actual.set_title("Actual vs predicted")
        st.pyplot(fig_actual)

    with diag_right:
        fig_resid, ax_resid = plt.subplots(figsize=(5, 5))
        ax_resid.hist(residuals, bins=30)
        ax_resid.axvline(0, linestyle="--")
        ax_resid.set_xlabel("Prediction error ($)")
        ax_resid.set_ylabel("Number of houses")
        ax_resid.set_title("Residual distribution")
        st.pyplot(fig_resid)

    _section("6. Important variables for the selected model", "This quick explanation helps connect the model prediction to house features.")
    importance = _feature_importance(selected_model)
    if importance.empty:
        st.info("Feature importance is not available for this model.")
    else:
        st.dataframe(importance, use_container_width=True, hide_index=True)
        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        imp_ordered = importance.sort_values("Importance", ascending=True)
        ax_imp.barh(imp_ordered["Feature"], imp_ordered["Importance"])
        ax_imp.set_xlabel("Importance / absolute coefficient")
        ax_imp.set_title(f"Top drivers in {selected_model_name}")
        st.pyplot(fig_imp)

    st.info(
        "Business use: a buyer, seller, or real-estate agency can use this page to "
        "estimate a fair house price before listing, buying, or negotiating. The model "
        "does not replace professional appraisal, but it gives a data-based starting point."
    )


if __name__ == "__main__":
    render()
