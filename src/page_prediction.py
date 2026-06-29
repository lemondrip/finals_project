"""
House Price Prediction Page
===========================
Streamlit prediction/model page for the group project.

Goal: predict house prices from property characteristics using
Scikit-Learn regression models. The page trains the models,
compares their performance, lets the user switch between models,
and provides a what-if predictor.
"""

from __future__ import annotations

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

from data_loader import load_data


TARGET = "Price"
NUMERIC_FEATURES = ["Area_SqFt", "Rooms", "Build_Year"]
CATEGORICAL_FEATURES = ["Location", "Street_Type", "Furnishing", "Property_Type", "Has_Pool"]
RANDOM_STATE = 42


# ── UI helpers ───────────────────────────────────────────────────────
def _money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def _section(title: str, description: str | None = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


# ── Model pipeline ───────────────────────────────────────────────────
def _make_one_hot_encoder() -> OneHotEncoder:
    """OneHotEncoder that works on old and new sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_one_hot_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _available_models() -> Dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def _build_model(estimator: object) -> TransformedTargetRegressor:
    """Pipeline that learns log(price) internally and returns dollars."""
    regressor = Pipeline([
        ("preprocessor", _preprocessor()),
        ("model", estimator),
    ])
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1,
    )


# ── Training ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _train_models(selected_models: Tuple[str, ...], test_size: float, cv_folds: int) -> dict:
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE,
    )

    all_models = _available_models()
    fitted: Dict[str, TransformedTargetRegressor] = {}
    rows = []

    for name in selected_models:
        model = _build_model(all_models[name])
        model.fit(X_train, y_train)
        y_pred = np.maximum(model.predict(X_test), 0)
        cv_scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv_folds)
        rows.append({
            "Model": name,
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAPE (%)": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
            "CV R²": float(np.mean(cv_scores)),
        })
        fitted[name] = model

    leaderboard = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    return {
        "df": df, "X_test": X_test, "y_test": y_test,
        "leaderboard": leaderboard, "models": fitted,
    }


# ── Feature importance helper ────────────────────────────────────────
def _clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "").replace("_", " ")


def _feature_importance(model: TransformedTargetRegressor) -> pd.DataFrame:
    pipe = model.regressor_
    pre = pipe.named_steps["preprocessor"]
    est = pipe.named_steps["model"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    df = pd.DataFrame({
        "Feature": [_clean_feature_name(n) for n in names],
        "Importance": vals,
    })
    return df.sort_values("Importance", ascending=False).head(15)


# ── Page ─────────────────────────────────────────────────────────────
def render() -> None:
    st.title("🏠 House Price Prediction")
    st.write(
        "This page trains multiple regression models to estimate house prices. "
        "Users can compare model performance, switch between models, and enter "
        "property details to generate a predicted fair market price."
    )

    try:
        df = load_data()
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

    # ── Model selection ─────────────────────────────────────────────
    _section(
        "2. Prediction Models",
        "There are 4 models: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting. Choose at least 2 for comparison.",
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

    if st.button("🚀 Train selected models", type="primary"):
        if len(chosen_models) < 2:
            st.warning("Please select at least two models to match the project requirement.")
        else:
            with st.spinner("Training models and building leaderboard..."):
                st.session_state["house_prediction_state"] = _train_models(
                    tuple(chosen_models), test_size, cv_folds,
                )
            st.success("Models trained successfully.")

    state = st.session_state.get("house_prediction_state")
    if state is None:
        st.info("Click **Train selected models** to create the leaderboard and unlock the predictor.")
        return

    leaderboard = state["leaderboard"]
    models = state["models"]
    best_model_name = leaderboard.iloc[0]["Model"]

    # ── Leaderboard ─────────────────────────────────────────────────
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

    fig_mae, ax_mae = plt.subplots(figsize=(8, 4))
    ordered = leaderboard.sort_values("MAE", ascending=True)
    ax_mae.barh(ordered["Model"], ordered["MAE"])
    ax_mae.set_xlabel("Mean Absolute Error ($)")
    ax_mae.set_title("Model comparison: lower MAE is better")
    ax_mae.invert_yaxis()
    st.pyplot(fig_mae)

    # ── What-if predictor ───────────────────────────────────────────
    _section("4. Choose a model and predict a house price", "Change the inputs below to test different house scenarios.")

    model_names = list(models.keys())
    default_index = model_names.index(best_model_name) if best_model_name in model_names else 0
    selected_model_name = st.selectbox(
        "Model used for the main prediction", model_names, index=default_index,
        help="The user can switch between trained models here.",
    )

    input_left, input_right = st.columns(2)
    with input_left:
        area = st.slider(
            "Area / size (square feet)",
            min_value=int(max(200, df["Area_SqFt"].min())),
            max_value=int(min(12000, df["Area_SqFt"].max())),
            value=int(df["Area_SqFt"].median()), step=50,
        )
        rooms = st.slider(
            "Number of rooms",
            min_value=int(max(1, df["Rooms"].min())),
            max_value=int(max(df["Rooms"].max(), 8)),
            value=int(df["Rooms"].median()), step=1,
        )
        build_year = st.slider(
            "Build year",
            min_value=int(df["Build_Year"].min()),
            max_value=int(df["Build_Year"].max()),
            value=int(df["Build_Year"].median()), step=1,
        )
        has_pool = st.radio("Has pool?", sorted(df["Has_Pool"].dropna().unique()), horizontal=True)
    with input_right:
        location = st.selectbox("Location", sorted(df["Location"].dropna().unique()))
        street_type = st.selectbox("Street type", sorted(df["Street_Type"].dropna().unique()))
        furnishing = st.selectbox("Furnishing", sorted(df["Furnishing"].dropna().unique()))
        property_type = st.selectbox("Property type", sorted(df["Property_Type"].dropna().unique()))

    input_row = pd.DataFrame([{
        "Area_SqFt": float(area),
        "Rooms": float(rooms),
        "Build_Year": int(build_year),
        "Location": location,
        "Street_Type": street_type,
        "Furnishing": furnishing,
        "Property_Type": property_type,
        "Has_Pool": has_pool,
    }])

    selected_model = models[selected_model_name]
    predicted_price = max(float(selected_model.predict(input_row)[0]), 0)
    st.metric(
        f"Predicted price using {selected_model_name}",
        _money(predicted_price),
        delta=f"{_money(predicted_price - df[TARGET].median())} vs dataset median",
    )

    # All-model comparison for the same house.
    rows = [{"Model": n, "Predicted Price": max(float(m.predict(input_row)[0]), 0)}
            for n, m in models.items()]
    cmp_df = pd.DataFrame(rows).sort_values("Predicted Price")
    display_cmp = cmp_df.copy()
    display_cmp["Predicted Price"] = display_cmp["Predicted Price"].map(_money)
    st.markdown("#### Predictions from all trained models")
    st.dataframe(display_cmp, use_container_width=True, hide_index=True)

    fig_pred, ax_pred = plt.subplots(figsize=(8, 4))
    ax_pred.barh(cmp_df["Model"], cmp_df["Predicted Price"])
    ax_pred.set_xlabel("Predicted price ($)")
    ax_pred.set_title("Same house, different model predictions")
    st.pyplot(fig_pred)

    # ── Diagnostics ────────────────────────────────────────────────
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
        mn = float(min(y_test.min(), y_pred.min()))
        mx = float(max(y_test.max(), y_pred.max()))
        ax_actual.plot([mn, mx], [mn, mx], linestyle="--")
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

    # ── Importance ────────────────────────────────────────────────
    _section("6. Important variables for the selected model",
             "This quick explanation helps connect the model prediction to house features.")
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
