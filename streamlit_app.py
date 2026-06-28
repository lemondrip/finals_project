import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
from streamlit_shap import st_shap
import wandb


st.set_page_config(page_title="Housing Price Prediction", layout="wide")

try:
    wandb.login(key="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")   # EDIT: paste your W&B API key
except Exception:
    pass


def _max_width_():
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container {
            max-width: 1100px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_max_width_()


st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

model_mode = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest"])

app_mode = st.sidebar.selectbox(
    "Select Page",
    ["Introduction", "Visualization", "Prediction", "Feature Importance", "W&B Tracking", "Conclusion"],
)


@st.cache_data
def load_data():
    return pd.read_csv("dataset_2.csv").dropna()


def get_xy(df):
    y = df["Price"]
    X = df.drop(columns=["Price"])
    cats = X.select_dtypes(exclude=["number"]).columns.tolist()
    if cats:
        X = pd.get_dummies(X, columns=cats, drop_first=True)
    return X, y


def scale(X):
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)


def train(model_mode, X, y, params=None):
    params = params or {}
    if model_mode == "Linear Regression":
        m = LinearRegression()
    elif model_mode == "Gradient Boosting":
        m = GradientBoostingRegressor(random_state=42, **params)
    else:
        m = RandomForestRegressor(random_state=42, **params)
    m.fit(X, y)
    return m


def scores(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


if app_mode == "Introduction":
    st.title("Housing Price Prediction")

    st.markdown("## Business Case")
    st.markdown(
        """
        Buyers, sellers, and agents struggle to price homes fairly. Mispricing means
        properties sit unsold or owners leave money on the table.

        A reliable estimate reduces negotiation friction, speeds up transactions, and
        protects both sides from over- or under-paying.

        We use linear regression, with a Random Forest baseline for comparison, to
        predict house price from attributes such as area, rooms, build year, location,
        and furnishing.
        """
    )

    st.markdown("## Data Presentation")
    df = load_data()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    st.subheader("Statistical Description")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())


if app_mode == "Visualization":
    st.title("Visualization")

    df = load_data()
    num_df = df.select_dtypes(include=["number"])
    list_vars = list(num_df.columns)

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Scatter vs Price", "Correlation", "Pairplot"])

    with tab1:
        var = st.selectbox("Select a variable", list_vars, key="dist_var")
        fig, ax = plt.subplots()
        sns.histplot(df[var], kde=True, ax=ax)
        ax.set_title("Distribution of " + var)
        st.pyplot(fig)

    with tab2:
        feature = st.selectbox("Select a feature", [c for c in list_vars if c != "Price"], key="scatter_x")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feature, y="Price", ax=ax)
        sns.regplot(data=df, x=feature, y="Price", scatter=False, color="red", ax=ax)
        ax.set_title(feature + " vs Price")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    with tab4:
        chosen = st.multiselect("Select up to 5 variables", list_vars, default=list_vars[: min(4, len(list_vars))])
        if 2 <= len(chosen) <= 5:
            sample = num_df[chosen].sample(min(300, len(num_df)), random_state=42)
            st.pyplot(sns.pairplot(sample))
        else:
            st.info("Pick between 2 and 5 variables.")


if app_mode == "Prediction":
    st.title("Prediction")

    df = load_data()
    X, y = get_xy(df)

    features = st.multiselect("Select Features", list(X.columns), default=list(X.columns))
    if not features:
        st.warning("Select at least one feature.")
        st.stop()
    X = X[features]

    train_size = st.sidebar.number_input("Train Size", 0.1, 0.9, 0.7, 0.05)
    do_scale = st.sidebar.checkbox("Standardize features", value=True)
    track_wandb = st.checkbox("Track this run with W&B")

    if not st.button("Train Model"):
        st.stop()

    if do_scale:
        X = scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    model = train(model_mode, X_train, y_train)
    preds = model.predict(X_test)
    s = scores(y_test, preds)

    st.subheader("Results - " + model_mode)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{s['MAE']:,.0f}")
    c2.metric("RMSE", f"{s['RMSE']:,.0f}")
    c3.metric("R2", f"{s['R2']:.3f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, preds, alpha=0.5)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    if track_wandb:
        wandb.init(
            project="group-project",
            entity="YOUR_WANDB_USERNAME",   # EDIT: set your W&B username
            name=model_mode,
            config={"model": model_mode, "features": features, "train_size": train_size},
            reinit=True,
        )
        wandb.log(s)
        wandb.finish()
        st.success("Logged to W&B")


if app_mode == "Feature Importance":
    st.title("Feature Importance")

    df = load_data()
    X, y = get_xy(df)
    X = scale(X)

    model = train(model_mode, X, y)

    st.subheader("Driving Variables")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X.columns)
    else:
        imp = pd.Series(np.abs(model.coef_), index=X.columns)
    imp = imp.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax)
    ax.set_title("Top Feature Importances")
    st.pyplot(fig)

    st.subheader("SHAP Explainability")
    try:
        sample = X.sample(min(100, len(X)), random_state=42)
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)

        st.markdown("**Summary (beeswarm)**")
        st_shap(shap.plots.beeswarm(shap_values), height=500)

        st.markdown("**Single prediction (waterfall)**")
        st_shap(shap.plots.waterfall(shap_values[0]), height=500)
    except Exception as e:
        st.error("SHAP failed: " + str(e))


if app_mode == "W&B Tracking":
    st.title("Weights and Biases - Hyperparameter Tuning")

    st.markdown("Run a small sweep over Random Forest / Gradient Boosting hyperparameters, log each run to W&B, and pick the best R2.")

    df = load_data()
    X, y = get_xy(df)
    X = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    sweep_model = st.selectbox("Model to tune", ["Random Forest", "Gradient Boosting"])
    n_estimators_grid = st.multiselect("n_estimators", [50, 100, 200, 300], default=[100, 200])
    max_depth_grid = st.multiselect("max_depth", [3, 5, 10, None], default=[5, 10])

    if st.button("Run Sweep"):
        results = []
        progress = st.progress(0.0)
        combos = [(n, d) for n in n_estimators_grid for d in max_depth_grid]

        for i, (n, d) in enumerate(combos):
            params = {"n_estimators": n, "max_depth": d}
            model = train(sweep_model, X_train, y_train, params)
            preds = model.predict(X_test)
            s = scores(y_test, preds)
            results.append({"n_estimators": n, "max_depth": d, **s})

            wandb.init(
                project="group-project",
                entity="YOUR_WANDB_USERNAME",   # EDIT: set your W&B username
                name=f"{sweep_model}-n{n}-d{d}",
                config={"model": sweep_model, **params},
                reinit=True,
            )
            wandb.log(s)
            wandb.finish()

            progress.progress((i + 1) / len(combos))

        res_df = pd.DataFrame(results).sort_values("R2", ascending=False)
        st.subheader("Sweep Results")
        st.dataframe(res_df.reset_index(drop=True))

        best = res_df.iloc[0]
        st.success(f"Best: {sweep_model} | n_estimators={best['n_estimators']} | max_depth={best['max_depth']} | R2={best['R2']:.3f}")


if app_mode == "Conclusion":
    st.title("Conclusion")
    st.markdown(
        """
        ### What we learned
        - Area is the strongest driver of price; location and property type matter too.
        - State which model performed best here and its R2 / error.
        - Note any surprising relationships from the visualizations.

        ### Business impact
        - Agents and buyers get a fast, defensible price estimate before negotiation.

        ### Limitations and next steps
        - Add features like neighbourhood amenities, condition, or sale date.
        """
    )


st.markdown("---")
st.markdown("### Made by Abhi, Eric, Jessie, and Joe")   # EDIT: put your team name here