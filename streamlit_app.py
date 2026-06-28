import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk


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
    wandb.login(key="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")   
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

#Introduction Page

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

    st.subheader("Statistical Description")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())


if app_mode == "Visualization":
    st.title("Visualization")

    df = load_data()
    num_df = df.select_dtypes(include=["number"])
    list_vars = list(num_df.columns)

    tab_corr, tab_dist, tab_scatter, tab_pair, tab_pie, tab_map = st.tabs(
        ["Correlation", "Distribution", "Scatter vs Price", "Pairplot", "Pie Charts", "Map"]
    )

    with tab_corr:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    with tab_dist:
        var = st.selectbox("Select a variable", list_vars, key="dist_var")
        fig, ax = plt.subplots()
        sns.histplot(df[var], kde=True, ax=ax)
        ax.set_title("Distribution of " + var)
        st.pyplot(fig)

    with tab_scatter:
        feature = st.selectbox("Select a feature", [c for c in list_vars if c != "Price"], key="scatter_x")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feature, y="Price", ax=ax)
        sns.regplot(data=df, x=feature, y="Price", scatter=False, color="red", ax=ax)
        ax.set_title(feature + " vs Price")
        st.pyplot(fig)

    with tab_pair:
        chosen = st.multiselect("Select up to 5 variables", list_vars, default=list_vars[: min(4, len(list_vars))])
        if 2 <= len(chosen) <= 5:
            sample = num_df[chosen].sample(min(300, len(num_df)), random_state=42)
            st.pyplot(sns.pairplot(sample))
        else:
            st.info("Pick between 2 and 5 variables.")

    with tab_pie:
        cat_cols = ["Street_Type", "Furnishing", "Property_Type", "Has_Pool"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for ax, col in zip(axes.ravel(), cat_cols):
            counts = df[col].value_counts()
            ax.pie(
                counts, labels=counts.index, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white"},
            )
            ax.set_title(col)
        fig.tight_layout()
        st.pyplot(fig)

    with tab_map:
        st.markdown("Heatmap across the 8 cities in the dataset. **Red = high, green = low**, relative to the chosen metric.")

        CITY_COORDS = {
            "Delhi": (28.6139, 77.2090), "Gurugram": (28.4595, 77.0266),
            "Noida": (28.5355, 77.3910), "Jaipur": (26.9124, 75.7873),
            "Lucknow": (26.8467, 80.9462), "Kanpur": (26.4499, 80.3319),
            "Prayagraj": (25.4358, 81.8463), "Indore": (22.7196, 75.8577),
        }

        metric = st.selectbox(
            "Color the map by",
            [
                "Average price (most expensive)",
                "Oldest homes (avg build year)",
                "Largest homes (avg area)",
                "Most rooms (avg rooms)",
                "Number of listings",
                "Share with a pool",
            ],
            key="map_metric",
        )

        g = df.groupby("Location")
        if metric == "Average price (most expensive)":
            val, hotter_high, fmt = g["Price"].mean(), True, lambda v: f"{v:,.0f}"
        elif metric == "Oldest homes (avg build year)":
            val, hotter_high, fmt = g["Build_Year"].mean(), False, lambda v: f"{v:.0f}"
        elif metric == "Largest homes (avg area)":
            val, hotter_high, fmt = g["Area_SqFt"].mean(), True, lambda v: f"{v:,.0f} sqft"
        elif metric == "Most rooms (avg rooms)":
            val, hotter_high, fmt = g["Rooms"].mean(), True, lambda v: f"{v:.1f}"
        elif metric == "Number of listings":
            val, hotter_high, fmt = g.size(), True, lambda v: f"{v:.0f}"
        else:
            val, hotter_high, fmt = g["Has_Pool"].apply(lambda s: (s == "Yes").mean()), True, lambda v: f"{v*100:.0f}%"

        mp = pd.DataFrame({"Location": val.index, "value": val.values})
        mp["lat"] = mp["Location"].map(lambda c: CITY_COORDS[c][0])
        mp["lon"] = mp["Location"].map(lambda c: CITY_COORDS[c][1])
        mp["display"] = mp["value"].map(fmt)

        lo, hi = mp["value"].min(), mp["value"].max()
        norm = (mp["value"] - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=mp.index)
        if not hotter_high:
            norm = 1 - norm
        mp["weight"] = 0.15 + 0.85 * norm   # floor keeps the coolest city visible

        COLOR_RANGE = [
            [26, 152, 80], [145, 207, 96], [217, 239, 139],
            [254, 224, 139], [252, 141, 89], [215, 48, 39],
        ]

        heat = pdk.Layer(
            "HeatmapLayer", data=mp, get_position="[lon, lat]",
            get_weight="weight", radius_pixels=90, intensity=1,
            threshold=0.05, color_range=COLOR_RANGE,
        )
        dots = pdk.Layer(
            "ScatterplotLayer", data=mp, get_position="[lon, lat]",
            get_radius=12000, get_fill_color=[255, 255, 255, 40], pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[heat, dots],
                initial_view_state=pdk.ViewState(latitude=26.8, longitude=78.5, zoom=4.3),
                map_provider="carto",
                map_style="light",
                tooltip={"html": "<b>{Location}</b><br/>" + metric + ": {display}"},
            )
        )
        st.caption("Coloring is relative within the selected metric (min = green, max = red), not an absolute scale.")

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
st.markdown("### Made by Abhi, Eric, Jessie, and Joe")