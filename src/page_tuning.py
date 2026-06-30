"""
Page 5 — Hyperparameter Tuning
================================
Automated hyperparameter optimization using Optuna,
with experiment tracking (Weights & Biases) and visualization.

Mirrors the data handling of the Prediction page: features arrive already
numeric from data_loader, are used as `df[features].values`, and are optionally
standardized — so the hyperparameters found here describe the same model the
Prediction page trains.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data_loader import dataset_selector, get_target, get_features
from src import wandb_tracker

RANDOM_STATE = 42

# Tunable models — names kept consistent with the Prediction page.
TUNABLE_MODELS = [
    "🧠 MLP (Neural Net)",
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
]

# Human-readable search spaces (for the summary table).
SEARCH_SPACES = {
    "🧠 MLP (Neural Net)": {
        "n_hidden_layers": "1 — 4",
        "neurons_per_layer": "16 — 256",
        "activation": "relu, tanh, logistic",
        "learning_rate_init": "0.0001 — 0.01",
        "alpha (L2 penalty)": "0.0001 — 0.1",
        "batch_size": "16 — 128",
        "max_iter": "200 — 1000",
    },
    "Ridge Regression": {"alpha": "0.001 — 100"},
    "Lasso Regression": {"alpha": "0.001 — 100"},
    "Elastic Net": {"alpha": "0.001 — 100", "l1_ratio": "0.0 — 1.0"},
    "Decision Tree": {
        "max_depth": "2 — 30",
        "min_samples_split": "2 — 20",
        "min_samples_leaf": "1 — 10",
        "max_features": "sqrt, log2, None",
    },
    "Random Forest": {
        "n_estimators": "50 — 500",
        "max_depth": "3 — 30",
        "min_samples_split": "2 — 20",
        "min_samples_leaf": "1 — 10",
        "max_features": "sqrt, log2, None",
    },
    "Gradient Boosting": {
        "n_estimators": "50 — 500",
        "max_depth": "2 — 10",
        "learning_rate": "0.01 — 0.3",
        "subsample": "0.6 — 1.0",
        "min_samples_split": "2 — 20",
    },
}


def _build_estimator(model_name: str, trial):
    """Construct the sklearn estimator for this trial's sampled hyperparameters."""
    if model_name == "🧠 MLP (Neural Net)":
        n_layers = trial.suggest_int("n_hidden_layers", 1, 4)
        hidden_layers = tuple(
            trial.suggest_int(f"neurons_layer_{i}", 16, 256, log=True)
            for i in range(n_layers)
        )
        return MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            alpha=trial.suggest_float("alpha", 1e-4, 0.1, log=True),
            batch_size=trial.suggest_int("batch_size", 16, 128, log=True),
            max_iter=trial.suggest_int("max_iter", 200, 1000, step=100),
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
        )
    if model_name == "Ridge Regression":
        return Ridge(alpha=trial.suggest_float("alpha", 0.001, 100, log=True))
    if model_name == "Lasso Regression":
        return Lasso(alpha=trial.suggest_float("alpha", 0.001, 100, log=True))
    if model_name == "Elastic Net":
        return ElasticNet(
            alpha=trial.suggest_float("alpha", 0.001, 100, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
        )
    if model_name == "Decision Tree":
        return DecisionTreeRegressor(
            max_depth=trial.suggest_int("max_depth", 2, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=RANDOM_STATE,
        )
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    # Gradient Boosting
    return GradientBoostingRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        random_state=RANDOM_STATE,
    )


def _rebuild_best(model_name: str, params: dict):
    """Rebuild the winning estimator from study.best_params and return
    (estimator, display_params) ready for a final fit/predict."""
    params = dict(params)
    if model_name == "🧠 MLP (Neural Net)":
        n_layers = params.pop("n_hidden_layers")
        hidden_layers = tuple(params.pop(f"neurons_layer_{i}") for i in range(n_layers))
        for k in list(params.keys()):
            if k.startswith("neurons_layer_"):
                params.pop(k)
        est = MLPRegressor(
            hidden_layer_sizes=hidden_layers, **params,
            random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1,
        )
        display = dict(params)
        display["architecture"] = " → ".join(str(n) for n in hidden_layers)
        return est, display
    if model_name == "Ridge Regression":
        return Ridge(**params), params
    if model_name == "Lasso Regression":
        return Lasso(**params), params
    if model_name == "Elastic Net":
        return ElasticNet(**params), params
    if model_name == "Decision Tree":
        return DecisionTreeRegressor(**params, random_state=RANDOM_STATE), params
    if model_name == "Random Forest":
        return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1), params
    return GradientBoostingRegressor(**params, random_state=RANDOM_STATE), params


def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## ⚙️ Hyperparameter Tuning")
    st.caption(
        "Optimize model hyperparameters with Optuna and track every experiment. "
        "This replaces manual trial-and-error with automated Bayesian search."
    )
    st.markdown("---")

    # ── Config ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox("Model to tune", TUNABLE_MODELS)
    with col2:
        n_trials = st.slider("Number of trials", 5, 100, 20, step=5)
    with col3:
        cv_folds = st.slider("CV folds", 3, 10, 5)

    col_f, col_s = st.columns([3, 1])
    with col_f:
        selected_features = st.multiselect(
            "Explanatory variables", features, default=features,
        )
    with col_s:
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        scale_data = st.checkbox("Standardize features", value=True)

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    # ── Prepare data (mirrors the Prediction page) ──────────────────
    X = df[selected_features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    st.markdown(
        f"**Training set:** {len(X_train):,} samples · "
        f"**Test set:** {len(X_test):,} samples · "
        f"**Features:** {len(selected_features)}"
    )

    # ── Search space ────────────────────────────────────────────────
    st.markdown("### 🔧 Search Space")
    space_df = pd.DataFrame(
        [{"Parameter": k, "Range": v} for k, v in SEARCH_SPACES[model_name].items()]
    )
    st.dataframe(space_df, use_container_width=True, hide_index=True)

    # ── MLP architecture preview ────────────────────────────────────
    if model_name == "🧠 MLP (Neural Net)":
        st.markdown("### 🏗️ Neural Network Architecture Preview")
        st.markdown(
            "The MLP is a fully-connected feedforward network. Optuna searches over "
            "the number of hidden layers, neurons per layer, activation, learning rate, "
            "regularization, and batch size."
        )
        st.markdown(
            "```\n"
            "Input Layer ──▶ Hidden Layer(s) ──▶ Output Layer\n"
            "  (features)    (relu/tanh/logistic)   (prediction)\n"
            "```"
        )

    # ── W&B toggle ──────────────────────────────────────────────────
    track_wandb = st.checkbox(
        "📡 Log study to Weights & Biases",
        value=wandb_tracker.is_available(),
        disabled=not wandb_tracker.is_available(),
        help="Set WANDB_API_KEY in .env to enable.",
    )

    # ── Run optimization ────────────────────────────────────────────
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            st.error("Install Optuna: `pip install optuna`")
            return

        wb_run = None
        if track_wandb:
            wb_run = wandb_tracker.init_run(
                run_name=f"{ds_key}-tune-{model_name}",
                config={
                    "dataset": ds_key,
                    "model": model_name,
                    "target": target,
                    "n_features": len(selected_features),
                    "features": selected_features,
                    "n_trials": n_trials,
                    "cv_folds": cv_folds,
                    "test_size": test_size,
                    "scale_data": scale_data,
                },
                job_type="hparam-search",
            )

        def objective(trial):
            model = _build_estimator(model_name, trial)
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="r2")
            return scores.mean()

