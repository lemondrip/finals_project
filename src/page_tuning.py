import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data_loader import dataset_selector, get_target, get_features
from src import wandb_tracker

# ──────────────────────────────────────────────────────────────────────
#  WEIGHTS & BIASES API KEY  ── where you add your token
#  ----------------------------------------------------------------------
#  Do NOT put the key in this file (the repo is public on GitHub/HF).
#  wandb_tracker reads it from the environment — add it as a SECRET:
#     HF Space → Settings → Variables and secrets → New secret
#        Name:  WANDB_API_KEY     Value: <key from https://wandb.ai/authorize>
#     Local dev:  export WANDB_API_KEY=your_key   (before streamlit run)
#  With no key set, tuning still runs — it just skips W&B logging.
# ──────────────────────────────────────────────────────────────────────

TEAL = "#0D9488"
TEAL_BRIGHT = "#2DD4BF"

# Linear models train on scaled inputs; trees use the raw matrix.
_SCALED_MODELS = {"Linear Regression", "Ridge"}


def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    st.markdown("## ⚙️ Hyperparameter Tuning")
    st.caption(
        "Optimize model hyperparameters using Optuna and track all experiments. "
        "This replaces manual trial-and-error with automated Bayesian search."
    )
    st.markdown("---")

    # ── Data prep (one-hot categoricals + log-price target) ─────────
    X_raw = df[features]
    cat_cols = X_raw.select_dtypes(exclude="number").columns.tolist()
    X_df = pd.get_dummies(X_raw, columns=cat_cols) if cat_cols else X_raw.copy()
    X = X_df.values
    y = df[target].values
    y_log = np.log1p(y)  # learn on log-price; invert with expm1 for $ metrics

    X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
        X, y, y_log, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Config ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox(
            "Model to tune",
            ["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting"],
        )
    with col2:
        n_trials = st.slider("Number of trials", 5, 100, 15, step=5)
    with col3:
        cv_folds = st.slider("CV folds", 3, 10, 5)

    # ── Hyperparameter search spaces ────────────────────────────────
    st.markdown("### 🔧 Search Space")
    search_spaces = {
        "Linear Regression": {
            "fit_intercept": "True / False  (baseline — no continuous hyperparameters)",
        },
        "Ridge": {"alpha": "0.001 — 100"},
        "Random Forest": {
            "n_estimators": "50 — 500",
            "max_depth": "3 — 30",
            "min_samples_split": "2 — 20",
            "min_samples_leaf": "1 — 10",
        },
        "Gradient Boosting": {
            "n_estimators": "50 — 500",
            "max_depth": "2 — 10",
            "learning_rate": "0.01 — 0.3",
            "subsample": "0.6 — 1.0",
            "min_samples_split": "2 — 20",
        },
    }
    st.dataframe(
        pd.DataFrame([{"Parameter": k, "Range": v} for k, v in search_spaces[model_name].items()]),
        use_container_width=True, hide_index=True,
    )
    if model_name == "Linear Regression":
        st.caption(
            "ℹ️ Linear Regression has no continuous hyperparameters to search, so it acts as a "
            "baseline — Optuna only flips `fit_intercept`."
        )

    # ── W&B toggle ──────────────────────────────────────────────────
    track_wandb = st.checkbox(
        "📡 Log study to Weights & Biases",
        value=wandb_tracker.is_available(),
        disabled=not wandb_tracker.is_available(),
        help="Set WANDB_API_KEY as a Space secret to enable.",
    )
    if not wandb_tracker.is_available():
        st.caption("📡 W&B is off — add `wandb` to requirements.txt and set `WANDB_API_KEY` as a Space secret.")

    # ── Run optimization ────────────────────────────────────────────
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            st.error("Install Optuna: add `optuna` to requirements.txt")
            return

        wb_run = None
        if track_wandb:
            wb_run = wandb_tracker.init_run(
                run_name=f"{ds_key}-tune-{model_name}",
                config={
                    "dataset": ds_key,
                    "model": model_name,
                    "n_trials": n_trials,
                    "cv_folds": cv_folds,
                    "target": target,
                    "n_features": int(X.shape[1]),
                },
                job_type="hparam-search",
            )

        scaled = model_name in _SCALED_MODELS
        X_obj = X_train_s if scaled else X_train

        def objective(trial):
            if model_name == "Linear Regression":
                model = LinearRegression(
                    fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
                )
            elif model_name == "Ridge":
                model = Ridge(alpha=trial.suggest_float("alpha", 0.001, 100, log=True))
            elif model_name == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 500),
                    max_depth=trial.suggest_int("max_depth", 3, 30),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=42, n_jobs=-1,
                )
            else:  # Gradient Boosting
                model = GradientBoostingRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 500),
                    max_depth=trial.suggest_int("max_depth", 2, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    random_state=42,
                )
            scores = cross_val_score(model, X_obj, y_train_log, cv=cv_folds, scoring="r2")
            return scores.mean()

        progress = st.progress(0, text="Optimizing...")
        live_log = st.empty()
        log_lines: list[str] = []
        study = optuna.create_study(direction="maximize", study_name=model_name)

        def callback(study, trial):
            progress.progress(
                (trial.number + 1) / n_trials,
                text=f"Trial {trial.number + 1}/{n_trials} — Best R²: {study.best_value:.4f}",
            )
            score = trial.value if trial.value is not None else float("nan")
            params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
            log_lines.append(
                f"Trial {trial.number + 1:>3}/{n_trials} │ R²={score:.4f} │ best={study.best_value:.4f} │ {params_str}"
            )
            live_log.code("\n".join(log_lines[-15:]), language="text")
            wandb_tracker.log_metrics(wb_run, {
                "trial/r2": score if score == score else 0.0,
                "trial/best_r2": study.best_value,
            }, step=trial.number)

        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        progress.empty()

        # ── Rebuild + evaluate best model on the test set ───────────
        best_params = dict(study.best_params)
        X_fit = X_train_s if scaled else X_train
        X_eval = X_test_s if scaled else X_test

        if model_name == "Linear Regression":
            best_model = LinearRegression(**best_params)
        elif model_name == "Ridge":
            best_model = Ridge(**best_params)
        elif model_name == "Random Forest":
            best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        else:
            best_model = GradientBoostingRegressor(**best_params, random_state=42)

        best_model.fit(X_fit, y_train_log)
        y_pred = np.expm1(best_model.predict(X_eval))   # back to dollars
        y_pred = np.maximum(y_pred, 0)

        # Store results
        trials_data = []
        for t in study.trials:
            row = {"Trial": t.number, "R² (CV)": t.value}
            row.update(t.params)
            trials_data.append(row)

        st.session_state["tune_study"] = study
        st.session_state["tune_trials"] = pd.DataFrame(trials_data)
        st.session_state["tune_best_params"] = best_params
        st.session_state["tune_test_metrics"] = {
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }
        st.session_state["tune_y_test"] = y_test
        st.session_state["tune_y_pred"] = y_pred
        st.session_state["tune_model_name"] = model_name
        st.session_state["tune_ready"] = True

        if wb_run is not None:
            wandb_tracker.log_metrics(wb_run, {
                "final/best_cv_r2": study.best_value,
                "final/test_r2": st.session_state["tune_test_metrics"]["R²"],
                "final/test_mae": st.session_state["tune_test_metrics"]["MAE"],
                "final/test_rmse": st.session_state["tune_test_metrics"]["RMSE"],
            })
            try:
                wb_run.summary["best_params"] = {
                    k: v for k, v in best_params.items() if isinstance(v, (int, float, str, bool))
                }
            except Exception:
                pass
            wandb_tracker.finish_run(wb_run)

    # ── Display results ─────────────────────────────────────────────
    if not st.session_state.get("tune_ready"):
        st.info("Click **Start Optimization** to begin hyperparameter search.")
        return

    trials_df = st.session_state["tune_trials"]
    best_params = st.session_state["tune_best_params"]
    test_metrics = st.session_state["tune_test_metrics"]
    y_test = st.session_state["tune_y_test"]
    y_pred = st.session_state["tune_y_pred"]
    tuned_model = st.session_state["tune_model_name"]

    st.markdown("---")

    # ── Best parameters ─────────────────────────────────────────────
    st.markdown("### 🏆 Best Hyperparameters")
    st.success(f"**{tuned_model}** — Best CV R² (log-price): {st.session_state['tune_study'].best_value:.4f}")

    param_cols = st.columns(max(len(best_params), 1))
    for i, (k, v) in enumerate(best_params.items()):
        with param_cols[i]:
            display_val = f"{v:.4f}" if isinstance(v, float) else str(v)
            st.metric(k, display_val)

    # ── Test set performance ────────────────────────────────────────
    st.markdown("### 📈 Test Set Performance (Best Model)")
    m_cols = st.columns(3)
    m_cols[0].metric("R²", f"{test_metrics['R²']:.3f}")
    m_cols[1].metric("MAE", f"${test_metrics['MAE']:,.0f}")
    m_cols[2].metric("RMSE", f"${test_metrics['RMSE']:,.0f}")

    st.markdown("---")

    # ── Optimization history ────────────────────────────────────────
    st.markdown("### 📉 Optimization History")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        best_so_far = trials_df["R² (CV)"].cummax()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trials_df["Trial"], y=trials_df["R² (CV)"],
            mode="markers", name="Trial score",
            marker=dict(color=TEAL_BRIGHT, size=6, opacity=0.6),
        ))
        fig.add_trace(go.Scatter(
            x=trials_df["Trial"], y=best_so_far,
            mode="lines", name="Best so far",
            line=dict(color=TEAL, width=3),
        ))
        fig.update_layout(
            height=400, title="Optimization Progress",
            xaxis_title="Trial", yaxis_title="R² (CV)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_h2:
        fig = px.scatter(
            x=y_test, y=y_pred, opacity=0.4,
            color_discrete_sequence=[TEAL_BRIGHT],
            title=f"Best {tuned_model} — Actual vs Predicted",
            labels={"x": "Actual ($)", "y": "Predicted ($)"},
        )
        mn = float(min(y_test.min(), y_pred.min()))
        mx = float(max(y_test.max(), y_pred.max()))
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", line=dict(color="#F43F5E", dash="dash"),
            showlegend=False,
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Parallel coordinates ────────────────────────────────────────
    st.markdown("### 🔀 Hyperparameter Exploration")
    param_names = [c for c in trials_df.columns if c not in ("Trial", "R² (CV)")]
    numeric_params = [p for p in param_names
                      if pd.to_numeric(trials_df[p], errors="coerce").notna().any()]
    if len(numeric_params) >= 2:
        dims = [dict(label="R² (CV)", values=trials_df["R² (CV)"])]
        for p in numeric_params:
            dims.append(dict(label=p, values=pd.to_numeric(trials_df[p], errors="coerce")))
        fig = go.Figure(go.Parcoords(
            line=dict(
                color=trials_df["R² (CV)"], colorscale="Teal", showscale=True,
                cmin=trials_df["R² (CV)"].min(), cmax=trials_df["R² (CV)"].max(),
            ),
            dimensions=dims,
        ))
        fig.update_layout(height=500, title="Parallel Coordinates — All Trials")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Parallel coordinates need at least two numeric hyperparameters (RF / Gradient Boosting).")

    # ── Experiment log ──────────────────────────────────────────────
    st.markdown("### 📋 Full Experiment Log")
    st.dataframe(
        trials_df.sort_values("R² (CV)", ascending=False).reset_index(drop=True),
        use_container_width=True, height=400,
    )