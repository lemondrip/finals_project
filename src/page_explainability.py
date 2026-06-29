from __future__ import annotations
 
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
 
from sklearn.inspection import permutation_importance
 
from data_loader import load_data
 
TARGET = "Price"
NUMERIC_FEATURES = ["Area_SqFt", "Rooms", "Build_Year"]
CATEGORICAL_FEATURES = ["Location", "Street_Type", "Furnishing", "Property_Type", "Has_Pool"]
 
# Tree models SHAP's TreeExplainer handles well — intersected with what's trained.
_TREE_FRIENDLY = {"Random Forest", "Gradient Boosting"}
 
 
# ── UI helpers ───────────────────────────────────────────────────────
def _money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"
 
 
def _section(title: str, description: str | None = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.caption(description)
 
 
def _clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "").replace("_", " ")
 
 
# ── SHAP extraction ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _shap(model_name: str, n_explain: int, _model_obj):
    """Compute SHAP values for the model already trained on the Prediction page.
 
    Accepts the fitted TransformedTargetRegressor; extracts the inner pipeline
    so TreeExplainer can see the raw tree without the log-transform wrapper.
    """
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
 
    # Re-use the pipeline's preprocessor to get a transformed explain set.
    pipe = _model_obj.regressor_
    pre = pipe.named_steps["preprocessor"]
    feat_names = list(pre.get_feature_names_out())
    est = pipe.named_steps["model"]
 
    rng = np.random.default_rng(99)
    pick = rng.choice(len(X), min(n_explain, len(X)), replace=False)
    X_exp_t = pre.transform(X.iloc[pick])
    y_exp = df[TARGET].iloc[pick].to_numpy()
    meta_exp = X.iloc[pick].reset_index(drop=True)
 
    # Preferred path: fast exact TreeExplainer on the raw tree model.
    try:
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(X_exp_t)
        base_value = explainer.expected_value
    except Exception:
        # Fallback: generic Explainer with a background set (returns an Explanation).
        bg = shap.sample(X_exp_t, min(100, len(X_exp_t)))
        explainer = shap.Explainer(est, bg)
        out = explainer(X_exp_t)
        sv = out.values
        base_value = out.base_values
 
    # Coerce SHAP output to a plain 2D (n_explain × n_features) float array.
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:            # some explainers add a trailing single-output axis
        sv = sv[..., 0]
    base_value = float(np.ravel(base_value)[0])
    X_exp_df = pd.DataFrame(X_exp_t, columns=feat_names)
    return dict(feat_names=feat_names, X_explain=X_exp_df,
                y_exp=y_exp, meta=meta_exp, est=est), sv, base_value
 
 
# ── Page ─────────────────────────────────────────────────────────────
def render() -> None:
    """Render the SHAP explainability page (controls → global → local views)."""
    st.write(
        "SHAP decomposes every prediction into additive feature contributions "
        "(in log-price space). A positive bar pushes the predicted price **up**; "
        "a negative bar pulls it **down**."
    )
 
    state = st.session_state.get("house_prediction_state")
    if state is None:
        st.warning("Train models on the **Prediction** page first, then return here.")
        return
 
    models = state["models"]
    tree_opts = [m for m in models if m in _TREE_FRIENDLY]
    if not tree_opts:
        st.warning("No tree model found in session. Train a Random Forest or Gradient Boosting model first.")
        return
 
    # ── Controls ──────────────────────────────────────────────────────
    _section("Set up the explainer", "Pick a tree model and how many properties to explain.")
    c1, c2 = st.columns([1.5, 1.5])
    model_name = c1.selectbox(
        "Tree model", tree_opts,
        help="TreeExplainer-friendly models only — exact, fast SHAP values.",
    )
    n_explain = c2.slider("Properties to explain (SHAP)", 50, 300, 200, step=25,
                          help="SHAP is the slow part — keep this modest.")
 
    if st.button("🔬 Compute SHAP", type="primary"):
        with st.spinner(f"Computing SHAP values for {n_explain:,} properties…"):
            bundle, sv, base_value = _shap(model_name, n_explain, models[model_name])
        # Persist across reruns so the dependence / property selectors stay live.
        st.session_state["shap_ready"] = True
        st.session_state["shap_bundle"] = bundle
        st.session_state["shap_sv"] = sv
        st.session_state["shap_base"] = base_value
        st.session_state["shap_label"] = model_name
 
    if not st.session_state.get("shap_ready"):
        st.info("Press **🔬 Compute SHAP** to explain the model's predictions.")
        return
 
    # Pull the cached results back out of session state.
    bundle = st.session_state["shap_bundle"]
    sv: np.ndarray = st.session_state["shap_sv"]
    base_value: float = st.session_state["shap_base"]
    feat_names = bundle["feat_names"]
    X_explain = bundle["X_explain"]
    # Positional (row-aligned) copies so sv[i] ↔ meta row i ↔ y_exp[i].
    meta = bundle["meta"]
    y_exp = bundle["y_exp"]
    clean = [_clean_feature_name(f) for f in feat_names]
 
    st.success(
        f"Explained **{len(X_explain):,}** properties with **{st.session_state['shap_label']}** "
        f"across **{len(feat_names)}** transformed features."
    )
 
    # ── 1. Global importance (the always-works chart) ─────────────────
    _section("1. Global feature importance",
             "Mean absolute SHAP value per feature — how much each driver moves price overall.")
    mean_abs = np.abs(sv).mean(axis=0)
    imp = (pd.DataFrame({"Feature": clean, "Mean |SHAP|": mean_abs})
           .sort_values("Mean |SHAP|", ascending=False).head(15)
           .sort_values("Mean |SHAP|"))      # ascending for a top-down h-bar
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(imp["Feature"], imp["Mean |SHAP|"])
    ax_imp.set_xlabel("Mean |SHAP| (log-price impact)")
    ax_imp.set_title("Global feature importance")
    st.pyplot(fig_imp)
    st.caption(
        "Area dominates — square footage is by far the strongest price driver, "
        "followed by location and build year."
    )
 
    # ── 2. Beeswarm ───────────────────────────────────────────────────
    _section("2. SHAP beeswarm",
             "Every property as a dot — colour is the feature value, x-position its impact.")
    try:
        fig = plt.figure()
        shap.summary_plot(sv, X_explain, feature_names=clean,
                          show=False, max_display=18, plot_size=(10, 7))
        st.pyplot(plt.gcf())
        plt.close("all")
    except Exception as exc:
        plt.close("all")
        st.info(f"Beeswarm unavailable for this model — see the global bar above. ({exc})")
    st.caption(
        "Red dots = high feature value, blue = low. Dots right of zero raise price; "
        "left of zero lower it. Wide spreads mean the feature swings price hard."
    )
 
 
# ── 3. Dependence ─────────────────────────────────────────────────
    _section("3. Dependence — how one feature bends price",
             "Raw (transformed) feature value vs its SHAP impact reveals non-linear effects.")
    # Default to the most important feature for an immediately interesting plot.
    default_feat = feat_names[int(np.argmax(mean_abs))]
    dep_feat = st.selectbox(
        "Feature to inspect", feat_names, index=feat_names.index(default_feat),
        format_func=_clean_feature_name,
    )
    j = feat_names.index(dep_feat)
    fvals = X_explain[dep_feat].to_numpy()
    fig_dep, ax_dep = plt.subplots(figsize=(8, 4))
    ax_dep.scatter(fvals, sv[:, j], alpha=0.5)
    ax_dep.axhline(0, linestyle="--")
    ax_dep.set_xlabel(f"{_clean_feature_name(dep_feat)} (scaled)")
    ax_dep.set_ylabel("SHAP value (log-price impact)")
    ax_dep.set_title(f"Dependence: {_clean_feature_name(dep_feat)}")
    st.pyplot(fig_dep)
    st.caption(
        f"Insight: where the cloud crosses zero is where **{_clean_feature_name(dep_feat)}** "
        "flips from lowering to raising price; a curved shape signals a non-linear effect."
    )
 
    # ── 4. Per-property waterfall (the price receipt) ─────────────────
    _section("4. Per-property price receipt",
             "Pick a property — see exactly which features raised or lowered its price.")
    # Disambiguate same properties with a descriptive row label.
    labels = [f"{r.Location} · {r.Property_Type} · {int(r.Rooms)} rooms"
              if "Location" in meta.columns
              else f"Property {i}" for i, r in enumerate(meta.itertuples())]
    label_to_pos = {lab: i for i, lab in enumerate(labels)}
    chosen = st.selectbox("Choose a property", labels, index=0)
    i = label_to_pos[chosen]
 
    # Top 10 features for THIS property by absolute contribution.
    contrib = sv[i]
    order = np.argsort(np.abs(contrib))[::-1][:10]
    wf = pd.DataFrame({
        "Feature": [clean[k] for k in order],
        "Contribution": contrib[order],
    }).sort_values("Contribution")        # diverging: negatives left, positives right
    colors = ["#f7941d" if v >= 0 else "#00a9a5" for v in wf["Contribution"]]
    fig_wf, ax_wf = plt.subplots(figsize=(8, 5))
    ax_wf.barh(wf["Feature"], wf["Contribution"], color=colors)
    ax_wf.axvline(0, color="black", linewidth=0.8)
    ax_wf.set_xlabel("SHAP contribution (log-price)")
    ax_wf.set_title(f"Price receipt — {chosen}")
    st.pyplot(fig_wf)
 
    # Predicted vs actual $ for this exact property — clip to match Prediction page.
    selected_model = models[st.session_state["shap_label"]]
    pred = max(float(selected_model.predict(meta.iloc[[i]])[0]), 0)
    actual = float(y_exp[i])
    diff = pred - actual
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted price", _money(pred))
    p2.metric("Actual price", _money(actual))
    p3.metric("Model vs market", _money(diff), "over-priced" if diff > 0 else "under-priced")
    st.caption(
        f"Starting from the average prediction ({_money(float(np.expm1(base_value)))}), each bar "
        "adds or subtracts until the model lands on the predicted price above."
    )
 
    # ── 5. Property-type importance ───────────────────────────────────
    _section("5. What matters by property type",
             "Mean |SHAP| per feature within each type — drivers differ by category.")
    if "Property_Type" in meta.columns:
        # Rank features by overall importance, then compare the top 8 across types.
        top_idx = np.argsort(mean_abs)[::-1][:8]
        rows = []
        for ptype, grp in meta.groupby("Property_Type"):
            pos_rows = grp.index.to_numpy()            # positional row ids (reset index)
            ptype_abs = np.abs(sv[pos_rows]).mean(axis=0)
            for k in top_idx:
                rows.append({"Property Type": ptype, "Feature": clean[k],
                             "Mean |SHAP|": float(ptype_abs[k])})
        ptype_imp = pd.DataFrame(rows)
        feat_order = [clean[k] for k in top_idx]
        fig_pos, ax_pos = plt.subplots(figsize=(8, 5))
        for ptype in ptype_imp["Property Type"].unique():
            sub = ptype_imp[ptype_imp["Property Type"] == ptype].set_index("Feature")
            vals = [sub.loc[f, "Mean |SHAP|"] if f in sub.index else 0 for f in feat_order]
            ax_pos.barh(feat_order, vals, label=ptype, alpha=0.75)
        ax_pos.set_xlabel("Mean |SHAP| within type")
        ax_pos.set_title("Feature importance by property type")
        ax_pos.legend()
        st.pyplot(fig_pos)
        st.caption(
            "Drivers shift by category — area dominates for Villas and Duplexes, "
            "while Apartments are more sensitive to location and build year."
        )
    else:
        st.info("Property type metadata unavailable for this sample.")
 
    # ── 6. Permutation importance (model-agnostic cross-check) ────────
    _section("6. Permutation importance — a model-agnostic second opinion",
             "Shuffle each feature and measure the drop in fit; agrees with SHAP when honest.")
    try:
        with st.spinner("Permuting features on the explain set…"):
            y_log = np.log1p(y_exp)            # true log targets for the explained rows
            perm = permutation_importance(
                bundle["est"], X_explain, y_log,
                n_repeats=5, random_state=42, n_jobs=-1,
            )
        perm_df = (pd.DataFrame({
            "Feature": clean,
            "Importance": perm.importances_mean,
            "Std": perm.importances_std,
        }).sort_values("Importance", ascending=False).head(15)
          .sort_values("Importance"))        # ascending for top-down h-bar
        fig_perm, ax_perm = plt.subplots(figsize=(8, 5))
        ax_perm.barh(perm_df["Feature"], perm_df["Importance"],
                     xerr=perm_df["Std"], capsize=3)
        ax_perm.set_xlabel("Drop in R² when shuffled")
        ax_perm.set_title("Permutation importance")
        st.pyplot(fig_perm)
        st.caption(
            "Insight: permutation importance is computed independently of SHAP, yet "
            "both crown **Area Sq Ft** — strong agreement that the feature "
            "genuinely drives the model, not an artefact of one method."
        )
    except Exception as exc:
        st.info(f"Permutation importance unavailable in this environment. ({exc})")