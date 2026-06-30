"""
Page 2 — Data Visualization
Leads with a data exploration report (which includes the correlation
matrix), then interactive charts and a city heatmap.
"""

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from sklearn.preprocessing import LabelEncoder
from data_loader import dataset_selector, get_target, get_features

# Teal palette (matches the accent used on the intro cards)
TEAL = "#0D9488"        # readable teal on white (report)
TEAL_BRIGHT = "#2DD4BF" # bright teal for charts on the dark app

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090), "Gurugram": (28.4595, 77.0266),
    "Noida": (28.5355, 77.3910), "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462), "Kanpur": (26.4499, 80.3319),
    "Prayagraj": (25.4358, 81.8463), "Indore": (22.7196, 75.8577),
}
COLOR_RANGE = [
    [26, 152, 80], [145, 207, 96], [217, 239, 139],
    [254, 224, 139], [252, 141, 89], [215, 48, 39],
]


# ── Report helpers ──────────────────────────────────────────────────
def _histogram_svg(values, bins=20, width=220, height=60, color=TEAL):
    vals = pd.to_numeric(values.dropna(), errors="coerce").dropna()
    if len(vals) < 2:
        return "<span style='color:#999'>—</span>"
    counts, _ = np.histogram(vals, bins=bins)
    mx = counts.max() if counts.max() > 0 else 1
    bar_w = width / len(counts)
    bars = []
    for i, c in enumerate(counts):
        bh = max(1, c / mx * (height - 4))
        x = round(i * bar_w, 2)
        y = round(height - bh, 2)
        op = 0.5 + 0.5 * (c / mx)
        bars.append(
            f'<rect x="{x}" y="{y}" width="{round(bar_w-1,2)}" '
            f'height="{round(bh,2)}" rx="1" fill="{color}" opacity="{op:.2f}"/>'
        )
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">' + "".join(bars) + "</svg>"


def _correlation_html(df, features, target):
    # Per project requirement: use scikit-learn's LabelEncoder().fit_transform()
    # to turn the text (categorical) features into numbers so they can appear in
    # the correlation matrix alongside the numeric columns. Numeric columns are
    # passed through unchanged.
    encoded = pd.DataFrame(index=df.index)
    for c in list(features) + [target]:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            encoded[c] = df[c]
        else:
            encoded[c] = LabelEncoder().fit_transform(df[c].astype(str))
    corr = encoded.corr()

    def cell(v):
        if v >= 0:
            return f"rgba(13,148,136,{abs(v):.2f})"   # teal = positive
        return f"rgba(220,90,40,{abs(v):.2f})"        # warm = negative

    header = "".join(
        f'<th style="font-size:.62rem;color:{TEAL};padding:4px;white-space:nowrap;">{c}</th>'
        for c in corr.columns
    )
    rows = ""
    for rn in corr.index:
        cells = f'<td style="font-weight:600;font-size:.7rem;color:{TEAL};white-space:nowrap;padding:4px 6px;">{rn}</td>'
        for cn in corr.columns:
            v = corr.loc[rn, cn]
            tc = "#fff" if abs(v) > 0.5 else "#333"
            cells += (
                f'<td style="background:{cell(v)};color:{tc};text-align:center;'
                f'font-size:.68rem;padding:4px 6px;border-radius:3px;min-width:42px;">{v:.2f}</td>'
            )
        rows += f"<tr>{cells}</tr>"
    return (
        '<div style="overflow-x:auto;"><table style="border-collapse:separate;border-spacing:2px;">'
        f"<tr><th></th>{header}</tr>{rows}</table></div>"
    )


def _outlier_summary(df, features):
    out = []
    for col in features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        out.append((col, n, n / len(df) * 100))
    return out


def _build_report(df, features, target):
    n_rows, n_cols = df.shape
    mem_kb = df.memory_usage(deep=True).sum() / 1024
    missing = int(df.isnull().sum().sum())
    dups = int(df.duplicated().sum())
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    cards = (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px;">'
        f'<div class="rc"><div class="rl">Observations</div><div class="rv">{n_rows:,}</div></div>'
        f'<div class="rc"><div class="rl">Variables</div><div class="rv">{n_cols}</div></div>'
        f'<div class="rc"><div class="rl">Missing</div><div class="rv">{missing:,}</div></div>'
        f'<div class="rc"><div class="rl">Duplicates</div><div class="rv">{dups:,}</div></div>'
        f'<div class="rc"><div class="rl">Memory</div><div class="rv">{mem_kb:.0f} KB</div></div>'
        f'<div class="rc"><div class="rl">Numeric</div><div class="rv">{len(numeric)}</div></div>'
        "</div>"
    )

    var_rows = ""
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        n_unique = s.nunique()
        if is_num:
            badge = '<span class="bn">Numeric</span>'
            stats = (
                f'<span class="pill">&#956;={s.mean():.2f}</span>'
                f'<span class="pill">&#963;={s.std():.2f}</span>'
                f'<span class="pill">min={s.min():.2f}</span>'
                f'<span class="pill">max={s.max():.2f}</span>'
            )
            chart = _histogram_svg(s)
        else:
            badge = '<span class="bc">Categorical</span>'
            top = s.mode().iloc[0] if len(s.mode()) else "—"
            stats = f'<span class="pill">top={top}</span><span class="pill">unique={n_unique}</span>'
            chart = ""
        tag = ' <span class="tt">TARGET</span>' if col == target else ""
        var_rows += (
            f'<div class="vr"><div class="vh"><span class="vn">{col}</span>{tag} {badge}'
            f'<span style="float:right;color:#888;font-size:.75rem;">{n_unique} unique</span></div>'
            f'<div class="vb"><div>{stats}</div><div>{chart}</div></div></div>'
        )

    corr_html = _correlation_html(df, features, target)

    orows = ""
    for feat, n, pct in sorted(_outlier_summary(df, features), key=lambda x: -x[2]):
        color = "#2ea043" if pct < 1 else ("#f0ad4e" if pct < 5 else "#d73a49")
        sev = "Low" if pct < 1 else ("Medium" if pct < 5 else "High")
        orows += (
            f'<tr><td>{feat}</td><td style="text-align:center;">{n:,}</td>'
            f'<td><div style="height:6px;background:#eee;border-radius:3px;overflow:hidden;width:120px;display:inline-block;vertical-align:middle;">'
            f'<div style="width:{min(pct,100):.1f}%;height:6px;background:{color};"></div></div> {pct:.1f}%</td>'
            f'<td style="text-align:center;"><span style="background:{color}22;color:{color};padding:2px 8px;border-radius:10px;font-size:.7rem;font-weight:600;">{sev}</span></td></tr>'
        )

    ts = df[target]
    target_hist = _histogram_svg(ts, bins=30, width=320, height=70, color="#14B8A6")

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#333;margin:0;padding:8px;background:#fff;}"
        f".st{{font-size:1rem;font-weight:700;color:{TEAL};margin:18px 0 10px;border-bottom:2px solid #ccfbf1;padding-bottom:5px;}}"
        ".rc{background:#f0fdfa;border:1px solid #99f6e4;border-radius:10px;padding:12px;text-align:center;}"
        f".rl{{font-size:.7rem;text-transform:uppercase;color:#888;}}.rv{{font-size:1.3rem;font-weight:700;color:{TEAL};}}"
        ".vr{background:#fafafa;border:1px solid #eee;border-radius:10px;margin-bottom:8px;padding:10px 14px;}"
        ".vn{font-weight:700;}.bn{background:#e8f4fd;color:#0969da;padding:2px 8px;border-radius:10px;font-size:.65rem;}"
        ".bc{background:#fdf4e8;color:#b35900;padding:2px 8px;border-radius:10px;font-size:.65rem;}"
        f".tt{{background:{TEAL};color:#fff;padding:2px 6px;border-radius:10px;font-size:.6rem;}}"
        ".vb{display:flex;justify-content:space-between;align-items:center;margin-top:8px;}"
        f".pill{{background:#ccfbf1;color:{TEAL};padding:2px 8px;border-radius:8px;font-size:.72rem;margin-right:4px;}}"
        "table{width:100%;border-collapse:collapse;}td,th{padding:6px 10px;font-size:.8rem;}"
        "th{color:#888;text-transform:uppercase;font-size:.7rem;text-align:left;}"
        ".tbox{background:linear-gradient(135deg,#f0fdfa,#ccfbf1);border:2px solid #5eead4;border-radius:12px;padding:16px;text-align:center;}"
        "</style></head><body>"
        f'<div class="st">📋 Overview</div>{cards}'
        f'<div class="st">🎯 Target — {target}</div>'
        f'<div class="tbox">{target_hist}<div style="margin-top:8px;">mean {ts.mean():,.0f} &#183; median {ts.median():,.0f} &#183; std {ts.std():,.0f}</div></div>'
        f'<div class="st">🔬 Variable Explorer</div>{var_rows}'
        f'<div class="st">🔗 Correlation Matrix</div>{corr_html}'
        f'<div class="st">⚠️ Outlier Analysis (IQR method)</div>'
        '<table><tr><th>Feature</th><th style="text-align:center;">Count</th><th>Percentage</th><th style="text-align:center;">Severity</th></tr>'
        f"{orows}</table>"
        "</body></html>"
    )


# ── Map heatmap ─────────────────────────────────────────────────────
def _render_map(df):
    if "Location" not in df.columns:
        st.info("This dataset has no Location column, so there is no map view.")
        return

    st.markdown("Every house, scattered around its city. **Green = low, red = high** for the chosen metric.")

    metric = st.selectbox(
        "Color by",
        ["Average price (most expensive)", "Oldest homes (build year)",
         "Largest homes (area)", "Most rooms", "Number of listings", "Share with a pool"],
        key="map_metric",
    )

    g = df.groupby("Location")
    if metric == "Average price (most expensive)":
        row_val, hotter_high, fmt = df["Price"], True, lambda v: f"{v:,.0f}"
    elif metric == "Oldest homes (build year)":
        row_val, hotter_high, fmt = df["Build_Year"], False, lambda v: f"{v:.0f}"
    elif metric == "Largest homes (area)":
        row_val, hotter_high, fmt = df["Area_SqFt"], True, lambda v: f"{v:,.0f} sqft"
    elif metric == "Most rooms":
        row_val, hotter_high, fmt = df["Rooms"], True, lambda v: f"{v:.1f}"
    elif metric == "Number of listings":
        row_val, hotter_high, fmt = df["Location"].map(g.size()), True, lambda v: f"{v:.0f}"
    else:
        share = g["Has_Pool"].apply(lambda s: (s == "Yes").mean())
        row_val, hotter_high, fmt = df["Location"].map(share), True, lambda v: f"{v*100:.0f}%"

    known = df["Location"].isin(CITY_COORDS)
    sub = df[known]
    vals = pd.Series(row_val.values, index=df.index)[known]

    rng = np.random.default_rng(42)
    pts = pd.DataFrame({"value": vals.values})
    pts["lat"] = [CITY_COORDS[c][0] for c in sub["Location"]] + rng.normal(0, 0.05, len(sub))
    pts["lon"] = [CITY_COORDS[c][1] for c in sub["Location"]] + rng.normal(0, 0.05, len(sub))

    lo, hi = pts["value"].min(), pts["value"].max()
    norm = (pts["value"] - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=pts.index)
    if not hotter_high:
        norm = 1 - norm
    pts["weight"] = 0.15 + 0.85 * norm

    heat = pdk.Layer(
        "HeatmapLayer", data=pts, get_position="[lon, lat]",
        get_weight="weight", radius_pixels=60, intensity=1,
        threshold=0.05, color_range=COLOR_RANGE,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[heat],
        initial_view_state=pdk.ViewState(latitude=26.8, longitude=78.5, zoom=4.3),
        map_provider="carto", map_style="light",
    ))

    left_lab, right_lab = (fmt(lo), fmt(hi)) if hotter_high else (fmt(hi), fmt(lo))
    gradient = ", ".join(f"rgb({r},{gc},{b})" for r, gc, b in COLOR_RANGE)
    st.markdown(
        f'<div style="margin-top:8px;"><div style="height:18px;border-radius:4px;'
        f'background:linear-gradient(to right,{gradient});"></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:.8rem;margin-top:2px;">'
        f"<span>{left_lab}</span><span>{metric}</span><span>{right_lab}</span></div></div>",
        unsafe_allow_html=True,
    )
    st.caption("Coloring is relative to the chosen metric. Dot positions are jittered around each city, not exact addresses.")


# ── Page ────────────────────────────────────────────────────────────
def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    st.markdown("## 📊 Data Visualization")
    st.caption("A profiling report, interactive charts, and a geographic heatmap.")

    tab_report, tab_dist, tab_rel, tab_cat, tab_map = st.tabs(
        ["📑 Data Report", "📈 Distributions", "🔗 Relationships", "🥧 Categories", "🗺️ Map"]
    )

    with tab_report:
        components.html(_build_report(df, features, target),
                        height=760 + len(df.columns) * 70, scrolling=True)

    with tab_dist:
        var = st.selectbox("Variable", num_cols,
                           index=num_cols.index(target) if target in num_cols else 0)
        fig = px.histogram(df, x=var, nbins=40, marginal="box",
                           color_discrete_sequence=[TEAL_BRIGHT])
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    with tab_rel:
        feat = st.selectbox("Feature (x) vs Price (y)", [c for c in num_cols if c != target])
        d = df[[feat, target]].dropna()
        fig = px.scatter(d, x=feat, y=target, opacity=0.5,
                         color_discrete_sequence=[TEAL_BRIGHT])
        if len(d) > 1:
            m, b = np.polyfit(d[feat], d[target], 1)
            xs = np.array([d[feat].min(), d[feat].max()])
            fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines",
                                     line=dict(color="#F43F5E", dash="dash"), name="trend"))
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    with tab_cat:
        if cat_cols:
            cols = st.columns(2)
            for i, col in enumerate(cat_cols):
                counts = df[col].value_counts()
                fig = px.pie(values=counts.values, names=counts.index, title=col, hole=0.35,
                             color_discrete_sequence=px.colors.sequential.Teal)
                fig.update_layout(height=320, margin=dict(t=40, b=0, l=0, r=0))
                cols[i % 2].plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns in this dataset.")

        st.markdown("**Box plots — numeric features**")
        choices = [c for c in num_cols if c != target]
        bf = st.multiselect("Features", choices, default=choices[:3], key="boxsel")
        if bf:
            fig = go.Figure()
            for c in bf:
                fig.add_trace(go.Box(y=df[c], name=c, marker_color=TEAL_BRIGHT))
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab_map:
        _render_map(df)