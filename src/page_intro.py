"""
Page 1 — Business Case & Data Presentation
"""

import streamlit as st
import pandas as pd
from data_loader import dataset_selector, get_target, get_features


def render():
    ds_key, df, info = dataset_selector()
    target = get_target(ds_key)
    features = get_features(df, target)

    st.title(info["title"])

    st.markdown("## Business Problem")
    st.markdown(info["problem"])
    st.markdown("---")

    st.markdown("## Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", len(features))
    c3.metric("Target", info["target"])
    c4.metric("Source", info["source"])

    st.markdown("## Data Preview")
    tab1, tab2, tab3 = st.tabs(["First rows", "Last rows", "Random sample"])
    tab1.dataframe(df.head(10), use_container_width=True)
    tab2.dataframe(df.tail(10), use_container_width=True)
    tab3.dataframe(df.sample(min(10, len(df)), random_state=42), use_container_width=True)

    st.markdown("## Feature Dictionary")
    feat_df = pd.DataFrame(
        [{"Feature": k, "Description": v, "Type": str(df[k].dtype)}
         for k, v in info["features_desc"].items() if k in df.columns]
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("## Descriptive Statistics")
    st.markdown("**Numeric columns**")
    st.dataframe(df.describe().T, use_container_width=True)
    cat_cols = df.select_dtypes(exclude="number").columns
    if len(cat_cols):
        st.markdown("**Categorical columns**")
        st.dataframe(df[cat_cols].describe().T, use_container_width=True)

    st.markdown("## Data Quality Check")
    col_a, col_b = st.columns(2)
    with col_a:
        missing = df.isnull().sum()
        miss_pct = (missing / len(df) * 100).round(2)
        st.dataframe(pd.DataFrame({"Missing": missing, "% Missing": miss_pct}),
                     use_container_width=True)
    with col_b:
        completeness = (1 - df.isnull().mean().mean()) * 100
        st.metric("Overall Completeness", f"{completeness:.1f}%")
        st.metric("Duplicate Rows", int(df.duplicated().sum()))
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB")