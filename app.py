import streamlit as st

st.set_page_config(page_title="Housing Price Prediction", page_icon="🏡", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetric"]{
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-left: 4px solid #2DD4BF;
  border-radius: 10px;
  padding: 14px 16px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{ color:#2DD4BF; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p{ color:#9CA3AF; }
</style>
""", unsafe_allow_html=True)

from src import (
    page_intro,
    page_visualization,
    page_prediction,
    page_explainability,
    page_tuning,
    page_conclusion,
)

PAGES = {
    "Introduction":       page_intro,
    "Visualization":      page_visualization,
    "Prediction":         page_prediction,
    "Feature Importance": page_explainability,
    "W&B Tracking":       page_tuning,
    "Conclusion":         page_conclusion,
}

st.sidebar.title("🏡 Housing Price Prediction")
choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

module = PAGES[choice]
if hasattr(module, "render"):
    module.render()
else:
    st.title(choice)
    st.info("🚧 This page is under construction.")