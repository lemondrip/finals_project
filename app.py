import streamlit as st

st.set_page_config(page_title="Housing Price Prediction", page_icon="🏡", layout="wide")

from src import page_intro

PAGES = ["Introduction", "Visualization", "Prediction",
         "Feature Importance", "W&B Tracking", "Conclusion"]

st.sidebar.title("🏡 Housing Price Prediction")
choice = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

if choice == "Introduction":
    page_intro.render()
else:
    st.title(choice)
    st.info("🚧 This page is under construction.")