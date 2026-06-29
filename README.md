---
title: Housing Price Prediction
emoji: 🏡
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: apache-2.0
---

# 🏡 Housing Price Prediction

A multi-page Streamlit app that predicts housing prices using linear regression and tree-based models, with data exploration, visualization, explainability (SHAP), and hyperparameter tuning.

Built for DS-UA 9111 — Data Science for Everyone.

## Pages

- Introduction — business case and data presentation
- Visualization — exploratory charts
- Prediction — model training and comparison
- Feature Importance — explainability
- W&B Tracking — hyperparameter tuning
- Conclusion

## Run locally

Install dependencies with `pip install -r requirements.txt`, then run `streamlit run app.py`.