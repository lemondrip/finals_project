"""
Shared data loading utilities for the Housing Price Prediction app.

Single-dataset version: loads dataset_2.csv and exposes the same
interface (load_data / get_target / get_features / dataset_selector)
that the page modules expect, so no page code needs to change.
"""

from pathlib import Path

import streamlit as st
import pandas as pd

# Resolve the CSV path relative to THIS file, so it loads no matter
# what the current working directory is.
DATA_PATH = Path(__file__).resolve().parent / "dataset_2.csv"

DEFAULT_KEY = "housing"

# A single dataset, still stored as a dict so helper scripts that loop
# over datasets (e.g. precompute_importance.py) keep working unchanged.
DATASETS = {
    "🏠 Housing Prices": "housing",
}

DATASET_DESCRIPTIONS = {
    "housing": {
        "title": "Housing Price Prediction",
        "problem": (
            "**Business Problem:** Buyers, sellers, and agents struggle to price "
            "homes fairly. Mispricing means properties sit unsold or owners leave "
            "money on the table. A reliable price estimate reduces negotiation "
            "friction, speeds up transactions, and protects both sides from over- "
            "or under-paying."
        ),
        "target": "Price",
        "target_desc": "Sale price of the property",
        "source": "Course-provided dataset (dataset_2.csv)",
        "rows": "—",
        "features_desc": {
            "Area_SqFt": "Floor area of the property (square feet)",
            "Rooms": "Number of rooms",
            "Build_Year": "Year the property was built",
            "Location": "City where the property is located",
            "Street_Type": "Type of street / road the property faces",
            "Furnishing": "Furnishing status (furnished / semi / unfurnished)",
            "Property_Type": "Type of property (apartment, villa, duplex, etc.)",
            "Has_Pool": "Whether the property has a pool (Yes / No)",
        },
    },
}


@st.cache_data
def load_data(dataset_key: str = DEFAULT_KEY) -> pd.DataFrame:
    """Load and return the cleaned housing DataFrame.

    `dataset_key` is accepted for compatibility with the original
    template interface but ignored — there is only one dataset.
    """
    df = pd.read_csv(DATA_PATH).dropna()
    return df


def get_target(dataset_key: str = DEFAULT_KEY) -> str:
    return DATASET_DESCRIPTIONS[dataset_key]["target"]


def get_features(df: pd.DataFrame, target: str) -> list[str]:
    return [c for c in df.columns if c != target]


def dataset_selector() -> tuple[str, pd.DataFrame, dict]:
    """Return (key, df, info) for the single dataset.

    No selectbox is rendered (only one dataset). A small static label
    is shown in the sidebar for context — delete it if you prefer.
    """
    key = DEFAULT_KEY
    info = DATASET_DESCRIPTIONS[key]
    with st.sidebar:
        st.markdown("### 📂 Dataset")
        st.caption(f"**{info['title']}**")
    df = load_data(key)
    return key, df, info