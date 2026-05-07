import streamlit as st
import pandas as pd
import pickle
import gzip
from sklearn.metrics.pairwise import linear_kernel

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }

    .title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #EAEAEA;
        margin-bottom: 30px;
    }

    .movie-card {
        background-color: rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stButton > button {
        width: 100%;
        background-color: #FFD700;
        color: black;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }

    .stButton > button:hover {
        background-color: #ffcc00;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# LOAD FILES
# -------------------------------------------------
)
