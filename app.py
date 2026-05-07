import streamlit as st
import pickle
import gzip
import time
from sklearn.metrics.pairwise import linear_kernel

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------
st.markdown(
    """
    <style>

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b, #111827);
        color: white;
    }

    /* Hide Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Title */
    .main-title {
        text-align: center;
        font-size: 70px;
        font-weight: 800;
        background: linear-gradient(to right, #ff512f, #dd2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -20px;
        animation: fadeIn 2s ease-in-out;
    }

    /* Subtitle */
    .sub-title {
        text-align: center;
        color: #cbd5e1;
        font-size: 22px;
        margin-bottom: 40px;
        animation: fadeIn 3s ease-in-out;
    }

    /* Recommendation Cards */
    .movie-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 15px;
        transition: 0.4s;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .movie-card:hover {
        transform: scale(1.03);
)
