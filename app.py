import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel


# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
        }
        .main-title {
            font-size: 55px;
            font-weight: bold;
            color: #ff4b4b;
            text-align: center;
        }
        .sub-title {
            font-size: 20px;
            color: #ffffff;
            text-align: center;
            margin-bottom: 30px;
        }
        .movie-box {
            background-color: #1c1f26;
            padding: 15px;
            border-radius: 15px;
            margin: 10px;
            text-align: center;
            box-shadow: 0px 0px 10px rgba(255, 75, 75, 0.2);
        }
        .movie-name {
            font-size: 20px;
            font-weight: bold;
            color: white;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: gray;
            font-size: 15px;
        }
    </style>
""", unsafe_allow_html=True)


# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_files():
    df = pickle.load(open("df.pkl", "rb"))
    indices = pickle.load(open("indices.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
    return df, indices, tfidf_matrix


df, indices, tfidf_matrix = load_files()


# ==========================
# RECOMMEND FUNCTION
# ==========================
def recommend_movies(title, num_recommendations=10):
    try:
        idx = indices[title]
        cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort()[-(num_recommendations+1):][::-1]
        similar_indices = similar_indices[1:]
        return df["title"].iloc[similar_indices].tolist()
    except:
        return []


# ==========================
# UI DESIGN
# ==========================
st.markdown("<div class='main-title'>🎬 CineMatch</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Find movies similar to your favorite one using AI Recommendation System</div>", unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns([3, 1])

with col1:
    selected_movie = st.selectbox("🎥 Select a Movie", df["title"].values)

with col2:
    num_movies = st.slider("⭐ Recommendations", 5, 20, 10)


if st.button("🚀 Recommend Movies"):
    recommendations = recommend_movies(selected_movie, num_movies)

    if recommendations:
        st.success(f"Top {num_movies} movies similar to **{selected_movie}**:")

        cols = st.columns(5)
        for i, movie in enumerate(recommendations):
            with cols[i % 5]:
                st.markdown(
                    f"""
                    <div class="movie-box">
                        <div class="movie-name">{movie}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.error("❌ Movie not found! Please select another movie.")


# ==========================
# FOOTER
# ==========================
st.markdown("<div class='footer'>Made with ❤️ using Streamlit | CineMatch Movie Recommender</div>", unsafe_allow_html=True)
