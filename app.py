import streamlit as st
import pickle
import gzip
import time
from sklearn.metrics.pairwise import linear_kernel

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #111827);
    color: white;
}

/* Hide Streamlit Branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 70px;
    font-weight: 800;
    background: linear-gradient(to right, #ff512f, #dd2476);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: -20px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 22px;
    margin-bottom: 40px;
}

/* Movie Cards */
.movie-card {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    transition: 0.3s;
}

.movie-card:hover {
    transform: scale(1.02);
    border: 1px solid #ff4b4b;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(to right, #ff512f, #dd2476);
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.stButton > button:hover {
    box-shadow: 0 0 20px rgba(255,75,75,0.7);
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------
@st.cache_resource
def load_data():

    # Load DataFrame
    try:
        with gzip.open("df.pkl", "rb") as file:
            movies = pickle.load(file)
    except:
        with open("df.pkl", "rb") as file:
            movies = pickle.load(file)

    # Load TF-IDF Matrix
    with open("tfidf_matrix.pkl", "rb") as file:
        tfidf_matrix = pickle.load(file)

    # Load Indices
    with open("indices.pkl", "rb") as file:
        indices = pickle.load(file)

    return movies, tfidf_matrix, indices


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
try:
    movies, tfidf_matrix, indices = load_data()

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ---------------------------------------------------
# FIND TITLE COLUMN
# ---------------------------------------------------
possible_columns = ["title", "movie_title", "name", "movie"]

movie_column = None

for col in possible_columns:
    if col in movies.columns:
        movie_column = col
        break

if movie_column is None:
    st.error("Movie title column not found in dataset")
    st.write("Available columns:", movies.columns.tolist())
    st.stop()

# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------
def recommend(movie_name, top_n=10):

    if movie_name not in indices:
        return []

    idx = indices[movie_name]

    cosine_sim = linear_kernel(
        tfidf_matrix[idx:idx + 1],
        tfidf_matrix
    ).flatten()

    sim_scores = list(enumerate(cosine_sim))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    recommendations = movies.iloc[movie_indices][movie_column].tolist()

    return recommendations

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    '<div class="main-title">🎬 CineMatch AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Discover Movies You Will Love Using AI Recommendations ✨</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# STATS
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎥 Movies", f"{len(movies):,}")

with col2:
    st.metric("🤖 Model", "TF-IDF")

with col3:
    st.metric("⚡ Speed", "Instant")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------
# MOVIE SELECT
# ---------------------------------------------------
movie_list = sorted(
    movies[movie_column].dropna().unique()
)

selected_movie = st.selectbox(
    "🍿 Select Your Favorite Movie",
    movie_list
)

num_recommendations = st.slider(
    "🎯 Number of Recommendations",
    5,
    20,
    10
)

# ---------------------------------------------------
# BUTTON
# ---------------------------------------------------
if st.button("🚀 Recommend Movies"):

    progress_bar = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    recommendations = recommend(
        selected_movie,
        num_recommendations
    )

    st.markdown("## ✨ Recommended Movies")

    for i, movie in enumerate(recommendations, start=1):

        st.markdown(
            f"""
            <div class="movie-card">
                <h3>🎬 {i}. {movie}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <center>
        <h4 style='color:gray;'>
            Made with ❤️ using Streamlit & Machine Learning
        </h4>
    </center>
    """,
    unsafe_allow_html=True
)
