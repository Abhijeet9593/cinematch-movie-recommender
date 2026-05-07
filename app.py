import streamlit as st
import pickle
import gzip
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #FFD700;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: gray;
    margin-bottom: 30px;
}

.movie-card {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------
@st.cache_resource
def load_data():

    # Load dataframe
    try:
        with gzip.open("df.pkl", "rb") as f:
            movies = pickle.load(f)
    except:
        with open("df.pkl", "rb") as f:
            movies = pickle.load(f)

    # Load TF-IDF matrix
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load indices
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)

    return movies, tfidf_matrix, indices

# ---------------------------------------------------
# CALL FUNCTION
# ---------------------------------------------------
try:
    movies, tfidf_matrix, indices = load_data()

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ---------------------------------------------------
# FIND MOVIE TITLE COLUMN
# ---------------------------------------------------
possible_columns = ["title", "movie_title", "name", "movie"]

movie_column = None

for col in possible_columns:
    if col in movies.columns:
        movie_column = col
        break

if movie_column is None:
    st.error("Movie title column not found in dataframe")
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
        tfidf_matrix[idx:idx+1],
        tfidf_matrix
    ).flatten()

    sim_scores = list(enumerate(cosine_sim))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]

    recommendations = movies.iloc[movie_indices][movie_column].tolist()

    return recommendations

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    '<div class="title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find Similar Movies Instantly</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("About")

st.sidebar.info(
    "This project uses TF-IDF and cosine similarity for movie recommendations."
)

# ---------------------------------------------------
# MOVIE DROPDOWN
# ---------------------------------------------------
movie_list = sorted(
    movies[movie_column].dropna().unique()
)

selected_movie = st.selectbox(
    "Select a Movie",
    movie_list
)

num_recommendations = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

# ---------------------------------------------------
# BUTTON
# ---------------------------------------------------
if st.button("Recommend Movies"):

    recommendations = recommend(
        selected_movie,
        num_recommendations
    )

    if recommendations:

        st.success(
            f"Top {num_recommendations} recommendations for {selected_movie}"
        )

        for i, movie in enumerate(recommendations, start=1):

            st.markdown(
                f"""
                <div class="movie-card">
                    <h4>{i}. {movie}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.warning("No recommendations found.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
