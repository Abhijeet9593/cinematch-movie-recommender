import streamlit as st
        movie_column = col
        break

if movie_column is None:
    st.error("Movie title column not found in dataset.")
    st.stop()


# ------------------------------
# RECOMMENDATION FUNCTION
# ------------------------------
def recommend(movie_name, top_n=10):
    movie_name = movie_name.strip()

    if movie_name not in indices:
        return []

    idx = indices[movie_name]

    cosine_sim = linear_kernel(
        tfidf_matrix[idx:idx+1],
        tfidf_matrix
    ).flatten()

    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices][movie_column].tolist()


# ------------------------------
# HEADER
# ------------------------------
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Get smart movie recommendations instantly using Machine Learning</div>',
    unsafe_allow_html=True
)

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("📌 About")
st.sidebar.info(
    """
    This Movie Recommendation System uses:

    ✅ TF-IDF Vectorization
    ✅ Cosine Similarity
    ✅ Content-Based Filtering

    Built with Streamlit & Scikit-learn.
    """
)

# ------------------------------
# SELECT MOVIE
# ------------------------------
movie_list = sorted(movies[movie_column].dropna().unique())

selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movie_list
)

num_recommendations = st.slider(
    "📊 Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

# ------------------------------
# BUTTON
# ------------------------------
)
