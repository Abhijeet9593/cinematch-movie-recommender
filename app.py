import streamlit as st
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("About")

st.sidebar.info(
    "Movie recommendation system using TF-IDF and cosine similarity."
)

# --------------------------------------------------
# MOVIE SELECT
# --------------------------------------------------
movie_list = sorted(movies[movie_column].dropna().unique())

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

# --------------------------------------------------
# BUTTON
# --------------------------------------------------
if st.button("Recommend Movies"):

    recommendations = recommend(selected_movie, num_recommendations)

    if recommendations:

        st.success(
            f"Top {num_recommendations} recommendations for {selected_movie}"
        )

        for i, movie in enumerate(recommendations, start=1):

            st.markdown(
                f'''
                <div class="movie-card">
                    <h4>{i}. {movie}</h4>
                </div>
                ''',
                unsafe_allow_html=True
            )

    else:
        st.error("Movie not found")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Made with Streamlit")
