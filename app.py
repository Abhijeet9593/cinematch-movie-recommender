import streamlit as st
    ✅ Content-Based Filtering

    Built with Streamlit and Scikit-learn.
    """
)

# -------------------------------------------------
# MOVIE SELECTION
# -------------------------------------------------
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

# -------------------------------------------------
# BUTTON ACTION
# -------------------------------------------------
if st.button("🚀 Recommend Movies"):

    recommendations = recommend(selected_movie, num_recommendations)

    if recommendations:

        st.success(
            f"Top {num_recommendations} recommendations for '{selected_movie}'"
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
        st.error("Movie not found in recommendation database.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")

st.markdown(
    "<center>Made with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)
