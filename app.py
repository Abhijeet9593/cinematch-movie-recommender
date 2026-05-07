import streamlit as st
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4221/4221484.png",
        width=120,
    )

    st.header("⚡ About App")
    st.write(
        "This movie recommendation system uses **TF-IDF** and **Cosine Similarity** to recommend movies based on content."
    )

    st.markdown("---")
    st.write("👨‍💻 Built with Streamlit")
    st.write("⭐ Deploy on Streamlit Cloud")

# =========================
# MOVIE SELECTOR
# =========================
movie_list = movies["title"].sort_values().unique()

selected_movie = st.selectbox(
    "🎥 Select Your Favorite Movie",
    movie_list,
)

num_recommendations = st.slider(
    "📌 Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10,
)

# =========================
# BUTTON ACTION
# =========================
if st.button("🚀 Recommend Movies"):

    recommendations = recommend_movies(
        selected_movie,
        num_recommendations,
    )

    if recommendations is None:
        st.error("Movie not found in dataset.")

    else:
        st.success(f"Top {num_recommendations} movies similar to '{selected_movie}'")

        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">🎬 {row['title']}</div>
                    <div class="movie-info">
                        <b>Genre:</b> {row['genres']} <br>
                        <b>⭐ Rating:</b> {row['vote_average']} <br>
                        <b>🔥 Popularity:</b> {row['popularity']} <br>
                        <b>📝 Tagline:</b> {row['tagline']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ using Streamlit & Machine Learning</center>",
    unsafe_allow_html=True,
)
