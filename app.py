import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def cosine_similarity_sparse(vec, matrix):
    """Memory-efficient cosine similarity using sparse matrix dot product.
    TF-IDF vectors are already L2-normalized, so dot product = cosine similarity."""
    return (matrix * vec.T).toarray().flatten()

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --gold: #F5C842;
    --deep: #0A0A0F;
    --surface: #12121A;
    --card: #1A1A26;
    --border: #2A2A3A;
    --text: #E8E8F0;
    --muted: #888899;
    --accent: #E8405A;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--deep);
    color: var(--text);
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem; max-width: 1300px; }

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, #0A0A0F 0%, #1a0a1f 50%, #0f0a1a 100%);
    border-bottom: 1px solid var(--border);
    padding: 3.5rem 2rem 2.5rem;
    margin: 0 -2rem 3rem -2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(245,200,66,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(232,64,90,0.06) 0%, transparent 50%);
}
.hero-inner { position: relative; z-index: 1; }
.hero-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 900;
    line-height: 1.05;
    color: #fff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero-title span { color: var(--gold); }
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.02em;
}
.film-strip {
    position: absolute;
    right: -20px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    opacity: 0.04;
    user-select: none;
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
}
.stat-item { display: flex; flex-direction: column; }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--gold);
}
.stat-lbl { font-size: 0.7rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }

/* Search section */
.search-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2.5rem;
}
.search-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #fff;
    margin-bottom: 1rem;
}

/* Selectbox override */
div[data-baseweb="select"] > div {
    background-color: #0f0f1a !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="select"]:focus-within > div {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(245,200,66,0.15) !important;
}

/* Recommend button */
div.stButton > button {
    background: linear-gradient(135deg, var(--gold), #e8a820) !important;
    color: #0A0A0F !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(245,200,66,0.25) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(245,200,66,0.4) !important;
}

/* Movie card */
.movie-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 1.25rem;
}
.movie-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 1.25rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--accent));
    opacity: 0;
    transition: opacity 0.25s ease;
}
.movie-card:hover {
    border-color: rgba(245,200,66,0.35);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}
.movie-card:hover::before { opacity: 1; }

.card-rank {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: rgba(245,200,66,0.12);
    line-height: 1;
    margin-bottom: 0.5rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.5rem;
    line-height: 1.3;
}
.card-genres {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-bottom: 0.75rem;
}
.genre-tag {
    background: rgba(245,200,66,0.1);
    color: var(--gold);
    border: 1px solid rgba(245,200,66,0.2);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.card-overview {
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.55;
    margin-bottom: 0.85rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
}
.rating {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--gold);
}
.popularity {
    font-size: 0.72rem;
    color: var(--muted);
}

/* Selected movie panel */
.selected-panel {
    background: linear-gradient(135deg, var(--card), #1e1a2e);
    border: 1px solid rgba(245,200,66,0.2);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 2rem;
}
.panel-label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
}
.panel-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 900;
    color: #fff;
    margin-bottom: 0.4rem;
}
.panel-tagline {
    font-style: italic;
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 0.85rem;
}
.panel-overview { font-size: 0.9rem; color: #ccc; line-height: 1.65; }

/* Section header */
.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #fff;
    white-space: nowrap;
}

/* No results */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
}
.empty-icon { font-size: 4rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-text { font-size: 1rem; }

/* Slider overrides */
div[data-testid="stSlider"] { padding: 0 0.5rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    import gzip, os
    from scipy.sparse import load_npz

    # Load df — supports df.pkl.gz (compressed) or df.pkl
    if os.path.exists("df.pkl.gz"):
        with gzip.open("df.pkl.gz", "rb") as f:
            df = pickle.load(f)
    else:
        with open("df.pkl", "rb") as f:
            df = pickle.load(f)

    # Load indices
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)

    # Load tfidf_matrix — prefer .npz (version-safe) over .pkl
    if os.path.exists("tfidf_matrix.npz"):
        tfidf_matrix = load_npz("tfidf_matrix.npz")
    else:
        with open("tfidf_matrix.pkl", "rb") as f:
            tfidf_matrix = pickle.load(f)

    return df, indices, tfidf_matrix


@st.cache_data(show_spinner=False)
def get_recommendations(title, _df, _indices, _tfidf_matrix, n=10):
    # Handle duplicate titles — always use first occurrence
    raw_idx = _indices[title]
    idx = int(raw_idx.iloc[0]) if hasattr(raw_idx, "iloc") else int(raw_idx)
    sim_scores = cosine_similarity_sparse(_tfidf_matrix[idx], _tfidf_matrix)
    sim_scores_enum = list(enumerate(sim_scores))
    sim_scores_sorted = sorted(sim_scores_enum, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores_sorted]
    scores = [round(i[1] * 100, 1) for i in sim_scores_sorted]
    result = _df.iloc[movie_indices][["title", "overview", "genres", "tagline", "vote_average", "popularity"]].copy()
    result["similarity"] = scores
    result = result.reset_index(drop=True)
    return result


def render_movie_card(row, rank):
    genres_html = ""
    if pd.notna(row.get("genres")) and str(row["genres"]).strip():
        genres = str(row["genres"]).split()[:3]
        genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in genres)

    overview = str(row.get("overview", "")) if pd.notna(row.get("overview")) else ""
    rating = row.get("vote_average", 0) or 0
    pop = row.get("popularity", 0) or 0
    similarity = row.get("similarity", 0)

    stars = "★" * min(5, round(rating / 2)) + "☆" * max(0, 5 - round(rating / 2))

    return f"""
    <div class="movie-card">
        <div class="card-rank">#{rank:02d}</div>
        <div class="card-title">{row['title']}</div>
        <div class="card-genres">{genres_html}</div>
        <div class="card-overview">{overview if overview else 'No overview available.'}</div>
        <div class="card-footer">
            <div class="rating">⭐ {rating:.1f} <span style="font-size:0.65rem;color:#888;font-weight:400">&nbsp;{stars}</span></div>
            <div class="popularity">🔥 {similarity}% match</div>
        </div>
    </div>
    """


# ─── Main App ──────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Loading cinema database..."):
        df, indices, tfidf_matrix = load_data()

    movie_titles = sorted(indices.index.tolist())

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="film-strip">🎬🎞️🎥</div>
        <div class="hero-inner">
            <div class="hero-label">✦ AI-Powered Discovery</div>
            <h1 class="hero-title">Cine<span>Match</span></h1>
            <p class="hero-sub">Discover your next obsession — powered by content intelligence across 45,000+ films</p>
            <div class="stats-bar">
                <div class="stat-item">
                    <span class="stat-num">45K+</span>
                    <span class="stat-lbl">Movies</span>
                </div>
                <div class="stat-item">
                    <span class="stat-num">TF-IDF</span>
                    <span class="stat-lbl">Engine</span>
                </div>
                <div class="stat-item">
                    <span class="stat-num">50K</span>
                    <span class="stat-lbl">Features</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search Panel ──────────────────────────────────────────────────────────
    st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="search-title">🔍 Find your movie</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([4, 1.5, 1])
    with col1:
        selected_movie = st.selectbox(
            "Select a movie you love",
            options=["— Choose a movie —"] + movie_titles,
            label_visibility="collapsed",
        )
    with col2:
        n_recommendations = st.slider("Results", min_value=5, max_value=20, value=10, step=1)
    with col3:
        st.write("")
        recommend_btn = st.button("✦ Recommend", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if recommend_btn or (selected_movie != "— Choose a movie —" and "last_movie" in st.session_state and st.session_state.last_movie != selected_movie):
        if selected_movie == "— Choose a movie —":
            st.warning("Please select a movie first!")
            return

        st.session_state.last_movie = selected_movie

        with st.spinner("Finding your perfect matches..."):
            # Selected movie info
            movie_data = df[df["title"] == selected_movie].iloc[0]
            tagline = str(movie_data.get("tagline", "")) if pd.notna(movie_data.get("tagline")) else ""
            overview = str(movie_data.get("overview", "")) if pd.notna(movie_data.get("overview")) else ""
            genres_raw = str(movie_data.get("genres", "")) if pd.notna(movie_data.get("genres")) else ""

            genres_tags = "".join(f'<span class="genre-tag">{g}</span>' for g in genres_raw.split()[:5]) if genres_raw else ""

            st.markdown(f"""
            <div class="selected-panel">
                <div class="panel-label">✦ Now finding matches for</div>
                <div class="panel-title">{selected_movie}</div>
                {"<div class='panel-tagline'>\"" + tagline + "\"</div>" if tagline else ""}
                <div class="card-genres" style="margin-bottom:0.85rem">{genres_tags}</div>
                <div class="panel-overview">{overview if overview else 'No overview available.'}</div>
            </div>
            """, unsafe_allow_html=True)

            # Get recommendations
            recs = get_recommendations(selected_movie, df, indices, tfidf_matrix, n=n_recommendations)

        # Section header
        st.markdown(f"""
        <div class="section-header">
            <span class="section-title">🎞️ Top {len(recs)} Recommendations</span>
            <div class="section-line"></div>
            <span style="font-size:0.75rem;color:#888;white-space:nowrap">{len(recs)} matches found</span>
        </div>
        """, unsafe_allow_html=True)

        # Render grid
        cols_per_row = 5
        for row_start in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, col in enumerate(cols):
                movie_idx = row_start + col_idx
                if movie_idx < len(recs):
                    row = recs.iloc[movie_idx]
                    with col:
                        st.markdown(render_movie_card(row, movie_idx + 1), unsafe_allow_html=True)

    elif selected_movie == "— Choose a movie —":
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <div class="empty-text">Select a movie above and hit <strong>Recommend</strong> to discover your next watch.</div>
        </div>
        """, unsafe_allow_html=True)

        # Show popular picks
        st.markdown("""
        <div class="section-header">
            <span class="section-title">🔥 Popular Starting Points</span>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        popular_titles = [
            "The Dark Knight", "Inception", "The Matrix",
            "Interstellar", "Pulp Fiction", "The Godfather",
            "Forrest Gump", "The Silence of the Lambs", "Schindler's List", "Titanic"
        ]
        valid_popular = [t for t in popular_titles if t in indices.index]

        pop_cols = st.columns(5)
        for i, title in enumerate(valid_popular[:10]):
            with pop_cols[i % 5]:
                if st.button(f"🎬 {title}", key=f"pop_{i}", use_container_width=True):
                    st.session_state["auto_select"] = title
                    st.rerun()

    # Handle auto-select from popular picks
    if "auto_select" in st.session_state:
        title = st.session_state.pop("auto_select")
        st.session_state.last_movie = title


if __name__ == "__main__":
    main()
