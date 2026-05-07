# 🎬 CineMatch — Movie Recommendation System

> A content-based movie recommender powered by TF-IDF vectorization and cosine similarity, deployed with a cinematic Streamlit interface across 45,000+ films.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

CineMatch is a **content-based filtering** movie recommendation system that analyzes movie metadata — genres, overview, tagline, and keywords — to find films most similar to what you love. Built with `scikit-learn` TF-IDF vectorization and cosine similarity, it delivers fast, accurate recommendations from a dataset of over **45,000 movies**.

---

## 🚀 Live Project Link

https://cinematch-movie-recommender-fbwuzhndnb93f9qjlnubuq.streamlit.app/

---

## ✨ Features

- 🔍 **Smart Search** — Searchable dropdown across 45,447 movie titles
- 🎯 **Content-Based Recommendations** — TF-IDF + Cosine Similarity engine
- 🎚️ **Adjustable Results** — Choose between 5 to 20 recommendations
- 🃏 **Rich Movie Cards** — Genres, overview, IMDb rating & match percentage
- ⚡ **Quick-Start Buttons** — Popular titles like Inception, The Dark Knight, etc.
- 💾 **Cached Loading** — Data loads once per session for instant response
- 🎨 **Cinematic Dark UI** — Playfair Display serif + gold accent design

---

## 🗂️ Project Structure

```
cinematch-movie-recommender/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
│
├── df.pkl                  # Movie DataFrame (title, overview, genres, etc.)
├── indices.pkl             # Title → Index mapping (pandas Series)
├── tfidf.pkl               # Fitted TF-IDF Vectorizer (sklearn)
├── tfidf_matrix.pkl        # TF-IDF sparse matrix (45447 × 50000)
│
└── README.md               # Project documentation
```

---

## 🧠 How It Works

```
User selects a movie
        ↓
Look up movie index via indices.pkl
        ↓
Retrieve TF-IDF vector from tfidf_matrix.pkl
        ↓
Compute cosine similarity against all 45,447 movies
        ↓
Return top-N most similar movies
        ↓
Display with metadata from df.pkl
```

### Algorithm: Content-Based Filtering

| Step | Detail |
|------|--------|
| **Feature Extraction** | Combines `genres`, `overview`, `tagline`, and `keywords` into a `tags` column |
| **Vectorization** | TF-IDF with 50,000 features converts text to numerical vectors |
| **Similarity** | Cosine similarity measures angle between movie vectors |
| **Ranking** | Top-N highest similarity scores returned as recommendations |

---

## 📦 Dataset & Model Files

| File | Type | Size | Description |
|------|------|------|-------------|
| `df.pkl` | DataFrame | — | 45,447 movies × 7 columns |
| `indices.pkl` | Series | — | Movie title to row-index mapping |
| `tfidf.pkl` | Vectorizer | — | Fitted `TfidfVectorizer` (sklearn) |
| `tfidf_matrix.pkl` | Sparse Matrix | — | Shape: (45447, 50000) |

**DataFrame columns:** `title`, `overview`, `genres`, `tagline`, `vote_average`, `popularity`, `tags`

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cinematch-movie-recommender.git
cd cinematch-movie-recommender
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to GitHub (include all `.pkl` files)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repository
4. Set **Main file path** to `app.py`
5. Click **Deploy** 🚀

> ⚠️ **Note:** `.pkl` files must be committed to the repository for Streamlit Cloud to access them. If files exceed GitHub's 100MB limit, use [Git LFS](https://git-lfs.github.com/).

---

## 📋 Requirements

```
streamlit
pandas
scikit-learn
scipy
numpy
```

---

## 🖼️ Screenshots

| Home Screen | Recommendations |
|-------------|-----------------|
| Select a movie from 45K+ titles | Get top-N matched movies with ratings |

---

## 🔮 Future Improvements

- [ ] Add movie poster images via TMDB API
- [ ] Hybrid filtering (content + collaborative)
- [ ] User rating & watchlist feature
- [ ] Genre-based filtering sidebar
- [ ] Search by actor or director

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) — Kaggle
- [scikit-learn](https://scikit-learn.org/) — TF-IDF & Cosine Similarity
- [Streamlit](https://streamlit.io/) — App framework
- [TMDB](https://www.themoviedb.org/) — Movie metadata source

---

<p align="center">Made with ❤️ and 🎬</p>
