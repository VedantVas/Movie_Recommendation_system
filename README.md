# 🎬 Content-Based Movie Recommendation System

A **Streamlit web app** that recommends movies based on metadata such as **genres, cast, crew, director, keywords, and overview text** using **content-based filtering**.

---

## 🚀 Features
- Search for any movie from the **TMDB 5000 Movies Dataset**  
- Get **similar movie recommendations** based on:
  - Movie overview (TF-IDF)
  - Metadata (Genres, Cast, Director, Keywords)
- Displays posters (if available from TMDB’s free image API)
- User-friendly Streamlit interface

---

## 📂 Dataset
This project uses the **TMDB 5000 Movies Dataset**, which you can download here:  
🔗 [TMDB 5000 Movies & Credits Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Place both files in the same folder as the script:
- `tmdb_5000_movies.csv`  
- `tmdb_5000_credits.csv`

---

## 🛠️ Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/CBF_Movie_Recommend.git
cd CBF_Movie_Recommend
pip install -r requirements.txt
