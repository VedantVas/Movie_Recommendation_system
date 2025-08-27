# app.py
# ----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import ast, re, difflib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Sample Data ---
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")  # corrected
    credits = pd.read_csv("tmdb_5000_credits.csv")

    credits.rename(columns={'movie_id': 'id'}, inplace=True)
    df = movies.merge(
        credits[['id','cast','crew','title']].rename(columns={'title':'title_from_credits'}),
        on='id', how='left'
    )
    df['title'] = df['title'].fillna(df['title_from_credits'])
    df.drop(columns=['title_from_credits'], inplace=True)
    return df

df = load_data()

# --- Helpers ---
def parse_names(x):
    if pd.isna(x): return []
    try:
        return [d.get('name','') for d in ast.literal_eval(x) if isinstance(d, dict) and d.get('name')]
    except: return []

def parse_top_cast(x, top=3):
    if pd.isna(x): return []
    try:
        L = ast.literal_eval(x)
        return [d.get('name','') for d in L[:top] if isinstance(d, dict) and d.get('name')]
    except: return []

def parse_director(x):
    if pd.isna(x): return []
    try:
        L = ast.literal_eval(x)
        for d in L:
            if isinstance(d, dict) and d.get('job') == 'Director' and d.get('name'):
                return [d['name']]
    except: pass
    return []

def normalize_tokens(tokens):
    cleaned = []
    for t in tokens:
        t = t.lower()
        t = re.sub(r'\s+', '', t)
        t = re.sub(r"[^a-z0-9#+]", "", t)
        if t: cleaned.append(t)
    return cleaned

# --- Build metadata ---
df['genres_list']   = df['genres'].apply(parse_names).apply(normalize_tokens)
df['keywords_list'] = df['keywords'].apply(parse_names).apply(normalize_tokens)
df['cast_list']     = df['cast'].apply(parse_top_cast).apply(normalize_tokens)
df['director_list'] = df['crew'].apply(parse_director).apply(normalize_tokens)

df['soup'] = (df['genres_list'] + df['keywords_list'] + df['cast_list'] + df['director_list']).apply(lambda x: ' '.join(x))
df['overview'] = df['overview'].fillna('')

# --- Vectorize ---
count_vec = CountVectorizer(stop_words='english', max_features=5000)
soup_mat = count_vec.fit_transform(df['soup'])

tfidf_vec = TfidfVectorizer(stop_words='english', min_df=2)
overview_mat = tfidf_vec.fit_transform(df['overview'])

# --- Title index ---
df['title_lower'] = df['title'].str.lower()
indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()

def resolve_title(q):
    ql = (q or '').lower().strip()
    if ql in indices.index:
        return df.loc[indices[ql], 'title']
    cand = difflib.get_close_matches(ql, df['title_lower'].tolist(), n=1, cutoff=0.4)
    if cand:
        match_idx = df[df['title_lower'] == cand[0]].index[0]
        return df.loc[match_idx, 'title']
    subset = df[df['title_lower'].str.contains(ql, na=False)]
    if not subset.empty:
        return subset.iloc[0]['title']
    return None

# --- Recommend ---
def recommend(title, top_n=5, w_soup=0.6, w_overview=0.4):
    true_title = resolve_title(title)
    if not true_title or true_title.lower() not in indices:
        return pd.DataFrame(columns=['title','score','genres','vote_average','runtime','release_date'])
    idx = indices[true_title.lower()]
    sim_soup = cosine_similarity(soup_mat[idx], soup_mat).ravel()
    sim_over = cosine_similarity(overview_mat[idx], overview_mat).ravel()
    sim = w_soup * sim_soup + w_overview * sim_over
    sim[idx] = -1
    top_idx = np.argsort(-sim)[:top_n]
    out = df.loc[top_idx, ['title','genres','vote_average','runtime','release_date']].copy()
    out.insert(1, 'score', sim[top_idx].round(4))
    return out.reset_index(drop=True)

# --- Fetch poster ---
def fetch_poster(title):
    # removed API call; use placeholder
    return "https://via.placeholder.com/300x450?text=No+Poster"

# --- Streamlit UI ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Content-Based Movie Recommendation System")
st.markdown("Search for a movie and get **similar recommendations** based on metadata + description.")

movie_name = st.text_input("Enter a movie name", "Avatar")

if st.button("Recommend"):
    resolved = resolve_title(movie_name)
    if resolved:
        st.subheader(f"Recommendations for **{resolved}**")
        recs = recommend(resolved, top_n=6)
        
        cols = st.columns(3)
        for i, row in recs.iterrows():
            col = cols[i % 3]
            with col:
                poster_url = fetch_poster(row['title'])
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                year = row['release_date'][:4] if pd.notna(row['release_date']) else "N/A"
                runtime = f"{int(row['runtime'])} min" if pd.notna(row['runtime']) else "Unknown"
                st.markdown(
                    f"**{row['title']}**  \n"
                    f"⭐ {row['vote_average']} | {year}  \n"
                    f"⏱️ {runtime}"
                )
    else:
        st.error("Movie not found in dataset. Try another title.")
