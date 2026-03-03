import pandas as pd
import pickle
from flask import Flask, render_template, request
import re
import os
import gc  # Garbage Collector to free memory

app = Flask(__name__)

# -----------------------------
# LOAD DATA (Optimized)
# -----------------------------
# We load, convert to dict, then DELETE the dataframe to save space
_movies_df = pd.read_csv(
    "data/movies.csv",
    dtype={"movieId": "int32"},
    usecols=["movieId", "title", "genres"]
)
movie_dict = _movies_df.set_index('movieId').to_dict('index')
all_titles = _movies_df['title'].tolist()

# Precompute regex-friendly movie list for Series Finder
# Keeping this as a small DF for string operations
movies_for_search = _movies_df[['movieId', 'title', 'genres']].copy()
del _movies_df # Free memory immediately

_ratings_df = pd.read_csv(
    "data/ratings_small.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}
)

# Convert to dictionaries (Sets/Lists are more memory efficient than DFs)
user_watched = _ratings_df.groupby('userId')['movieId'].apply(set).to_dict()
movie_likers = _ratings_df[_ratings_df['rating'] >= 4].groupby('movieId')['userId'].apply(list).to_dict()
user_liked_movies = _ratings_df[_ratings_df['rating'] >= 4].groupby('userId')['movieId'].apply(list).to_dict()

del _ratings_df # Free memory
gc.collect()    # Force garbage collection

# -----------------------------
# Load artifacts
# -----------------------------
with open("artifacts/content_similarity.pkl", "rb") as f:
    content_similarity = pickle.load(f)

with open("artifacts/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# -----------------------------
# Series Finder (Updated to use search DF)
# -----------------------------
def get_series_movies(movie_name, limit=6):
    clean = re.sub(r"\(.*?\)", "", movie_name).strip()
    pattern = r'[:\-]|Season\s*\d+|S\d+|Part\s*[IVX\d]+|Vol|Volume|Episode\s*[IVX\d]+'
    parts = [p.strip() for p in re.split(pattern, clean, flags=re.IGNORECASE) if p.strip()]
    if not parts:
        return pd.DataFrame()
    core_name = parts[0]
    # Search in our lightweight search DF
    related = movies_for_search[movies_for_search['title'].str.contains(re.escape(core_name), case=False)]
    related = related[related['title'] != movie_name]
    return related.head(limit)

# -----------------------------
# Content-Based Recommendations
# -----------------------------
def get_content_recs(movie_id, watched_ids, series_titles, top_n=9):
    recs = []
    # Logic remains exactly the same
    for mid, score in content_similarity.get(movie_id, []):
        if mid in movie_dict and mid not in watched_ids and movie_dict[mid]['title'] not in series_titles:
            recs.append({'title': movie_dict[mid]['title'], 'genres': movie_dict[mid]['genres'].replace('|', ', ')})
        if len(recs) >= top_n:
            break
    return recs

# -----------------------------
# Routes (Logic Unchanged)
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html", suggestions=all_titles)

@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form['movie_name']
    user_id = int(request.form.get('user_id', 1))

    # Search against our search-ready DF
    movie_row = movies_for_search[movies_for_search['title'] == movie_name]
    if movie_row.empty:
        return render_template("home.html", suggestions=all_titles, error="Movie not found.")

    movie_id = movie_row['movieId'].values[0]
    liked_genres = movie_row['genres'].values[0].replace('|', ', ')
    watched_ids = user_watched.get(user_id, set())

    series_df = get_series_movies(movie_name)
    series_list = [{'title': r['title'], 'genres': r['genres'].replace('|', ', ')} for _, r in series_df.iterrows()]
    series_titles = {s['title'] for s in series_list}

    more_like_this = get_content_recs(movie_id, watched_ids, series_titles)

    # -----------------------------
    # Production-Level CF Ranking
    # -----------------------------
    others_enjoyed = []
    if user_id in user_watched:
        similar_users = movie_likers.get(movie_id, [])
        
        # Candidate Generation: Find movies liked by people who liked THIS movie
        cf_candidates = {}
        for u in similar_users[:100]:
            for mid in user_liked_movies.get(u, []):
                # Count frequency: how many 'similar' users liked this candidate?
                cf_candidates[mid] = cf_candidates.get(mid, 0) + 1
        
        # Sort by co-occurrence frequency first (the 'People also liked' signal)
        # then use SVD to refine the top 20 candidates
        top_candidates = sorted(cf_candidates.items(), key=lambda x: x[1], reverse=True)[:20]
        
        for mid, freq in top_candidates:
            if mid != movie_id and mid not in watched_ids and mid in movie_dict:
                if movie_dict[mid]['title'] not in series_titles:
                    # SVD prediction acts as a personalized re-ranker
                    est = svd.predict(user_id, mid).est
                    others_enjoyed.append({
                        'title': movie_dict[mid]['title'], 
                        'genres': movie_dict[mid]['genres'].replace('|', ', '), 
                        'score': est * (freq / 10) # Weighted by frequency to ensure relevance to the movie
                    })
        
        # Final Sort
        others_enjoyed = sorted(others_enjoyed, key=lambda x: x['score'], reverse=True)[:6]

    return render_template("recommend.html", movie_name=movie_name, liked_genres=liked_genres,
                            series_list=series_list, more_like_this=more_like_this, others_enjoyed=others_enjoyed)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)