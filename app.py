import pandas as pd
import pickle
from flask import Flask, render_template, request
import re
import os

app = Flask(__name__)

# -----------------------------
# 🔥 LOAD DATA (Optimized dtypes)
# -----------------------------
movies = pd.read_csv(
    "data/movies.csv",
    dtype={"movieId": "int32"},
    usecols=["movieId", "title", "genres"]
)

ratings = pd.read_csv(
    "data/ratings_small.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    usecols=["userId", "movieId", "rating"]
)

# -----------------------------
# Precompute lookup tables
# -----------------------------
user_watched = ratings.groupby('userId')['movieId'].apply(set).to_dict()
movie_likers = ratings[ratings['rating'] >= 4].groupby('movieId')['userId'].apply(list).to_dict()
user_liked_movies = ratings[ratings['rating'] >= 4].groupby('userId')['movieId'].apply(list).to_dict()

movie_dict = movies.set_index('movieId').to_dict('index')
all_titles = movies['title'].tolist()

# -----------------------------
# Load trained artifacts (ONLY LOAD, DO NOT TRAIN HERE)
# -----------------------------
with open("artifacts/content_similarity.pkl", "rb") as f:
    content_similarity = pickle.load(f)

with open("artifacts/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# -----------------------------
# Series Finder
# -----------------------------
def get_series_movies(movie_name, limit=6):
    clean = re.sub(r"\(.*?\)", "", movie_name).strip()
    pattern = r'[:\-]|Season\s*\d+|S\d+|Part\s*[IVX\d]+|Vol|Volume|Episode\s*[IVX\d]+'
    parts = [p.strip() for p in re.split(pattern, clean, flags=re.IGNORECASE) if p.strip()]
    if not parts:
        return pd.DataFrame()

    core_name = parts[0]

    related = movies[movies['title'].str.contains(re.escape(core_name), case=False)]

    related = related[related['title'] != movie_name]

    return related.head(limit)

# -----------------------------
# Content-Based Recommendations
# -----------------------------
def get_content_recs(movie_id, watched_ids, series_titles, top_n=9):
    similar = content_similarity.get(movie_id, [])
    recs = []

    for mid, score in similar:
        if mid in movie_dict:
            title = movie_dict[mid]['title']
            if mid not in watched_ids and title not in series_titles:
                recs.append({
                    'title': title,
                    'genres': movie_dict[mid]['genres'].replace('|', ', ')
                })

        if len(recs) >= top_n:
            break

    return recs

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html", suggestions=all_titles)

@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form['movie_name']
    user_id = int(request.form.get('user_id', 1))

    movie_row = movies[movies['title'] == movie_name]
    if movie_row.empty:
        return render_template("home.html", suggestions=all_titles, error="Movie not found.")

    movie_id = movie_row['movieId'].values[0]
    liked_genres = movie_row['genres'].values[0].replace('|', ', ')

    watched_ids = user_watched.get(user_id, set())

    # -------- Series --------
    series_df = get_series_movies(movie_name)
    series_list = [
        {'title': r['title'], 'genres': r['genres'].replace('|', ', ')}
        for _, r in series_df.iterrows()
    ]
    series_titles = {s['title'] for s in series_list}

    # -------- Content-Based --------
    more_like_this = get_content_recs(movie_id, watched_ids, series_titles)

    # -------- Collaborative Filtering --------
    others_enjoyed = []

    if user_id in user_watched:  # Cold-start protection

        similar_users = movie_likers.get(movie_id, [])

        cf_candidates = set()
        for u in similar_users[:100]:
            cf_candidates.update(user_liked_movies.get(u, []))

        # 🎯 If very few CF candidates → fallback to global SVD ranking
        if len(cf_candidates) < 20:
            candidate_pool = movies['movieId'].values[:500]
        else:
            candidate_pool = cf_candidates

        for mid in candidate_pool:
            if mid != movie_id and mid not in watched_ids:
                if mid in movie_dict and movie_dict[mid]['title'] not in series_titles:
                    est = svd.predict(user_id, mid).est
                    others_enjoyed.append({
                        'title': movie_dict[mid]['title'],
                        'genres': movie_dict[mid]['genres'].replace('|', ', '),
                        'score': est
                    })

        others_enjoyed = sorted(
            others_enjoyed,
            key=lambda x: x['score'],
            reverse=True
        )[:6]

    return render_template(
        "recommend.html",
        movie_name=movie_name,
        liked_genres=liked_genres,
        series_list=series_list,
        more_like_this=more_like_this,
        others_enjoyed=others_enjoyed
    )

# -----------------------------
# Render Dynamic Port
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)