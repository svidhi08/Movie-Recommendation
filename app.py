import pandas as pd
import pickle
from flask import Flask, render_template, request
import re
import os
import gc 
import math

app = Flask(__name__)

# -----------------------------
# 1️⃣ LOAD MOVIES & GLOBAL STATS
# -----------------------------
# We load the global counts first to identify "blockbuster" movies that cause bias
_ratings_temp = pd.read_csv("data/ratings_small.csv", usecols=["movieId"])
# Dictionary of {movieId: total_number_of_ratings}
global_popularity = _ratings_temp["movieId"].value_counts().to_dict()
del _ratings_temp
gc.collect()

_movies_df = pd.read_csv(
    "data/movies.csv",
    dtype={"movieId": "int32"},
    usecols=["movieId", "title", "genres"]
)
movie_dict = _movies_df.set_index('movieId').to_dict('index')
all_titles = _movies_df['title'].tolist()

# Lightweight DF for sequel searching
movies_for_search = _movies_df[['movieId', 'title', 'genres']].copy()
del _movies_df
gc.collect()

# -----------------------------
# 2️⃣ LOAD USER DATA
# -----------------------------
_ratings_df = pd.read_csv(
    "data/ratings_small.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}
)

# Optimization: Pre-filter liked movies (>= 3.5) to keep dictionaries small
user_watched = _ratings_df.groupby('userId')['movieId'].apply(set).to_dict()
liked_mask = _ratings_df['rating'] >= 3.5
movie_likers = _ratings_df[liked_mask].groupby('movieId')['userId'].apply(list).to_dict()
user_liked_movies = _ratings_df[liked_mask].groupby('userId')['movieId'].apply(list).to_dict()

del _ratings_df
gc.collect()

# -----------------------------
# 3️⃣ LOAD ARTIFACTS
# -----------------------------
with open("artifacts/content_similarity.pkl", "rb") as f:
    content_similarity = pickle.load(f)

with open("artifacts/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# -----------------------------
# 4️⃣ IMPROVED SEQUEL FUNCTION
# -----------------------------
def get_series_movies(movie_name, limit=6):
    # Explanation: Instead of strict regex (Part 1, Vol 2), we find the 'root' name.
    # We remove the year and any subtitles after a colon.
    clean = re.sub(r"\(.*?\)", "", movie_name).strip()
    core_name = clean.split(':')[0].strip() 
    
    # We find movies containing the core name that aren't the exact movie searched
    related = movies_for_search[
        (movies_for_search['title'].str.contains(re.escape(core_name), case=False)) & 
        (movies_for_search['title'] != movie_name)
    ]
    return related.head(limit)

# -----------------------------
# 5️⃣ CONTENT-BASED RECOMMENDATIONS
# -----------------------------
def get_content_recs(movie_id, watched_ids, series_titles, top_n=9):
    recs = []
    added_titles = set()
    
    # Similarity scores from your pickle file
    for mid, score in content_similarity.get(movie_id, []):
        if mid not in movie_dict: continue
        title = movie_dict[mid]['title']
        
        # If the movie is already in the 'Sequels' list, don't show it in 'Similar Vibe'
        if title in series_titles:
            continue
            
        if mid not in watched_ids and title not in added_titles:
            added_titles.add(title)
            recs.append({
                'title': title, 
                'genres': movie_dict[mid]['genres'].replace('|', ', ')
            })
            
        if len(recs) >= top_n:
            break
    return recs

# -----------------------------
# 6️⃣ ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html", suggestions=all_titles)

# -----------------------------
# 5️⃣ UPDATED RECOMMEND ROUTE (HYBRID CF)
# -----------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form['movie_name']
    user_id_raw = request.form.get('user_id')
    user_id = int(user_id_raw) if user_id_raw and user_id_raw.isdigit() else 1

    movie_row = movies_for_search[movies_for_search['title'] == movie_name]
    if movie_row.empty:
        return render_template("home.html", suggestions=all_titles, error="Movie not found.")

    movie_id = movie_row['movieId'].values[0]
    watched_ids = user_watched.get(user_id, set())

    # --- 1. SEQUELS ---
    series_df = get_series_movies(movie_name)
    series_titles = {r['title'] for _, r in series_df.iterrows()}

    # --- 2. MORE LIKE THIS ---
    more_like_this = get_content_recs(movie_id, watched_ids, series_titles)
    cb_titles = {m['title'] for m in more_like_this}

    # --- 3. OTHERS ALSO ENJOYED (Hybrid Logic) ---
    others_enjoyed = []
    cf_candidates = {}
    
    # STEP A: Try to find User-based Soulmates
    my_liked_movies = user_liked_movies.get(user_id, [])
    potential_soulmates = []
    for mid in my_liked_movies:
        potential_soulmates.extend(movie_likers.get(mid, []))
    
    # Remove the user themselves
    potential_soulmates = [u for u in potential_soulmates if u != user_id]

    # STEP B: If no soulmates (User IDs 6, 7 etc.), fallback to Item-based neighbors
    # This ensures the list is never empty!
    if not potential_soulmates:
        # Fallback: Find users who liked THIS specific movie
        potential_soulmates = movie_likers.get(movie_id, [])

    # Count similar users
    from collections import Counter
    soulmate_counts = Counter(potential_soulmates)
    top_peers = [u for u, count in soulmate_counts.most_common(50)]

    for peer_id in top_peers:
        peer_likes = user_liked_movies.get(peer_id, [])
        for mid in peer_likes:
            # Don't suggest if watched, or if it's the current movie, or already in CB
            if mid not in watched_ids and mid != movie_id:
                cf_candidates[mid] = cf_candidates.get(mid, 0) + 1

    candidate_list = []
    for mid, freq in cf_candidates.items():
        if mid in movie_dict:
            title = movie_dict[mid]['title']
            if title in series_titles or title in cb_titles:
                continue

            # Calculate score: SVD Prediction * Popularity Penalty
            est_rating = svd.predict(user_id, mid).est
            global_pop = global_popularity.get(mid, 1)
            
            # Using freq + 1 to ensure even low-frequency items get a chance if SVD likes them
            score = est_rating * ((freq + 1) / math.log1p(global_pop + 1))
            
            candidate_list.append({
                'title': title,
                'genres': movie_dict[mid]['genres'].replace('|', ', '),
                'final_score': score
            })

    # Sort and take top 6
    others_enjoyed = sorted(candidate_list, key=lambda x: x['final_score'], reverse=True)[:6]

    # FINAL FAILSAFE: If list is STILL short (very rare), fill with generic popular movies in genre
    if len(others_enjoyed) < 6:
        # Get the first genre of the current movie
        primary_genre = movie_row['genres'].values[0].split('|')[0]
        for mid, pop in sorted(global_popularity.items(), key=lambda x: x[1], reverse=True):
            if len(others_enjoyed) >= 6: break
            title = movie_dict[mid]['title']
            if mid not in watched_ids and mid != movie_id and title not in cb_titles and primary_genre in movie_dict[mid]['genres']:
                if not any(o['title'] == title for o in others_enjoyed):
                    others_enjoyed.append({
                        'title': title,
                        'genres': movie_dict[mid]['genres'].replace('|', ', '),
                        'final_score': 0
                    })

    return render_template("recommend.html",
                           movie_name=movie_name,
                           liked_genres=movie_row['genres'].values[0].replace('|', ', '),
                           series_list=[{'title': r['title'], 'genres': r['genres'].replace('|', ', ')} for _, r in series_df.iterrows()],
                           more_like_this=more_like_this,
                           others_enjoyed=others_enjoyed)

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)