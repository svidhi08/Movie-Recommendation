import pandas as pd
import pickle
from flask import Flask, render_template, request
import re

app = Flask(__name__)

# -----------------------------
# 🔥 LOAD DATA & ARTIFACTS
# -----------------------------
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings_small.csv")

# Precompute fast lookup tables
user_watched = ratings.groupby('userId')['movieId'].apply(set).to_dict()
movie_likers = ratings[ratings['rating'] >= 4].groupby('movieId')['userId'].apply(list).to_dict()
user_liked_movies = ratings[ratings['rating'] >= 4].groupby('userId')['movieId'].apply(list).to_dict()

movie_dict = movies.set_index('movieId').to_dict('index')
content_similarity = pickle.load(open("artifacts/content_similarity.pkl", "rb"))
svd = pickle.load(open("artifacts/svd_model.pkl", "rb"))

# -----------------------------
# Helper: Series Search (Year Descending)
# -----------------------------
def get_series_movies(movie_name, limit=6):
    clean = re.sub(r"\(.*?\)", "", movie_name).strip()
    pattern = r'[:\-]|Season\s*\d+|S\d+|Part\s*[IVX\d]+|Vol|Volume|Episode\s*[IVX\d]+'
    parts = [p.strip() for p in re.split(pattern, clean, flags=re.IGNORECASE) if p.strip()]
    if not parts:
        return pd.DataFrame()
    core_name = parts[0]

    related = movies[movies['title'].str.contains(re.escape(core_name), case=False, regex=True)].copy()

    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0

    related['year'] = related['title'].apply(extract_year)

    garbage = ["robot chicken", "fan film", "parody", "documentary", "lego", "holiday special"]
    related = related[~related['title'].str.lower().str.contains('|'.join(garbage))]

    related = related[related['title'] != movie_name].sort_values(by='year', ascending=False)

    return related.head(limit)

# -----------------------------
# Pure Content-Based Filtering
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
# Flask Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html", suggestions=movies['title'].tolist())

@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form['movie_name']
    user_id = int(request.form.get('user_id', 1))

    movie_row = movies[movies['title'] == movie_name]
    if movie_row.empty:
        return render_template("home.html", suggestions=movies['title'].tolist(), error="Movie not found. Please try again.")

    movie_id = movie_row['movieId'].values[0]
    liked_genres = movie_row['genres'].values[0].replace('|', ', ')

    watched_ids = user_watched.get(user_id, set())

    # Section 1: Continue the Story
    series_df = get_series_movies(movie_name)
    series_list = [{'title': r['title'], 'genres': r['genres'].replace('|', ', ')} for _, r in series_df.iterrows()]
    series_titles = {s['title'] for s in series_list}

    # Section 2: More Like This (Content-Based)
    more_like_this = get_content_recs(movie_id, watched_ids, series_titles)

    # Section 3: Others Also Enjoyed (Hybrid CF using SVD)
    similar_users = movie_likers.get(movie_id, [])

    cf_candidates = set()
    for u in similar_users[:50]:
        cf_candidates.update(user_liked_movies.get(u, []))

    others_enjoyed = []
    for mid in cf_candidates:
        if mid != movie_id and mid not in watched_ids:
            if mid in movie_dict and movie_dict[mid]['title'] not in series_titles:
                est = svd.predict(user_id, mid).est
                others_enjoyed.append({
                    'title': movie_dict[mid]['title'],
                    'genres': movie_dict[mid]['genres'].replace('|', ', '),
                    'score': est
                })

    others_enjoyed = sorted(others_enjoyed, key=lambda x: x['score'], reverse=True)[:6]

    return render_template(
        "recommend.html",
        movie_name=movie_name,
        liked_genres=liked_genres,
        series_list=series_list,
        more_like_this=more_like_this,
        others_enjoyed=others_enjoyed
    )

if __name__ == "__main__":
    app.run(debug=False)


# -----------------------------
# Content-based filtering (CBF) can only recommend movies similar in genres, titles, or description.
# Example: If a user likes Jumanji (1995) → CBF will suggest other “Adventure, Children, Fantasy” movies.
# ❌ Problem: It cannot recommend movies outside these genres that other users with similar tastes loved.

# Collaborative filtering (CF) looks at user behavior (ratings, watch history) and finds patterns:

# “People who liked movie A also liked movie B.”
# Sequels first->similar content-based recommendations → Movies of the same genre, similar description.->collaborative filtering suggestions → Movies that similar users liked, even outside the original genre.
# SVD learns user taste vectors and movie feature vectors from past ratings and predicts a movie’s rating by matching the user’s preferences with the movie’s features.
# A user taste vector represents what type of movies a user generally likes based on their past ratings.
# A movie feature vector represents what type of content a movie has based on how users rated it.
# During prediction, SVD combines the user taste vector and movie feature vector to estimate the rating.(dot product)
# SVD uses two vectors because a rating depends on both user preference and movie characteristics.
# Traditional collaborative filtering relies on user–user or item–item comparisons, which becomes slow and unreliable when data is sparse or users are new.
# SVD instead learns user preferences and movie characteristics once during training and predicts ratings directly, making it faster and more scalable.
