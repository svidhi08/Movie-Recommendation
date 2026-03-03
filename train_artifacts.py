import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD

# -----------------------------
# CONFIG
# -----------------------------
CONTENT_TOP_K = 15   # content-based candidates

# -----------------------------
# 1️⃣ Load movies
# -----------------------------
movies = pd.read_csv("data/movies.csv")
movies['combined'] = (
    movies['title'].astype(str) + " " +
    movies['genres'].astype(str).str.replace('|', ' ', regex=False)
)

# -----------------------------
# 2️⃣ Content-Based Filtering
# -----------------------------
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 1),
    max_features=8000
)
tfidf_matrix = tfidf.fit_transform(movies['combined'])

nn = NearestNeighbors(metric="cosine", n_neighbors=CONTENT_TOP_K + 1)
nn.fit(tfidf_matrix)
distances, indices = nn.kneighbors(tfidf_matrix)

# Map index → movieId
index_to_movie = dict(enumerate(movies['movieId']))

content_similarity = {}
for idx, movie_id in enumerate(movies['movieId']):
    sims = 1 - distances[idx]
    similar_movies = []
    for i, sim in zip(indices[idx][1:], sims[1:]):
        similar_movies.append((index_to_movie[i], sim))
    content_similarity[movie_id] = similar_movies

pickle.dump(content_similarity, open("artifacts/content_similarity.pkl", "wb"))
print("✅ content_similarity.pkl saved")

# -----------------------------
# 3️⃣ Load small ratings
# -----------------------------
ratings = pd.read_csv("data/ratings_small.csv")

# -----------------------------
# 4️⃣ Collaborative Filtering (SVD)
# -----------------------------
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

svd = SVD(n_factors=50, n_epochs=20, random_state=42)
svd.fit(trainset)

pickle.dump(svd, open("artifacts/svd_model.pkl", "wb"))
print("✅ svd_model.pkl saved")