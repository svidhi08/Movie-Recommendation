import pandas as pd

# -----------------------------
# Load full ratings
# -----------------------------
ratings = pd.read_csv("data/ratings.csv")
print("Original shape:", ratings.shape)

# -----------------------------
# Keep only active users with >= 50 ratings
# -----------------------------
user_counts = ratings['userId'].value_counts()
active_users = user_counts[user_counts >= 50].index
ratings = ratings[ratings['userId'].isin(active_users)]

# -----------------------------
# Keep only popular movies with >= 50 ratings
# -----------------------------
movie_counts = ratings['movieId'].value_counts()
popular_movies = movie_counts[movie_counts >= 50].index
ratings = ratings[ratings['movieId'].isin(popular_movies)]

# -----------------------------
# Optional: limit total rows for free deploy
# -----------------------------
if len(ratings) > 100_000:
    ratings = ratings.sample(100_000, random_state=42)

# -----------------------------
# Save small CSV
# -----------------------------
ratings.to_csv("data/ratings_small.csv", index=False)

# -----------------------------
# Info for sanity check
# -----------------------------
print("✅ ratings_small.csv created")
print("Shape:", ratings.shape)
print("Unique users:", ratings['userId'].nunique())
print("Unique movies:", ratings['movieId'].nunique())