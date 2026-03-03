import pandas as pd

print("Loading full ratings...")
ratings = pd.read_csv("data/ratings.csv")

print("Original size:", ratings.shape)

# Keep only 200k rows (safe for free deployment)
ratings_small = ratings.sample(200000, random_state=42)

print("New size:", ratings_small.shape)

ratings_small.to_csv("data/ratings_small.csv", index=False)

print("✅ ratings_small.csv created")