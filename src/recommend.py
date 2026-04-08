import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import requests
import json
from collections import defaultdict

# Load config
with open('src/config.json') as f:
    config = json.load(f)
OMDB_API_KEY = config['OMDB_API_KEY']

print("Loading processed data...")
df = pd.read_pickle('src/data/movies_processed.pkl')

# Create movieId if not present (use index)
if 'movieId' not in df.columns:
    df['movieId'] = df.index

# Prepare ratings matrix for Surprise (using vote_average as proxy for ratings)
df['userId'] = 1  # Single user for simplicity, in real: multiple users
ratings = df[['userId', 'movieId', 'vote_average']].rename(columns={'movieId': 'itemId', 'vote_average': 'rating'})

reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(ratings[['userId', 'itemId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

print("Training SVD model...")
algo = SVD()
algo.fit(trainset)

# Test predictions
predictions = algo.test(testset)
print("RMSE:", accuracy.rmse(predictions))

print("Generating top recommendations for user 1...")
user_id = 1
movies_not_seen = [i for i in trainset.all_items() if i not in trainset.ur[user_id]]
predictions = [algo.predict(user_id, i) for i in movies_not_seen[:100]]
predictions.sort(key=lambda x: x.est, reverse=True)

top_recs = predictions[:10]
print("Top 10 Recommendations:")
for pred in top_recs:
    movie_title = df[df['movieId'] == pred.iid]['title'].iloc[0]
    print(f"{movie_title} (predicted rating: {pred.est:.2f})")

# OMDB integration for details
def get_movie_details(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    resp = requests.get(url)
    return resp.json()

print("\\nSample movie detail (top rec):")
print(json.dumps(get_movie_details(df[df['movieId'] == top_recs[0].iid]['title'].iloc[0]), indent=2))

print("Recommendation complete!")
