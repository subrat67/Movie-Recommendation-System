import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import json
from sklearn.preprocessing import MultiLabelBinarizer

# Create data dir
os.makedirs('src/data', exist_ok=True)

print("Checking for dataset...")
if not os.path.exists('src/data/tmdb_5000_movies.csv') or not os.path.exists('src/data/tmdb_5000_credits.csv'):
    print("Dataset not found. Run python src/download_dataset.py or download manually.")
    exit(1)

print("Loading dataset...")
movies_df = pd.read_csv('src/data/tmdb_5000_movies.csv')
credits_df = pd.read_csv('src/data/tmdb_5000_credits.csv')

print("Merging movies and credits...")
df = movies_df.merge(credits_df, on='title')

print("Processing genres...")
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in json.loads(x)])

print("Processing cast and crew...")
df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in json.loads(x)][:3])
df['crew'] = df['crew'].apply(lambda x: [i['name'] for i in json.loads(x) if i['job'] == 'Director'])

print("Processing overview...")
df['overview'] = df['overview'].fillna('')

print("Computing weighted averages...")
def weighted_rating(x, m=2456537, C=8.521):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

df['score'] = df.apply(weighted_rating, axis=1)

print("Computing genre averages...")
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df['genres'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=df.index)

genre_means = {}
for genre in mlb.classes_:
    genre_mask = genre_df[genre] == 1
    genre_means[genre] = df.loc[genre_mask, 'vote_average'].mean() if genre_mask.sum() > 0 else 0

print("Creating soup for TF-IDF...")
df['soup'] = (df['genres'].apply(lambda x: ' '.join(x)) + ' ' +
              df['cast'].apply(lambda x: ' '.join(x)) + ' ' +
              df['crew'].apply(lambda x: ' '.join(x)) + ' ' +
              df['overview'])

print("Saving processed data...")
df.to_pickle('src/data/movies_processed.pkl')
with open('src/data/genre_means.json', 'w') as f:
    json.dump(genre_means, f)

print("Preprocessing complete!")
