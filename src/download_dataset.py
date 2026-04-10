import os

os.makedirs('src/data', exist_ok=True)

print("TMDB 5000 Movies dataset required in src/data/")
print("Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
print("Files needed: tmdb_5000_movies.csv, tmdb_5000_credits.csv")
print("")
print("Using Kaggle CLI (recommended):")
print("1. pip install kaggle")
print("2. kaggle datasets download -d tmdb/tmdb-movie-metadata -p src/data --unzip")
print("3. Rename extracted files if needed.")
print("")
print("Manual download if no Kaggle account.")
print("Run python src/preprocess.py after dataset is ready.")
