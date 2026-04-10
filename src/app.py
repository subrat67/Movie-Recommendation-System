
import streamlit as st
import pandas as pd
import json
import requests
import difflib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Page config
st.set_page_config(page_title="Movie Recommender", layout="wide")

def load_data():
    df = pd.read_pickle('src/data/movies_processed.pkl')
    with open('src/data/genre_means.json') as f:
        genre_means = json.load(f)
    with open('src/config.json') as f:
        config = json.load(f)
    return df, genre_means, config['OMDB_API_KEY']

df, genre_means, OMDB_API_KEY = load_data()

def get_movie_details(title):
    url = f"http://www.omdbapi.com/?t={title.replace(' ', '+')}&apikey={OMDB_API_KEY}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

# Title
st.title("🎬 Movie Recommender")

# Tabs
tab1, tab2, tab3 = st.tabs(["Top Movies", "Get Recommendations", "Browse All"])

with tab1:
    st.subheader("Top 10 Highest Rated Movies")
    top_movies = df.nlargest(10, 'score')[['title', 'vote_average', 'score', 'genres']]
    st.dataframe(top_movies)

with tab2:
    st.subheader("Find Movies Similar to Your Favorite")
    movie_title = st.text_input("Enter a movie title:")
    
    if movie_title:
        # Find closest match
        titles = df['title'].tolist()
        match = difflib.get_close_matches(movie_title, titles, n=1, cutoff=0.6)
        
        if not match:
            st.error("❌ Movie not found. Try another title.")
        else:
            movie = match[0]
            movie_row = df[df['title'] == movie].iloc[0]
            
            st.success(f"Found: **{movie}**")
            st.write(f"**Score:** {movie_row['score']:.2f} | **Rating:** {movie_row['vote_average']:.1f}")
            
            # Simple rec: high score movies with overlapping genres
            movie_genres = set(movie_row['genres']) if isinstance(movie_row['genres'], (list, set)) else set()
            candidates = df[df['score'] > movie_row['score'] * 0.9].copy()
            candidates['genre_overlap'] = candidates['genres'].apply(
                lambda g: len(set(g) & movie_genres) if isinstance(g, (list, set)) else 0
            )
            recs = candidates.nlargest(5, 'genre_overlap')[['title', 'score', 'genres', 'genre_overlap']]
            
            st.subheader("📽️ Recommended Movies:")
            for idx, (_, rec) in enumerate(recs.iterrows(), 1):
                with st.expander(f"{idx}. {rec['title']} (Score: {rec['score']:.2f})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Score:** {rec['score']:.2f}")
                        st.write(f"**Genres:** {', '.join(rec['genres']) if isinstance(rec['genres'], (list, set)) else rec['genres']}")
                    with col2:
                        details = get_movie_details(rec['title'])
                        if details and 'Poster' in details and details['Poster'] not in ['N/A', '']:
                            st.image(details['Poster'], width=150)

with tab3:
    st.subheader("Browse All Movies")
    search_term = st.text_input("Search movies:")
    if search_term:
        filtered_df = df[df['title'].str.contains(search_term, case=False, na=False)]
        if filtered_df.empty:
            st.warning(f"⚠️ No movies found matching '{search_term}'. Try a different search term.")
        else:
            st.dataframe(filtered_df[['title', 'vote_average', 'score', 'genres']])
    else:
        st.dataframe(df[['title', 'vote_average', 'score', 'genres']].head(50))

