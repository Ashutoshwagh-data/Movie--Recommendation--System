import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

st.title("🎬 Movie Recommendation System by Aniket Gund")

# --------------------------------
# Load data
# --------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df.dropna(subset=["title", "genres", "Year"])
    df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)
    df["content"] = df["title"] + " " + df["genres_clean"]
    return df

df = load_data()

# --------------------------------
# Load model + embeddings
# --------------------------------
@st.cache_resource
def load_model_and_embeddings(data):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        data["content"].tolist(),
        show_progress_bar=False
    )
    return model, embeddings

model, embeddings = load_model_and_embeddings(df)

# --------------------------------
# Recommendation logic
# --------------------------------
def recommend_movies(movie_query, top_n, year_filter, genre_filter):

    # Partial, case-insensitive match
    matches = df[df["title"].str.lower().str.contains(movie_query.lower())]

    if matches.empty:
        st.error("No matching movie found. Try a different name.")
        return None, pd.DataFrame()

    # Take the first relevant match
    idx = matches.index[0]
    matched_movie = matches.iloc[0]

    similarity_scores = cosine_similarity(
        embeddings[idx].reshape(1, -1),
        embeddings
    ).flatten()

    result_df = df.copy()
    result_df["similarity"] = similarity_scores

    # Remove selected movie
    result_df = result_df[result_df.index != idx]

    # Year filter
    if year_filter is not None:
        result_df = result_df[result_df["Year"] == year_filter]

    # Genre filter (any match)
    if genre_filter:
        pattern = "|".join(genre_filter)
        result_df = result_df[
            result_df["genres"].str.contains(pattern, case=False, regex=True)
        ]

    result_df = result_df.sort_values(
        by="similarity",
        ascending=False
    )

    return matched_movie, result_df.head(top_n)[
        ["title", "Year", "genres", "similarity"]
    ]

# --------------------------------
# UI controls
# --------------------------------
movie_input = st.text_input(
    "Enter movie name",
    placeholder="e.g. toy story, jumanji, godfather"
)

year_option = st.selectbox(
    "Filter by year",
    ["All"] + sorted(df["Year"].unique().tolist())
)

common_genres = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"
]

selected_genres = st.multiselect(
    "Filter by genres (optional)",
    common_genres
)

top_n = st.number_input(
    "Number of recommendations",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# --------------------------------
# Action
# --------------------------------
if st.button("Recommend"):

    if movie_input.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        year_filter = None if year_option == "All" else year_option

        matched_movie, recommendations = recommend_movies(
            movie_input,
            top_n,
            year_filter,
            selected_genres
        )

        if matched_movie is not None:
            st.subheader("🎥 Suggested Movie")
            st.markdown(f"### {matched_movie['title']}")
            st.markdown(f"**Year:** {matched_movie['Year']}")
            st.markdown(f"**Genres:** {matched_movie['genres']}")

        if not recommendations.empty:
            st.subheader("🎯 Recommended Movies")
            st.dataframe(
                recommendations.reset_index(drop=True),
                use_container_width=True
            )
