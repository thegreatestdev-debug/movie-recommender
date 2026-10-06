"""Streamlit dashboard for the content-based movie recommender."""

import streamlit as st

from explore_data import load_data
from features import build_genre_matrix
from recommender import recommend
from similarity import build_similarity_matrix


@st.cache_resource(show_spinner="Building similarity matrix...")
def load_model():
    """Load data and build the similarity matrix once, then reuse it."""
    movies, _ = load_data()
    similarity = build_similarity_matrix(build_genre_matrix(movies))
    return movies, similarity


st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Content-Based Movie Recommender")
st.write("Pick a movie and get others with the most similar genres.")

movies, similarity = load_model()

title = st.selectbox(
    "Choose a movie",
    options=sorted(movies["title"].unique()),
    index=None,
    placeholder="Start typing a title...",
)
top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if title:
    try:
        results = recommend(title, movies, similarity, top_n=top_n)
    except ValueError as err:  # e.g. a movie with no genre data
        st.warning(str(err))
    else:
        st.subheader(f"Because you liked {title}")
        st.dataframe(results, hide_index=True)