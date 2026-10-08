"""Streamlit dashboard for the content-based movie recommender."""
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI2N2Q5NDI1YzkwY2FmOGViYTNlMGM0NGUxNmJmYTliMSIsIm5iZiI6MTc5MTMyNDU4MC4wNDQsInN1YiI6IjZhYzU3MWE0M2FmMWFmZTkxOTY0ZWRmNyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.YDDrw40-RAYrnjgF0wpm5XYifbIbsgQlmyc4fa5SNuk"
import streamlit as st
from explore_data import load_data
from features import build_genre_matrix
from poster import get_poster_url, load_links
from recommender import recommend
from similarity import build_similarity_matrix
 
# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Movie Recommender | Your Name",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

POSTERS_PER_ROW = 5

# --- Cached Data Loading Functions ---

@st.cache_resource(show_spinner="Building similarity matrix...")
def load_model():
    """Loads data and builds the similarity matrix once."""
    movies, _ = load_data()
    return movies, build_similarity_matrix(build_genre_matrix(movies))

@st.cache_resource
def load_link_table():
    """Loads the TMDB link table once."""
    return load_links()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_poster(movie_id: int, token: str) -> str | None:
    """One TMDB call per movie per hour, no matter how often the page reruns."""
    return get_poster_url(movie_id, load_link_table(), token)

# --- Helper Functions ---

def get_token() -> str | None:
    """Retrieve TMDB token from Streamlit secrets."""
    try:
        return st.secrets["TMDB_TOKEN"]
    except (FileNotFoundError, KeyError):
        return None

def show_results(results, token):
    """Display the recommendation results in a grid."""
    for start in range(0, len(results), POSTERS_PER_ROW):
        row = results.iloc[start : start + POSTERS_PER_ROW]
        cols = st.columns(POSTERS_PER_ROW)
        
        for col, movie in zip(cols, row.itertuples()):
            with col:
                poster = cached_poster(int(movie.movieId), token) if token else None
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown("🖼️ *No poster*")
                
                st.caption(f"**{movie.title}**\n{movie.genres.replace('|', ' · ')}")

# --- Main App Execution ---

# Header
st.title("🎬 Content-Based Movie Recommender")
st.markdown("""
    **Find movies with similar genres instantly.** 
    *Built with Python, Pandas, Scikit-Learn (Cosine Similarity), and Streamlit.*
""")
st.divider()

# Load data
movies, similarity = load_model()
token = get_token()

if not token:
    st.info("💡 No TMDB token found, so posters are turned off.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    title = st.selectbox(
        "Choose a movie",
        options=sorted(movies["title"].unique()),
        index=None,
        placeholder="Start typing a title...",
    )
    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Main Content Area
if title:
    try:
        results = recommend(title, movies, similarity, top_n=top_n)
    except ValueError as err:
        st.warning(str(err))
    else:
        st.subheader(f"Because you liked {title}")
        with st.spinner("Fetching posters..."):
            show_results(results, token)
else:
    st.write("⬅️ Select a movie from the sidebar to get recommendations!")

# Footer
st.divider()
st.caption("Built by [Your Name](https://your-portfolio-link.com) | Data from MovieLens | Images from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.")
