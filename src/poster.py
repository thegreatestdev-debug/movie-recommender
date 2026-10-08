"""Look up movie poster URLs from TMDB."""

import pandas as pd
import requests
from download_data import ensure_dataset
from explore_data import DATA_DIR

TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w185"
from download_data import ensure_dataset


def load_links() -> pd.DataFrame:
    """movieId -> tmdbId lookup from links.csv (some tmdbIds are missing)."""
    ensure_dataset()
    return pd.read_csv(DATA_DIR / "links.csv").set_index("movieId")


def load_links() -> pd.DataFrame:
    """movieId -> tmdbId lookup from links.csv (some tmdbIds are missing)."""
    return pd.read_csv(DATA_DIR / "links.csv").set_index("movieId")


def get_poster_url(movie_id: int, links: pd.DataFrame, token: str) -> str | None:
    """Return a poster URL, or None if anything is missing or fails."""
    if movie_id not in links.index:
        return None
    tmdb_id = links.loc[movie_id, "tmdbId"]
    if pd.isna(tmdb_id):
        return None

    try:
        response = requests.get(
            TMDB_MOVIE_URL.format(tmdb_id=int(tmdb_id)),
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None  # network error, bad token, unknown id...

    poster_path = response.json().get("poster_path")
    return f"{IMAGE_BASE_URL}{poster_path}" if poster_path else None


if __name__ == "__main__":
    import tomllib
    from pathlib import Path

    secrets_file = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
    token = tomllib.loads(secrets_file.read_text())["TMDB_TOKEN"]

    links_df = load_links()
    # Toy Story, Jumanji, La cravate, and a movieId that doesn't exist
    for movie_id in [1, 2, 114335, 999999999]:
        print(movie_id, "->", get_poster_url(movie_id, links_df, token))
