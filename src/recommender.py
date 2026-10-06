"""Recommend similar movies using a precomputed cosine similarity matrix."""

import numpy as np
import pandas as pd

from explore_data import load_data
from features import build_genre_matrix
from similarity import build_similarity_matrix


def find_movie_index(title: str, movies: pd.DataFrame) -> int:
    """Return the row position of a movie (case-insensitive exact title match).

    Row positions line up with the similarity matrix because both are built
    from `movies` in the same order.
    """
    query = title.strip()
    matches = movies.index[movies["title"].str.lower() == query.lower()]

    if len(matches) == 0:
        suggestions = (
            movies.loc[movies["title"].str.contains(query, case=False, regex=False), "title"]
            .head(5)
            .tolist()
        )
        raise ValueError(f"Movie '{title}' not found. Did you mean: {suggestions}")

    return int(matches[0])  # a few titles are duplicated; we take the first


def recommend(
    title: str, movies: pd.DataFrame, similarity: np.ndarray, top_n: int = 5
) -> pd.DataFrame:
    """Return the top_n movies most similar to `title`."""
    idx = find_movie_index(title, movies)
    scores = similarity[idx]  # similarity of this movie to all others

    if scores.max() == 0:
        raise ValueError(f"'{movies.loc[idx, 'title']}' has no genre data to compare.")

    # Negate so a stable ascending sort gives high-to-low order with
    # predictable tie-breaking (ties keep their original row order).
    ranked = np.argsort(-scores, kind="stable")
    ranked = ranked[ranked != idx][:top_n]  # drop the movie itself

    result = movies.iloc[ranked][["title", "genres"]].copy()
    result["similarity"] = scores[ranked].round(3)
    return result.reset_index(drop=True)


if __name__ == "__main__":
    movies_df, _ = load_data()
    sim = build_similarity_matrix(build_genre_matrix(movies_df))

    for query in ["Toy Story (1995)", "toy story (1995)", "Heat (1995)"]:
        print(f"\nBecause you liked: {query}")
        print(recommend(query, movies_df, sim).to_string())

    # Error handling demo: no year in the title
    try:
        recommend("Toy Story", movies_df, sim)
    except ValueError as err:
        print("\nExpected error:", err)