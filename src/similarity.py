"""Compute cosine similarity between every pair of movies."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from explore_data import load_data
from features import build_genre_matrix


def build_similarity_matrix(genre_matrix: pd.DataFrame) -> np.ndarray:
    """Return an (n_movies x n_movies) array of cosine similarities.

    Row i / column j holds the similarity between movie i and movie j,
    in the same order as genre_matrix's rows.
    """
    # float32 uses half the memory of the default float64.
    features = genre_matrix.to_numpy(dtype=np.float32)
    return cosine_similarity(features)


def position_of(genre_matrix: pd.DataFrame, movie_id: int) -> int:
    """Row number in the matrix for a given movieId."""
    return genre_matrix.index.get_loc(movie_id)


if __name__ == "__main__":
    movies_df, _ = load_data()
    genre_matrix = build_genre_matrix(movies_df)
    sim = build_similarity_matrix(genre_matrix)

    print("Shape:", sim.shape, "| dtype:", sim.dtype)
    print(f"Memory: {sim.nbytes / 1e6:.0f} MB")

    toy_story = position_of(genre_matrix, 1)
    grumpier = position_of(genre_matrix, 3)
    exhale = position_of(genre_matrix, 4)
    print("\nToy Story vs Grumpier Old Men:", round(float(sim[toy_story, grumpier]), 3))
    print("Grumpier vs Waiting to Exhale:", round(float(sim[grumpier, exhale]), 3))

    print("\nSymmetric:", np.allclose(sim, sim.T))
    print("Movies with self-similarity 1.0:", int(np.isclose(np.diag(sim), 1.0).sum()))

    zero_rows = np.flatnonzero(genre_matrix.sum(axis=1).to_numpy() == 0)
    print("Max similarity for a no-genre movie:", float(sim[zero_rows[0]].max()))