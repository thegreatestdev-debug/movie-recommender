"""Turn movie genres into a binary (one-hot) feature matrix."""

import pandas as pd

from explore_data import load_data

PLACEHOLDER_GENRE = "(no genres listed)"


def build_genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of 0/1 columns, one per genre, indexed by movieId."""
    # Split "Comedy|Romance" on "|" and create one 0/1 column per genre.
    genre_matrix = movies["genres"].str.get_dummies(sep="|")

    # "(no genres listed)" is a placeholder, not a real genre. Dropping its
    # column leaves those movies as all-zero rows.
    genre_matrix = genre_matrix.drop(columns=PLACEHOLDER_GENRE)

    # Keep track of which row belongs to which movie.
    genre_matrix.index = movies["movieId"]
    return genre_matrix


if __name__ == "__main__":
    movies_df, _ = load_data()
    matrix = build_genre_matrix(movies_df)

    print("Matrix shape:", matrix.shape)
    print("Columns:", list(matrix.columns))
    print(matrix.head())

    print("\nGenre totals (should match your EDA counts):")
    print(matrix.sum().sort_values(ascending=False).head(5))

    all_zero_rows = (matrix.sum(axis=1) == 0).sum()
    print("\nMovies with no genre at all:", all_zero_rows)