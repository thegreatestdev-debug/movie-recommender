"""Load MovieLens data and run basic exploratory data analysis (EDA)."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-latest-small"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read movies.csv and ratings.csv into DataFrames."""
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    return movies, ratings


def explore(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Print a quick profile of both DataFrames."""
    print("=== MOVIES ===")
    print("Shape (rows, columns):", movies.shape)
    print(movies.head(), "\n")
    movies.info()
    print("\nMissing values per column:\n", movies.isna().sum())
    print("\nDuplicate titles:", movies["title"].duplicated().sum())

    print("\n=== GENRE COUNTS ===")
    genre_counts = movies["genres"].str.split("|").explode().value_counts()
    print(genre_counts)

    print("\n=== RATINGS ===")
    print("Shape:", ratings.shape)
    print(ratings.head(), "\n")
    print(ratings["rating"].describe())
    print("\nUnique users:", ratings["userId"].nunique())
    print("Unique rated movies:", ratings["movieId"].nunique())


if __name__ == "__main__":
    movies_df, ratings_df = load_data()
    explore(movies_df, ratings_df)