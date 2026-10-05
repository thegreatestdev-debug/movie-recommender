"""Download and extract the MovieLens Latest Small dataset."""

import zipfile
from pathlib import Path

import requests

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"
EXTRACTED_DIR = DATA_DIR / "ml-latest-small"


def download_dataset() -> None:
    """Download the zip file if we don't already have it."""
    DATA_DIR.mkdir(exist_ok=True)

    if ZIP_PATH.exists():
        print("Zip already downloaded, skipping.")
        return

    print("Downloading dataset...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()  # stop with a clear error if download fails
    ZIP_PATH.write_bytes(response.content)
    print(f"Saved to {ZIP_PATH}")


def extract_dataset() -> None:
    """Unzip the dataset if it isn't already extracted."""
    if EXTRACTED_DIR.exists():
        print("Already extracted, skipping.")
        return

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {EXTRACTED_DIR}")


if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    print("Files:", sorted(p.name for p in EXTRACTED_DIR.iterdir()))