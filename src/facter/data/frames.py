"""Load raw dataset files into pandas DataFrames and construct item databases.

This module provides thin wrappers around raw dataset files (e.g., MovieLens-1M
and Amazon Movies&TV) to:

- Load the raw files into pandas DataFrames with canonicalized column names.
- Apply lightweight preprocessing consistent with the repository's expectations.
- Build an item database mapping item IDs to text fields used downstream.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class MovieLensFrames:
    """Load MovieLens-1M raw files as pandas DataFrames.

    Attributes:
        raw_dir (Path): Directory containing MovieLens-1M raw files.
        ratings (pd.DataFrame): Ratings DataFrame created in `__post_init__`.
        users (pd.DataFrame): Users DataFrame created in `__post_init__`.
        movies (pd.DataFrame): Movies DataFrame created in `__post_init__`.
    """

    raw_dir: Path

    def __post_init__(self):
        """Load MovieLens-1M from `raw_dir` and attach DataFrames.

        The computed DataFrames are stored as attributes on this frozen dataclass
        via `object.__setattr__`.

        Canonical DataFrame columns:
            ratings: uid, mid, rating, timestamp
            users: uid, gender, age, occupation, zip
            movies: mid, title, genres

        Returns:
            None
        """
        ratings_path = self.raw_dir / "ratings.dat"
        users_path = self.raw_dir / "users.dat"
        movies_path = self.raw_dir / "movies.dat"

        ratings = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["uid", "mid", "rating", "timestamp"],
        )
        users = pd.read_csv(
            users_path,
            sep="::",
            engine="python",
            names=["uid", "gender", "age", "occupation", "zip"],
        )
        movies = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["mid", "title", "genres"],
            encoding="latin-1",
        )
        ratings = ratings[ratings["rating"] >= 4].copy()


        # Assign dataframes as class attributes
        object.__setattr__(self, "ratings", ratings)
        object.__setattr__(self, "users", users)
        object.__setattr__(self, "movies", movies)

    def build_item_db(self) -> Dict[int, Dict[str, str]]:
        """Build an item database from the movies table.

        Returns:
            Dict[int, Dict[str, str]]: Mapping ``mid -> {"title": ..., "genres": ...}``.
        """
        m = self.movies.copy()
        m["mid"] = m["mid"].astype(int)
        return m.set_index("mid")[["title", "genres"]].to_dict(orient="index")


@dataclass(frozen=True)
class AmazonFrames:
    """Load Amazon Movies&TV ratings and metadata as pandas DataFrames.

    Attributes:
        raw_dir (Path): Directory containing extracted Amazon raw JSON files.
        ratings (pd.DataFrame): Ratings/reviews DataFrame created in `__post_init__`.
        metadata (pd.DataFrame): Item metadata DataFrame created in `__post_init__`.
    """

    raw_dir: Path

    def __post_init__(self):
        """Load Amazon Movies&TV 5-core data from `raw_dir` and attach DataFrames.

        The ratings JSON is loaded and renamed into canonical columns. The
        metadata JSON is reduced to the item identifier and title.

        Columns (as used by this repository):
            ratings: rating, verified, reviewTime, uid, mid, style, reviewerName,
                text, summary, timestamp, vote, image
            metadata: mid, title

        Notes:
            This implementation maps the original item IDs (ASIN strings) to
            contiguous integer IDs based on the order of unique IDs in the
            metadata table.

        Returns:
            None
        """
        ratings_path = self.raw_dir / "Movies_and_TV_5.json"
        metadata_path = self.raw_dir / "meta_Movies_and_TV_5.json"

        ratings = pd.read_json(ratings_path, lines=True)
        metadata = pd.read_json(metadata_path, lines=True)

        # Preprocessing
        ratings = ratings.rename(
            columns={
                "overall": "rating",
                "reviewerID": "uid",
                "asin": "mid",
                "reviewText": "text",
                "unixReviewTime": "timestamp",
            }
        )
        ratings = ratings[ratings["rating"] >= 4].copy()

        metadata = metadata[["asin", "title"]].drop_duplicates()
        metadata = metadata.rename(columns={"asin": "mid"})

        # map MID to simple index integer IDs
        unique_mids = metadata["mid"].unique()
        mid_to_int = {mid: idx for idx, mid in enumerate(unique_mids)}
        ratings["mid"] = ratings["mid"].map(mid_to_int)
        metadata["mid"] = metadata["mid"].map(mid_to_int)

        # Assign dataframes as class attributes
        object.__setattr__(self, "ratings", ratings)
        object.__setattr__(self, "metadata", metadata)

    def build_item_db(self) -> Dict[int, Dict[str, str]]:
        """Build an item database from the metadata table.

        Returns:
            Dict[int, Dict[str, str]]: Mapping ``mid -> {"title": ...}``.
        """
        m = self.metadata.copy()
        m["mid"] = m["mid"].astype(int)
        return m.set_index("mid")[["title"]].to_dict(orient="index")
