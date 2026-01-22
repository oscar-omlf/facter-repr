from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class MovieLensFrames:
    raw_dir: Path

    def __post_init__(self):
        """
        Loads MovieLens-1M from raw_dir (data/raw/ml-1m).
        Returns ratings/users/movies dataframes with canonical columns.

        ratings: uid, mid, rating, timestamp
        users: uid, gender, age, occupation, zip
        movies: mid, title, genres
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
        """
        Returns {mid: {"title": ..., "genres": ...}}
        """
        m = self.movies.copy()
        m["mid"] = m["mid"].astype(int)
        return m.set_index("mid")[["title", "genres"]].to_dict(orient="index")


@dataclass(frozen=True)
class AmazonFrames:
    raw_dir: Path

    def __post_init__(self):
        """
        Loads Amazon Movies&TV 5-core + metadata from raw_dir (data/raw/amazon).
        Returns ratings/users/movies dataframes with canonical columns.

        ratings: 'rating', 'verified', 'reviewTime', 'uid', 'mid', 'style',
                'reviewerName', 'text', 'summary', 'timestamp', 'vote',
                'image'
        metadata: asin, title
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
        """
        Returns {mid: {"title": ...}}
        """
        m = self.metadata.copy()
        m["mid"] = m["mid"].astype(int)
        return m.set_index("mid")[["title"]].to_dict(orient="index")
