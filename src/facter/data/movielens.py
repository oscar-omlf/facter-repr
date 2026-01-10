from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class MovieLensFrames:
    ratings: pd.DataFrame
    users: pd.DataFrame
    movies: pd.DataFrame


def load_ml1m(raw_dir: Path) -> MovieLensFrames:
    """
    Loads MovieLens-1M from raw_dir (data/raw/ml-1m).
    Returns ratings/users/movies dataframes with canonical columns.

    ratings: uid, mid, rating, timestamp
    users: uid, gender, age, occupation, zip
    movies: mid, title, genres
    """
    ratings_path = raw_dir / "ratings.dat"
    users_path = raw_dir / "users.dat"
    movies_path = raw_dir / "movies.dat"

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
    return MovieLensFrames(ratings=ratings, users=users, movies=movies)


def build_item_db(movies: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    """
    Returns {mid: {"title": ..., "genres": ...}}
    """
    m = movies.copy()
    m["mid"] = m["mid"].astype(int)
    return m.set_index("mid")[["title", "genres"]].to_dict(orient="index")
