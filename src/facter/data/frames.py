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


@dataclass(frozen=True)
class Sushi3Frames:
    """Loader for the Sushi3 dataset (as shipped in data/raw/sushi3-2016).

    We use the item set B (100 items) by default:
      - `sushi3.idata`: item metadata (tab-separated)
      - `sushi3.udata`: user features (tab-separated)
      - `sushi3b.5000.10.order`: top-10 preference order for item set B

    Notes:
      - The .order files encode *most preferred first*.
      - User features are numeric codes; we map a few into the pipeline's
        protected columns (gender/age/occupation) in a simple, deterministic way.
    """

    raw_dir: Path
    variant: str = "b"  # "a" (10 items) | "b" (100 items)

    def __post_init__(self):
        idata_path = self.raw_dir / "sushi3.idata"
        udata_path = self.raw_dir / "sushi3.udata"
        if self.variant == "a":
            order_path = self.raw_dir / "sushi3a.5000.10.order"
        else:
            order_path = self.raw_dir / "sushi3b.5000.10.order"

        if not idata_path.exists() or not udata_path.exists() or not order_path.exists():
            raise FileNotFoundError(
                f"Missing Sushi3 files under {self.raw_dir}. Expected: sushi3.idata, sushi3.udata, and sushi3{self.variant}.5000.10.order"
            )

        # Items: tab-separated. Columns documented in README-en.txt.
        # We only need (item_id, name) for the pipeline.
        items = pd.read_csv(
            idata_path,
            sep="\t",
            header=None,
        )
        items = items.rename(columns={0: "mid", 1: "title"})
        items["mid"] = items["mid"].astype(int)
        items["title"] = items["title"].astype(str)

        # Users: tab-separated numeric codes.
        # Sushi3 has many user fields; for our fairness experiments we treat
        # only gender+age as "protected" by default.
        # The Sushi3 udata format is described in README; we map:
        #   col1: gender (0/1) -> {"F","M"}
        #   col2: age group code -> int
        #   col3: other demographic/region code -> used as occupation proxy
        users = pd.read_csv(
            udata_path,
            sep="\t",
            header=None,
        )
        users = users.rename(columns={0: "uid"})
        users["uid"] = users["uid"].astype(int)

        # Defensive: udata should have at least 4 cols.
        if users.shape[1] < 4:
            raise ValueError(f"Unexpected sushi3.udata shape: {users.shape}")

        # gender
        users["gender"] = users[1].map({0: "F", 1: "M"}).fillna("U")
        # age (keep as int-ish code)
        users["age"] = users[2].astype(int)

        # NOTE: we intentionally do NOT set an "occupation" column here.
        # If the pipeline needs one for stratification, it can be synthesized
        # deterministically during dataset building (see protocol.py), similar
        # to how we handle Amazon.

        # Orders: whitespace-separated text; first line is header.
        # Each subsequent line: "0 <Xi> <item_id_1> ... <item_id_Xi>".
        orders_rows = []
        with order_path.open("r", encoding="utf-8", errors="ignore") as f:
            f.readline()  # header, e.g., "100 1" (ignored)
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                # parts[0] is a constant 0 in the release; parts[1] is Xi
                xi = int(parts[1])
                mids = [int(x) for x in parts[2 : 2 + xi]]
                orders_rows.append({"row_idx": int(line_idx), "ranked_mids": mids})

        orders = pd.DataFrame(orders_rows)
        # Align orders to users by file row order (README: each line corresponds to user line)
        # We'll create a stable row index and merge.
        users = users.reset_index(drop=True).reset_index().rename(columns={"index": "row_idx"})
        orders = orders.merge(users[["row_idx", "uid", "gender", "age"]], on="row_idx", how="inner")

        object.__setattr__(self, "items", items)
        object.__setattr__(self, "users", users)
        object.__setattr__(self, "orders", orders)

    def build_item_db(self) -> Dict[int, Dict[str, str]]:
        m = self.items.copy()
        m["mid"] = m["mid"].astype(int)
        return m.set_index("mid")[["title"]].to_dict(orient="index")
