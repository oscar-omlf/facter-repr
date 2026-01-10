import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests

from .paths import DOWNLOADS_DIR, RAW_DIR, ensure_dirs


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    url: str
    # For zip datasets: the folder name created inside the zip (e.g., "ml-1m")
    extracted_subdir: Optional[str] = None


MOVIELENS_1M = DownloadSpec(
    name="ml-1m",
    url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    extracted_subdir="ml-1m",
)

AMAZON_MOVIES_TV_5 = DownloadSpec(
    name="amazon-movies-tv-5",
    url="https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Movies_and_TV_5.json.gz",
    extracted_subdir=None,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(manifest_path: Path, payload: Dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def download_file(url: str, out_path: Path, timeout: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_movielens_1m(force: bool = False) -> Path:
    """
    Downloads and extracts MovieLens-1M into data/raw/ml-1m/.
    Writes data/raw/ml-1m/manifest.json with URL + hashes.
    Returns the extracted directory path.
    """
    ensure_dirs()
    target_dir = RAW_DIR / "ml-1m"
    manifest_path = target_dir / "manifest.json"
    zip_path = DOWNLOADS_DIR / "ml-1m.zip"

    if target_dir.exists() and (target_dir / "ratings.dat").exists() and not force:
        return target_dir

    download_file(MOVIELENS_1M.url, zip_path)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)  # zip contains "ml-1m/" folder

    if not target_dir.exists():
        raise RuntimeError(f"Expected extracted dir not found: {target_dir}")

    # Hash key files
    key_files = ["ratings.dat", "users.dat", "movies.dat"]
    file_hashes = {}
    for fn in key_files:
        p = target_dir / fn
        if not p.exists():
            raise RuntimeError(f"Missing expected file after extraction: {p}")
        file_hashes[fn] = sha256_file(p)

    payload = {
        "dataset": MOVIELENS_1M.name,
        "url": MOVIELENS_1M.url,
        "downloaded_at_unix": int(time.time()),
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "files_sha256": file_hashes,
    }
    write_manifest(manifest_path, payload)
    return target_dir


def download_amazon_movies_tv_5(force: bool = False) -> Path:
    """
    Downloads Amazon Movies&TV 5-core json.gz into data/raw/amazon/Movies_and_TV_5.json.gz
    Writes manifest with URL + hash.
    Returns the directory path containing the gz.
    """
    ensure_dirs()
    target_dir = RAW_DIR / "amazon"
    target_dir.mkdir(parents=True, exist_ok=True)
    gz_path = target_dir / "Movies_and_TV_5.json.gz"
    manifest_path = target_dir / "manifest.json"

    if gz_path.exists() and not force:
        return target_dir

    download_file(AMAZON_MOVIES_TV_5.url, gz_path)

    payload = {
        "dataset": AMAZON_MOVIES_TV_5.name,
        "url": AMAZON_MOVIES_TV_5.url,
        "downloaded_at_unix": int(time.time()),
        "gz_path": str(gz_path),
        "gz_sha256": sha256_file(gz_path),
    }
    write_manifest(manifest_path, payload)
    return target_dir
