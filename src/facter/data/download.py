import gzip
import hashlib
import json
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

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

META_AMAZON_MOVIES_TV_5 = DownloadSpec(
    name="meta-amazon-movies-tv-5",
    url="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz",
    extracted_subdir=None,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_gz(archive_path: Path, out_path: Path) -> None:
    """Decompresses a .gz file to the specified out_path."""

    print(f"Extracting {archive_path.name} to {out_path.name}...")

    with gzip.open(archive_path, "rb") as f_in:
        with out_path.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def write_manifest(manifest_path: Path, payload: Dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def download_file(
    url: str, out_path: Path, timeout: int = 60, verify: Optional[bool] = None
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout, verify=verify) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024 * 1024)):
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
    Downloads Amazon Movies&TV 5-core & Metadata into DOWNLOADS_DIR.
    Extracts them into data/raw/amazon/ as .json files.
    Writes manifest with hashes of the EXTRACTED files.
    Returns the directory path containing the gz.
    """
    ensure_dirs()
    target_dir = RAW_DIR / "amazon"
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.json"

    # Define the files to process: (DownloadSpec, Archive Path, Extracted File Path)
    # We download to DOWNLOADS_DIR, but extract to RAW_DIR/amazon
    files_to_process = [
        (
            AMAZON_MOVIES_TV_5,
            DOWNLOADS_DIR / "Movies_and_TV_5.json.gz",
            target_dir / "Movies_and_TV_5.json",
        ),
        (
            META_AMAZON_MOVIES_TV_5,
            DOWNLOADS_DIR / "meta_Movies_and_TV_5.json.gz",
            target_dir / "meta_Movies_and_TV_5.json",
        ),
    ]

    # Check if all extracted files exist
    all_exist = all(extracted.exists() for _, _, extracted in files_to_process)
    if all_exist and not force:
        return target_dir

    file_hashes = {}
    archive_info = {}

    for spec, archive_path, extracted_path in files_to_process:
        # 1. Download archive to DOWNLOADS_DIR
        download_file(spec.url, archive_path, verify=False)

        # 2. Extract to data/raw/amazon/
        extract_gz(archive_path, extracted_path)

        # 3. Hash the extracted .json file
        file_hash = sha256_file(extracted_path)
        file_hashes[extracted_path.name] = file_hash

        # Store archive info for the manifest
        archive_info[spec.name] = {
            "url": spec.url,
            "archive_path": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
        }

    # 4. Write Manifest
    # We structure this to look like ML-1M, but accommodating multiple source files
    payload = {
        "dataset": "amazon-movies-tv-5-plus-meta",
        "downloaded_at_unix": int(time.time()),
        "files_sha256": file_hashes,  # Hashes of the .json files
        "archives": archive_info,  # Details about the source .gz files
    }
    write_manifest(manifest_path, payload)

    return target_dir


def download_dataset(dataset: str = "ml-1m", force: bool = False) -> Path:
    if dataset == "ml-1m":
        target_dir = download_movielens_1m(force=force)

    elif dataset == "amazon":
        target_dir = download_amazon_movies_tv_5(force=force)

    return target_dir
