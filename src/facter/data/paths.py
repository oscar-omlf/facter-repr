from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOWNLOADS_DIR = RAW_DIR / "_downloads"
PROCESSED_DIR = DATA_DIR / "processed"

def ensure_dirs() -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)