"""Read and write small JSON files.

This module provides thin wrappers around :mod:`json` for common file I/O
patterns used in this repository.
"""

import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file into a Python dictionary.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON object.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write a JSON-serializable object to disk.

    The parent directory is created if it does not already exist.

    Args:
        path (Path): Output JSON file path.
        obj (Dict[str, Any]): JSON-serializable object to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
