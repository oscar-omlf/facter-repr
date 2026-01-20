import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    Eager-loading Dataset. Loads all JSONL lines into memory.
    Necessary because run_facter.py requires access to raw .data for DataFrame creation.
    """

    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

        self.data: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def dict_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Optimized collate function:
    - Numeric scalar fields -> torch.Tensor (via numpy for speed)
    - List of numerics -> torch.Tensor (if shapes match, via numpy)
    - Everything else -> List[Any]
    """
    if not batch:
        return {}

    keys = batch[0].keys()
    collated = {}

    for key in keys:
        sample_val = batch[0][key]

        # 1. Scalar Numerics
        if isinstance(sample_val, (int, float)):
            arr = np.array([item[key] for item in batch])
            # Infer correct type based on numpy dtype
            if np.issubdtype(arr.dtype, np.integer):
                collated[key] = torch.from_numpy(arr).long()

            else:
                collated[key] = torch.from_numpy(arr).float()

        # 2. Lists of Numerics (e.g. candidate_mids)
        elif (
            isinstance(sample_val, list)
            and len(sample_val) > 0
            and isinstance(sample_val[0], (int, float))
        ):
            # Attempt to tensorize.
            is_fixed_len = all(len(item[key]) == len(sample_val) for item in batch)
            if is_fixed_len:
                arr = np.array([item[key] for item in batch])
                if np.issubdtype(arr.dtype, np.integer):
                    collated[key] = torch.from_numpy(arr).long()

                else:
                    collated[key] = torch.from_numpy(arr).float()
            else:
                collated[key] = [item[key] for item in batch]

        # 3. Tensors (already tensors in dataset? unlikely but possible)
        elif isinstance(sample_val, torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])

        # 4. Everything else
        else:
            collated[key] = [item[key] for item in batch]

    return collated
