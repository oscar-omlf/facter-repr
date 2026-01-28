"""Seed common RNG sources for best-effort reproducibility.

This module centralizes seeding for Python's RNG, NumPy, and PyTorch (CPU and
CUDA) and optionally configures PyTorch for deterministic algorithms.
"""

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class SeedConfig:
    """Configure seeding and determinism options for :func:`seed_all`.

    Attributes:
        seed (int): Seed value used for Python, NumPy, and PyTorch RNGs.
        deterministic (bool): Whether to enable PyTorch deterministic settings
            where possible.
        warn_only (bool): Whether PyTorch should warn instead of raising an
            error when nondeterministic operations are encountered.
        disable_tf32 (bool): Whether to disable TF32 for matmul/cuDNN to reduce
            numeric variability across GPUs.
    """
    seed: int
    deterministic: bool = True
    warn_only: bool = True  # if True, torch will warn instead of error for nondeterministic ops
    disable_tf32: bool = True  # improves determinism across GPUs at some perf cost


def seed_all(cfg: SeedConfig) -> Dict[str, Any]:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for best-effort determinism.

    Notes:
        - Full determinism on GPU is not always possible depending on
          ops/kernels.
        - If cfg.deterministic=True and warn_only=False, torch may raise errors
          when encountering nondeterministic operations.

    Args:
        cfg (SeedConfig): Seeding and determinism configuration.

    Returns:
        Dict[str, Any]: A dictionary summarizing the applied configuration and
        selected runtime backend flags.
    """
    # Must be set before many libraries initialize internal state
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    # cuBLAS determinism (matmul). Must be set before CUDA context init.
    # Valid values: ":16:8" or ":4096:8"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.disable_tf32:
        # TF32 can introduce small numeric differences
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if cfg.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # This enforces deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=cfg.warn_only)

    return {
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "warn_only": cfg.warn_only,
        "disable_tf32": cfg.disable_tf32,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }
