from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from facter.data.prompts import AGE_ID2LABEL, OCC_ID2LABEL
from facter.data.prompts import PromptConfig, build_ranking_prompt
from facter.fairness.scoring import item_text
from facter.models.embedder import TextEmbedder
from facter.models.ranker import Ranker


_ML_AGE_BUCKETS = sorted(AGE_ID2LABEL.keys())   # [1, 18, 25, 35, 45, 50, 56]
_ML_OCC_IDS = sorted(OCC_ID2LABEL.keys())       # [0..20]

@dataclass(frozen=True)
class CFRConfig:
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    k: int = 10
    # which attribute to flip for CFR. You can run multiple and average externally.
    flip_attr: str = "gender"


def flip_protected_value(attr: str, value) -> str:
    """
    Minimal in-domain flips (MovieLens):
    - gender: M <-> F
    - age: move to adjacent valid age bucket (never produce invalid codes)
    - occupation: (o + 1) mod 21
    """
    if attr == "gender":
        v = str(value)
        if v.upper() == "M":
            return "F"
        if v.upper() == "F":
            return "M"
        return v

    if attr == "age":
        try:
            a = int(value)
        except Exception:
            return str(value)

        if a not in _ML_AGE_BUCKETS:
            # snap to nearest bucket
            a = min(_ML_AGE_BUCKETS, key=lambda x: abs(x - a))

        idx = _ML_AGE_BUCKETS.index(a)
        if idx < len(_ML_AGE_BUCKETS) - 1:
            return str(_ML_AGE_BUCKETS[idx + 1])
        else:
            return str(_ML_AGE_BUCKETS[max(idx - 1, 0)])

    if attr == "occupation":
        try:
            o = int(value)
        except Exception:
            return str(value)
        if o not in _ML_OCC_IDS:
            o = 0
        return str((o + 1) % (max(_ML_OCC_IDS) + 1))

    return str(value)


def _embed_list_mean(embedder: TextEmbedder, mids: Sequence[int], item_db: Dict[int, Dict[str, str]]) -> np.ndarray:
    texts = [item_text(int(m), item_db) for m in mids]
    embs = embedder.encode_texts(texts)  # [K,D], normalized
    v = np.mean(embs, axis=0).astype(np.float32)
    # normalize mean vector to compare with cosine/dot
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def _l2_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v))


def compute_cfr(
    df: pd.DataFrame,
    ranker: Ranker,
    embedder: TextEmbedder,
    item_db: Dict[int, Dict[str, str]],
    prompt_cfg: PromptConfig,
    cfg: CFRConfig,
    iter: Optional[int] = None,
) -> float:
    """
    CFR proxy via counterfactual flips:
      CFR = mean_{examples} || f(x,a) - f(x,a') ||_2
    where f(.) is the mean embedding of top-k recommended items.
    """
    if cfg.flip_attr not in cfg.protected_cols:
        raise ValueError(f"flip_attr must be one of {cfg.protected_cols}")

    dists: List[float] = []
    for _, row in df.iterrows():
        cand_titles = row["candidate_titles"]
        prompt_orig = row["prompt_rank"]
        system_prompt = row[f"system_prompt_iter{iter}" if iter is not None else "system_prompt"]

        # counterfactual prompt
        row_cf = row.to_dict()
        row_cf[cfg.flip_attr] = flip_protected_value(cfg.flip_attr, row_cf[cfg.flip_attr])
        prompt_cf = build_ranking_prompt(row_cf, cand_titles, prompt_cfg)

        idx_orig, _ = ranker.rank(prompt_orig, cand_titles, system_prompt=system_prompt)
        idx_orig = idx_orig[: cfg.k]
        idx_cf, _ = ranker.rank(prompt_cf, cand_titles, system_prompt=system_prompt)
        idx_cf = idx_cf[: cfg.k]

        mids_orig = [int(row["candidate_mids"][i]) for i in idx_orig]
        mids_cf = [int(row["candidate_mids"][i]) for i in idx_cf]

        v_orig = _embed_list_mean(embedder, mids_orig, item_db)
        v_cf = _embed_list_mean(embedder, mids_cf, item_db)
        dists.append(_l2_distance(v_orig, v_cf))

    return float(np.mean(dists)) if dists else 0.0

