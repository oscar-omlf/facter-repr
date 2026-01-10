from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from facter.data.prompts import PromptConfig, build_ranking_prompt
from facter.fairness.scoring import item_text
from facter.models.embedder import TextEmbedder
from facter.models.ranker import Ranker


@dataclass(frozen=True)
class CFRConfig:
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    k: int = 10
    # which attribute to flip for CFR. You can run multiple and average externally.
    flip_attr: str = "gender"


def flip_protected_value(attr: str, value) -> str:
    """
    Minimal flips (counterfactual):
    - gender: M <-> F
    - age: +1 (bounded)
    - occupation: +1 mod 1000 (safe integer wrap)
    """
    if attr == "gender":
        v = str(value)
        if v.upper() == "M":
            return "F"
        if v.upper() == "F":
            return "M"
        # unknown -> leave unchanged
        return v

    if attr == "age":
        try:
            a = int(value)
        except Exception:
            return str(value)
        # minimal perturbation
        a2 = a + 1
        if a2 > 100:
            a2 = a - 1
        return str(a2)

    if attr == "occupation":
        try:
            o = int(value)
        except Exception:
            return str(value)
        return str(o + 1)

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


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    # since normalized, cosine sim = dot
    sim = float(np.sum(u * v))
    return float(1.0 - sim)


def compute_cfr(
    df: pd.DataFrame,
    ranker: Ranker,
    embedder: TextEmbedder,
    item_db: Dict[int, Dict[str, str]],
    prompt_cfg: PromptConfig,
    cfg: CFRConfig,
    system_prompt: str | None = None,
) -> float:
    """
    CFR proxy via counterfactual flips:
      CFR = mean_{examples} [ 1 - cos( f(x,a), f(x,a') ) ]
    where f(.) is the mean embedding of top-k recommended items.
    """
    if cfg.flip_attr not in cfg.protected_cols:
        raise ValueError(f"flip_attr must be one of {cfg.protected_cols}")

    dists: List[float] = []
    for _, row in df.iterrows():
        # original prompt
        cand_titles = row["candidate_titles"]
        prompt_orig = row["prompt_rank"]

        # counterfactual prompt: rebuild with one attr flipped, everything else constant
        row_cf = row.to_dict()
        row_cf[cfg.flip_attr] = flip_protected_value(cfg.flip_attr, row_cf[cfg.flip_attr])
        prompt_cf = build_ranking_prompt(row_cf, cand_titles, prompt_cfg)

        # rank both
        idx_orig = ranker.rank(prompt_orig, cand_titles, system_prompt=system_prompt)[: cfg.k]
        idx_cf = ranker.rank(prompt_cf, cand_titles, system_prompt=system_prompt)[: cfg.k]

        mids_orig = [int(row["candidate_mids"][i]) for i in idx_orig]
        mids_cf = [int(row["candidate_mids"][i]) for i in idx_cf]

        v_orig = _embed_list_mean(embedder, mids_orig, item_db)
        v_cf = _embed_list_mean(embedder, mids_cf, item_db)
        dists.append(_cosine_distance(v_orig, v_cf))

    return float(np.mean(dists)) if dists else 0.0
