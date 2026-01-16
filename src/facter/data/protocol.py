from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProtocolConfig:
    min_history: int = 10          # how many history items before predicting next
    sample_interactions: int = 2500
    test_size: float = 0.30
    seed: int = 42
    n_candidates: int = 100       # for ranking-style prompts; includes 1 positive
    stratify: bool = True
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")


def build_interactions_ml(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    item_db: Dict[int, Dict[str, str]],
    cfg: ProtocolConfig,
    filter_ratings: bool = False,
) -> pd.DataFrame:
    """
    Build interaction rows:
      uid, gender, age, occupation,
      history_mids (list[int]), history_titles (list[str]),
      target_mid (int), target_title (str)

    For each user, sort by timestamp. For each position t, if t >= min_history,
    history = previous min_history items, target = current item.
    """
    if filter_ratings:
        ratings = ratings[ratings["rating"] >= 4.0].reset_index(drop=True)

    # Merge + sort
    df = ratings.merge(users, on="uid", how="inner")
    df = df.sort_values(["uid", "timestamp"]).reset_index(drop=True)

    rows: List[Dict] = []
    for uid, g in df.groupby("uid", sort=False):
        mids = g["mid"].astype(int).tolist()
        if len(mids) <= cfg.min_history:
            continue
        gender = g["gender"].iloc[0]
        age = int(g["age"].iloc[0])
        occupation = int(g["occupation"].iloc[0])

        for t in range(cfg.min_history, len(mids)):
            hist = mids[t - cfg.min_history : t]
            target = mids[t]

            # titles
            try:
                hist_titles = [item_db[m]["title"] for m in hist]
                target_title = item_db[target]["title"]
            except KeyError:
                # skip if missing in db
                continue

            rows.append(
                {
                    "uid": int(uid),
                    "gender": gender,
                    "age": age,
                    "occupation": occupation,
                    "history_mids": hist,
                    "history_titles": hist_titles,
                    "target_mid": int(target),
                    "target_title": target_title,
                }
            )

    return pd.DataFrame(rows)


def sample_and_split(
    interactions: pd.DataFrame,
    cfg: ProtocolConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministically sample cfg.sample_interactions interactions, then split into calibration/test.

    Stratification is by joint protected attributes if cfg.stratify=True and feasible.
    """
    rng = np.random.default_rng(cfg.seed)

    if len(interactions) < cfg.sample_interactions:
        raise ValueError(
            f"Not enough interactions: have {len(interactions)}, need {cfg.sample_interactions}"
        )

    sampled_idx = rng.choice(len(interactions), size=cfg.sample_interactions, replace=False)
    sampled = interactions.iloc[sampled_idx].reset_index(drop=True)

    # Split
    n_test = int(round(cfg.test_size * len(sampled)))
    n_cal = len(sampled) - n_test

    if cfg.stratify:
        strata = sampled[list(cfg.protected_cols)].astype(str).agg("_".join, axis=1)
        # If any stratum is too small, fallback to unstratified
        if strata.value_counts().min() < 2:
            cfg = ProtocolConfig(**{**cfg.__dict__, "stratify": False})
        else:
            # stratified split by sampling within strata
            cal_parts = []
            test_parts = []
            for s, g in sampled.groupby(strata, sort=False):
                idx = np.arange(len(g))
                rng.shuffle(idx)
                # proportional allocation
                t_count = max(1, int(round(len(g) * cfg.test_size)))
                test_parts.append(g.iloc[idx[:t_count]])
                cal_parts.append(g.iloc[idx[t_count:]])
            cal = pd.concat(cal_parts, ignore_index=True)
            test = pd.concat(test_parts, ignore_index=True)

            # Fix exact sizes if rounding drifted
            if len(test) > n_test:
                test = test.sample(n=n_test, random_state=cfg.seed).reset_index(drop=True)
            if len(test) < n_test:
                extra = cal.sample(n=n_test - len(test), random_state=cfg.seed)
                test = pd.concat([test, extra], ignore_index=True)
                cal = cal.drop(index=extra.index).reset_index(drop=True)

            cal = cal.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
            test = test.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
            return cal, test

    # Unstratified fallback
    perm = rng.permutation(len(sampled))
    test_idx = perm[:n_test]
    cal_idx = perm[n_test:]
    cal = sampled.iloc[cal_idx].reset_index(drop=True)
    test = sampled.iloc[test_idx].reset_index(drop=True)
    return cal, test


def build_candidate_sets(
    df: pd.DataFrame,
    item_pool: np.ndarray,
    cfg: ProtocolConfig,
) -> pd.DataFrame:
    """
    Adds:
      candidate_mids: list[int] length cfg.n_candidates (includes target_mid)
    Strategy: sample negatives uniformly from item_pool excluding history and target.
    """
    rng = np.random.default_rng(cfg.seed)

    def sample_one(history: List[int], target: int) -> List[int]:
        banned = set(history)
        banned.add(target)
        # filter pool
        allowed = [int(m) for m in item_pool if int(m) not in banned]
        if len(allowed) < cfg.n_candidates - 1:
            # fallback: allow replacement from allowed (still exclude banned)
            negs = rng.choice(allowed, size=cfg.n_candidates - 1, replace=True).tolist()
        else:
            negs = rng.choice(allowed, size=cfg.n_candidates - 1, replace=False).tolist()
        cand = [int(target)] + [int(x) for x in negs]
        rng.shuffle(cand)
        return cand

    out = df.copy()
    out["candidate_mids"] = out.apply(
        lambda r: sample_one(r["history_mids"], int(r["target_mid"])), axis=1
    )
    return out
