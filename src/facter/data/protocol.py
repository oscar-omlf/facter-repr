"""Construct interaction datasets and candidate sets for evaluation protocols.

This module builds per-user interaction rows containing histories, targets, and
optionally multi-target relevance sets. It also provides utilities to
deterministically sample/split interactions and to construct ranking candidate
pools.
"""

# src/facter/data/protocol.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProtocolConfig:
    """Configure interaction construction, splitting, and candidate sampling.

    Attributes:
        min_history (int): Minimum history length required before a target can be
            formed for a user.
        sample_interactions (int): Number of interactions to sample prior to
            splitting.
        test_size (float): Fraction of sampled interactions allocated to the test
            split.
        seed (int): Random seed used for deterministic sampling/shuffling.
        n_candidates (int): Candidate set size for ranking-style evaluation.
        stratify (bool): Whether to attempt stratified splitting by protected
            attributes.
        protected_cols (Tuple[str, ...]): Column names used when computing
            stratification strata.
        relevance_mode (str): Relevance definition mode.
        relevance_window (int): Window size used when `relevance_mode` is
            "future_window".
        max_pos_in_candidates (Optional[int]): Optional cap on how many positive
            items to include in the candidate pool.
    """

    min_history: int = 10  # how many history items before predicting next
    sample_interactions: int = 2500
    test_size: float = 0.30
    seed: int = 42

    # Ranking-style candidate pool
    n_candidates: int = 100  # total candidates shown to ranker
    stratify: bool = True
    protected_cols: Tuple[str, ...] = ("gender", "age", "occupation")
    relevance_mode: str = "single"  # "single" | "future_window" | "all_future"
    relevance_window: int = 10      # only used when relevance_mode="future_window"
    max_pos_in_candidates: Optional[int] = None


def _process_ml(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merge and sort MovieLens-style frames into a single interaction table.

    Args:
        df1 (pd.DataFrame): Ratings-like DataFrame containing at least ``uid`` and
            ``timestamp``.
        df2 (pd.DataFrame): Users-like DataFrame keyed by ``uid``.

    Returns:
        pd.DataFrame: Merged and time-sorted DataFrame.
    """
    df = df1.merge(df2, on="uid", how="inner")
    df = df.sort_values(["uid", "timestamp"]).reset_index(drop=True)
    return df


def _process_amazon(df1: pd.DataFrame, df2: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Merge and preprocess Amazon-style frames and synthesize protected columns.

    This function merges ratings with metadata on ``mid``, sorts by ``uid`` and
    ``timestamp``, coerces timestamps to numeric, drops rows missing key fields,
    and then creates synthetic protected attributes.

    Args:
        df1 (pd.DataFrame): Ratings-like DataFrame containing at least ``uid``,
            ``mid``, and ``timestamp``.
        df2 (pd.DataFrame): Metadata-like DataFrame keyed by ``mid``.
        seed (int): Seed used for deterministic random attribute generation.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with synthesized protected columns.
    """
    df = pd.merge(df1, df2, on="mid", how="inner")
    df = df.sort_values(["uid", "timestamp"]).reset_index(drop=True)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["uid", "mid", "timestamp"])

    # Synthesize protected attributes (Amazon lacks them)
    rng = np.random.default_rng(seed)
    df["gender"] = rng.choice(["M", "F"], size=len(df))
    df["age"] = rng.choice([1, 18, 25, 35, 45, 50, 56], size=len(df)).astype(int)
    df["occupation"] = rng.integers(0, 21, size=len(df)).astype(str)

    return df


def _compute_relevant_mids(mids: List[int], t: int, cfg: ProtocolConfig) -> List[int]:
    """Compute the relevance set for a target position within a user's sequence.

    Always includes ``mids[t]`` when ``t`` is a valid index.

    Args:
        mids (List[int]): Sequence of item IDs for a user.
        t (int): Target index into `mids`.
        cfg (ProtocolConfig): Protocol configuration.

    Returns:
        List[int]: List of relevant item IDs for the given position.

    Raises:
        ValueError: If `cfg.relevance_mode` is not recognized.
    """
    if t < 0 or t >= len(mids):
        return []

    mode = str(cfg.relevance_mode).lower().strip()
    if mode == "single":
        return [int(mids[t])]

    if mode == "future_window":
        w = int(cfg.relevance_window)
        if w <= 0:
            # fall back to single if window is invalid
            return [int(mids[t])]
        end = min(len(mids), t + w)
        return [int(x) for x in mids[t:end]]

    if mode == "all_future":
        return [int(x) for x in mids[t:]]

    raise ValueError(f"Unknown relevance_mode: {cfg.relevance_mode}. Expected 'single', 'future_window', or 'all_future'.")


def build_interactions_ml(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    item_db: Dict[int, Dict[str, str]],
    cfg: ProtocolConfig,
) -> pd.DataFrame:
    """Build MovieLens interaction rows with history, target, and relevance sets.

    Output columns include:
        uid, gender, age, occupation,
        history_mids (list[int]), history_titles (list[str]),
        target_mid (int), target_title (str),
        relevant_mids (list[int]), relevant_titles (list[str]).

    For each user and each position ``t >= cfg.min_history``:
        - history consists of the previous ``cfg.min_history`` items.
        - target is the item at position `t`.
        - relevant items are computed via `_compute_relevant_mids` and include the
          target.

    Args:
        ratings (pd.DataFrame): Ratings DataFrame.
        users (pd.DataFrame): Users DataFrame.
        item_db (Dict[int, Dict[str, str]]): Mapping from item ID to item fields.
        cfg (ProtocolConfig): Protocol configuration.

    Returns:
        pd.DataFrame: Interaction rows.
    """
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
            target = int(mids[t])

            # Compute relevants (may be multi-target)
            relevant = _compute_relevant_mids(mids, t, cfg)
            if not relevant:
                continue

            # Titles: ensure target exists; for relevants, filter missing items conservatively
            try:
                hist_titles = [item_db[int(m)]["title"] for m in hist]
                target_title = item_db[int(target)]["title"]
            except KeyError:
                continue

            rel_titles: List[str] = []
            rel_mids_clean: List[int] = []
            for m in relevant:
                info = item_db.get(int(m))
                if info is None:
                    continue
                title = str(info.get("title", "")).strip()
                if not title:
                    continue
                rel_mids_clean.append(int(m))
                rel_titles.append(title)

            # Guarantee the target is in relevant_mids (if it exists in item_db, it should)
            if target not in rel_mids_clean:
                rel_mids_clean.insert(0, target)
                rel_titles.insert(0, target_title)

            # Dedup relevants while preserving order
            seen = set()
            rel_mids_dedup: List[int] = []
            rel_titles_dedup: List[str] = []
            for m, tt in zip(rel_mids_clean, rel_titles):
                if int(m) in seen:
                    continue
                seen.add(int(m))
                rel_mids_dedup.append(int(m))
                rel_titles_dedup.append(tt)

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
                    "relevant_mids": rel_mids_dedup,
                    "relevant_titles": rel_titles_dedup,
                }
            )

    return pd.DataFrame(rows)


def build_interactions(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    item_db: Dict[int, Dict[str, str]],
    cfg: ProtocolConfig,
    dataset: str = "ml-1m",
) -> pd.DataFrame:
    """Build interaction rows for a supported dataset.

    Output columns include:
        uid, gender, age, occupation,
        history_mids (list[int]), history_titles (list[str]),
        target_mid (int), target_title (str),
        relevant_mids (list[int]), relevant_titles (list[str]).

    Dataset handling:
        - ``"ml-1m"``: merges ratings/users via `_process_ml`.
        - ``"amazon"``: merges ratings/metadata via `_process_amazon` and
          synthesizes protected attributes.

    Args:
        df1 (pd.DataFrame): Dataset-specific ratings-like table.
        df2 (pd.DataFrame): Dataset-specific users/metadata table.
        item_db (Dict[int, Dict[str, str]]): Mapping from item ID to item fields.
        cfg (ProtocolConfig): Protocol configuration.
        dataset (str): Dataset identifier.

    Returns:
        pd.DataFrame: Interaction rows.

    Raises:
        ValueError: If `dataset` is not recognized.
    """
    if dataset == "ml-1m":
        df = _process_ml(df1=df1, df2=df2)
    elif dataset == "amazon":
        df = _process_amazon(df1=df1, df2=df2, seed=cfg.seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

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
            target = int(mids[t])

            relevant = _compute_relevant_mids(mids, t, cfg)
            if not relevant:
                continue

            try:
                hist_titles = [item_db[int(m)]["title"] for m in hist]
                target_title = item_db[int(target)]["title"]
            except KeyError:
                continue

            rel_titles: List[str] = []
            rel_mids_clean: List[int] = []
            for m in relevant:
                info = item_db.get(int(m))
                if info is None:
                    continue
                title = str(info.get("title", "")).strip()
                if not title:
                    continue
                rel_mids_clean.append(int(m))
                rel_titles.append(title)

            if target not in rel_mids_clean:
                rel_mids_clean.insert(0, target)
                rel_titles.insert(0, target_title)

            seen = set()
            rel_mids_dedup: List[int] = []
            rel_titles_dedup: List[str] = []
            for m, tt in zip(rel_mids_clean, rel_titles):
                if int(m) in seen:
                    continue
                seen.add(int(m))
                rel_mids_dedup.append(int(m))
                rel_titles_dedup.append(tt)

            rows.append(
                {
                    "uid": str(uid),
                    "gender": gender,
                    "age": age,
                    "occupation": occupation,
                    "history_mids": hist,
                    "history_titles": hist_titles,
                    "target_mid": int(target),
                    "target_title": target_title,
                    "relevant_mids": rel_mids_dedup,
                    "relevant_titles": rel_titles_dedup,
                }
            )

    return pd.DataFrame(rows)


def sample_and_split(
    interactions: pd.DataFrame,
    cfg: ProtocolConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample interactions and split into calibration and test sets.

    Sampling is performed deterministically using `cfg.seed`, selecting exactly
    `cfg.sample_interactions` rows without replacement. The sampled set is then
    split into calibration and test.

    If `cfg.stratify` is True, the function attempts to stratify by the joint
    protected attributes in `cfg.protected_cols` when feasible.

    Args:
        interactions (pd.DataFrame): Interaction rows.
        cfg (ProtocolConfig): Protocol configuration.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (calibration_df, test_df).

    Raises:
        ValueError: If there are fewer than `cfg.sample_interactions` rows.
    """
    rng = np.random.default_rng(cfg.seed)

    if len(interactions) < cfg.sample_interactions:
        raise ValueError(
            f"Not enough interactions: have {len(interactions)}, need {cfg.sample_interactions}"
        )

    sampled_idx = rng.choice(
        len(interactions), size=cfg.sample_interactions, replace=False
    )
    sampled = interactions.iloc[sampled_idx].reset_index(drop=True)

    n_test = int(round(cfg.test_size * len(sampled)))

    if cfg.stratify:
        strata = sampled[list(cfg.protected_cols)].astype(str).agg("_".join, axis=1)
        if strata.value_counts().min() < 2:
            cfg = ProtocolConfig(**{**cfg.__dict__, "stratify": False})
        else:
            cal_parts = []
            test_parts = []
            for s, g in sampled.groupby(strata, sort=False):
                idx = np.arange(len(g))
                rng.shuffle(idx)
                t_count = max(1, int(round(len(g) * cfg.test_size)))
                test_parts.append(g.iloc[idx[:t_count]])
                cal_parts.append(g.iloc[idx[t_count:]])
            cal = pd.concat(cal_parts, ignore_index=True)
            test = pd.concat(test_parts, ignore_index=True)

            if len(test) > n_test:
                test = test.sample(n=n_test, random_state=cfg.seed).reset_index(drop=True)
            if len(test) < n_test:
                extra = cal.sample(n=n_test - len(test), random_state=cfg.seed)
                test = pd.concat([test, extra], ignore_index=True)
                cal = cal.drop(index=extra.index).reset_index(drop=True)

            cal = cal.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
            test = test.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
            return cal, test

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
    """Add ranking candidate sets to interaction rows.

    Adds a ``candidate_mids`` column containing a list of length
    ``cfg.n_candidates``.

    Strategy:
        - positives: from ``relevant_mids`` if present, else ``[target_mid]``
        - optionally clip positives via `cfg.max_pos_in_candidates` (or a default
          derived from the relevance mode)
        - negatives: sample from `item_pool` while excluding history and positives
        - combine positives and negatives, then shuffle

    Notes:
        - If `cfg.relevance_mode == "single"`, this reduces to the common
          "1 positive + negatives" construction.

    Args:
        df (pd.DataFrame): Interaction rows.
        item_pool (np.ndarray): Pool of item IDs eligible as negatives.
        cfg (ProtocolConfig): Protocol configuration.

    Returns:
        pd.DataFrame: Copy of `df` with a new `candidate_mids` column.
    """
    rng = np.random.default_rng(cfg.seed)
    pool = np.asarray(item_pool, dtype=np.int64)

    def _dedup_preserve_order(xs: List[int]) -> List[int]:
        seen = set()
        out = []
        for x in xs:
            x = int(x)
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def sample_one(history: List[int], target: int, relevant: Optional[List[int]]) -> List[int]:
        hist_set = {int(x) for x in history}
        target = int(target)

        # positives
        if relevant is None:
            pos = [target]
        else:
            pos = [int(x) for x in relevant]
            if target not in pos:
                pos.insert(0, target)

        # remove any positives that appear in history (rare, but safe)
        pos = [x for x in pos if x not in hist_set]
        pos = _dedup_preserve_order(pos)
        if not pos:
            pos = [target]

        # decide max positives to include in candidates
        if cfg.max_pos_in_candidates is not None:
            max_pos = int(cfg.max_pos_in_candidates)
        else:
            mode = str(cfg.relevance_mode).lower().strip()
            if mode == "single":
                max_pos = 1
            else:
                # default: up to relevance_window positives, but always leave room for negatives
                max_pos = int(max(1, min(cfg.relevance_window, cfg.n_candidates - 1)))

        # always leave at least 1 slot for negatives (unless n_candidates==1)
        if cfg.n_candidates > 1:
            max_pos = min(max_pos, cfg.n_candidates - 1)
        else:
            max_pos = min(max_pos, cfg.n_candidates)

        pos = pos[:max_pos]

        banned = set(hist_set) | set(pos)

        # sample negatives
        n_negs = int(cfg.n_candidates - len(pos))
        if n_negs < 0:
            n_negs = 0

        allowed = pool[~np.isin(pool, np.array(list(banned), dtype=np.int64))]
        if len(allowed) == 0:
            # pathological fallback: allow sampling from pool excluding history only
            allowed = pool[~np.isin(pool, np.array(list(hist_set), dtype=np.int64))]

        if len(allowed) == 0:
            # worst-case fallback
            negs = []
        elif len(allowed) < n_negs:
            negs = rng.choice(allowed, size=n_negs, replace=True).tolist()
        else:
            negs = rng.choice(allowed, size=n_negs, replace=False).tolist()

        cand = [int(x) for x in pos] + [int(x) for x in negs]
        # ensure exact length
        cand = cand[: cfg.n_candidates]
        rng.shuffle(cand)
        return cand

    out = df.copy()

    # Use relevant_mids if present, else None
    has_relevant = "relevant_mids" in out.columns

    out["candidate_mids"] = out.apply(
        lambda r: sample_one(
            history=r["history_mids"],
            target=r["target_mid"],
            relevant=(r["relevant_mids"] if has_relevant else None),
        ),
        axis=1,
    )
    return out
