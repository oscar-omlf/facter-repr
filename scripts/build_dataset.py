"""Build and write processed protocol datasets with prompts.

This script downloads a supported raw dataset, constructs an item database and
interaction protocol splits, attaches generation and ranking prompts, and writes
JSONL files under the processed data directory.

The supported datasets are: "ml-1m" and "amazon".
"""

import argparse
import json

from facter.data.download import download_dataset
from facter.data.frames import AmazonFrames, MovieLensFrames
from facter.data.paths import PROCESSED_DIR
from facter.data.prompts import (
    PromptConfig,
    build_open_prompt,
    build_ranking_prompt,
)
from facter.data.protocol import (
    ProtocolConfig,
    build_candidate_sets,
    build_interactions,
    sample_and_split,
)


def main() -> None:
    """Build calibration/test splits and write them to the processed data folder.

    The script:

    - Downloads the selected dataset if needed.
    - Builds an item database.
    - Samples interactions, splits into calibration/test, and creates candidate
      sets.
    - Builds prompts for open-ended generation and ranking.
    - Writes JSONL files for each split plus a small metadata JSON.

    Returns:
        None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "amazon"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--n_candidates", type=int, default=100)

    # multi-target relevance config
    p.add_argument(
        "--relevance_mode",
        type=str,
        default="single",
        choices=["single", "future_window", "all_future"],
        help="How to define relevant items for Recall/NDCG evaluation.",
    )
    p.add_argument(
        "--relevance_window",
        type=int,
        default=10,
        help="Only used when relevance_mode='future_window'. Includes the target and next (window-1) items.",
    )
    p.add_argument(
        "--max_pos_in_candidates",
        type=int,
        default=None,
        help="Max number of positives to include in rank-mode candidate pool. Default is sensible per relevance_mode.",
    )

    args = p.parse_args()

    raw_dir = download_dataset(dataset=args.dataset, force=False)

    if args.dataset == "ml-1m":
        frames = MovieLensFrames(raw_dir)
        df1, df2 = frames.ratings, frames.users
        domain = "movie"
    elif args.dataset == "amazon":
        frames = AmazonFrames(raw_dir=raw_dir)
        df1, df2 = frames.ratings, frames.metadata
        domain = "product"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    item_db = frames.build_item_db()

    # IMPORTANT: item_pool should be a concrete sequence/array (not dict_keys)
    item_pool = list(item_db.keys())

    cfg = ProtocolConfig(
        sample_interactions=args.n,
        seed=args.seed,
        n_candidates=args.n_candidates,
        relevance_mode=args.relevance_mode,
        relevance_window=args.relevance_window,
        max_pos_in_candidates=args.max_pos_in_candidates,
    )

    interactions = build_interactions(df1, df2, item_db, cfg, dataset=args.dataset)

    # Split then build candidate sets for both splits
    cal, test = sample_and_split(interactions, cfg)
    cal = build_candidate_sets(cal, item_pool=item_pool, cfg=cfg)
    test = build_candidate_sets(test, item_pool=item_pool, cfg=cfg)

    # Add prompts
    pcfg = PromptConfig(k_recs=10, include_demographics=True, domain=domain)

    def titles_from_mids(mids):
        """Map item IDs from an interaction row to item titles.

        Args:
            mids: Iterable of item IDs used as keys into `item_db`.

        Returns:
            list[str]: Titles corresponding to the provided IDs.
        """
        return [item_db[int(m)]["title"] for m in mids]

    for split_name, df in [("cal", cal), ("test", test)]:
        df["prompt_gen"] = df.apply(
            lambda r: build_open_prompt(r.to_dict(), pcfg), axis=1
        )
        df["candidate_titles"] = df["candidate_mids"].apply(titles_from_mids)
        df["prompt_rank"] = df.apply(
            lambda r: build_ranking_prompt(r.to_dict(), r["candidate_titles"], pcfg),
            axis=1,
        )

        out = PROCESSED_DIR / args.dataset / "dataset.jsonl"
        out_dir = out.parent
        (out_dir / split_name).mkdir(parents=True, exist_ok=True)
        out_path = out_dir / split_name / "dataset.jsonl"
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)

    meta = {
        "seed": args.seed,
        "n_sample": args.n,
        "n_candidates": args.n_candidates,
        "n_cal": len(cal),
        "n_test": len(test),
        "protocol": cfg.__dict__,
    }
    meta_path = (PROCESSED_DIR / args.dataset / "meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"- {PROCESSED_DIR / args.dataset / 'cal/dataset.jsonl'}")
    print(f"- {PROCESSED_DIR / args.dataset / 'test/dataset.jsonl'}")
    print(f"- {meta_path}")


if __name__ == "__main__":
    main()
