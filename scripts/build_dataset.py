import argparse
import json

from facter.data.download import download_dataset
from facter.data.frames import AmazonFrames, MovieLensFrames, Sushi3Frames
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
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "amazon", "sushi3-2016"])
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
    elif args.dataset == "sushi3-2016":
        frames = Sushi3Frames(raw_dir=raw_dir, variant="b")
        # Sushi is an explicit ranking dataset; we treat per-user top-k order as a pseudo-sequence.
        # We pass a single df through build_interactions via a new dataset handler.
        df1, df2 = frames.orders, frames.users
        domain = "sushi"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    item_db = frames.build_item_db()

    # IMPORTANT: item_pool should be a concrete sequence/array (not dict_keys)
    # build_candidate_sets expects a numpy array.
    import numpy as np

    item_pool = np.asarray(list(item_db.keys()), dtype=np.int64)

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
        return [item_db[int(m)]["title"] for m in mids]

    for split_name, df in [("cal", cal), ("test", test)]:
        df["prompt_gen"] = df.apply(lambda r: build_open_prompt(r.to_dict(), pcfg), axis=1)
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
    meta_path = PROCESSED_DIR / args.dataset / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"- {PROCESSED_DIR / args.dataset / 'cal/dataset.jsonl'}")
    print(f"- {PROCESSED_DIR / args.dataset / 'test/dataset.jsonl'}")
    print(f"- {meta_path}")


if __name__ == "__main__":
    main()
