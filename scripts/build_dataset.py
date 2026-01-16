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
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "amazon"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--n_candidates", type=int, default=100)
    args = p.parse_args()

    raw_dir = download_dataset(dataset=args.dataset, force=False)

    if args.dataset == "ml-1m":
        frames = MovieLensFrames(raw_dir)
        df1, df2 = frames.ratings, frames.users

    elif args.dataset == "amazon":
        frames = AmazonFrames(raw_dir=raw_dir)
        df1, df2 = frames.ratings, frames.metadata

    item_db = frames.build_item_db()
    item_pool = item_db.keys()

    cfg = ProtocolConfig(
        sample_interactions=args.n, seed=args.seed, n_candidates=args.n_candidates
    )
    interactions = build_interactions(df1, df2, item_db, cfg, dataset=args.dataset)

    # Build candidate sets for both splits (ranking-style support)
    cal, test = sample_and_split(interactions, cfg)
    cal = build_candidate_sets(cal, item_pool=item_pool, cfg=cfg)
    test = build_candidate_sets(test, item_pool=item_pool, cfg=cfg)

    # Add prompts
    pcfg = PromptConfig(k_recs=10, include_demographics=True, domain="movie")

    def titles_from_mids(mids):
        return [item_db[m]["title"] for m in mids]

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
    meta_path = out.parent / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"- {out.parent / 'cal/dataset.jsonl'}")
    print(f"- {out.parent / 'test/dataset.jsonl'}")
    print(f"- {meta_path}")


if __name__ == "__main__":
    main()
