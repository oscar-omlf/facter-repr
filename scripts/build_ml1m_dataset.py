import argparse
import json
from pathlib import Path

import pandas as pd

from facter.data.download import download_movielens_1m
from facter.data.movielens import build_item_db, load_ml1m
from facter.data.protocol import ProtocolConfig, build_candidate_sets, build_interactions_ml, sample_and_split
from facter.data.prompts import PromptConfig, build_generation_prompt, build_ranking_prompt
from facter.data.paths import PROCESSED_DIR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--n_candidates", type=int, default=100)
    p.add_argument("--out", type=str, default=str(PROCESSED_DIR / "ml-1m" / "dataset.jsonl"))
    args = p.parse_args()

    raw_dir = download_movielens_1m(force=False)
    frames = load_ml1m(raw_dir)
    item_db = build_item_db(frames.movies)

    cfg = ProtocolConfig(sample_interactions=args.n, seed=args.seed, n_candidates=args.n_candidates)
    interactions = build_interactions_ml(frames.ratings, frames.users, item_db, cfg)

    cal, test = sample_and_split(interactions, cfg)

    # Build candidate sets for both splits (ranking-style support)
    item_pool = frames.movies["mid"].astype(int).unique()
    cal = build_candidate_sets(cal, item_pool=item_pool, cfg=cfg)
    test = build_candidate_sets(test, item_pool=item_pool, cfg=cfg)

    # Add prompts
    pcfg = PromptConfig(k_recs=10, include_demographics=True, domain="movie")

    def titles_from_mids(mids):
        return [item_db[int(m)]["title"] for m in mids]

    for split_name, df in [("cal", cal), ("test", test)]:
        df["prompt_gen"] = df.apply(lambda r: build_generation_prompt(r.to_dict(), pcfg), axis=1)
        df["candidate_titles"] = df["candidate_mids"].apply(titles_from_mids)
        df["prompt_rank"] = df.apply(lambda r: build_ranking_prompt(r.to_dict(), r["candidate_titles"], pcfg), axis=1)

        out_dir = Path(args.out).parent
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
    meta_path = Path(args.out).parent / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"- {Path(args.out).parent/'cal/dataset.jsonl'}")
    print(f"- {Path(args.out).parent/'test/dataset.jsonl'}")
    print(f"- {meta_path}")


if __name__ == "__main__":
    main()
