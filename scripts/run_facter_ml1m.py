import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from facter.data.download import download_movielens_1m
from facter.data.movielens import load_ml1m, build_item_db
from facter.fairness.calibration import OfflineCalibrator, OfflineCalibConfig
from facter.fairness.context_encoder import ContextEncoder, ContextEncodingConfig
from facter.fairness.monitor import FACTEROnlineMonitor, OnlineMonitorConfig
from facter.fairness.online import CalibrationArtifacts, OnlineScorer, OnlineScoringConfig
from facter.models.embedder import EmbedderConfig, TextEmbedder
from facter.models.hf_ranker import HFChatRanker, HFChatRankerConfig
from facter.prompting.repair import PromptRepairConfig, PromptRepairEngine
from facter.eval.metrics import mean_recall_ndcg
from facter.eval.baselines import evaluate_zero_shot
from facter.tracking.mlflow import MLflowConfig, start_run, log_params, log_metrics, log_text
from facter.utils.seeding import seed_all, SeedConfig


def _read_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / split / "dataset.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_json(path, lines=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed/ml-1m")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    # FACTER hyperparams (paper defaults)
    p.add_argument("--alpha", type=float, default=0.10)       # miscoverage in Eq.(6)
    p.add_argument("--lambda_fairness", type=float, default=0.7)
    p.add_argument("--tau_rho", type=float, default=0.90)
    p.add_argument("--tau_x_l2", type=float, default=None)    # optional locality
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--buffer_size", type=int, default=50)
    p.add_argument("--min_feature_count", type=int, default=3)
    p.add_argument("--max_iterations", type=int, default=3)

    # Which protected attribute to enforce/repair on (single-attr mode)
    p.add_argument("--protected_attr", type=str, default="gender", choices=["gender", "age", "occupation"])

    # Metrics top-k (for Recall/NDCG)
    p.add_argument("--k", type=int, default=10)

    args = p.parse_args()

    seed_all(SeedConfig(seed=args.seed))

    processed_dir = Path(args.processed_dir)
    cal_df = _read_split(processed_dir, "cal")
    test_df = _read_split(processed_dir, "test")

    # Load item DB
    raw_dir = download_movielens_1m(force=False)
    frames = load_ml1m(raw_dir)
    item_db = build_item_db(frames.movies)

    # Models
    embedder = TextEmbedder(EmbedderConfig(device="cuda" if True else "cpu"))
    ranker = HFChatRanker(HFChatRankerConfig(model_id=args.model_id))

    # Context encoder Enc(x)
    ctx = ContextEncoder(embedder, ContextEncodingConfig(max_history_items=5))

    # Consistent protected columns for fairness neighborhoods
    protected_cols = (args.protected_attr,)

    # Offline calibration
    off_cfg = OfflineCalibConfig(
        alpha=args.alpha,
        lambda_fairness=args.lambda_fairness,
        tau_rho=args.tau_rho,
        tau_x_l2=args.tau_x_l2,
        protected_cols=protected_cols,
        top_k_neighbors=None,
    )
    calibrator = OfflineCalibrator(ranker=ranker, embedder=embedder, context_encoder=ctx, cfg=off_cfg)
    cal_res = calibrator.run(cal_df=cal_df, item_db=item_db, system_prompt=None)

    cal_art = CalibrationArtifacts(
        cal_df=cal_res.cal_df,
        cal_context_emb=cal_res.cal_context_emb,
        cal_pred_emb=cal_res.cal_pred_emb,
        q_alpha0=cal_res.q_alpha0,
    )

    # Online scorer + repair + monitor
    scorer = OnlineScorer(embedder, ctx, OnlineScoringConfig(
        protected_cols=protected_cols,
        tau_rho=args.tau_rho,
        tau_x_l2=args.tau_x_l2,
        lambda_fairness=args.lambda_fairness,
    ))
    repair = PromptRepairEngine(
        PromptRepairConfig(
            buffer_size=args.buffer_size,
            protected_key=args.protected_attr,
            min_feature_count=args.min_feature_count,
            max_rules=5,
            domain="movielens",
        ),
        item_db=item_db,
    )
    monitor = FACTEROnlineMonitor(
        ranker=ranker,
        scorer=scorer,
        repair=repair,
        cfg=OnlineMonitorConfig(max_iterations=args.max_iterations, gamma=args.gamma, protected_key=args.protected_attr),
    )

    # Baseline: zero-shot ranking (no fairness prompt)
    baseline_metrics = evaluate_zero_shot(test_df, ranker, k=args.k)

    # Run FACTER iterations
    out_df, logs = monitor.run(
        test_df=test_df,
        item_db=item_db,
        cal_artifacts=cal_art,
        q_alpha0=cal_res.q_alpha0,
    )

    # Compute Recall/NDCG per iteration using the stored top-1 preds (as a degenerate ranked list)
    # NOTE: If you want true Recall@10/NDCG@10, modify monitor to store top-k mids; see note below.
    facter_metrics = {}
    for it in range(1, args.max_iterations + 1):
        preds = out_df[f"pred_mid_iter{it}"].astype(int).tolist()
        ranked_lists = [[m] for m in preds]  # top-1 only
        targets = out_df["target_mid"].astype(int).tolist()
        m = mean_recall_ndcg(ranked_lists, targets, k=args.k)
        v = int(np.sum(out_df[f"is_violation_iter{it}"].to_numpy()))
        facter_metrics[f"iter{it}.violations"] = float(v)
        facter_metrics[f"iter{it}.q_alpha_end"] = float([x for x in logs if x.iteration == it][0].q_alpha)
        facter_metrics[f"iter{it}.Recall@{args.k}"] = m[f"Recall@{args.k}"]
        facter_metrics[f"iter{it}.NDCG@{args.k}"] = m[f"NDCG@{args.k}"]

    # MLflow logging
    mcfg = MLflowConfig(experiment_name="facter-repro", run_name=f"ml1m_{args.model_id}_{args.protected_attr}")
    with start_run(mcfg, tags={"dataset": "ml-1m", "model_id": args.model_id, "protected_attr": args.protected_attr}):
        log_params({
            "seed": args.seed,
            "alpha": args.alpha,
            "lambda_fairness": args.lambda_fairness,
            "tau_rho": args.tau_rho,
            "tau_x_l2": args.tau_x_l2,
            "gamma": args.gamma,
            "buffer_size": args.buffer_size,
            "min_feature_count": args.min_feature_count,
            "max_iterations": args.max_iterations,
            "k": args.k,
        })
        log_metrics({f"baseline.{k}": v for k, v in baseline_metrics.items()}, step=0)
        log_metrics(facter_metrics, step=1)
        log_text(json.dumps({"baseline": baseline_metrics, "facter": facter_metrics}, indent=2), "results/summary.json")

    # Save outputs
    out_path = processed_dir / "runs" / f"run_{args.model_id.replace('/', '_')}_{args.protected_attr}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    print("Baseline:", baseline_metrics)
    print("FACTER:", facter_metrics)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
