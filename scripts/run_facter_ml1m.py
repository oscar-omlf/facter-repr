import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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


@contextmanager
def stage(name: str, timings: dict, log_to_mlflow: bool = True):
    t0 = time.perf_counter()
    print(f"\n[stage] START {name}", flush=True)
    if log_to_mlflow:
        # small heartbeat so you can see the run is alive
        log_metrics({f"stage.{name}.started": 1.0})
    try:
        yield
        ok = True
    except Exception as e:
        ok = False
        if log_to_mlflow:
            # store exception info as a small artifact
            log_text(str(e), f"stage_errors/{name}.txt")
        raise
    finally:
        dt = time.perf_counter() - t0
        timings[name] = dt
        print(f"[stage] END   {name}  ({dt:.2f}s)  ok={ok}", flush=True)
        if log_to_mlflow:
            log_metrics({f"stage.{name}.seconds": float(dt)})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed/ml-1m")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--progress", action="store_true", help="Enable tqdm progress bars in heavy loops")

    # FACTER hyperparams
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--lambda_fairness", type=float, default=0.7)
    p.add_argument("--tau_rho", type=float, default=0.90)
    p.add_argument("--tau_x_l2", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--buffer_size", type=int, default=50)
    p.add_argument("--min_feature_count", type=int, default=3)
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--protected_attr", type=str, default="gender", choices=["gender", "age", "occupation"])
    p.add_argument("--k", type=int, default=10)

    args = p.parse_args()

    # Choose device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Make MLflow DB path absolute (avoids “wrong CWD” issues)
    repo_root = Path(__file__).resolve().parents[1]
    db_path = (repo_root / "mlflow.db").resolve()

    mcfg = MLflowConfig(
        tracking_uri=f"sqlite:///{db_path}",
        experiment_name="facter-repro",
        run_name=f"ml1m_{args.model_id}_{args.protected_attr}_seed{args.seed}",
    )

    timings: dict[str, float] = {}
    total_t0 = time.perf_counter()

    with start_run(mcfg, tags={"dataset": "ml-1m", "model_id": args.model_id, "protected_attr": args.protected_attr}):
        # Log params immediately so the run appears right away
        log_params({
            "seed": args.seed,
            "device": device,
            "model_id": args.model_id,
            "alpha": args.alpha,
            "lambda_fairness": args.lambda_fairness,
            "tau_rho": args.tau_rho,
            "tau_x_l2": args.tau_x_l2,
            "gamma": args.gamma,
            "buffer_size": args.buffer_size,
            "min_feature_count": args.min_feature_count,
            "max_iterations": args.max_iterations,
            "protected_attr": args.protected_attr,
            "k": args.k,
            "progress": bool(args.progress),
        })

        with stage("seeding", timings):
            seed_all(SeedConfig(seed=args.seed))

        with stage("load_processed_splits", timings):
            processed_dir = Path(args.processed_dir)
            cal_df = _read_split(processed_dir, "cal")
            test_df = _read_split(processed_dir, "test")
            log_metrics({"data.cal_n": float(len(cal_df)), "data.test_n": float(len(test_df))})

        with stage("load_raw_item_db", timings):
            raw_dir = download_movielens_1m(force=False)
            frames = load_ml1m(raw_dir)
            item_db = build_item_db(frames.movies)
            log_metrics({"data.items_n": float(len(item_db))})

        with stage("init_models", timings):
            embedder = TextEmbedder(EmbedderConfig(device=device))
            ranker = HFChatRanker(HFChatRankerConfig(model_id=args.model_id))

        with stage("init_facter_components", timings):
            ctx = ContextEncoder(embedder, ContextEncodingConfig(max_history_items=5))
            protected_cols = (args.protected_attr,)

            off_cfg = OfflineCalibConfig(
                alpha=args.alpha,
                lambda_fairness=args.lambda_fairness,
                tau_rho=args.tau_rho,
                tau_x_l2=args.tau_x_l2,
                protected_cols=protected_cols,
                top_k_neighbors=None,
            )
            calibrator = OfflineCalibrator(ranker=ranker, embedder=embedder, context_encoder=ctx, cfg=off_cfg)

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

        with stage("offline_calibration", timings):
            # you will add tqdm inside OfflineCalibrator.run in Patch 2
            cal_res = calibrator.run(cal_df=cal_df, item_db=item_db, system_prompt=None, progress=args.progress)
            log_metrics({
                "offline.q_alpha0": float(cal_res.q_alpha0),
                "offline.S_mean": float(np.mean(cal_res.scores_S)),
                "offline.S_max": float(np.max(cal_res.scores_S)),
            })

        with stage("prepare_online_artifacts", timings):
            cal_art = CalibrationArtifacts(
                cal_df=cal_res.cal_df,
                cal_context_emb=cal_res.cal_context_emb,
                cal_pred_emb=cal_res.cal_pred_emb,
                q_alpha0=cal_res.q_alpha0,
            )

        with stage("baseline_zero_shot", timings):
            # you will add tqdm inside evaluate_zero_shot in Patch 2
            baseline_metrics = evaluate_zero_shot(test_df, ranker, k=args.k, progress=args.progress)
            log_metrics({f"baseline.{k}": v for k, v in baseline_metrics.items()})

        with stage("online_monitor", timings):
            # you will add tqdm inside monitor.run in Patch 2
            out_df, logs = monitor.run(
                test_df=test_df,
                item_db=item_db,
                cal_artifacts=cal_art,
                q_alpha0=cal_res.q_alpha0,
                progress=args.progress,
            )
            # log per-iteration “heartbeat” metrics
            for it_log in logs:
                log_metrics({
                    f"iter{it_log.iteration}.q_alpha_end": float(it_log.q_alpha),
                    f"iter{it_log.iteration}.violations": float(it_log.violations),
                    f"iter{it_log.iteration}.S_mean": float(it_log.mean_S),
                }, step=it_log.iteration)

        with stage("compute_facter_metrics", timings):
            facter_metrics = {}
            for it in range(1, args.max_iterations + 1):
                preds = out_df[f"pred_mid_iter{it}"].astype(int).tolist()
                ranked_lists = [[m] for m in preds]  # top-1 only
                targets = out_df["target_mid"].astype(int).tolist()
                m = mean_recall_ndcg(ranked_lists, targets, k=args.k)
                v = int(np.sum(out_df[f"is_violation_iter{it}"].to_numpy()))
                facter_metrics[f"iter{it}.violations"] = float(v)
                facter_metrics[f"iter{it}.Recall@{args.k}"] = m[f"Recall@{args.k}"]
                facter_metrics[f"iter{it}.NDCG@{args.k}"] = m[f"NDCG@{args.k}"]

            log_metrics(facter_metrics)
            log_text(json.dumps({"baseline": baseline_metrics, "facter": facter_metrics}, indent=2), "results/summary.json")

        with stage("save_outputs", timings):
            out_path = Path(args.processed_dir) / "runs" / f"run_{args.model_id.replace('/', '_')}_{args.protected_attr}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(out_path, index=False)
            log_text(str(out_path), "results/output_path.txt")

        total_dt = time.perf_counter() - total_t0
        timings["TOTAL"] = total_dt
        log_metrics({"stage.TOTAL.seconds": float(total_dt)})
        log_text(json.dumps(timings, indent=2), "results/timings.json")

        print("\nBaseline:", baseline_metrics)
        print("FACTER:", facter_metrics)
        print("Timings:", timings)


if __name__ == "__main__":
    main()
