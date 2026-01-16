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
from facter.models.hf_generator import HFOpenGenerator, HFGenConfig
from facter.models.hf_ranker import HFChatRanker, HFChatRankerConfig
from facter.prompting.repair import PromptRepairConfig, PromptRepairEngine
from facter.eval.metrics import mean_recall_ndcg, snsr_snsv_proxy_from_title_lists, snsr_snsv_proxy_from_mid_lists
from facter.eval.baselines import evaluate_zero_shot, run_zero_shot
from facter.eval.counterfactual import compute_cfr, CFRConfig
from facter.tracking.mlflow import MLflowConfig, start_run, log_params, log_metrics, log_text, log_dataframe
from facter.utils.seeding import seed_all, SeedConfig
from facter.data.prompts import PromptConfig
from facter.eval.catalogue_map import CatalogMapper

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
    p.add_argument("--cfr_flip_attr", type=str, default="gender", choices=["gender", "age", "occupation"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--predict_mode", type=str, default="rank", choices=["rank", "open"])

    args = p.parse_args()

    # Choose device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Make MLflow DB path absolute
    repo_root = Path(__file__).resolve().parents[1]
    db_path = (repo_root / "mlflow.db").resolve()

    mcfg = MLflowConfig(
        tracking_uri=f"sqlite:///{db_path}",
        experiment_name="facter-repro",
        run_name=f"ml1m_{args.model_id}_{args.protected_attr}_seed{args.seed}_{args.predict_mode}",
    )

    timings: dict[str, float] = {}
    total_t0 = time.perf_counter()

    def _norm_title(s: str) -> str:
        return str(s).strip().lower()

    with start_run(mcfg, tags={"dataset": "ml-1m", "model_id": args.model_id, "protected_attr": args.protected_attr, "predict_mode": args.predict_mode}):
        log_params({
            "seed": args.seed,
            "device": device,
            "model_id": args.model_id,
            "predict_mode": args.predict_mode,
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

            # title->mid index for open mode
            title_to_mid = {}
            for mid, info in item_db.items():
                title = info.get("title", "")
                if title:
                    title_to_mid[_norm_title(title)] = int(mid)

        with stage("init_models", timings):
            embedder = TextEmbedder(EmbedderConfig(model_name="JJTsao/fine-tuned_movie_retriever-all-mpnet-base-v2", device=device))       
            ranker = HFChatRanker(HFChatRankerConfig(model_id=args.model_id))
            
            generator = None
            if args.predict_mode == "open":
                generator = HFOpenGenerator(
                    HFGenConfig(model_id=args.model_id),
                    tokenizer=ranker.tokenizer,
                    model=ranker.model,
                )

        with stage("init_facter_components", timings):
            ctx = ContextEncoder(embedder, ContextEncodingConfig(max_history_items=10))
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

            prompt_cfg = PromptConfig(k_recs=args.k)
            cfr_cfg = CFRConfig(flip_attr=args.cfr_flip_attr, k=args.k)

            catalog_mapper = CatalogMapper(
                embedder=embedder, item_db=item_db, title_key="title"
            )
            catalog_mapper.build(dedup=True)
            log_metrics({"catalog.items_n": float(len(catalog_mapper.catalog_titles))})

        with stage("offline_calibration", timings):
            cal_res = calibrator.run(
                cal_df=cal_df,
                item_db=item_db,
                system_prompt=None,
                progress=args.progress,
                predict_mode=args.predict_mode,
                generator=generator,
                prompt_cfg=prompt_cfg,
                catalog_mapper=catalog_mapper,
            )
            log_metrics({
                "offline.q_alpha0": float(cal_res.q_alpha0),
                "offline.S_mean": float(np.mean(cal_res.scores_S)),
                "offline.S_max": float(np.max(cal_res.scores_S)),
            })
            # In open mode, track fraction of generated titles mapped to catalog during calibration
            if args.predict_mode == "open" and "valid_at_k" in cal_res.cal_df.columns:
                try:
                    offline_valid_mean = float(np.mean(cal_res.cal_df["valid_at_k"]))
                    log_metrics({"offline.ValidAtK.mean": offline_valid_mean})
                except Exception:
                    pass
            log_dataframe(cal_res.cal_df, "data/calibration_df.json", format="json")

        with stage("prepare_online_artifacts", timings):
            cal_art = CalibrationArtifacts(
                cal_df=cal_res.cal_df,
                cal_context_emb=cal_res.cal_context_emb,
                cal_pred_emb=cal_res.cal_pred_emb,
                q_alpha0=cal_res.q_alpha0,
            )

        with stage("baseline_zero_shot", timings):
            baseline_df = run_zero_shot(
            test_df.copy(),
            ranker=ranker if args.predict_mode == "rank" else None,
            generator=generator if args.predict_mode == "open" else None,
            item_db=item_db,
            predict_mode=args.predict_mode,
            k=args.k,
            catalog_mapper=catalog_mapper,
            title_to_mid=title_to_mid,
            progress=args.progress,
            )

            baseline_metrics = evaluate_zero_shot(baseline_df, k=args.k)

            # In open mode, track fraction of generated titles mapped to catalog for baseline
            if args.predict_mode == "open" and "valid_at_k" in baseline_df.columns:
                try:
                    baseline_metrics["ValidAtK.mean"] = float(np.mean(baseline_df["valid_at_k"]))
                except Exception:
                    baseline_metrics["ValidAtK.mean"] = 0.0

            # CFR for baseline (supports both rank and open modes)
            cfr_kwargs = {
                "df": baseline_df,
                "embedder": embedder,
                "item_db": item_db,
                "prompt_cfg": prompt_cfg,
                "cfg": cfr_cfg,
                "predict_mode": args.predict_mode,
                "iter": None,
            }
            
            if args.predict_mode == "rank":
                cfr_kwargs["ranker"] = ranker
            elif args.predict_mode == "open":
                cfr_kwargs["generator"] = generator
                cfr_kwargs["catalog_mapper"] = catalog_mapper
                cfr_kwargs["title_to_mid"] = title_to_mid

            baseline_cfr = compute_cfr(**cfr_kwargs)
            baseline_metrics[f"CFR_{args.cfr_flip_attr}"] = float(baseline_cfr)
            
            # Compute SNSR/SNSV fairness metrics for baseline
            group_keys = baseline_df.apply(
                lambda row: "|".join(f"{col}={row[col]}" for col in protected_cols),
                axis=1,
            ).tolist()
            
            if args.predict_mode == "rank":
                # Use ranked_mids for rank mode
                sns_metrics = snsr_snsv_proxy_from_mid_lists(
                    rec_mid_lists=baseline_df["ranked_mids"].tolist(),
                    group_keys=group_keys,
                    embedder=embedder,
                    item_db=item_db,
                    k=args.k,
                    min_group_size=30,
                )
            else:
                # Use generated_titles for open mode
                if "generator_response" in baseline_df.columns:
                    title_lists = []
                    for resp in baseline_df["generator_response"]:
                        try:
                            titles = json.loads(resp) if isinstance(resp, str) else resp
                            title_lists.append(titles if isinstance(titles, list) else [])
                        except:
                            title_lists.append([])
                    sns_metrics = snsr_snsv_proxy_from_title_lists(
                        rec_title_lists=title_lists,
                        group_keys=group_keys,
                        embedder=embedder,
                        k=args.k,
                        min_group_size=30,
                    )
                else:
                    sns_metrics = None
            
            if sns_metrics:
                baseline_metrics["SNSR"] = float(sns_metrics.SNSR)
                baseline_metrics["SNSV"] = float(sns_metrics.SNSV)
            
            log_metrics({f"baseline.{k}": v for k, v in baseline_metrics.items()})
            log_dataframe(baseline_df, "data/baseline_df.json", format="json")

        with stage("online_monitor", timings):
            out_df, logs = monitor.run(
                test_df=test_df,
                item_db=item_db,
                cal_artifacts=cal_art,
                q_alpha0=cal_res.q_alpha0,
                progress=args.progress,
                predict_mode=args.predict_mode,
                generator=generator,
                prompt_cfg=prompt_cfg,
                title_to_mid=title_to_mid if args.predict_mode == "open" else None,
            )

            for it_log in logs:
                log_metrics({
                    f"iter{it_log.iteration}.q_alpha_end": float(it_log.q_alpha),
                    f"iter{it_log.iteration}.violations": float(it_log.violations),
                    f"iter{it_log.iteration}.S_mean": float(it_log.mean_S),
                }, step=it_log.iteration)

            log_dataframe(out_df, "data/online_monitor_df.json", format="json")

        with stage("compute_facter_metrics", timings):
            facter_metrics = {}
            targets = out_df["target_mid"].astype(int).tolist()
            group_keys = out_df[list(protected_cols)].astype(str).apply(
                lambda r: "|".join([f"{c}={r[c]}" for c in protected_cols]),
                axis=1,
            ).tolist()

            for it in range(1, args.max_iterations + 1):
                if args.predict_mode == "rank":
                    ranked_lists = out_df[f"ranked_mids_iter{it}"].tolist()
                else:
                    ranked_lists = out_df[f"generated_mids_iter{it}"].tolist()

                m = mean_recall_ndcg(ranked_lists, targets, k=args.k)
                v = int(np.sum(out_df[f"is_violation_iter{it}"].to_numpy()))

                facter_metrics[f"iter{it}.violations"] = float(v)
                facter_metrics[f"iter{it}.Recall{args.k}"] = m[f"Recall@{args.k}"]
                facter_metrics[f"iter{it}.NDCG{args.k}"] = m[f"NDCG@{args.k}"]
                
                # Compute SNSR/SNSV fairness metrics for this iteration
                if args.predict_mode == "rank":
                    # Use ranked_mids for rank mode
                    sns_metrics = snsr_snsv_proxy_from_mid_lists(
                        rec_mid_lists=out_df[f"ranked_mids_iter{it}"].tolist(),
                        group_keys=group_keys,
                        embedder=embedder,
                        item_db=item_db,
                        k=args.k,
                        min_group_size=30,
                    )
                else:
                    # Use generated_titles for open mode
                    if f"generated_titles_iter{it}" in out_df.columns:
                        title_lists = out_df[f"generated_titles_iter{it}"].tolist()
                        sns_metrics = snsr_snsv_proxy_from_title_lists(
                            rec_title_lists=title_lists,
                            group_keys=group_keys,
                            embedder=embedder,
                            k=args.k,
                            min_group_size=30,
                        )
                    else:
                        sns_metrics = None
                
                if sns_metrics:
                    facter_metrics[f"iter{it}.SNSR"] = float(sns_metrics.SNSR)
                    facter_metrics[f"iter{it}.SNSV"] = float(sns_metrics.SNSV)

                # In open mode, track fraction of generated titles mapped to catalog per iteration
                if args.predict_mode == "open":
                    col_name = f"valid_at_k_iter{it}"
                    if col_name in out_df.columns:
                        try:
                            facter_metrics[f"iter{it}.ValidAtK.mean"] = float(np.mean(out_df[col_name]))
                        except Exception:
                            facter_metrics[f"iter{it}.ValidAtK.mean"] = 0.0

                # CFR for online monitor (supports both rank and open modes)
                cfr_kwargs = {
                    "df": out_df,
                    "embedder": embedder,
                    "item_db": item_db,
                    "prompt_cfg": prompt_cfg,
                    "cfg": cfr_cfg,
                    "predict_mode": args.predict_mode,
                    "iter": it,
                }
                if args.predict_mode == "rank":
                    cfr_kwargs["ranker"] = ranker
                elif args.predict_mode == "open":
                    cfr_kwargs["generator"] = generator
                    cfr_kwargs["catalog_mapper"] = catalog_mapper
                    cfr_kwargs["title_to_mid"] = title_to_mid
                cfr_metric = compute_cfr(**cfr_kwargs)
                facter_metrics[f"iter{it}.CFR_{args.cfr_flip_attr}"] = float(cfr_metric)

            log_metrics(facter_metrics)
            log_text(json.dumps({"baseline": baseline_metrics, "facter": facter_metrics}, indent=2), "results/summary.json")

        with stage("save_outputs", timings):
            out_path = Path(args.processed_dir) / "runs" / f"run_{args.model_id.replace('/', '_')}_{args.protected_attr}_{args.predict_mode}.parquet"
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
