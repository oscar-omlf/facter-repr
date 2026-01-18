import argparse
import itertools
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from facter.data.download import download_dataset
from facter.data.frames import AmazonFrames, MovieLensFrames

from facter.fairness.calibration import OfflineCalibrator, OfflineCalibConfig
from facter.fairness.context_encoder import ContextEncoder, ContextEncodingConfig
from facter.fairness.monitor import FACTEROnlineMonitor, OnlineMonitorConfig
from facter.fairness.online import CalibrationArtifacts, OnlineScorer, OnlineScoringConfig

from facter.models.embedder import EmbedderConfig, TextEmbedder
from facter.models.hf_generator import HFOpenGenerator, HFGenConfig
from facter.models.hf_ranker import HFChatRanker, HFChatRankerConfig

from facter.prompting.repair import PromptRepairConfig, PromptRepairEngine

from facter.eval.metrics import mean_recall_ndcg, snsr_snsv_proxy_from_mid_lists, count_violations
from facter.eval.baselines import evaluate_zero_shot, run_zero_shot
from facter.eval.counterfactual import compute_cfr, CFRConfig
from facter.eval.catalogue_map import CatalogueMapper

from facter.tracking.mlflow import (
    MLflowConfig,
    start_run,
    log_params,
    log_metrics,
    log_text,
    log_dataframe,
)
from facter.utils.seeding import seed_all, SeedConfig
from facter.data.prompts import PromptConfig


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
        log_metrics({f"stage.{name}.started": 1.0})
    try:
        yield
        ok = True
    except Exception as e:
        ok = False
        if log_to_mlflow:
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
    p.add_argument(
        "--datasets",
        type=str,
        default="ml-1m",
        help="Comma-separated list of datasets to run: ml-1m and/or amazon. Default: ml-1m"
    )
    p.add_argument("--processed_dir_template", type=str, default="data/processed/{dataset}")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--progress", action="store_true")

    # FACTER hyperparams
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--lambda_fairness", type=float, default=0.7)
    p.add_argument("--tau_rho", type=float, default=0.90)
    p.add_argument("--tau_x_l2", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--buffer_size", type=int, default=50)
    p.add_argument("--min_feature_count", type=int, default=3)
    p.add_argument("--max_iterations", type=int, default=3)

    # single-attribute options (still works)
    p.add_argument("--protected_attr", type=str, default="gender", choices=["gender", "age", "occupation"])
    p.add_argument("--cfr_flip_attr", type=str, default="gender", choices=["gender", "age", "occupation"])

    p.add_argument("--k", type=int, default=10)
    p.add_argument("--predict_mode", type=str, default="rank", choices=["rank", "open"])

    # multi-attribute options
    p.add_argument(
        "--protected_attrs",
        type=str,
        default=None,
        help="Comma-separated attrs to treat jointly, e.g. gender,age or gender,age,occupation. "
             "If not provided, uses --protected_attr.",
    )
    p.add_argument(
        "--sweep_protected_sets",
        action="store_true",
        help="Run all non-empty subsets of protected_attrs (or of gender,age,occupation if protected_attrs not provided).",
    )
    p.add_argument(
        "--cfr_flip_attrs",
        type=str,
        default=None,
        help="Comma-separated attrs to compute CFR for (per protected set). Default: attrs in protected set.",
    )

    p.add_argument(
        "--cfr_flip_strategy",
        type=str,
        default="random",
        choices=["random", "minimal"],
    )

    args = p.parse_args()

    # device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Parse dataset list
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        datasets = ["ml-1m"]
    for ds in datasets:
        if ds not in ["ml-1m", "amazon"]:
            raise ValueError(f"Unknown dataset: {ds}. Allowed: ml-1m, amazon")

    # protected set planning
    ALLOWED = ("gender", "age", "occupation")

    if args.protected_attrs:
        base_attrs = [a.strip() for a in args.protected_attrs.split(",") if a.strip()]
    else:
        base_attrs = [args.protected_attr]

    for a in base_attrs:
        if a not in ALLOWED:
            raise ValueError(f"Unknown protected attr: {a}. Allowed: {ALLOWED}")

    if args.sweep_protected_sets:
        attrs_for_sweep = base_attrs if args.protected_attrs else list(ALLOWED)
        protected_sets: List[Tuple[str, ...]] = []
        for r in range(1, len(attrs_for_sweep) + 1):
            for comb in itertools.combinations(attrs_for_sweep, r):
                protected_sets.append(tuple(comb))
    else:
        protected_sets = [tuple(base_attrs)]

    total_t0 = time.perf_counter()

    def _norm_title(s: str) -> str:
        return str(s).strip().lower()

    # Loop over datasets
    for dataset_name in datasets:
        print(f"\n\n{'='*80}")
        print(f"Processing dataset: {dataset_name.upper()}")
        print(f"{'='*80}\n")

        timings: dict[str, float] = {}

        # Determine processed_dir based on dataset
        if args.processed_dir_template == "data/processed/{dataset}":
            if dataset_name == "ml-1m":
                processed_dir = Path("data/processed/ml-1m")
            elif dataset_name == "amazon":
                processed_dir = Path("data/processed/amazon")
            else:
                processed_dir = Path(args.processed_dir_template.format(dataset=dataset_name))
        else:
            processed_dir = Path(args.processed_dir_template.format(dataset=dataset_name))

        # MLflow
        repo_root = Path(__file__).resolve().parents[1]
        db_path = (repo_root / "mlflow.db").resolve()
        mcfg = MLflowConfig(
            tracking_uri=f"sqlite:///{db_path}",
            experiment_name="facter-repro",
            run_name=f"{dataset_name}_{args.model_id}_seed{args.seed}_{args.predict_mode}",
        )

        with start_run(
            mcfg,
            tags={
                "dataset": dataset_name,
                "model_id": args.model_id,
                "predict_mode": args.predict_mode,
            },
        ):
            log_params({
                "dataset": dataset_name,
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
                "k": args.k,
                "progress": bool(args.progress),
                "protected.base_attrs": ",".join(base_attrs),
                "protected.sweep": bool(args.sweep_protected_sets),
                "protected.sets_count": len(protected_sets),
            })
            log_text(json.dumps({"protected_sets": protected_sets}, indent=2), "config/protected_sets.json")

            with stage("seeding", timings):
                seed_all(SeedConfig(seed=args.seed))

            with stage("load_processed_splits", timings):
                cal_df = _read_split(processed_dir, "cal")
                test_df = _read_split(processed_dir, "test")
                log_metrics({"data.cal_n": float(len(cal_df)), "data.test_n": float(len(test_df))})

            with stage("load_raw_item_db", timings):
                raw_dir = download_dataset(dataset=dataset_name, force=False)
                if dataset_name == "ml-1m":
                    frames = MovieLensFrames(raw_dir)
                    domain = "movielens"
                elif dataset_name == "amazon":
                    frames = AmazonFrames(raw_dir=raw_dir)
                    domain = "amazon"
                else:
                    raise ValueError(f"Unknown dataset: {dataset_name}")
                
                item_db = frames.build_item_db()
                log_metrics({"data.items_n": float(len(item_db))})

                title_to_mid: Dict[str, int] = {}
                for mid, info in item_db.items():
                    title = info.get("title", "")
                    if title:
                        title_to_mid[_norm_title(title)] = int(mid)

            with stage("init_models", timings):
                embedder = TextEmbedder(
                    EmbedderConfig(
                        model_name="JJTsao/fine-tuned_movie_retriever-all-mpnet-base-v2",
                        device=device,
                        progress=args.progress,
                    )
                )
                ranker = HFChatRanker(HFChatRankerConfig(model_id=args.model_id))

                generator = None
                if args.predict_mode == "open":
                    generator = HFOpenGenerator(
                        HFGenConfig(model_id=args.model_id),
                        tokenizer=ranker.tokenizer,
                        model=ranker.model,
                    )

            with stage("build_catalogue_mapper", timings):
                catalogue_mapper = CatalogueMapper(embedder=embedder, item_db=item_db, title_key="title")
                catalogue_mapper.build(dedup=True)
                log_metrics({"catalog.items_n": float(len(catalogue_mapper.catalog_titles))})

            # Sweep over protected sets (single attrs and combinations)
            for protected_cols in protected_sets:
                pset = "+".join(protected_cols)

                def P(name: str) -> str:
                    return f"pset.{pset}.{name}" if args.sweep_protected_sets else name

                # Default CFR flips: each attribute in the current protected set
                if args.cfr_flip_attrs:
                    cfr_flips = [a.strip() for a in args.cfr_flip_attrs.split(",") if a.strip()]
                else:
                    cfr_flips = list(protected_cols)

                for a in cfr_flips:
                    if a not in ALLOWED:
                        raise ValueError(f"Unknown CFR flip attr: {a}. Allowed: {ALLOWED}")

                log_metrics({P("protected_cols_count"): float(len(protected_cols))})
                log_text(",".join(protected_cols), f"config/protected_cols_{pset}.txt")

                with stage(f"init_facter_components[{pset}]", timings):
                    ctx = ContextEncoder(embedder, ContextEncodingConfig(max_history_items=10))

                    off_cfg = OfflineCalibConfig(
                        alpha=args.alpha,
                        lambda_fairness=args.lambda_fairness,
                        tau_rho=args.tau_rho,
                        tau_x_l2=args.tau_x_l2,
                        protected_cols=protected_cols,
                        top_k_neighbors=None,
                    )
                    calibrator = OfflineCalibrator(
                        ranker=ranker, embedder=embedder, context_encoder=ctx, cfg=off_cfg
                    )

                    scorer = OnlineScorer(
                        embedder,
                        ctx,
                        OnlineScoringConfig(
                            protected_cols=protected_cols,
                            tau_rho=args.tau_rho,
                            tau_x_l2=args.tau_x_l2,
                            lambda_fairness=args.lambda_fairness,
                        ),
                    )

                    repair = PromptRepairEngine(
                        PromptRepairConfig(
                            buffer_size=args.buffer_size,
                            protected_key=protected_cols[0],
                            min_feature_count=args.min_feature_count,
                            max_rules=5,
                            domain=domain,
                        ),
                        item_db=item_db,
                    )

                    monitor = FACTEROnlineMonitor(
                        ranker=ranker,
                        scorer=scorer,
                        repair=repair,
                        cfg=OnlineMonitorConfig(
                            max_iterations=args.max_iterations,
                            gamma=args.gamma,
                            protected_key=protected_cols[0],
                        ),
                    )

                    prompt_cfg = PromptConfig(k_recs=args.k)

                with stage(f"offline_calibration[{pset}]", timings):
                    cal_res = calibrator.run(
                        cal_df=cal_df,
                        item_db=item_db,
                        system_prompt=None,
                        progress=args.progress,
                        predict_mode=args.predict_mode,
                        generator=generator,
                        prompt_cfg=prompt_cfg,
                        catalogue_mapper=catalogue_mapper,
                    )
                    log_metrics({
                        P("offline.q_alpha0"): float(cal_res.q_alpha0),
                        P("offline.S_mean"): float(np.mean(cal_res.scores_S)),
                        P("offline.S_max"): float(np.max(cal_res.scores_S)),
                    })
                    if args.predict_mode == "open" and "valid_at_k" in cal_res.cal_df.columns:
                        try:
                            log_metrics({P("offline.ValidAtK.mean"): float(np.mean(cal_res.cal_df["valid_at_k"]))})
                        except Exception:
                            pass

                    log_dataframe(cal_res.cal_df, f"data/calibration_df_{pset}.json", format="json")

                cal_art = CalibrationArtifacts(
                    cal_df=cal_res.cal_df,
                    cal_context_emb=cal_res.cal_context_emb,
                    cal_pred_emb=cal_res.cal_pred_emb,
                    q_alpha0=cal_res.q_alpha0,
                )

                with stage(f"baseline_zero_shot[{pset}]", timings):
                    NEUTRAL_SYSTEM_PROMPT = (
                    "You are a helpful recommendation assistant.\n"
                    "Recommend items based on the user's watch history.\n"
                    f"Return ONLY a JSON array of exactly {args.k} item titles (strings), ranked best-first.\n"
    )

                    baseline_df = run_zero_shot(
                        test_df.copy(),
                        ranker=ranker if args.predict_mode == "rank" else None,
                        generator=generator if args.predict_mode == "open" else None,
                        item_db=item_db,
                        predict_mode=args.predict_mode,
                        k=args.k,
                        catalogue_mapper=catalogue_mapper,
                        title_to_mid=title_to_mid,
                        progress=args.progress,
                        system_prompt=NEUTRAL_SYSTEM_PROMPT,
                    )

                    baseline_metrics = evaluate_zero_shot(baseline_df, k=args.k)

                    if args.predict_mode == "open" and "valid_at_k" in baseline_df.columns:
                        baseline_metrics["ValidAtK.mean"] = float(np.mean(baseline_df["valid_at_k"]))

                    group_keys_b = baseline_df[list(protected_cols)].astype(str).apply(
                        lambda r: "|".join([f"{c}={r[c]}" for c in protected_cols]), axis=1
                    ).tolist()

                    rec_lists_b = baseline_df["ranked_mids"].tolist()

                    sns_b = snsr_snsv_proxy_from_mid_lists(
                        rec_mid_lists=rec_lists_b,
                        group_keys=group_keys_b,
                        embedder=embedder,
                        item_db=item_db,
                        k=args.k,
                        min_group_size=30,
                    )
                    baseline_metrics["SNSR"] = float(sns_b.SNSR)
                    baseline_metrics["SNSV"] = float(sns_b.SNSV)

                    # CFR for all flip attrs at the same time:
                    cfr_cfg_all = CFRConfig(flip_attr=cfr_flips, k=args.k, flip_strategy=args.cfr_flip_strategy)
                    cfr_kwargs_all = {
                        "df": baseline_df,
                        "embedder": embedder,
                        "item_db": item_db,
                        "prompt_cfg": prompt_cfg,
                        "cfg": cfr_cfg_all,
                        "predict_mode": args.predict_mode,
                        "iter": None,
                    }
                    if args.predict_mode == "rank":
                        cfr_kwargs_all["ranker"] = ranker
                    else:
                        cfr_kwargs_all["generator"] = generator
                        cfr_kwargs_all["catalogue_mapper"] = catalogue_mapper
                        cfr_kwargs_all["title_to_mid"] = title_to_mid
                    
                    baseline_metrics["CFR_all"] = float(compute_cfr(**cfr_kwargs_all))

                    # CFR baseline: compute for each flip attr
                    for flip_attr in cfr_flips:
                        cfr_cfg = CFRConfig(flip_attr=flip_attr, k=args.k, flip_strategy=args.cfr_flip_strategy)
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
                        else:
                            cfr_kwargs["generator"] = generator
                            cfr_kwargs["catalogue_mapper"] = catalogue_mapper
                            cfr_kwargs["title_to_mid"] = title_to_mid

                        baseline_metrics[f"CFR_{flip_attr}"] = float(compute_cfr(**cfr_kwargs))

                    log_metrics({P(f"baseline.{k}"): v for k, v in baseline_metrics.items()})
                    log_dataframe(baseline_df, f"data/baseline_df_{pset}.json", format="json")
                    log_text(json.dumps(baseline_metrics, indent=2), f"results/baseline_metrics_{pset}.json")


                # ---- online monitor ----
                with stage(f"online_monitor[{pset}]", timings):
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
                        catalogue_mapper=catalogue_mapper if args.predict_mode == "open" else None,
                        min_sim=0.65,
                        group_cols=protected_cols,
                    )

                    for it_log in logs:
                        log_metrics({
                            P(f"iter{it_log.iteration}.q_alpha_end"): float(it_log.q_alpha),
                            P(f"iter{it_log.iteration}.violations"): float(it_log.violations),
                            P(f"iter{it_log.iteration}.S_mean"): float(it_log.mean_S),
                        }, step=it_log.iteration)

                    log_dataframe(out_df, f"data/online_monitor_df_{pset}.json", format="json")

                with stage(f"compute_facter_metrics[{pset}]", timings):
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

                        facter_metrics[P(f"iter{it}.violations")] = float(v)
                        facter_metrics[P(f"iter{it}.Recall{args.k}")] = m[f"Recall@{args.k}"]
                        facter_metrics[P(f"iter{it}.NDCG{args.k}")] = m[f"NDCG@{args.k}"]

                        # SNSR/SNSV (paper-aligned proxy) on mids
                        sns = snsr_snsv_proxy_from_mid_lists(
                            rec_mid_lists=ranked_lists,
                            group_keys=group_keys,
                            embedder=embedder,
                            item_db=item_db,
                            k=args.k,
                            min_group_size=30,
                        )
                        facter_metrics[P(f"iter{it}.SNSR")] = float(sns.SNSR)
                        facter_metrics[P(f"iter{it}.SNSV")] = float(sns.SNSV)

                        # Valid@K (open mode)
                        if args.predict_mode == "open":
                            col_name = f"valid_at_k_iter{it}"
                            if col_name in out_df.columns:
                                facter_metrics[P(f"iter{it}.ValidAtK.mean")] = float(np.mean(out_df[col_name]))
                        
                    # CFR for all flip attrs at the same time:
                    cfr_cfg_all = CFRConfig(flip_attr=cfr_flips, k=args.k, flip_strategy=args.cfr_flip_strategy)
                    cfr_kwargs_all = {
                        "df": out_df,
                        "embedder": embedder,
                        "item_db": item_db,
                        "prompt_cfg": prompt_cfg,
                        "cfg": cfr_cfg_all,
                        "predict_mode": args.predict_mode,
                        "iter": it,
                    }
                    if args.predict_mode == "rank":
                        cfr_kwargs_all["ranker"] = ranker
                    else:
                        cfr_kwargs_all["generator"] = generator
                        cfr_kwargs_all["catalogue_mapper"] = catalogue_mapper
                        cfr_kwargs_all["title_to_mid"] = title_to_mid
                    
                    facter_metrics[P(f"iter{it}.CFR_all")] = float(compute_cfr(**cfr_kwargs_all))

                    # CFR per flip attribute
                    for flip_attr in cfr_flips:
                        cfr_cfg = CFRConfig(flip_attr=flip_attr, k=args.k)
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
                        else:
                            cfr_kwargs["generator"] = generator
                            cfr_kwargs["catalogue_mapper"] = catalogue_mapper
                            cfr_kwargs["title_to_mid"] = title_to_mid

                        facter_metrics[P(f"iter{it}.CFR_{flip_attr}")] = float(compute_cfr(**cfr_kwargs))

                    log_metrics(facter_metrics)
                    log_text(json.dumps(facter_metrics, indent=2), f"results/facter_metrics_{pset}.json")

                # save parquet per protected set
                with stage(f"save_outputs[{pset}]", timings):
                    out_path = processed_dir / "runs" / f"run_{args.model_id.replace('/', '_')}_{pset}_{args.predict_mode}.parquet"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_df.to_parquet(out_path, index=False)
                    log_text(str(out_path), f"results/output_path_{pset}.txt")

            log_metrics({"stage.TOTAL.seconds": float(time.perf_counter() - total_t0)})
            log_text(json.dumps(timings, indent=2), "results/timings.json")

            print(f"Timings for {dataset_name}:", timings)
            print(f"Baseline metrics: {baseline_metrics}")
            print(f"FACTER metrics: {facter_metrics}")



if __name__ == "__main__":
    main()
