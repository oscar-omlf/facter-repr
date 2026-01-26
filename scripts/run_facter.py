import argparse
import itertools
import json
import os
import time
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker

from facter.data.download import download_dataset
from facter.data.frames import AmazonFrames, MovieLensFrames
from facter.data.prompts import PromptConfig
from facter.eval.baselines import evaluate_zero_shot, run_zero_shot
from facter.eval.catalogue_map import CatalogueMapper
from facter.eval.counterfactual import CFRConfig, compute_cfr
from facter.eval.metrics import mean_recall_ndcg, mean_recall_ndcg_multi, snsr_snsv_proxy_from_mid_lists
from facter.fairness.calibration import OfflineCalibConfig, OfflineCalibrator
from facter.fairness.context_encoder import ContextEncoder, ContextEncodingConfig
from facter.fairness.monitor import FACTEROnlineMonitor, OnlineMonitorConfig
from facter.fairness.neighbors import CrossGroupNeighborIndex, NeighborConfig
from facter.fairness.online import CalibrationArtifacts, OnlineScorer, OnlineScoringConfig
from facter.fairness.scoring import NonconformityScorer, ScoreConfig
from facter.models.embedder import EmbedderConfig, TextEmbedder
from facter.models.hf_generator import HFGenConfig, HFOpenGenerator
from facter.models.hf_ranker import HFChatRanker, HFChatRankerConfig
from facter.models.item_embedder import ItemEmbedder
from facter.models.model_registry import BASE_MODELS
from facter.prompting.repair import PromptRepairConfig, PromptRepairEngine
from facter.tracking.mlflow import (
    MLflowConfig,
    log_dataframe,
    log_metrics,
    log_params,
    log_text,
    start_run,
)
from facter.utils.seeding import SeedConfig, seed_all


ALLOWED_PROTECTED = ("gender", "age", "occupation")

NEUTRAL_PROMPT_TEMPLATE = (
    "You are a helpful recommendation assistant.\n"
    "Recommend items based on the user's watch history.\n"
)


FAIR_PROMPT_TEMPLATE = "\n".join([
    "You are a fair recommendation system.",
    "Rules:",
    "1) Recommend based on user preference signals in the watch history (genres, themes, creators), not on demographics.",
    "2) Do NOT reinforce stereotypes or demographic-based assumptions.",
])


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated list of seeds to loop over (default: 42).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="ml-1m",
        help="Comma-separated list of datasets to run: ml-1m and/or amazon. Default: ml-1m",
    )
    p.add_argument("--processed_dir_template", type=str, default="data/processed/{dataset}")
    p.add_argument(
        "--base_model",
        type=str,
        default="llama3",
        choices=sorted(BASE_MODELS.keys()),
        help=(
            "Short name for a local Hugging Face baseline model. "
            "Used to resolve --model_id via src/facter/models/model_registry.py (BASE_MODELS)."
        ),
    )
    p.add_argument(
        "--model_id",
        type=str,
        default=None,
        help=(
            "Hugging Face model id or local path. If omitted, resolved from --base_model. "
            "If provided, it overrides --base_model."
        ),
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--progress", action="store_true")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--lambda_fairness", type=float, default=0.7)
    p.add_argument("--tau_rho", type=float, default=0.90)
    p.add_argument("--tau_x_l2", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--buffer_size", type=int, default=50)
    p.add_argument("--min_feature_count", type=int, default=3)
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--protected_attr", type=str, default="gender", choices=list(ALLOWED_PROTECTED))
    p.add_argument("--cfr_flip_attr", type=str, default="gender", choices=list(ALLOWED_PROTECTED))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--predict_mode", type=str, default="rank", choices=["rank", "open"])
    p.add_argument("--llm_batch_size", type=int, default=16, help="Batch size used when batching LLM calls")
    p.add_argument("--embedder_batch_size", type=int, default=256, help="Batch size used when batching LLM calls")
    p.add_argument("--temperature", type=float, default=0.7, help="Temperature for open-ended generation")
    p.add_argument(
        "--protected_attrs",
        type=str,
        default=None,
        help="Comma-separated attrs to treat jointly, e.g. gender,age or gender,age,occupation. If not provided, uses --protected_attr.",
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
    p.add_argument("--cfr_flip_strategy", type=str, default="random", choices=["random", "minimal"])
    p.add_argument(
        "--repair_keying",
        type=str,
        default="per_attr",
        choices=["per_attr", "tuple"],
        help=(
            "How prompt-repair mines AVOID rules from the violation buffer: "
            "'per_attr' mines per attribute (gender-only / age-only / occupation-only) without duplicating buffer entries; "
            "'tuple' mines only for the full protected tuple (interaction key)."
        ),
    )
    p.add_argument(
        "--baseline_prompts",
        type=str,
        default="neutral",
        choices=["neutral", "fair", "both"],
        help="Which baseline prompt(s) to run: 'neutral' (standard), 'fair' (fairness-aware), or 'both'.",
    )
    p.add_argument(
        "--skip_online",
        action="store_true",
        help="Skip the FACTER online monitoring phase (only run baselines).",
    )

    args = p.parse_args()

    # Resolve model id from registry unless explicitly overridden.
    if args.model_id is None:
        args.model_id = str(BASE_MODELS[args.base_model]["model_id"])

    return args


def resolve_device(arg_device: str) -> str:
    if arg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg_device


def parse_datasets(raw: str) -> List[str]:
    datasets = [d.strip().lower() for d in raw.split(",") if d.strip()]
    if not datasets:
        return ["ml-1m"]
    for ds in datasets:
        if ds not in ["ml-1m", "amazon"]:
            raise ValueError(f"Unknown dataset: {ds}. Allowed: ml-1m, amazon")
    return datasets


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(s) for s in raw.split(",") if s.strip()]
    if not seeds:
        raise ValueError("Provide at least one seed via --seeds (comma-separated).")
    return seeds


def parse_protected_sets(args: argparse.Namespace) -> Tuple[List[str], List[Tuple[str, ...]]]:
    if args.protected_attrs:
        base_attrs = [a.strip() for a in args.protected_attrs.split(",") if a.strip()]
    else:
        base_attrs = [args.protected_attr]

    for attr in base_attrs:
        if attr not in ALLOWED_PROTECTED:
            raise ValueError(f"Unknown protected attr: {attr}. Allowed: {ALLOWED_PROTECTED}")

    if args.sweep_protected_sets:
        attrs_for_sweep = base_attrs if args.protected_attrs else list(ALLOWED_PROTECTED)
        protected_sets: List[Tuple[str, ...]] = []
        for r in range(1, len(attrs_for_sweep) + 1):
            protected_sets.extend(tuple(comb) for comb in itertools.combinations(attrs_for_sweep, r))
    else:
        protected_sets = [tuple(base_attrs)]

    return base_attrs, protected_sets


def _norm_title(title: str) -> str:
    return str(title).strip().lower()


def build_processed_dir(template: str, dataset_name: str) -> Path:
    if template == "data/processed/{dataset}":
        if dataset_name == "ml-1m":
            return Path("data/processed/ml-1m")
        if dataset_name == "amazon":
            return Path("data/processed/amazon")
    return Path(template.format(dataset=dataset_name))


def load_item_db(dataset_name: str) -> Tuple[Dict[int, Dict[str, str]], str, Dict[str, int]]:
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
    title_to_mid: Dict[str, int] = {}
    for mid, info in item_db.items():
        title = info.get("title", "")
        if title:
            title_to_mid[_norm_title(title)] = int(mid)
    return item_db, domain, title_to_mid


def init_models(args: argparse.Namespace, device: str, dataset_name: str) -> Tuple[TextEmbedder, HFChatRanker, HFOpenGenerator | None]:
    embedder = TextEmbedder(
        EmbedderConfig(
            model_name="JJTsao/fine-tuned_movie_retriever-all-mpnet-base-v2",
            device=device,
            progress=args.progress,
            cache_dir=Path(f"data/cache/embeddings/{dataset_name}"),
            batch_size=args.embedder_batch_size,
        )
    )

    ranker = HFChatRanker(
        HFChatRankerConfig(
            model_id=args.model_id,
            batch_size=args.llm_batch_size,
            temperature=args.temperature,
            seed=args.seed,
        )
    )
    generator = None
    if args.predict_mode == "open":
        generator = HFOpenGenerator(
            HFGenConfig(
                model_id=args.model_id,
                batch_size=args.llm_batch_size,
                temperature=args.temperature,
                seed=args.seed,
            ),
            tokenizer=ranker.tokenizer,
            model=ranker.model,
        )
    return embedder, ranker, generator


def build_catalogue(embedder: TextEmbedder, item_db: Dict[int, Dict[str, str]]) -> CatalogueMapper:
    catalogue_mapper = CatalogueMapper(embedder=embedder, item_db=item_db, title_key="title")
    catalogue_mapper.build(dedup=True)
    return catalogue_mapper


def run_baseline(
    args: argparse.Namespace,
    protected_cols: Sequence[str],
    test_df: pd.DataFrame,
    ranker: HFChatRanker,
    generator: HFOpenGenerator | None,
    embedder: TextEmbedder,
    catalogue_mapper: CatalogueMapper,
    item_db: Dict[int, Dict[str, str]],
    title_to_mid: Dict[str, int],
    cal_q_alpha0: float,
    system_prompt: str,
    item_embedder: ItemEmbedder,
) -> pd.DataFrame:

    baseline_scorer = NonconformityScorer(
        embedder=embedder,
        cfg=ScoreConfig(lambda_fairness=args.lambda_fairness, tau_rho=args.tau_rho),
        item_embedder=item_embedder,
    )

    neighbor_cfg = NeighborConfig(protected_cols=list(protected_cols), tau_rho=args.tau_rho)
    neighbor_idx = CrossGroupNeighborIndex(neighbor_cfg)
    baseline_context_encoder = ContextEncoder(embedder=embedder, cfg=ContextEncodingConfig(max_history_items=10))

    baseline_df = run_zero_shot(
        test_df.copy(),
        ranker=ranker if args.predict_mode == "rank" else None,
        generator=generator if args.predict_mode == "open" else None,
        scorer=baseline_scorer,
        neighbor_index=neighbor_idx,
        context_encoder=baseline_context_encoder,
        item_db=item_db,
        predict_mode=args.predict_mode,
        k=args.k,
        catalogue_mapper=catalogue_mapper,
        title_to_mid=title_to_mid,
        progress=args.progress,
        system_prompt=system_prompt,
        threshold=cal_q_alpha0,
        protected_cols=protected_cols,
        buffer_size=args.buffer_size,
    )

    return baseline_df


def _resolve_cfr_flips(args: argparse.Namespace, protected_cols: Sequence[str]) -> List[str]:
    if args.cfr_flip_attrs:
        flips = [a.strip() for a in args.cfr_flip_attrs.split(",") if a.strip()]
    else:
        flips = list(protected_cols)
    for attr in flips:
        if attr not in ALLOWED_PROTECTED:
            raise ValueError(f"Unknown CFR flip attr: {attr}. Allowed: {ALLOWED_PROTECTED}")
    return flips


def _compute_cfr_metrics(
    df: pd.DataFrame,
    embedder: TextEmbedder,
    item_db: Dict[int, Dict[str, str]],
    prompt_cfg: PromptConfig,
    catalogue_mapper: CatalogueMapper,
    title_to_mid: Dict[str, int],
    args: argparse.Namespace,
    cfr_flips: Sequence[str],
    iteration: int | None,
    ranker: HFChatRanker | None,
    generator: HFOpenGenerator | None,
    item_embedder: ItemEmbedder,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    cfg_all = CFRConfig(flip_attr=cfr_flips, k=args.k, flip_strategy=args.cfr_flip_strategy, seed=args.seed)
    kwargs_all = {
        "df": df,
        "embedder": embedder,
        "item_db": item_db,
        "prompt_cfg": prompt_cfg,
        "cfg": cfg_all,
        "predict_mode": args.predict_mode,
        "iter": iteration,
        "progress": args.progress,
        "item_embedder": item_embedder,
    }
    if args.predict_mode == "rank":
        kwargs_all["ranker"] = ranker
    else:
        kwargs_all["generator"] = generator
        kwargs_all["catalogue_mapper"] = catalogue_mapper
        kwargs_all["title_to_mid"] = title_to_mid
    metrics["CFR_all"] = float(compute_cfr(**kwargs_all))

    for flip_attr in cfr_flips:
        cfg = CFRConfig(flip_attr=flip_attr, k=args.k, flip_strategy=args.cfr_flip_strategy, seed=args.seed)
        kwargs = {
            "df": df,
            "embedder": embedder,
            "item_db": item_db,
            "prompt_cfg": prompt_cfg,
            "cfg": cfg,
            "predict_mode": args.predict_mode,
            "iter": iteration,
            "progress": args.progress,
            "item_embedder": item_embedder,
        }
        if args.predict_mode == "rank":
            kwargs["ranker"] = ranker
        else:
            kwargs["generator"] = generator
            kwargs["catalogue_mapper"] = catalogue_mapper
            kwargs["title_to_mid"] = title_to_mid
        metrics[f"CFR_{flip_attr}"] = float(compute_cfr(**kwargs))
    return metrics


def compute_baseline_metrics(
    args: argparse.Namespace,
    protected_cols: Sequence[str],
    baseline_df: pd.DataFrame,
    embedder: TextEmbedder,
    item_db: Dict[int, Dict[str, str]],
    prompt_cfg: PromptConfig,
    catalogue_mapper: CatalogueMapper,
    title_to_mid: Dict[str, int],
    ranker: HFChatRanker,
    generator: HFOpenGenerator | None,
    item_embedder: ItemEmbedder,
) -> Dict[str, float]:
    metrics = evaluate_zero_shot(baseline_df, k=args.k)

    # Decide which recommendation lists to score
    if "ranked_mids" in baseline_df.columns:
        ranked_lists = baseline_df["ranked_mids"].tolist()
    elif "generated_mids" in baseline_df.columns:
        ranked_lists = baseline_df["generated_mids"].tolist()
    else:
        raise KeyError("baseline_df must contain 'ranked_mids' or 'generated_mids' for Recall/NDCG computation.")

    targets_single = baseline_df["target_mid"].astype(int).tolist()
    m_single = mean_recall_ndcg_multi(ranked_lists, targets_single, k=args.k)
    metrics[f"Recall@{args.k}.single"] = float(m_single[f"Recall@{args.k}"])
    metrics[f"NDCG@{args.k}.single"] = float(m_single[f"NDCG@{args.k}"])

    if "relevant_mids" in baseline_df.columns:
        relevants_multi = baseline_df["relevant_mids"].tolist()
        m_multi = mean_recall_ndcg_multi(ranked_lists, relevants_multi, k=args.k)
        metrics[f"Recall@{args.k}.multi"] = float(m_multi[f"Recall@{args.k}"])
        metrics[f"NDCG@{args.k}.multi"] = float(m_multi[f"NDCG@{args.k}"])
        try:
            metrics["RelevantSetSize.mean"] = float(np.mean([len(x) for x in relevants_multi]))
        except Exception:
            pass
    else:
        metrics[f"Recall@{args.k}.multi"] = metrics[f"Recall@{args.k}.single"]
        metrics[f"NDCG@{args.k}.multi"] = metrics[f"NDCG@{args.k}.single"]
        metrics["RelevantSetSize.mean"] = 1.0

    metrics[f"Recall@{args.k}"] = metrics[f"Recall@{args.k}.multi"]
    metrics[f"NDCG@{args.k}"] = metrics[f"NDCG@{args.k}.multi"]

    if args.predict_mode == "open" and "valid_at_k" in baseline_df.columns:
        metrics["ValidAtK.mean"] = float(np.mean(baseline_df["valid_at_k"]))

    group_keys_joint = baseline_df[list(protected_cols)].astype(str).apply(
        lambda r: "|".join([f"{c}={r[c]}" for c in protected_cols]), axis=1,
    ).tolist()

    sns = snsr_snsv_proxy_from_mid_lists(
        rec_mid_lists=ranked_lists,
        group_keys=group_keys_joint,
        embedder=embedder,
        item_db=item_db,
        k=args.k,
        min_group_size=30,
    )
    metrics["SNSR"] = float(sns.SNSR)
    metrics["SNSV"] = float(sns.SNSV)

    # Per-attribute SNS metrics in addition to the joint key.
    for attr in protected_cols:
        attr_keys = baseline_df[attr].astype(str).tolist()
        sns_attr = snsr_snsv_proxy_from_mid_lists(
            rec_mid_lists=ranked_lists,
            group_keys=attr_keys,
            embedder=embedder,
            item_db=item_db,
            k=args.k,
            min_group_size=30,
        )
        metrics[f"SNSR.{attr}"] = float(sns_attr.SNSR)
        metrics[f"SNSV.{attr}"] = float(sns_attr.SNSV)

    if "is_violation" in baseline_df.columns:
        metrics["n_violations"] = int(np.sum(baseline_df["is_violation"].to_numpy(dtype=bool)))
    else:
        metrics["n_violations"] = 0

    cfr_flips = _resolve_cfr_flips(args, protected_cols)
    metrics.update(
        _compute_cfr_metrics(
            df=baseline_df,
            embedder=embedder,
            item_db=item_db,
            prompt_cfg=prompt_cfg,
            catalogue_mapper=catalogue_mapper,
            title_to_mid=title_to_mid,
            args=args,
            cfr_flips=cfr_flips,
            iteration=None,
            ranker=ranker,
            generator=generator,
            item_embedder=item_embedder,
        )
    )
    return metrics


def run_online_monitor(
    monitor: FACTEROnlineMonitor,
    args: argparse.Namespace,
    protected_cols: Sequence[str],
    item_db: Dict[int, Dict[str, str]],
    cal_art: CalibrationArtifacts,
    cal_q_alpha0: float,
    test_df: pd.DataFrame,
    generator: HFOpenGenerator | None,
    prompt_cfg: PromptConfig,
    title_to_mid: Dict[str, int],
    catalogue_mapper: CatalogueMapper,
):
    return monitor.run(
        test_df=test_df,
        item_db=item_db,
        cal_artifacts=cal_art,
        q_alpha0=cal_q_alpha0,
        progress=args.progress,
        predict_mode=args.predict_mode,
        generator=generator,
        prompt_cfg=prompt_cfg,
        title_to_mid=title_to_mid if args.predict_mode == "open" else None,
        catalogue_mapper=catalogue_mapper if args.predict_mode == "open" else None,
        min_sim=0.65,
        group_cols=protected_cols,
    )


def compute_facter_metrics(
    args: argparse.Namespace,
    protected_cols: Sequence[str],
    embedder: TextEmbedder,
    item_db: Dict[int, Dict[str, str]],
    out_df: pd.DataFrame,
    prompt_cfg: PromptConfig,
    cfr_flips: Sequence[str],
    catalogue_mapper: CatalogueMapper,
    title_to_mid: Dict[str, int],
    ranker: HFChatRanker,
    generator: HFOpenGenerator | None,
    item_embedder: ItemEmbedder,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    targets_single = out_df["target_mid"].astype(int).tolist()
    relevants_multi = out_df["relevant_mids"].tolist() if "relevant_mids" in out_df.columns else None

    if relevants_multi is not None:
        try:
            metrics["RelevantSetSize.mean"] = float(np.mean([len(x) for x in relevants_multi]))
        except Exception:
            metrics["RelevantSetSize.mean"] = 1.0
    else:
        metrics["RelevantSetSize.mean"] = 1.0

    group_keys_joint = out_df[list(protected_cols)].astype(str).apply(
        lambda r: "|".join([f"{c}={r[c]}" for c in protected_cols]), axis=1,
    ).tolist()
    group_keys_per_attr = {attr: out_df[attr].astype(str).tolist() for attr in protected_cols}

    for iteration in range(1, args.max_iterations + 1):
        ranked_lists = (
            out_df[f"ranked_mids_iter{iteration}"].tolist()
            if args.predict_mode == "rank"
            else out_df[f"generated_mids_iter{iteration}"].tolist()
        )

        m_single = mean_recall_ndcg_multi(ranked_lists, targets_single, k=args.k)
        metrics[f"iter{iteration}.Recall{args.k}.single"] = float(m_single[f"Recall@{args.k}"])
        metrics[f"iter{iteration}.NDCG{args.k}.single"] = float(m_single[f"NDCG@{args.k}"])

        if relevants_multi is not None:
            m_multi = mean_recall_ndcg_multi(ranked_lists, relevants_multi, k=args.k)
            metrics[f"iter{iteration}.Recall{args.k}.multi"] = float(m_multi[f"Recall@{args.k}"])
            metrics[f"iter{iteration}.NDCG{args.k}.multi"] = float(m_multi[f"NDCG@{args.k}"])
            # Headline = multi-target
            metrics[f"iter{iteration}.Recall{args.k}"] = metrics[f"iter{iteration}.Recall{args.k}.multi"]
            metrics[f"iter{iteration}.NDCG{args.k}"] = metrics[f"iter{iteration}.NDCG{args.k}.multi"]
        else:
            metrics[f"iter{iteration}.Recall{args.k}.multi"] = metrics[f"iter{iteration}.Recall{args.k}.single"]
            metrics[f"iter{iteration}.NDCG{args.k}.multi"] = metrics[f"iter{iteration}.NDCG{args.k}.single"]
            # Headline = single-target
            metrics[f"iter{iteration}.Recall{args.k}"] = metrics[f"iter{iteration}.Recall{args.k}.single"]
            metrics[f"iter{iteration}.NDCG{args.k}"] = metrics[f"iter{iteration}.NDCG{args.k}.single"]

        v = int(np.sum(out_df[f"is_violation_iter{iteration}"].to_numpy(dtype=bool)))
        metrics[f"iter{iteration}.violations"] = float(v)

        sns = snsr_snsv_proxy_from_mid_lists(
            rec_mid_lists=ranked_lists,
            group_keys=group_keys_joint,
            embedder=embedder,
            item_db=item_db,
            k=args.k,
            min_group_size=30,
        )
        metrics[f"iter{iteration}.SNSR"] = float(sns.SNSR)
        metrics[f"iter{iteration}.SNSV"] = float(sns.SNSV)

        for attr, keys in group_keys_per_attr.items():
            sns_attr = snsr_snsv_proxy_from_mid_lists(
                rec_mid_lists=ranked_lists,
                group_keys=keys,
                embedder=embedder,
                item_db=item_db,
                k=args.k,
                min_group_size=30,
            )
            metrics[f"iter{iteration}.SNSR.{attr}"] = float(sns_attr.SNSR)
            metrics[f"iter{iteration}.SNSV.{attr}"] = float(sns_attr.SNSV)

        if args.predict_mode == "open":
            col_name = f"valid_at_k_iter{iteration}"
            if col_name in out_df.columns:
                metrics[f"iter{iteration}.ValidAtK.mean"] = float(np.mean(out_df[col_name]))

    metrics.update(
        _compute_cfr_metrics(
            df=out_df,
            embedder=embedder,
            item_db=item_db,
            prompt_cfg=prompt_cfg,
            catalogue_mapper=catalogue_mapper,
            title_to_mid=title_to_mid,
            args=args,
            cfr_flips=cfr_flips,
            iteration=args.max_iterations,
            ranker=ranker,
            generator=generator,
            item_embedder=item_embedder,
        )
    )
    return metrics



def save_outputs(processed_dir: Path, args: argparse.Namespace, pset: str, out_df: pd.DataFrame):
    out_path = processed_dir / "runs" / f"run_{args.model_id.replace('/', '_')}_{pset}_{args.predict_mode}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    log_text(str(out_path), f"results/output_path_{pset}.txt")


def compute_qstar_counterfactuals(out_df: pd.DataFrame, q_star: float, max_iterations: int) -> Dict[str, float]:
    """
    Diagnostics using already-computed S_iter{t}.
    Does NOT re-run the model or prompt repair; only re-evaluates the violation predicate.
    """
    metrics: Dict[str, float] = {}

    iters_with_any = 0
    first_zero = None

    for t in range(1, max_iterations + 1):
        col = f"S_iter{t}"
        if col not in out_df.columns:
            continue
        S = out_df[col].to_numpy(dtype=float)
        v = int(np.sum(S > float(q_star)))
        metrics[f"iter{t}.violations.fixedQstar"] = float(v)

        if v > 0:
            iters_with_any += 1
        if v == 0 and first_zero is None:
            first_zero = t

    metrics["qstar.iters_with_any_violation"] = float(iters_with_any)
    metrics["qstar.iters_to_first_zero_violation"] = float(first_zero if first_zero is not None else max_iterations)
    return metrics


def run_for_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    device: str,
    base_attrs: List[str],
    protected_sets: List[Tuple[str, ...]],
    total_t0: float,
):
    timings: dict[str, float] = {}

    processed_dir = build_processed_dir(args.processed_dir_template, dataset_name)
    repo_root = Path(__file__).resolve().parents[1]
    db_path = (repo_root / "mlflow.db").resolve()
    mcfg = MLflowConfig(
        tracking_uri=f"sqlite:///{db_path}",
        experiment_name="facter-repro",
        run_name=f"{dataset_name}_{args.model_id}_seed{args.seed}_{args.predict_mode}",
    )

    with start_run(
        mcfg,
        tags={"dataset": dataset_name, "model_id": args.model_id, "predict_mode": args.predict_mode},
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
            "repair.keying": args.repair_keying,
            "baseline_prompts": args.baseline_prompts,
            "skip_online": bool(args.skip_online),
            "cfr_flip_strategy": args.cfr_flip_strategy,
            "cfr_flip_attrs": args.cfr_flip_attrs or "default",
            "llm_batch_size": args.llm_batch_size,
            "embedder_batch_size": args.embedder_batch_size,
            "temperature": args.temperature,
        })
        log_text(json.dumps({"protected_sets": protected_sets}, indent=2), "config/protected_sets.json")

        with stage("seeding", timings):
            seed_all(SeedConfig(seed=args.seed))

        with stage("load_processed_splits", timings):
            cal_df = _read_split(processed_dir, "cal")
            test_df = _read_split(processed_dir, "test")
            log_metrics({"data.cal_n": float(len(cal_df)), "data.test_n": float(len(test_df))})

        with stage("load_raw_item_db", timings):
            item_db, domain, title_to_mid = load_item_db(dataset_name)
            log_metrics({"data.items_n": float(len(item_db))})

        with stage("init_models", timings):
            embedder, ranker, generator = init_models(args, device, dataset_name=dataset_name)

        with stage("build_catalogue_mapper", timings):
            catalogue_mapper = build_catalogue(embedder, item_db)
            log_metrics({"catalog.items_n": float(len(catalogue_mapper.catalog_titles))})

        with stage("build_item_embedder", timings):
            item_embedder = ItemEmbedder(embedder, item_db)
            log_metrics({"item_embedder.items_n": float(len(item_embedder._item_embeddings))})

        prompt_cfg = PromptConfig(k_recs=args.k)
        baseline_metrics_all = {}
        facter_metrics = {}

        # Determine which baseline prompts to run
        if args.baseline_prompts == "neutral":
            prompt_configs = [("neutral", NEUTRAL_PROMPT_TEMPLATE)]
        elif args.baseline_prompts == "fair":
            prompt_configs = [("fair", FAIR_PROMPT_TEMPLATE)]
        else:  # both
            prompt_configs = [
                ("neutral", NEUTRAL_PROMPT_TEMPLATE),
                ("fair", FAIR_PROMPT_TEMPLATE),
            ]

        for protected_cols in protected_sets:
            pset = "+".join(protected_cols)

            def P(name: str) -> str:
                return f"pset.{pset}.{name}" if args.sweep_protected_sets else name

            log_metrics({P("repair.keying.is_tuple"): 1.0 if args.repair_keying == "tuple" else 0.0})
            log_text(args.repair_keying, f"config/repair_keying_{pset}.txt")
            log_metrics({P("protected_cols_count"): float(len(protected_cols))})
            log_text(",".join(protected_cols), f"config/protected_cols_{pset}.txt")

            cfr_flips = _resolve_cfr_flips(args, protected_cols)

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
                calibrator = OfflineCalibrator(ranker=ranker, embedder=embedder, context_encoder=ctx, cfg=off_cfg)
                scorer = OnlineScorer(
                    embedder,
                    item_embedder,
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
                        protected_cols=protected_cols,
                        keying=args.repair_keying,
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
                        protected_key=args.protected_attr,
                    ),
                )

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

            # Run baseline(s) with selected prompt(s)
            for prompt_name, system_prompt in prompt_configs:
                prompt_suffix = f"_{prompt_name}"
                baseline_key = f"baseline_{prompt_name}"

                with stage(f"baseline_zero_shot[{pset}]{prompt_suffix}", timings):
                    baseline_df = run_baseline(
                        args=args,
                        protected_cols=protected_cols,
                        test_df=test_df,
                        ranker=ranker,
                        generator=generator,
                        embedder=embedder,
                        catalogue_mapper=catalogue_mapper,
                        item_db=item_db,
                        title_to_mid=title_to_mid,
                        cal_q_alpha0=cal_res.q_alpha0,
                        system_prompt=system_prompt,
                        item_embedder=item_embedder,
                    )
                    log_dataframe(baseline_df, f"data/baseline_df_{pset}{prompt_suffix}.json", format="json")

                with stage(f"baseline_metrics[{pset}]{prompt_suffix}", timings):
                    baseline_metrics = compute_baseline_metrics(
                        args=args,
                        protected_cols=protected_cols,
                        baseline_df=baseline_df,
                        embedder=embedder,
                        item_db=item_db,
                        prompt_cfg=prompt_cfg,
                        catalogue_mapper=catalogue_mapper,
                        title_to_mid=title_to_mid,
                        ranker=ranker,
                        generator=generator,
                        item_embedder=item_embedder,
                    )
                    log_metrics({P(f"{baseline_key}.{k}"): v for k, v in baseline_metrics.items()})
                    log_text(json.dumps(baseline_metrics, indent=2), f"results/baseline_metrics_{pset}{prompt_suffix}.json")
                    print(f"Baseline {prompt_name} metrics: {baseline_metrics}")
                    baseline_metrics_all[prompt_name] = baseline_metrics

            if args.skip_online:
                print(f"Skipping online FACTER phase for {pset} (--skip_online flag set)")
                continue

            with stage(f"online_monitor[{pset}]", timings):
                out_df, logs = run_online_monitor(
                    monitor=monitor,
                    args=args,
                    protected_cols=protected_cols,
                    item_db=item_db,
                    cal_art=cal_art,
                    cal_q_alpha0=cal_res.q_alpha0,
                    test_df=test_df,
                    generator=generator,
                    prompt_cfg=prompt_cfg,
                    title_to_mid=title_to_mid,
                    catalogue_mapper=catalogue_mapper,
                )

                # Log per-iteration Q start/end and violations, plus Q-trace artifacts
                for it_log in logs:
                    log_metrics({
                        P(f"iter{it_log.iteration}.q_alpha_start"): float(it_log.q_alpha_start),
                        P(f"iter{it_log.iteration}.q_alpha_end"): float(it_log.q_alpha_end),
                        P(f"iter{it_log.iteration}.q_updates.count"): float(max(0, len(it_log.q_trace) - 1)),
                        P(f"iter{it_log.iteration}.violations.dynamicQ"): float(it_log.violations),
                        P(f"iter{it_log.iteration}.violations.fixedQ0"): float(it_log.violations_at_Q0),
                        P(f"iter{it_log.iteration}.S_mean"): float(it_log.mean_S),
                        P(f"iter{it_log.iteration}.violations.dynamicQ.corrected"): float(it_log.violations_corr_dynamic),
                        P(f"iter{it_log.iteration}.violations.fixedQ0.corrected"): float(it_log.violations_corr_at_Q0),
                    }, step=it_log.iteration)

                    # Q trajectory
                    log_text(
                        json.dumps(
                            {"q_trace": it_log.q_trace, "q_update_steps": it_log.q_update_steps},
                            indent=2
                        ),
                        f"results/q_trace_{pset}_iter{it_log.iteration}.json"
                    )

                log_dataframe(out_df, f"data/online_monitor_df_{pset}.json", format="json")
            
            if logs:
                q_star = float(logs[-1].q_alpha_end)
                qstar_metrics = compute_qstar_counterfactuals(out_df, q_star=q_star, max_iterations=args.max_iterations)

                log_metrics({P(k): v for k, v in qstar_metrics.items()})
                log_text(json.dumps(qstar_metrics, indent=2), f"results/qstar_counterfactuals_{pset}.json")

            with stage(f"compute_facter_metrics[{pset}]", timings):
                facter_metrics = compute_facter_metrics(
                    args=args,
                    protected_cols=protected_cols,
                    embedder=embedder,
                    item_db=item_db,
                    out_df=out_df,
                    prompt_cfg=prompt_cfg,
                    cfr_flips=cfr_flips,
                    catalogue_mapper=catalogue_mapper,
                    title_to_mid=title_to_mid,
                    ranker=ranker,
                    generator=generator,
                    item_embedder=item_embedder,
                )
                log_metrics({P(k): v for k, v in facter_metrics.items()})
                log_text(json.dumps(facter_metrics, indent=2), f"results/facter_metrics_{pset}.json")

            with stage(f"save_outputs[{pset}]", timings):
                save_outputs(processed_dir, args, pset, out_df)

        # Flush any remaining embeddings to disk
        with stage("flush_embedder_cache", timings):
            embedder.flush()

        log_metrics({"stage.TOTAL.seconds": float(time.perf_counter() - total_t0)})
        log_text(json.dumps(timings, indent=2), "results/timings.json")
        print(f"Timings for {dataset_name}:", timings)
        print(f"Baseline metrics: {baseline_metrics_all}")
        if not args.skip_online:
            print(f"FACTER metrics: {facter_metrics}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    datasets = parse_datasets(args.datasets)
    base_attrs, protected_sets = parse_protected_sets(args)
    seeds = parse_seeds(args.seeds)

    total_t0 = time.perf_counter()

    for dataset_name in datasets:
        print(f"\n\n{'#'*80}")
        print(f"Processing dataset: {dataset_name.upper()}")
        print(f"{'#'*80}")

        for seed in seeds:
            seed_args = argparse.Namespace(**vars(args))
            seed_args.seed = seed

            print(f"\n\n{'='*80}")
            print(f"Running seed: {seed}")
            print(f"{'='*80}\n")
            run_for_dataset(dataset_name, seed_args, device, base_attrs, protected_sets, total_t0)


if __name__ == "__main__":
    output_dir = "./emissions"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_filename = f"emissions_{timestamp}.csv"

    tracker = EmissionsTracker(
        project_name="facter_repro",
        output_dir=output_dir,
        output_file=run_filename,
        save_to_api=False,
    )

    with tracker:
        main()

    print("\n[CodeCarbon] Energy tracking finished.")
    print(f" -> Results saved to: {os.path.join(output_dir, run_filename)}")
