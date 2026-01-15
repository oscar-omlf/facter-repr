"""
utils.py: Generation, parsing, and metrics for FACTER (paper-aligned).
- Generates Top-K ranked lists (open-vocabulary) and parses JSON arrays.
- Computes HitRate@K and NDCG@K for next-item prediction.
"""
from __future__ import annotations

import json
import logging
import re
import json
import pickle
import hashlib
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch

from .config import Config

logger = logging.getLogger(__name__)


def setup_logging():
    logger_root = logging.getLogger()
    if logger_root.handlers:
        return logger_root

    logger_root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger_root.addHandler(stream_handler)

    file_handler = logging.FileHandler("run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger_root.addHandler(file_handler)

    return logger_root


# -------------------------
# Prompt formatting (chat template if available)
# -------------------------
def _format_chat(tokenizer, system_msg: str, user_msg: str) -> torch.Tensor:
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # fallback
    text = f"<system>\n{system_msg}\n</system>\n<user>\n{user_msg}\n</user>\n<assistant>\n"
    return tokenizer(text, return_tensors="pt").input_ids


def _best_fuzzy_match(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def parse_ranked_list(text: str, k: int) -> List[str]:
    if not text:
        logger.info("[parse_ranked_list] Empty text")
        return []

    # 1. Pre-process to handle common LLM "JSON-ish" errors
    # Replace smart quotes with standard quotes
    sanitized_text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # 2. Try JSON array extraction
    m = re.search(r"\[[\s\S]*\]", sanitized_text)
    if m:
        json_str = m.group(0)
        logger.info(f"[parse_ranked_list] Found JSON: {json_str[:100]}")
        # Remove trailing commas before a closing bracket (common LLM error)
        json_str = re.sub(r",\s*\]", "]", json_str)
        try:
            arr = json.loads(json_str)
            if isinstance(arr, list):
                out = []
                seen = set()
                for x in arr:
                    val = str(x).strip()
                    if val and val not in seen:
                        out.append(val)
                        seen.add(val)
                    if len(out) >= k:
                        break
                logger.info(f"[parse_ranked_list] Parsed JSON: {out}")
                return out
        except Exception as e:
            logger.info(f"[parse_ranked_list] JSON parse failed: {e}")

    # 3. Improved Fallback Parse
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    logger.info(f"[parse_ranked_list] Fallback parse from {len(lines)} lines")
    out = []
    seen = set()
    
    # Common conversational prefixes to ignore
    ignore_patterns = [r"based on", r"here is", r"here're", r"recommended", r"note that"]

    for ln in lines:
        # Clean bullets/numbers
        clean_ln = re.sub(r"^\s*[\-\*\d\.\)\:\[\]]+\s*", "", ln).strip()
        # Remove wrapping quotes if they exist
        clean_ln = clean_ln.strip('"').strip("'")
        
        # Skip if line is empty, too long (likely a sentence), or matches ignore list
        if not clean_ln or len(clean_ln) > 100:
            continue
        if any(re.search(p, clean_ln, re.I) for p in ignore_patterns):
            continue
            
        if clean_ln not in seen:
            out.append(clean_ln)
            seen.add(clean_ln)
        
        if len(out) >= k:
            break

    logger.info(f"[parse_ranked_list] Final output: {out}")
    return out

def generate_recommendations(
    prompts: List[str],
    system_msg: str,
    tokenizer,
    model,
) -> List[List[str]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup cache directory
    cache_dir = Path("./data/cache/recommendations")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache key based on prompts and config
    cache_key_str = json.dumps({
        "prompts": prompts,
        "system_msg": system_msg,
        "max_tokens": Config.MAX_NEW_TOKENS,
        "temperature": Config.TEMPERATURE,
        "top_p": Config.TOP_P,
    }, sort_keys=True)
    cache_key = hashlib.sha256(cache_key_str.encode()).hexdigest()
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    # Try to load from cache
    if cache_file.exists():
        logger.info(f"[generate_recommendations] Loading from cache: {cache_file.name}")
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.info(f"[generate_recommendations] Cache load failed: {e}")
    
    all_recs: List[List[str]] = []

    for i in range(0, len(prompts), Config.BATCH_SIZE):
        batch = [p for p in prompts[i : i + Config.BATCH_SIZE] if p is not None]

        if not batch:
            continue

        # build batch input ids
        input_ids_list = [_format_chat(tokenizer, system_msg, p) for p in batch]
        # pad manually
        max_len = max(x.shape[-1] for x in input_ids_list)
        input_ids = torch.full((len(input_ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long)
        for j, x in enumerate(input_ids_list):
            input_ids[j, -x.shape[-1] :] = x[0]
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                repetition_penalty=Config.REPETITION_PENALTY,
                do_sample=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for txt in decoded:
            # extract generated part
            gen_part = txt.split("assistant")[-1].strip()
            logger.info(gen_part)
            recs = parse_ranked_list(gen_part, Config.TOP_K_RECS)
            logger.info(recs)
            all_recs.append(recs)

    # if any prompts were None, keep alignment by returning empty lists for them
    if len(all_recs) != len(prompts):
        # best-effort: pad
        while len(all_recs) < len(prompts):
            all_recs.append([])
        all_recs = all_recs[: len(prompts)]
    
    # Save to cache
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(all_recs, f)
        logger.info(f"[generate_recommendations] Saved to cache: {cache_file.name}")
    except Exception as e:
        logger.info(f"[generate_recommendations] Cache save failed: {e}")
    
    return all_recs


# -------------------------
# Metrics (@K)
# -------------------------
def hitrate_ndcg_at_k(preds: List[str], gold: str, k: int) -> Tuple[float, float]:
    if not preds:
        return 0.0, 0.0
    gold = gold.strip()
    for rank, p in enumerate(preds[:k], start=1):
        if _best_fuzzy_match(p, gold) >= 0.85:
            # Hit@K = 1; NDCG@K = 1/log2(rank+1)
            return 1.0, 1.0 / np.log2(rank + 1)
    return 0.0, 0.0


def evaluate_at_k(df, k: int = 10) -> dict:
    hits, ndcgs = [], []
    for _, row in df.iterrows():
        preds = row["recs"] if isinstance(row["recs"], list) else []
        gold = str(row["target_title"])
        h, n = hitrate_ndcg_at_k(preds, gold, k)
        hits.append(h)
        ndcgs.append(n)
    return {
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }

def _best_fuzzy_match(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower().strip(), (b or "").lower().strip()).ratio()


def hitrate_ndcg_at_k(preds: List[str], gold: str, k: int) -> Tuple[float, float]:
    if not preds:
        logger.info(f"[hitrate_ndcg_at_k] Empty preds, gold={gold}")
        return 0.0, 0.0
    gold = (gold or "").strip()
    logger.info(f"[hitrate_ndcg_at_k] Comparing {len(preds)} preds against gold='{gold}'")
    for rank, p in enumerate(preds[:k], start=1):
        score = _best_fuzzy_match(p, gold) if p else 0.0
        logger.info(f"  Rank {rank}: pred='{p}' -> score={score:.2f}")
        if p and score >= 0.85:
            ndcg = 1.0 / np.log2(rank + 1)
            logger.info(f"[hitrate_ndcg_at_k] HIT! rank={rank}, NDCG={ndcg:.4f}")
            return 1.0, ndcg
    logger.info(f"[hitrate_ndcg_at_k] No hit found")
    return 0.0, 0.0


def evaluate_at_k_from_lists(
    rec_lists: List[List[str]],
    gold_titles: List[str],
    k: int = 10,
) -> Dict[str, float]:
    logger.info(f"[evaluate_at_k_from_lists] Processing {len(rec_lists)} items")
    hits, ndcgs = [], []
    for i, (recs, gold) in enumerate(zip(rec_lists, gold_titles)):
        recs = recs if isinstance(recs, list) else []
        logger.info(f"  Item {i}: {len(recs)} recs, gold='{gold}'")
        h, n = hitrate_ndcg_at_k(recs, str(gold), k)
        hits.append(h)
        ndcgs.append(n)
    result = {
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
    logger.info(f"[evaluate_at_k_from_lists] Result: {result}")
    return result


def evaluate_valid_at_k(valid_at_k_list: List[float], k: int = 10) -> Dict[str, float]:
    # valid_at_k_list is already per-example fraction valid among top-k mapped
    return {f"Valid@{k}": float(np.mean(valid_at_k_list)) if valid_at_k_list else 0.0}
