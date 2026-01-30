"""Rank candidate items using a HuggingFace causal language model.

This module provides a small "ranker" wrapper around HuggingFace Transformers
to order a provided candidate set given a ranking prompt. Rankings can be
cached on disk (JSON files) to avoid repeated inference for identical inputs.

The parsing utilities in this module are designed to interpret LLM outputs that
attempt to return a JSON array of ranked titles.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prefer generator-style JSON parsing for consistency
from facter.models.hf_generator import parse_json_list


def _sha256(s: str) -> str:
    """Compute a SHA-256 hex digest for a string.

    Args:
        s (str): Input string.

    Returns:
        str: Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_title(s: str) -> str:
    """Normalize a title string for matching against candidate titles.

    The normalization is heuristic and intended to make matching more robust to
    formatting differences (quotes, whitespace, casing) and optional trailing
    year suffixes.

    Args:
        s (str): Input title.

    Returns:
        str: Normalized title.
    """
    s = s.strip().lower()
    s = s.replace("'", "'")
    s = re.sub(r'["""]', "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 \-:'&(),.!?]", "", s)
    s = s.strip()

    # Strip a trailing year suffix: "Heat (1995)" -> "Heat".
    # This makes matching robust when the model omits years but candidates include them.
    # We only remove a terminal 4-digit year in () or [] to avoid harming legitimate titles.
    s = re.sub(r"\s*(?:\(|\[)\s*(19\d{2}|20\d{2})\s*(?:\)|\])\s*$", "", s).strip()

    # Move determiners from end to beginning: "Fish Called Wanda, A" -> "A Fish Called Wanda"
    determiner_match = re.search(r",\s*(a|an|the)$", s, re.IGNORECASE)
    if determiner_match:
        determiner = determiner_match.group(1).lower()
        title_without_determiner = s[: determiner_match.start()].strip()
        s = f"{determiner} {title_without_determiner}"

    return s.strip()


def _try_parse_as_indices(titles: List[str], n: int) -> List[int]:
    """Extract numeric candidate references from parsed titles.

    This helper supports LLM outputs that rank by returning integer identifiers
    (1..n) rather than reproducing full candidate titles.

    Args:
        titles (List[str]): Parsed entries from the model output.
        n (int): Number of candidates.

    Returns:
        List[int]: De-duplicated 0-based candidate indices.
    """
    out: List[int] = []
    for title in titles:
        title = title.strip()
        if not title:
            continue
        m = re.match(r"^\s*(\d+)\s*$", title)
        if m:
            k = int(m.group(1))
            if 1 <= k <= n and (k - 1) not in out:
                out.append(k - 1)
    return out


def _parse_ranking_to_indices(titles: List[str], candidate_titles: Sequence[str]) -> List[int]:
    """Map parsed model outputs to a permutation of candidate indices.

    The function first attempts to interpret entries as numeric indices (1..n).
    If that fails, it performs a best-effort match by normalized title, with a
    substring fallback.

    Args:
        titles (List[str]): Parsed entries from the model output.
        candidate_titles (Sequence[str]): Candidate titles to be ranked.

    Returns:
        List[int]: A full permutation of ``range(len(candidate_titles))``,
        beginning with any indices recovered from the model output.
    """
    n = len(candidate_titles)
    norm_to_idx = {}
    for i, t in enumerate(candidate_titles):
        norm_to_idx[_normalize_title(t)] = i

    # 1) Try numeric index format
    idxs = _try_parse_as_indices(titles, n)
    if len(idxs) >= 2:
        remaining = [i for i in range(n) if i not in idxs]
        return idxs + remaining

    # 2) Title-based matching via normalization + substring fallback
    idxs = []
    seen = set()

    for title_str in titles:
        title_clean = title_str.strip().strip('"').strip("'")
        if not title_clean:
            continue

        nt = _normalize_title(title_clean)
        if nt in norm_to_idx:
            i = norm_to_idx[nt]
            if i not in seen:
                idxs.append(i)
                seen.add(i)
            continue

        # substring fallback
        for cand_norm, i in norm_to_idx.items():
            if i in seen:
                continue
            if cand_norm and (cand_norm in nt or nt in cand_norm):
                idxs.append(i)
                seen.add(i)
                break

    remaining = [i for i in range(n) if i not in seen]
    return idxs + remaining


@dataclass(frozen=True)
class HFChatRankerConfig:
    """Configure :class:`HFChatRanker`.

    Attributes:
        model_id (str): HuggingFace model identifier used with
            :func:`transformers.AutoTokenizer.from_pretrained` and
            :func:`transformers.AutoModelForCausalLM.from_pretrained`.
        cache_dir (Path): Root directory for on-disk rank cache.
        max_new_tokens (int): Maximum number of new tokens generated for the
            ranking response.
        temperature (float): Sampling temperature; values $\le 0$ disable
            sampling.
        top_p (float): Nucleus sampling parameter (used only when sampling is
            enabled).
        repetition_penalty (float): Repetition penalty forwarded to
            ``model.generate``.
        batch_size (int): Number of ranking prompts processed per generation
            batch.
        torch_dtype (str): Torch dtype passed to ``from_pretrained``.
        device_map (str): Device mapping passed to ``from_pretrained``.
        trust_remote_code (bool): Whether to allow custom model code in
            ``from_pretrained``.
        seed (Optional[int]): Optional seed included in the cache key.
    """
    model_id: str
    cache_dir: Path = Path("data/cache/ranker")
    max_new_tokens: int = 250
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    batch_size: int = 8
    torch_dtype: str = "bfloat16"  # "auto" | "float16" | "bfloat16"
    device_map: str = "auto"  # passed to transformers
    trust_remote_code: bool = False
    seed: Optional[int] = None


class HFChatRanker:
    """Rank candidates via a local HuggingFace causal language model.

    Deterministic settings are used when ``cfg.temperature == 0``.
    Rankings are cached on disk keyed by the model id, generation parameters,
    system prompt, user prompt, and candidates.
    """

    def __init__(self, cfg: HFChatRankerConfig):
        """Initialize the ranker and create model-specific cache directory.

        Args:
            cfg (HFChatRankerConfig): Ranker configuration.

        Raises:
            ValueError: If ``cfg.torch_dtype`` is not one of the supported
                string values.
        """
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

        dtype = cfg.torch_dtype
        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown torch_dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, use_fast=True, trust_remote_code=cfg.trust_remote_code, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.model.eval()

        self._model_cache_dir = self.cfg.cache_dir / _sha256(cfg.model_id)[:16]
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)

    # def _cache_key(self, prompt_rank: str, candidate_titles: Sequence[str], system_prompt: Optional[str]) -> str:
    #     blob = {
    #         "system": system_prompt or "",
    #         "user": prompt_rank,
    #         "candidates": list(candidate_titles),
    #     }
    #     return _sha256(json.dumps(blob, ensure_ascii=False, sort_keys=True))

    def _cache_key(self, prompt_rank: str, candidate_titles: Sequence[str], system_prompt: Optional[str]) -> str:
        """Compute the cache key for a ranking request.

        Args:
            prompt_rank (str): User prompt that instructs ranking.
            candidate_titles (Sequence[str]): Candidate titles to rank.
            system_prompt (Optional[str]): Optional system prompt.

        Returns:
            str: SHA-256 digest over a JSON-serialized request descriptor.
        """
        rank_cfg = {
            "max_new_tokens": int(self.cfg.max_new_tokens),
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "repetition_penalty": float(self.cfg.repetition_penalty),
        }
        if self.cfg.seed is not None:
            rank_cfg["seed"] = int(self.cfg.seed)

        blob = {
            "model_id": self.cfg.model_id,
            "rank_cfg": rank_cfg,
            "system": system_prompt or "",
            "user": prompt_rank,
            "candidates": list(candidate_titles),
        }
        return _sha256(json.dumps(blob, ensure_ascii=False, sort_keys=True))

    def _cache_path(self, key: str) -> Path:
        """Map a cache key to the corresponding on-disk JSON path.

        Args:
            key (str): Cache key produced by :meth:`_cache_key`.

        Returns:
            Path: Path to the cache file.
        """
        return self._model_cache_dir / f"{key}.json"

    @torch.inference_mode()
    def rank_batch(
        self,
        prompts: Sequence[str],
        candidate_titles_list: Sequence[Sequence[str]],
        system_prompts: Sequence[Optional[str]],
        progress: bool = False,
    ) -> List[Tuple[List[int], str]]:
        """Rank multiple candidate sets in batch.

        For each input, the model is prompted to return a JSON list. The parsed
        list is converted to a permutation of candidate indices via
        :func:`_parse_ranking_to_indices`.

        Args:
            prompts (Sequence[str]): Ranking prompts.
            candidate_titles_list (Sequence[Sequence[str]]): Candidate titles
                per prompt.
            system_prompts (Sequence[Optional[str]]): System prompts per prompt.
            progress (bool): Whether to show a tqdm progress bar.

        Returns:
            List[Tuple[List[int], str]]: For each prompt, a tuple of
            ``(ranked_indices, raw_text)`` where ``ranked_indices`` is a
            permutation of candidate indices and ``raw_text`` is the decoded
            model continuation.

        Raises:
            ValueError: If input sequences are not the same length.
        """
        if len(prompts) != len(system_prompts) or len(prompts) != len(candidate_titles_list):
            raise ValueError("prompts, system_prompts, and candidate_titles_list must have the same length")

        out_all: List[Tuple[List[int], str]] = [None] * len(prompts)
        device = self.model.device
        do_sample = self.cfg.temperature > 0.0

        # Check cache first and collect uncached indices
        uncached_indices = []
        cache_keys = {}
        for i, (prompt, candidates, sys_prompt) in enumerate(zip(prompts, candidate_titles_list, system_prompts)):
            key = self._cache_key(prompt, candidates, sys_prompt)
            cache_keys[i] = (key, candidates)
            cpath = self._cache_path(key)

            if cpath.exists():
                obj = json.loads(cpath.read_text(encoding="utf-8"))
                out_all[i] = (list(obj["ranked_indices"]), obj.get("raw_text", ""))
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return out_all

        # Process uncached items in batches
        if progress:
            it = tqdm(
                range(0, len(uncached_indices), self.cfg.batch_size),
                total=(len(uncached_indices) + self.cfg.batch_size - 1) // self.cfg.batch_size,
                desc="HFRanker: rank",
            )
        else:
            it = range(0, len(uncached_indices), self.cfg.batch_size)

        for batch_start in it:
            batch_end = min(batch_start + self.cfg.batch_size, len(uncached_indices))
            batch_indices = uncached_indices[batch_start:batch_end]

            batch_prompts = [prompts[i] for i in batch_indices]
            batch_systems = [system_prompts[i] for i in batch_indices]
            batch_candidates = [candidate_titles_list[i] for i in batch_indices]

            rendered_batch = []
            for p, sys in zip(batch_prompts, batch_systems):
                sys = sys or ""
                msgs = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": p},
                ]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    rendered = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                else:
                    rendered = f"SYSTEM:\n{sys}\n\nUSER:\n{p}\n\nASSISTANT:\n"
                rendered_batch.append(rendered)

            toks = self.tokenizer(rendered_batch, return_tensors="pt", padding=True, truncation=True).to(device)

            gen = self.model.generate(
                **toks,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=self.cfg.temperature if do_sample else None,
                top_p=self.cfg.top_p if do_sample else None,
                repetition_penalty=self.cfg.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Decode and process results
            for j, orig_idx in enumerate(batch_indices):
                cont = gen[j][toks["input_ids"].shape[-1] :]
                txt = self.tokenizer.decode(cont, skip_special_tokens=True)

                # Parse LLM output as JSON list of titles
                candidates = batch_candidates[j]
                # We want at least the top-k candidates to be recoverable.
                # Some models (notably Mistral) may produce invalid JSON-like arrays
                # plus trailing explanation text; parse_json_list is robust to this,
                # but if it returns empty, retry with a smaller k to reduce brittleness.
                json_titles = parse_json_list(txt, k=len(candidates))
                if not json_titles:
                    json_titles = parse_json_list(txt, k=min(10, len(candidates)))

                # Map parsed titles to candidate indices
                ranked = _parse_ranking_to_indices(json_titles, candidates)

                # print(f"Given prompt:\n{batch_prompts[j]} system prompt: {system_prompts[j] if system_prompts[j] else ''}\nGenerated text:\n{txt}\nParsed titles:\n{json_titles}\nRanked indices:\n{ranked}\n")

                # Cache result
                key, _ = cache_keys[orig_idx]
                cpath = self._cache_path(key)
                cpath.write_text(
                    json.dumps(
                        {"ranked_indices": ranked, "raw_text": txt, "parsed_titles": json_titles},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                out_all[orig_idx] = (ranked, txt)

        return out_all

    def rank(
        self, prompt_rank: str, candidate_titles: Sequence[str], system_prompt: Optional[str] = None
    ) -> Tuple[List[int], str]:
        """Rank a single candidate set.

        Args:
            prompt_rank (str): Ranking prompt.
            candidate_titles (Sequence[str]): Candidate titles to rank.
            system_prompt (Optional[str]): Optional system prompt passed with
                the request.

        Returns:
            Tuple[List[int], str]: A pair ``(ranked_indices, raw_text)``.
        """
        results = self.rank_batch([prompt_rank], [candidate_titles], [system_prompt], progress=False)
        return results[0]
