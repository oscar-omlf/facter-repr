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
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_title(s: str) -> str:
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
    """
    If titles contain numeric candidate IDs (1..n), extract and map them to indices.
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
    """
    Map parsed titles to candidate indices using catalogue-based matching.
    - Tries numeric indices first if titles contain numbers (1..n).
    - Falls back to normalized title matching with substring fallback.
    - Ensures output is a full permutation of indices.
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
    model_id: str
    cache_dir: Path = Path("data/cache/ranker")
    max_new_tokens: int = 250
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    batch_size: int = 8
    torch_dtype: str = "auto"  # "auto" | "float16" | "bfloat16"
    device_map: str = "auto"  # passed to transformers
    trust_remote_code: bool = False


class HFChatRanker:
    """
    Black-box ranker via local HF causal LM.

    Deterministic settings by default (do_sample=False when temperature==0).
    Caches rankings on disk keyed by (system_prompt, user_prompt, candidates).
    """

    def __init__(self, cfg: HFChatRankerConfig):
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
        blob = {
            "model_id": self.cfg.model_id,
            "rank_cfg": {
                "max_new_tokens": int(self.cfg.max_new_tokens),
                "temperature": float(self.cfg.temperature),
                "top_p": float(self.cfg.top_p),
                "repetition_penalty": float(self.cfg.repetition_penalty),
            },
            "system": system_prompt or "",
            "user": prompt_rank,
            "candidates": list(candidate_titles),
        }
        return _sha256(json.dumps(blob, ensure_ascii=False, sort_keys=True))

    def _cache_path(self, key: str) -> Path:
        return self._model_cache_dir / f"{key}.json"

    @torch.inference_mode()
    def rank_batch(
        self,
        prompts: Sequence[str],
        candidate_titles_list: Sequence[Sequence[str]],
        system_prompts: Sequence[Optional[str]],
        progress: bool = False,
    ) -> List[Tuple[List[int], str]]:
        """
        Rank multiple sets of candidates in batch.

        Args:
            prompts: Ranking prompts
            candidate_titles_list: Candidate titles per prompt
            system_prompts: System prompts per prompt
            progress: Show progress bar

        Returns:
            List of (ranked_indices, raw_text) tuples
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
        """
        Rank a single set of candidates. Wrapper for convenience.

        Args:
            prompt_rank: Ranking prompt
            candidate_titles: Candidate titles to rank
            system_prompt: Optional system prompt

        Returns:
            (ranked_indices, raw_text)
        """
        results = self.rank_batch([prompt_rank], [candidate_titles], [system_prompt], progress=False)
        return results[0]
