import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_title(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("'", "'")
    s = re.sub(r'["""]', "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 \-:'&(),.!?]", "", s)
    s = s.strip()

    # Move determiners from end to beginning: "Fish Called Wanda, A" -> "A Fish Called Wanda"
    # Handle patterns like ", A (year)", ", The (year)", ", An (year)"
    # First remove year in parentheses temporarily
    year_match = re.search(r"\s*\((\d{4})\)$", s)
    year_suffix = ""
    if year_match:
        year_suffix = f" ({year_match.group(1)})"
        s = s[: year_match.start()].strip()

    # Now handle determiner at end: ", A", ", The", ", An"
    determiner_match = re.search(r",\s*(a|an|the)$", s, re.IGNORECASE)
    if determiner_match:
        determiner = determiner_match.group(1).lower()
        title_without_determiner = s[: determiner_match.start()].strip()
        s = f"{determiner} {title_without_determiner}"

    # Re-add year if it existed
    s = s + year_suffix

    return s.strip()


def _try_parse_as_indices(text: str, n: int) -> List[int]:
    """
    If the model outputs numeric candidate IDs (1..n) per line, parse them.
    """
    out: List[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(\d+)\s*$", line)
        if m:
            k = int(m.group(1))
            if 1 <= k <= n and (k - 1) not in out:
                out.append(k - 1)
            continue
        m = re.match(r"^\s*(\d+)[\).\:\-]\s*(.*)$", line)
        if m and (m.group(2).strip() == ""):
            k = int(m.group(1))
            if 1 <= k <= n and (k - 1) not in out:
                out.append(k - 1)
    return out


def _parse_ranking_to_indices(text: str, candidate_titles: Sequence[str]) -> List[int]:
    """
    Robust-ish parsing:
    - Prefer exact/normalized title matches.
    - Fall back to numeric indices if the model outputs numbers.
    - Ensure output is a full permutation.
    """
    n = len(candidate_titles)
    norm_to_idx = {}
    for i, t in enumerate(candidate_titles):
        norm_to_idx[_normalize_title(t)] = i

    # 1) Try numeric-only format
    idxs = _try_parse_as_indices(text, n)
    if len(idxs) >= 2:  # strong signal
        pass
    else:
        idxs = []

        # 2) Title-based parsing
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue

            # strip leading bullets / numbering
            s = re.sub(r"^\s*[\-\*\u2022]\s*", "", s)
            s = re.sub(r"^\s*\d+\s*[\).\:\-]\s*", "", s)

            # remove surrounding quotes
            s = s.strip().strip('"').strip("'").strip()
            if not s:
                continue

            # exact normalized match
            ns = _normalize_title(s)
            if ns in norm_to_idx:
                i = norm_to_idx[ns]
                if i not in idxs:
                    idxs.append(i)
                continue

            # try substring match against candidates (common when model truncates)
            for cand_norm, i in norm_to_idx.items():
                if cand_norm and (cand_norm in ns or ns in cand_norm):
                    if i not in idxs:
                        idxs.append(i)
                    break

    # 3) Complete to a permutation
    remaining = [i for i in range(n) if i not in idxs]
    return idxs + remaining


@dataclass(frozen=True)
class HFChatRankerConfig:
    model_id: str
    cache_dir: Path = Path("data/cache/ranker")
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
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
            cfg.model_id, use_fast=True, trust_remote_code=cfg.trust_remote_code
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
            offload_folder="offload",
        )
        self.model.eval()

        self._model_cache_dir = self.cfg.cache_dir / _sha256(cfg.model_id)[:16]
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        prompt_rank: str,
        candidate_titles: Sequence[str],
        system_prompt: Optional[str],
    ) -> str:
        blob = {
            "system": system_prompt or "",
            "user": prompt_rank,
            "candidates": list(candidate_titles),
        }
        return _sha256(json.dumps(blob, ensure_ascii=False, sort_keys=True))

    def _cache_path(self, key: str) -> Path:
        return self._model_cache_dir / f"{key}.json"

    def rank(
        self,
        prompt_rank: str,
        candidate_titles: Sequence[str],
        system_prompt: str | None = None,
    ) -> Tuple[List[str], str]:
        key = self._cache_key(prompt_rank, candidate_titles, system_prompt)
        cpath = self._cache_path(key)

        if cpath.exists():
            obj = json.loads(cpath.read_text(encoding="utf-8"))
            return list(obj["ranked_indices"]), obj.get("raw_text", "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_rank})

        # Prefer chat template when available
        prompt = None
        chat_template = getattr(self.tokenizer, "chat_template", None)

        if chat_template:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except ValueError:
                prompt = None  # fall back below

        if prompt is None:
            # Fallback: plain text (works for base models)
            prompt = ""
            if system_prompt:
                prompt += f"SYSTEM:\n{system_prompt}\n\n"
            prompt += f"USER:\n{prompt_rank}\n\nASSISTANT:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        do_sample = not (self.cfg.temperature == 0.0)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=self.cfg.temperature if do_sample else None,
            top_p=self.cfg.top_p if do_sample else None,
            repetition_penalty=self.cfg.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        out = self.tokenizer.decode(
            gen[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        ranked = _parse_ranking_to_indices(out, candidate_titles)

        cpath.write_text(
            json.dumps(
                {"ranked_indices": ranked, "raw_text": out},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return ranked, out
