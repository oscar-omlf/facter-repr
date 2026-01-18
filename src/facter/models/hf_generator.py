import hashlib
import json, re
from dataclasses import dataclass
from typing import List, Sequence, Optional

from pathlib import Path

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HFGenConfig:
    model_id: str
    max_new_tokens: int = 250
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    batch_size: int = 8
    cache_dir: Path = Path("data/cache/generator")
    torch_dtype: str = "auto"       # "auto" | "float16" | "bfloat16"
    device_map: str = "auto"        # passed to transformers
    trust_remote_code: bool = False

def parse_json_list(text: str, k: int) -> List[str]:
    if not text:
        return []

    # 1. Pre-process (Smart Quotes)
    sanitized_text = (
        text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    )

    # 2. Try JSON array extraction
    m = re.search(r"\[[\s\S]*\]", sanitized_text)
    if m:
        json_str = m.group(0)
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
                return out
        except Exception:
            pass 

    # 3. Improved Fallback Parse
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    seen = set()

    # ADD "assistant" and "user" to ignore patterns
    ignore_patterns = [
        r"based on", r"here is", r"here're", r"recommended", 
        r"note that", r"^assistant$", r"^user$", r"^system$"
    ]

    for ln in lines:
        # 1. Skip Markdown code fences and JSON structural brackets
        if '```' in ln or ln.strip() in ['[', ']', '],', '],']:
            continue

        # 2. Clean bullets, numbers, and leading/trailing brackets/braces
        # This regex now also targets leading [ or { if they are part of the line
        clean_ln = re.sub(r"^\s*[\-\*\d\.\)\:\[\]\{\}]+\s*", "", ln).strip()
        
        # 3. Clean up trailing artifacts from the JSON-like format
        # This removes trailing commas, closing brackets, and quotes
        clean_ln = clean_ln.rstrip(',').rstrip(']').rstrip('}').strip('"').strip("'")

        # 4. Skip if line is empty, too long, or matches ignore list
        if not clean_ln or len(clean_ln) > 100:
            continue
        if any(re.search(p, clean_ln, re.I) for p in ignore_patterns):
            continue

        if clean_ln not in seen:
            out.append(clean_ln)
            seen.add(clean_ln)

        if len(out) >= k:
            break

    return out


class HFOpenGenerator:
    def __init__(self, cfg: HFGenConfig, tokenizer: AutoTokenizer = None, model: AutoModelForCausalLM = None):
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = model or AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
        )
        self.model.eval()

        self._model_cache_dir = self.cfg.cache_dir / _sha256(cfg.model_id)[:16]
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, prompt_rank: str, system_prompt: Optional[str]) -> str:
        blob = {
            "system": system_prompt or "",
            "user": prompt_rank,
        }
        return _sha256(json.dumps(blob, ensure_ascii=False, sort_keys=True))

    def _cache_path(self, key: str) -> Path:
        return self._model_cache_dir / f"{key}.json"

    @torch.inference_mode()
    def generate_topk(
        self,
        prompts: Sequence[str],
        system_prompts: Sequence[Optional[str]],
        k: int,
        progress: bool = False,
    ) -> List[List[str]]:
        if len(prompts) != len(system_prompts):
            raise ValueError("prompts and system_prompts must have the same length")
        
        key = self._cache_key(prompts, system_prompts)
        cpath = self._cache_path(key)
        
        if cpath.exists():
            obj = json.loads(cpath.read_text(encoding="utf-8"))
            return list(obj["json_list"])

        out_all: List[List[str]] = []
        generated_content_list: List[str] = []
        device = self.model.device
        do_sample = self.cfg.temperature > 0.0

        if progress:
            it = tqdm(range(0, len(prompts), self.cfg.batch_size), total=(len(prompts) + self.cfg.batch_size - 1) // self.cfg.batch_size, desc="HFGen: generate")
        else:
            it = range(0, len(prompts), self.cfg.batch_size)

        for i in it:
            batch_prompts = list(prompts[i : i + self.cfg.batch_size])
            batch_systems = list(system_prompts[i : i + self.cfg.batch_size])

            rendered_batch = []
            for p, sys in zip(batch_prompts, batch_systems):
                sys = sys or ""
                msgs = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": p},
                ]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    rendered = self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                else:
                    rendered = f"SYSTEM:\n{sys}\n\nUSER:\n{p}\n\nASSISTANT:\n"
                rendered_batch.append(rendered)

            toks = self.tokenizer(
                rendered_batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            gen = self.model.generate(
                **toks,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=self.cfg.temperature if do_sample else None,
                top_p=self.cfg.top_p if do_sample else None,
                repetition_penalty=self.cfg.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # decode only generated continuation
            for j in range(len(batch_prompts)):
                cont = gen[j][toks["input_ids"].shape[-1] :]
                txt = self.tokenizer.decode(cont, skip_special_tokens=True)

                json_list = parse_json_list(txt, k)

                out_all.append(json_list)
                generated_content_list.append(txt)

        cpath.write_text(
            json.dumps({"json_list": out_all, 'generated_content': generated_content_list}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return out_all
