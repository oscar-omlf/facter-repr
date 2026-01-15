import json, re
from dataclasses import dataclass
from typing import List, Sequence, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass(frozen=True)
class HFGenConfig:
    model_id: str
    max_new_tokens: int = 250
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    batch_size: int = 8

def parse_json_list(text: str, k: int) -> List[str]:
    if not text:
        return []
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                out = []
                seen = set()
                for x in arr:
                    s = str(x).strip()
                    if s and s not in seen:
                        out.append(s); seen.add(s)
                    if len(out) >= k: break
                return out
        except Exception:
            pass
    # fallback: parse lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        ln = re.sub(r"^\s*[\-\*\d\.\)\:]+\s*", "", ln).strip()
        if ln:
            out.append(ln)
        if len(out) >= k: break
    return out


class HFOpenGenerator:
    def __init__(self, cfg: HFGenConfig, tokenizer: AutoTokenizer = None, model: AutoModelForCausalLM = None):
        self.cfg = cfg
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

    @torch.inference_mode()
    def generate_topk(
        self,
        prompts: Sequence[str],
        system_prompts: Sequence[Optional[str]],
        k: int,
    ) -> List[List[str]]:
        if len(prompts) != len(system_prompts):
            raise ValueError("prompts and system_prompts must have the same length")

        out_all: List[List[str]] = []
        device = self.model.device
        do_sample = self.cfg.temperature > 0.0

        for i in range(0, len(prompts), self.cfg.batch_size):
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
                print(f"Generated text (prompt): {txt}")
                print(f"Parsed list: {json_list}")
                out_all.append(json_list)

        return out_all
