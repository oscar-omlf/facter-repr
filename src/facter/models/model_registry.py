"""Model registry for selecting local Hugging Face baseline models.

This module defines a small mapping from short names ("base models") to the
Hugging Face `model_id` used by HFChatRanker/HFOpenGenerator.

Notes:
- All models are assumed to be run locally via `transformers`.
- Ranker and generator share the same `model_id` in this repo.
- Parsing differences (e.g. Mistral vs Llama) will be handled later.
"""

from __future__ import annotations

from typing import Dict


BASE_MODELS: Dict[str, Dict[str, str]] = {
    "llama3": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct"},
    "llama2": {"model_id": "meta-llama/Llama-2-7b-chat-hf"},
    "mistral": {"model_id": "mistralai/Mistral-7B-Instruct-v0.2"},
}
