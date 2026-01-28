"""Provide a registry of HuggingFace model identifiers used by this repo.

This module defines a mapping from short model nicknames ("base models") to
HuggingFace ``model_id`` strings.

The values are used to configure local Transformers backends such as
:class:`~facter.models.hf_ranker.HFChatRanker` and
:class:`~facter.models.hf_generator.HFOpenGenerator`.
"""

from __future__ import annotations

from typing import Dict


BASE_MODELS: Dict[str, Dict[str, str]] = {
    "llama3": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct"},
    "llama2": {"model_id": "meta-llama/Llama-2-7b-chat-hf"},
    "mistral": {"model_id": "mistralai/Mistral-7B-Instruct-v0.2"},
}

# Public constant:
#   BASE_MODELS maps a short name to a dict currently holding ``model_id``.