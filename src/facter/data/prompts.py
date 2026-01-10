from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PromptConfig:
    k_recs: int = 10
    include_demographics: bool = True
    domain: str = "movie"  # "movie" | "product"


def _render_demographics(row: Dict) -> str:
    return (
        f"User demographics:\n"
        f"- gender: {row['gender']}\n"
        f"- age: {row['age']}\n"
        f"- occupation: {row['occupation']}\n"
    )


def build_generation_prompt(row: Dict, cfg: PromptConfig) -> str:
    """
    Generation-style prompt: ask the LLM to produce top-k recommendations (titles).
    This supports Recall@10/NDCG@10 evaluation by comparing the returned list to the target.
    """
    parts: List[str] = []
    if cfg.include_demographics:
        parts.append(_render_demographics(row).strip())

    parts.append("History (most recent last):")
    for i, title in enumerate(row["history_titles"], start=1):
        parts.append(f"{i}. {title}")

    parts.append("")
    parts.append(
        f"Task: Recommend the next {cfg.k_recs} {cfg.domain}s for this user."
    )
    parts.append(
        "Output format: titles only, one title per line. Do not include explanations."
    )
    return "\n".join(parts).strip()


def build_ranking_prompt(row: Dict, candidate_titles: List[str], cfg: PromptConfig) -> str:
    """
    Ranking-style prompt: ask the LLM to rank a candidate set (used for controlled evaluation).
    """
    parts: List[str] = []
    if cfg.include_demographics:
        parts.append(_render_demographics(row).strip())

    parts.append("History (most recent last):")
    for i, title in enumerate(row["history_titles"], start=1):
        parts.append(f"{i}. {title}")

    parts.append("")
    parts.append(f"Candidates ({cfg.domain}s):")
    for i, title in enumerate(candidate_titles, start=1):
        parts.append(f"{i}. {title}")

    parts.append("")
    parts.append(
        f"Task: Rank the candidates from most likely to be the next preferred {cfg.domain} to least likely."
    )
    parts.append("Output format: a numbered list of the candidate titles in ranked order, one per line.")
    return "\n".join(parts).strip()
