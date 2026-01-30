"""Build prompt strings for generation and ranking tasks.

This module contains small prompt templates used to construct LLM inputs from an
interaction row, optionally including demographic metadata.

"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PromptConfig:
    """Configure prompt rendering.

    Attributes:
        k_recs (int): Number of recommendations/titles to request in prompts.
        include_demographics (bool): Whether to include demographic metadata in
            the prompt.
        domain (str): Domain label used when describing candidates/items.
    """

    k_recs: int = 10
    include_demographics: bool = True
    domain: str = "movie"  # "movie" | "product"


# Mapping for age IDs to human-readable labels
AGE_ID2LABEL = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

# Mapping for occupation IDs to human-readable labels
OCC_ID2LABEL = {
    0: "not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


def _render_demographics(row: Dict) -> str:
    """Render a demographics block from a row dictionary.

    Args:
        row (Dict): Row-like mapping containing demographic keys such as
            ``gender``, ``age``, and ``occupation``.

    Returns:
        str: A formatted multi-line demographics string.
    """
    age_val_raw = row.get("age", "")
    try:
        age_val = int(age_val_raw)
    except Exception:
        age_val = age_val_raw
    age_label = AGE_ID2LABEL.get(age_val, str(age_val))

    occ_val_raw = row.get("occupation", "")
    try:
        occ_val = int(occ_val_raw)
    except Exception:
        occ_val = occ_val_raw
    occ_label = OCC_ID2LABEL.get(occ_val, str(occ_val))

    gender = str(row.get("gender", "")).strip()

    return f"User demographics:\n- gender: {gender}\n- age: {age_label}\n- occupation: {occ_label}\n"


def build_ranking_prompt(row: Dict, candidate_titles: List[str], cfg: PromptConfig) -> str:
    """Build a ranking-style prompt over a fixed candidate set.

    The prompt requests the model to rank the provided `candidate_titles` and
    return a JSON array of exactly `cfg.k_recs` titles.

    Args:
        row (Dict): Row-like mapping that must contain ``history_titles``.
        candidate_titles (List[str]): Candidate titles to rank.
        cfg (PromptConfig): Prompt configuration.

    Returns:
        str: Prompt string.
    """
    demo = ""
    if cfg.include_demographics:
        demo = _render_demographics(row) + "\n"

    hist = "\n".join([f"{i}. {t}" for i, t in enumerate(row["history_titles"], start=1)])
    context = f"Watch history:\n{hist}\n\n"

    cand = "\n".join([f"{i}. {t}" for i, t in enumerate(candidate_titles, start=1)])
    candidates_block = f"Candidates ({cfg.domain}s):\n{cand}\n\n"

    task = (
        f"Task: Rank the candidates from most likely to be the next preferred {cfg.domain} to least likely, as a ranked list.\n"
        f"Return ONLY a JSON array of exactly {cfg.k_recs} {cfg.domain} titles (strings), best-first.\n"
        f"Output format: titles only, do not include explanations. Only rank the candidates provided; do not add new titles or repeat titles from the history.\n"
    )
    return (demo + context + candidates_block + task).strip()


def build_open_prompt(row: Dict, cfg: PromptConfig) -> str:
    """Build an open-ended generation prompt.

    Asks the model to produce a ranked list of top-k titles.

    Output requirement is strict to make parsing reliable: return ONLY a JSON
    array of exactly `cfg.k_recs` strings.

    Args:
        row (Dict): Row-like mapping that must contain ``history_titles``.
        cfg (PromptConfig): Prompt configuration.

    Returns:
        str: Prompt string.
    """
    demo = ""
    if cfg.include_demographics:
        demo = _render_demographics(row) + "\n"

    hist = "\n".join([f"{i}. {t}" for i, t in enumerate(row["history_titles"], start=1)])
    context = f"Watch history:\n{hist}\n\n"

    task = (
        f"Task: Recommend the next {cfg.k_recs} movies the user would like, as a ranked list.\n"
        f"Return ONLY a JSON array of exactly {cfg.k_recs} movie titles (strings), best-first.\n"
        f"Output format: titles only, do not include explanations. Only recommend new titles, do not repeat titles from the history.\n"
    )
    return (demo + context + task).strip()
