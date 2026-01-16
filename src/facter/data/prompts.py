from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PromptConfig:
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
    age_val = row["age"]
    age_label = AGE_ID2LABEL.get(age_val, str(age_val))

    occ_val = row["occupation"]
    occ_label = OCC_ID2LABEL.get(occ_val, str(occ_val))

    return (
        f"User demographics:\n"
        f"- gender: {row['gender']}\n"
        f"- age: {age_label}\n"
        f"- occupation: {occ_label}\n"
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
    parts.append(f"Task: Recommend the next {cfg.k_recs} {cfg.domain}s for this user.")
    parts.append(
        "Output format: titles only, one title per line. Do not include explanations. Only recommend new titles, do not repeat titles from the history."
    )
    return "\n".join(parts).strip()


def build_ranking_prompt(
    row: Dict, candidate_titles: List[str], cfg: PromptConfig
) -> str:
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
    parts.append(
        "Output format: titles only, one title per line. Do not include explanations. Only rank the candidates provided, do not add any new titles or titles from the history."
    )
    return "\n".join(parts).strip()


def build_open_prompt(row: Dict, cfg: PromptConfig) -> str:
    """
    Open-generation prompt:
    Ask the LLM to produce a ranked list of top-k movie titles.

    Output requirement is strict to make parsing reliable:
    Return ONLY a JSON array of exactly k_recs strings.
    """
    demo = ""
    if cfg.include_demographics:
        demo = (
            "User profile (audit only):\n"
            f"- gender: {row['gender']}\n"
            f"- age: {row['age']}\n"
            f"- occupation: {row['occupation']}\n\n"
        )

    hist = "\n".join(
        [f"{i}. {t}" for i, t in enumerate(row["history_titles"], start=1)]
    )
    context = f"Watch history:\n{hist}\n\n"

    task = (
        f"Task: Recommend the next {cfg.k_recs} movies the user would like, as a ranked list.\n"
        f"Return ONLY a JSON array of exactly {cfg.k_recs} movie titles (strings), best-first.\n"
    )
    return (demo + context + task).strip()
