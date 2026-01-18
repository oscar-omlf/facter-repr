from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PromptRepairConfig:
    """
    Implements the paper's violation protocol:
    - FIFO buffer size M
    - match violations by same protected attribute value (one key)
    - if feature appears >=3 times, inject Avoid: (a)->feature-only
    """

    buffer_size: int = 50
    protected_key: str = (
        "gender"  # which protected attribute to filter by for pattern mining
    )
    min_feature_count: int = 3  # >=3 violations -> consistent pattern
    max_rules: int = 5  # limit injected avoid rules
    domain: str = "movielens"  # "movielens" | "amazon" (we implement movielens now)


@dataclass(frozen=True)
class ViolationEntry:
    a_value: str  # value for protected_key, e.g. "F"
    pred_mid: str  # predicted item id
    pred_title: str
    pred_genres: Tuple[str, ...]


class ViolationBuffer:
    def __init__(self, cfg: PromptRepairConfig):
        self.cfg = cfg
        self._buf: deque[ViolationEntry] = deque(maxlen=cfg.buffer_size)

    def add(self, entry: ViolationEntry) -> None:
        self._buf.append(entry)

    def recent(self) -> List[ViolationEntry]:
        return list(self._buf)

    def filtered_by_a(self, a_value: str) -> List[ViolationEntry]:
        return [e for e in self._buf if e.a_value == a_value]


class PromptRepairEngine:
    """
    Builds a system prompt I^(t) that injects “Avoid: (a) -> feature-only” rules.
    Paper Eq.(10) and protocol description in Section 3.3 stage 3.
    """

    def __init__(self, cfg: PromptRepairConfig, item_db: Dict[str, Dict[str, str]]):
        self.cfg = cfg
        self.item_db = item_db
        self.buffer = ViolationBuffer(cfg)

    def _extract_features(self, mid: str) -> Tuple[str, Tuple[str, ...]]:
        info = self.item_db.get(str(mid), {})
        title = info.get("title", f"UNKNOWN_ITEM_{mid}")
        genres_str = info.get("genres", "")
        genres = (
            tuple([g.strip() for g in genres_str.split("|") if g.strip()])
            if genres_str
            else tuple()
        )
        return title, genres

    def add_violation(
        self,
        protected_value: str,
        pred_mid: Optional[str] = None,
        pred_title: Optional[str] = None,
    ) -> None:
        """
        Record a violation in the FIFO buffer.

        Rank mode: supply pred_mid and we extract MovieLens title/genres from item_db.
        Open mode: if pred_mid is unavailable or unmapped, supply pred_title and we store
                title-only with empty genre features.
        """
        if pred_mid is not None and str(pred_mid) in self.item_db:
            title, genres = self._extract_features(str(pred_mid))
            self.buffer.add(
                ViolationEntry(
                    a_value=str(protected_value),
                    pred_mid=str(pred_mid),
                    pred_title=title,
                    pred_genres=genres,
                )
            )
            return

        t = (pred_title or "").strip()
        if not t:
            t = "UNKNOWN_GENERATION"
        self.buffer.add(
            ViolationEntry(
                a_value=str(protected_value),
                pred_mid=str(pred_mid) if pred_mid is not None else "-1",
                pred_title=t,
                pred_genres=tuple(),
            )
        )

    def mine_avoid_rules(self, a_value: str) -> List[str]:
        """
        Implements:
        - Filter buffer V for same protected attribute value a
        - Count metadata features (MovieLens: title and genre)
        - If any feature appears >=3 times, add Avoid rule
        """
        entries = self.buffer.filtered_by_a(a_value)
        if not entries:
            return []

        rules: List[str] = []
        # Count genre occurrences
        genre_counter = Counter()
        title_counter = Counter()
        for e in entries:
            for g in e.pred_genres:
                genre_counter[g] += 1
            title_counter[e.pred_title] += 1

        # Rule priority: genre-only first, then exact title
        for genre, c in genre_counter.most_common():
            if c >= self.cfg.min_feature_count:
                rules.append(
                    f"Avoid: ({a_value}) -> ({genre}-only)"
                )
            if len(rules) >= self.cfg.max_rules:
                return rules[: self.cfg.max_rules]

        for title, c in title_counter.most_common():
            if c >= self.cfg.min_feature_count:
                rules.append(
                    f"Avoid: ({a_value}) -> ({title})"
                )
            if len(rules) >= self.cfg.max_rules:
                return rules[: self.cfg.max_rules]

        return rules[: self.cfg.max_rules]

    def build_system_prompt(
        self,
        a_value: Optional[str],
        q_alpha: float,
        iteration: int,
        max_iterations: int,
    ) -> str:
        base = [
            "You are a fair recommendation system.",
            "Rank the candidates by relevance to the user's history, NOT demographics.",
            "Do not rely on protected attributes (gender, age, occupation) for stereotypes.",
        ]
        if a_value is not None:
            rules = self.mine_avoid_rules(a_value)
            if rules:
                base.append("Fairness constraints (learned from past violations):")
                base.extend([f"- {r}" for r in rules])

        base.append(f"Current fairness threshold Q_alpha: {q_alpha:.6f}")
        base.append(f"Iteration: {iteration}/{max_iterations}")
        base.append("Output must be a ranked list of the given candidates only.")
        return "\n".join(base)
