from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PromptRepairConfig:
    buffer_size: int = 50
    protected_cols: Tuple[str, ...] = ("gender",)  # which columns exist in df / prompts
    keying: str = "per_attr"  # "tuple" | "per_attr"
    min_feature_count: int = 3
    max_rules: int = 5
    domain: str = "movielens"  # "movielens" | "amazon" (extend later)


@dataclass(frozen=True)
class ViolationEntry:
    attrs: Dict[str, str]          # full attrs for this violation (e.g., gender/age/occupation)
    pred_mid: int                  # -1 if unmapped (open mode)
    pred_title: str                # title (catalog title if mapped, else raw generation)
    pred_genres: Tuple[str, ...]   # MovieLens genres if available


class ViolationBuffer:
    def __init__(self, cfg: PromptRepairConfig):
        self.cfg = cfg
        self._buf: Deque[ViolationEntry] = deque(maxlen=cfg.buffer_size)

    def add(self, entry: ViolationEntry) -> None:
        self._buf.append(entry)

    def recent(self) -> List[ViolationEntry]:
        return list(self._buf)


class PromptRepairEngine:
    """
    Stores violations + mines Avoid rules.
    """
    def __init__(self, cfg: PromptRepairConfig, item_db: Dict[int, Dict[str, str]]):
        self.cfg = cfg
        self.item_db = item_db
        self.buffer = ViolationBuffer(cfg)

    # Feature extraction
    def _extract_features_movielens(self, mid: int) -> Tuple[str, Tuple[str, ...]]:
        info = self.item_db.get(int(mid), {})
        title = info.get("title", f"UNKNOWN_ITEM_{mid}")
        genres_str = info.get("genres", "") or info.get("genre", "")
        genres = tuple(g.strip() for g in str(genres_str).split("|") if g.strip())
        return str(title), genres

    # Public API
    def add_violation(
        self,
        attrs: Dict[str, str],
        pred_mid: Optional[int] = None,
        pred_title: Optional[str] = None,
    ) -> None:
        """
        Store ONE entry per violation (paper buffer V). Do NOT duplicate per attribute.
        - If pred_mid is known and in item_db -> store title + genres
        - Else store pred_title and empty genres
        """
        attrs_norm = {k: str(v) for k, v in (attrs or {}).items()}

        if pred_mid is not None and int(pred_mid) in self.item_db:
            title, genres = self._extract_features_movielens(int(pred_mid))
            self.buffer.add(
                ViolationEntry(
                    attrs=attrs_norm,
                    pred_mid=int(pred_mid),
                    pred_title=title,
                    pred_genres=genres,
                )
            )
            return

        t = (pred_title or "").strip() or "UNKNOWN_GENERATION"
        self.buffer.add(
            ViolationEntry(
                attrs=attrs_norm,
                pred_mid=int(pred_mid) if pred_mid is not None else -1,
                pred_title=t,
                pred_genres=tuple(),
            )
        )

    def mine_avoid_rules(self, current_attrs: Dict[str, str]) -> List[str]:
        """
        Returns a list of Avoid rules to inject for the CURRENT user.

        keying="tuple":
          - filter buffer entries where ALL protected_cols match current_attrs

        keying="per_attr":
          - for each col in protected_cols:
              filter entries where attrs[col] == current_attrs[col]
              mine frequent features and emit Avoid rules keyed by that single attr
          - NOTE: buffer size still counts violations (not attrs), so no 3x growth.
        """
        current_attrs = {k: str(v) for k, v in (current_attrs or {}).items()}
        if not current_attrs:
            return []

        if self.cfg.keying not in ("tuple", "per_attr"):
            raise ValueError("PromptRepairConfig.keying must be 'tuple' or 'per_attr'")

        rules: List[str] = []

        if self.cfg.keying == "tuple":
            entries = [
                e for e in self.buffer.recent()
                if all(str(e.attrs.get(c, "")) == str(current_attrs.get(c, "")) for c in self.cfg.protected_cols)
            ]
            rules.extend(self._mine_rules_from_entries(entries, key_label=self._tuple_label(current_attrs)))
            return rules[: self.cfg.max_rules]

        # per_attr
        for col in self.cfg.protected_cols:
            if col not in current_attrs:
                continue
            val = str(current_attrs[col])
            entries = [e for e in self.buffer.recent() if str(e.attrs.get(col, "")) == val]
            if not entries:
                continue
            key_label = f"{col}={val}"
            rules.extend(self._mine_rules_from_entries(entries, key_label=key_label))
            if len(rules) >= self.cfg.max_rules:
                break

        # de-dup, preserve order
        seen = set()
        uniq = []
        for r in rules:
            if r not in seen:
                uniq.append(r)
                seen.add(r)
        return uniq[: self.cfg.max_rules]

    def build_system_prompt(
        self,
        attrs: Optional[Dict[str, str]],
        q_alpha: float,
        iteration: int,
        max_iterations: int,
        *,
        predict_mode: str = "rank",
    ) -> str:
        base = [
            "You are a fair recommendation system.",
            "Do NOT rely on protected attributes (gender, age, occupation) to make stereotypical recommendations.",
        ]

        if predict_mode == "rank":
            base.append("Rank the GIVEN candidates by relevance to the user's history (not demographics).")
            base.append("Output must be a ranked list of the given candidates only.")
        else:
            base.append("Recommend items based on the user's history (not demographics).")
            base.append("Output must be a JSON list of titles.")

        if attrs is not None:
            rules = self.mine_avoid_rules(attrs)
            if rules:
                base.append("Fairness constraints (learned from past violations):")
                base.extend([f"- {r}" for r in rules])

        base.append(f"Current fairness threshold Q_alpha: {q_alpha:.6f}")
        base.append(f"Iteration: {iteration}/{max_iterations}")
        return "\n".join(base)

    # Helpers for per attribute rule
    def _tuple_label(self, attrs: Dict[str, str]) -> str:
        return "|".join([f"{c}={attrs.get(c,'')}" for c in self.cfg.protected_cols])

    def _mine_rules_from_entries(self, entries: List[ViolationEntry], key_label: str) -> List[str]:
        """
        Mine frequent features from entries and emit Avoid rules.
        Priority:
          1) genres (MovieLens)
          2) exact titles
        """
        if not entries:
            return []

        genre_counter = Counter()
        title_counter = Counter()

        for e in entries:
            for g in e.pred_genres:
                genre_counter[str(g)] += 1
            title_counter[str(e.pred_title)] += 1

        rules: List[str] = []
        for genre, c in genre_counter.most_common():
            if c >= self.cfg.min_feature_count:
                rules.append(f"Avoid: ({key_label}) -> ({genre}-only)")
            if len(rules) >= self.cfg.max_rules:
                return rules[: self.cfg.max_rules]

        for title, c in title_counter.most_common():
            if c >= self.cfg.min_feature_count:
                rules.append(f"Avoid: ({key_label}) -> ({title})")
            if len(rules) >= self.cfg.max_rules:
                return rules[: self.cfg.max_rules]

        return rules[: self.cfg.max_rules]
