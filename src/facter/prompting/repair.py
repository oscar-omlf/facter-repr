"""Mine prompt repair rules from a rolling buffer of threshold violations.

This module implements a small in-memory buffer for storing "violation" events
and an engine for deriving "Avoid" rules that can be injected into system
prompts.
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from facter.data.prompts import AGE_ID2LABEL, OCC_ID2LABEL


@dataclass(frozen=True)
class PromptRepairConfig:
    """Configure prompt-repair rule mining.

    Attributes:
        buffer_size (int): Maximum number of recent violations to retain.
        protected_cols (Tuple[str, ...]): Protected attribute columns expected
            in input attribute dictionaries.
        keying (str): Rule keying strategy. Supported values are ``"tuple"``
            (all protected columns must match) and ``"per_attr"`` (match each
            protected column independently).
        min_feature_count (int): Minimum count required for a feature (genre or
            title) to be turned into an Avoid rule.
        max_rules (int): Maximum number of rules to mine per key.
        domain (str): Domain identifier used to select feature extraction
            behavior. Supported values are ``"movielens"`` and ``"amazon"``.
    """
    buffer_size: int = 50
    protected_cols: Tuple[str, ...] = ("gender",)  # which columns exist in df / prompts
    keying: str = "per_attr"  # "tuple" | "per_attr"
    min_feature_count: int = 3
    max_rules: int = 5
    domain: str = "movielens"  # "movielens" | "amazon" (extend later)


@dataclass(frozen=True)
class ViolationEntry:
    """Represent a single observed violation.

    Attributes:
        attrs (Dict[str, str]): Protected attributes for the violated example.
        pred_mid (int): Predicted item id if mapped; ``-1`` if not mapped.
        pred_title (str): Predicted title. This may be a catalogue title when
            mapped, or a raw generated string.
        pred_genres (Tuple[str, ...]): Parsed genre strings, if available.
    """
    attrs: Dict[str, str]          # full attrs for this violation (e.g., gender/age/occupation)
    pred_mid: int                  # -1 if unmapped (open mode)
    pred_title: str                # title (catalog title if mapped, else raw generation)
    pred_genres: Tuple[str, ...]   # MovieLens genres if available


class ViolationBuffer:
    def __init__(self, cfg: PromptRepairConfig):
        """Initialize the rolling violation buffer.

        Args:
            cfg (PromptRepairConfig): Buffer configuration.
        """
        self.cfg = cfg
        self._buf: Deque[ViolationEntry] = deque(maxlen=cfg.buffer_size)

    def add(self, entry: ViolationEntry) -> None:
        """Append a new violation entry to the buffer.

        Args:
            entry (ViolationEntry): Violation event to store.
        """
        self._buf.append(entry)

    def recent(self) -> List[ViolationEntry]:
        """Return the most recent violations currently stored.

        Returns:
            List[ViolationEntry]: Violations in buffer order.
        """
        return list(self._buf)


class PromptRepairEngine:
    """Store violations and mine "Avoid" rules for prompt injection.

    The engine maintains a :class:`ViolationBuffer` and can derive simple rules
    from frequent features seen in recent violations.
    """

    def __init__(self, cfg: PromptRepairConfig, item_db: Dict[int, Dict[str, str]]):
        """Initialize the prompt repair engine.

        Args:
            cfg (PromptRepairConfig): Rule mining configuration.
            item_db (Dict[int, Dict[str, str]]): Item metadata database. For
                MovieLens-style data, entries may contain ``"title"`` and
                ``"genres"`` (pipe-separated) keys.
        """
        self.cfg = cfg
        self.item_db = item_db
        self.buffer = ViolationBuffer(cfg)

    # Feature extraction
    def _extract_features_movielens(self, mid: int) -> Tuple[str, Tuple[str, ...]]:
        """Extract movie title and genres from the item database.

        Args:
            mid (int): Item id.

        Returns:
            Tuple[str, Tuple[str, ...]]: A pair ``(title, genres)`` where
            ``genres`` is a tuple of non-empty genre strings.
        """
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
        """Store one violation entry in the rolling buffer.

        This method stores a single buffer entry per violation (i.e., it does
        not duplicate entries per protected attribute).

        If ``pred_mid`` is provided and exists in ``item_db``, the stored entry
        includes title and genres extracted from the database; otherwise the
        entry stores ``pred_title`` and an empty genre tuple.

        Args:
            attrs (Dict[str, str]): Protected attributes for the violated
                example.
            pred_mid (Optional[int]): Predicted item id if available.
            pred_title (Optional[str]): Predicted title string if available.
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
        """Mine Avoid rules for the current user's protected attributes.

        The engine supports two keying modes:

        - ``keying="tuple"``: filter violation entries where all
          ``cfg.protected_cols`` match ``current_attrs``.
        - ``keying="per_attr"``: for each protected column, filter entries that
          match the value of that column and mine rules keyed by the single
          attribute.

        Args:
            current_attrs (Dict[str, str]): Protected attributes of the user
                being served.

        Returns:
            List[str]: A list of Avoid rules.

        Raises:
            ValueError: If ``cfg.keying`` is not one of the supported values.
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

            if col == "age":
                try:
                    mapped_val = AGE_ID2LABEL.get(int(val), val)
                except Exception:
                    mapped_val = val
            elif col == "occupation":
                try:
                    mapped_val = OCC_ID2LABEL.get(int(val), val)
                except Exception:
                    mapped_val = val
            else:
                mapped_val = val

            key_label = f"{col}={mapped_val}"
            col_rules = self._mine_rules_from_entries(entries, key_label=key_label)
            rules.extend(col_rules)

        # de-dup, preserve order
        seen = set()
        uniq = []
        for r in rules:
            if r not in seen:
                uniq.append(r)
                seen.add(r)
        return uniq[: self.cfg.max_rules * len(self.cfg.protected_cols)]

    def build_system_prompt(
        self,
        attrs: Optional[Dict[str, str]],
        q_alpha: float,
        iteration: int,
        max_iterations: int,
        ) -> str:
        """Build a system prompt with optional mined constraints.

        Args:
            attrs (Optional[Dict[str, str]]): Protected attributes of the
                current user. When provided, the engine mines Avoid rules from
                the buffer.
            q_alpha (float): Threshold value inserted into the system prompt.
            iteration (int): Current iteration number, included for logging.
            max_iterations (int): Total planned iterations, included for
                logging.

        Returns:
            str: System prompt string.
        """
        base = [
            "You are a fair recommendation system.",
            "Rules:",
            "1) Recommend based on user preference signals in the watch history (genres, themes, creators), not on demographics.",
            "2) Do NOT reinforce stereotypes or demographic-based assumptions.",
        ]

        if attrs is not None:
            rules = self.mine_avoid_rules(attrs)
            if rules:
                base.append("Fairness constraints (learned from past violations):")
                base.extend([f"- {r}" for r in rules])

        base.append(
                f"Fairness target: keep nonconformity S <= {q_alpha:.6f}."
            )
        base.append(f"Iteration: {iteration}/{max_iterations}")
        return "\n".join(base)

    # Helpers for per attribute rule
    def _tuple_label(self, attrs: Dict[str, str]) -> str:
        """Create a key label by concatenating protected attribute assignments.

        Args:
            attrs (Dict[str, str]): Protected attributes.

        Returns:
            str: Label string of the form ``"col=value|col=value|..."``.
        """
        return "|".join([f"{c}={attrs.get(c,'')}" for c in self.cfg.protected_cols])

    def _mine_rules_from_entries(self, entries: List[ViolationEntry], key_label: str) -> List[str]:
        """Mine feature-based Avoid rules from a filtered set of violations.

        The implementation prioritizes rules derived from genres first, then
        falls back to rules derived from exact titles.

        Args:
            entries (List[ViolationEntry]): Violations to mine from.
            key_label (str): Key label inserted into emitted rules.

        Returns:
            List[str]: Avoid rules derived from frequent genres/titles.
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
