"""Map open-ended LLM outputs (free-text titles) to catalogue items.

This module provides a small utility to map arbitrary title strings (e.g., LLM
outputs) to the nearest catalogue title using cosine similarity in an embedding
space.

The main entry point is :class:`CatalogueMapper`, which builds an embedding
index over the catalogue and then maps one title
(:meth:`CatalogueMapper.map_one`) or a list of titles
(:meth:`CatalogueMapper.map_list`).

The code assumes the embedder returns *normalized* vectors so that cosine
similarity can be computed via a dot product.

Paper context:
    The repository uses embedding-based mapping to convert generated text back
    into catalogue items for evaluation in open-generation settings.
    TODO(doc): Add a paper citation if/when you identify the specific section
    describing this mapping/evaluation step.
"""
from __future__ import annotations

from ast import pattern
import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_title(s: str) -> str:
    """Normalize a title string for mapping.

    The normalization performed is minimal:
    1) strip surrounding whitespace,
    2) collapse internal whitespace to single spaces.

    Args:
        s (str): Input title.

    Returns:
        str: Normalized title.
    """
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def rewrite_prompt_attrs(prompt: str, new_attrs: Dict[str, str]) -> str:
    """Rewrite a prompt's user-profile block with updated protected attributes.

    The function searches for lines of the form ``- <key>: <value>`` and
    replaces the value for each key in ``new_attrs``. If no such lines are
    found, it prepends a new "User profile (audit only)" block.

    Args:
        prompt (str): Prompt text to modify.
        new_attrs (Dict[str, str]): Mapping from attribute key to replacement
            value.

    Returns:
        str: Updated prompt text.
    """
    if not prompt:
        return prompt

    # If block exists, replace those lines
    out = prompt
    replaced_any = False
    for k, v in new_attrs.items():
        pattern = rf"(\-\s*{re.escape(k)}\s*:\s*)(.*)"
        if re.search(pattern, out, flags=re.IGNORECASE):
            out = re.sub(pattern, lambda m: m.group(1) + v, out, flags=re.IGNORECASE)
            replaced_any = True

    if replaced_any:
        return out

    # Otherwise prepend a new block
    profile = "User profile (audit only):\n" + "\n".join([f"- {k}: {v}" for k, v in new_attrs.items()]) + "\n\n"
    return profile + out


@dataclass
class MapResult:
    """Store catalogue-mapping outputs for a top-k list.

    Attributes:
        mapped_titles (List[str]): Canonical catalogue titles per position.
            Invalid or duplicate-disallowed positions are set to ``""``.
        mapped_mids (List[Optional[str]]): Catalogue item ids per position.
            Invalid or duplicate-disallowed positions are set to ``None``.
        sims (List[float]): Similarity scores per position (one per input
            position, even if invalid).
        valid_at_k (float): Fraction of the first ``k`` positions that mapped
            to a non-empty canonical title (and, when ``allow_duplicates`` is
            false, are not duplicates).
    """
    mapped_titles: List[str]
    mapped_mids: List[Optional[str]]
    sims: List[float]
    valid_at_k: float


class CatalogueMapper:
    """Map free-text titles to the nearest catalogue title in embedding space.

    The mapper builds an embedding index over catalogue titles via
    :meth:`build`, then provides:
    - :meth:`map_one` for mapping a single predicted title, and
    - :meth:`map_list` for mapping the first ``k`` positions of a predicted
      title list and computing ``valid_at_k``.

    TODO(doc): Clarify the expected embedder protocol (methods, return shapes)
    and where it is defined in the codebase.

    Attributes:
        embedder: Text embedder used to encode titles.
        item_db (Dict[str, Dict]): Catalogue mapping keyed by item id.
        title_key (str): Key in ``item_db`` used to read the catalogue title.
    """

    def __init__(
        self,
        embedder,  # TextEmbedder instance
        item_db: Dict[str, Dict],
        title_key: str = "title",
    ):
        """Initialize the catalogue mapper.

        Args:
            embedder: Embedder used for encoding titles.
            item_db (Dict[str, Dict]): Catalogue mapping keyed by item id.
            title_key (str): Key used to extract a title from each catalogue
                record.
        """
        self.embedder = embedder
        self.item_db = item_db
        self.title_key = title_key

        self._catalog_mids: List[str] = []
        self._catalog_titles: List[str] = []
        self._catalog_embeds: Optional[np.ndarray] = None

    @property
    def catalog_titles(self) -> List[str]:
        """Return the normalized catalogue titles used for mapping."""
        return self._catalog_titles

    @property
    def catalog_mids(self) -> List[str]:
        """Return the catalogue item ids aligned with :attr:`catalog_titles`."""
        return self._catalog_mids

    def build(self, dedup: bool = True) -> None:
        """Build the embedding index over catalogue titles.

        The function extracts titles from ``item_db`` using ``title_key``,
        normalizes them with :func:`_normalize_title`, optionally de-duplicates
        by title (keeping the first id), then encodes all titles via the
        embedder.

        Args:
            dedup (bool): Whether to de-duplicate catalogue titles.

        Returns:
            None
        """
        mids = []
        titles = []
        for mid, info in self.item_db.items():
            t = str(info.get(self.title_key, "")).strip()
            if not t:
                continue
            mids.append(str(mid))
            titles.append(_normalize_title(t))

        if dedup:
            # De-dup by title but keep the first mid
            seen = {}
            for mid, t in zip(mids, titles):
                if t not in seen:
                    seen[t] = mid
            titles = list(seen.keys())
            mids = [seen[t] for t in titles]

        self._catalog_mids = mids
        self._catalog_titles = titles

        logger.info(f"Building catalog embeddings for {len(self._catalog_titles)} items...")
        # Use TextEmbedder.encode_texts which returns normalized numpy array [N, D]
        self._catalog_embeds = self.embedder.encode_texts(self._catalog_titles)

    def map_one(self, title: str, min_sim: float = 0.65) -> Tuple[Optional[str], Optional[str], float]:
        """Map a single free-text title to its nearest catalogue neighbor.

        The method normalizes ``title``, embeds it with the configured embedder,
        computes cosine similarity against the precomputed catalogue embeddings,
        and returns the best match if its similarity is at least ``min_sim``.

        Args:
            title (str): Free-text title to map.
            min_sim (float): Minimum similarity required for a mapping to be
                considered valid.

        Returns:
            Tuple[Optional[str], Optional[str], float]: A tuple
            ``(mid, canonical_title, sim)``. If no mapping meets ``min_sim``,
            returns ``(None, None, sim)`` where ``sim`` is the best similarity
            found.

        Raises:
            RuntimeError: If :meth:`build` has not been called.
        """
        if self._catalog_embeds is None:
            raise RuntimeError("CatalogueMapper.build() must be called before mapping.")

        title = _normalize_title(title)
        if not title:
            return None, None, 0.0

        # Use TextEmbedder.encode_text which returns normalized numpy array [D]
        q = self.embedder.encode_text(title)
        
        # Compute cosine similarity with all catalog embeddings
        # Both q and catalog_embeds are already normalized, so dot product = cosine similarity
        sims = np.dot(self._catalog_embeds, q)  # [N]
        j = int(np.argmax(sims))
        sim = float(sims[j])

        if sim < min_sim:
            return None, None, sim
        return self._catalog_mids[j], self._catalog_titles[j], sim

    def map_list(
        self,
        titles: List[str],
        k: int,
        min_sim: float = 0.65,
        allow_duplicates: bool = False,
    ) -> MapResult:
        """Map the first ``k`` predicted titles to catalogue items.

        The method calls :meth:`map_one` independently for each position
        ``0..k-1``. If a position is invalid (below ``min_sim``) or duplicates
        a previously accepted canonical title (when ``allow_duplicates`` is
        false), that position is recorded as empty.

        Args:
            titles (List[str]): Predicted titles. If shorter than ``k``, missing
                positions are treated as empty strings.
            k (int): Number of positions to map.
            min_sim (float): Minimum similarity required for a mapping to be
                considered valid.
            allow_duplicates (bool): Whether to allow the same canonical title
                to appear multiple times in the mapped list.

        Returns:
            MapResult: Mapping outputs including ``valid_at_k``.

        Raises:
            RuntimeError: If :meth:`build` has not been called.
        """
        mapped_titles: List[str] = []
        mapped_mids: List[Optional[str]] = []
        sims_out: List[float] = []

        seen_titles = set()
        valid = 0

        for i in range(k):
            pred = titles[i] if i < len(titles) else ""
            mid, tcanon, sim = self.map_one(pred, min_sim=min_sim)
            sims_out.append(sim)

            if tcanon is None:
                mapped_titles.append("")
                mapped_mids.append(None)
            else:
                if (not allow_duplicates) and (tcanon in seen_titles):
                    mapped_titles.append("")
                    mapped_mids.append(None)
                else:
                    mapped_titles.append(tcanon)
                    mapped_mids.append(mid)
                    seen_titles.add(tcanon)
                    valid += 1

        valid_at_k = float(valid) / float(k) if k > 0 else 0.0
        return MapResult(mapped_titles=mapped_titles, mapped_mids=mapped_mids, sims=sims_out, valid_at_k=valid_at_k)
