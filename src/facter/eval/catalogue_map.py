from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import re

from facter.models.embedder import TextEmbedder

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

@dataclass
class MapResult:
    mapped_mids: List[Optional[int]]
    mapped_titles: List[str]
    sims: List[float]
    valid_at_k: float

class CatalogMapper:
    def __init__(self, embedder: TextEmbedder, item_db: Dict[int, Dict[str,str]]):
        self.embedder = embedder
        self.item_db = item_db
        self._titles: List[str] = []
        self._mids: List[int] = []
        self._E: Optional[np.ndarray] = None  # [N,D] normalized

    def build(self, dedup: bool = True) -> None:
        mids, titles = [], []
        seen = set()
        for mid, info in self.item_db.items():
            t = _norm(str(info.get("title","")))
            if not t: 
                continue
            if dedup and t in seen:
                continue
            seen.add(t)
            mids.append(int(mid))
            titles.append(t)
        self._mids, self._titles = mids, titles
        self._E = self.embedder.encode_texts(self._titles)  # should be cached and normalized (?)

    def map_one(self, title: str, min_sim: float) -> Tuple[Optional[int], str, float]:
        if self._E is None:
            raise RuntimeError("call build() first")
        q = self.embedder.encode_texts([_norm(title)])[0]
        sims = self._E @ q
        j = int(np.argmax(sims))
        sim = float(sims[j])
        if sim < min_sim:
            return None, "", sim
        return self._mids[j], self._titles[j], sim

    def map_list(self, preds: List[str], k: int, min_sim: float, allow_dupes: bool = False) -> MapResult:
        mapped_mids: List[Optional[int]] = []
        mapped_titles: List[str] = []
        sims_out: List[float] = []
        seen = set()
        valid = 0
        for i in range(k):
            s = preds[i] if i < len(preds) else ""
            mid, title, sim = self.map_one(s, min_sim=min_sim)
            sims_out.append(sim)
            if mid is None or (not allow_dupes and title in seen):
                mapped_mids.append(None)
                mapped_titles.append("")
            else:
                mapped_mids.append(mid)
                mapped_titles.append(title)
                seen.add(title)
                valid += 1
        return MapResult(mapped_mids, mapped_titles, sims_out, valid / k if k else 0.0)
