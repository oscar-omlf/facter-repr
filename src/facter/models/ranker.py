from dataclasses import dataclass
from typing import List, Protocol, Sequence


class Ranker(Protocol):
    """
    Ranking-based interface.

    rank() returns a permutation of candidate indices (0..len(candidates)-1) from best to worst.
    """
    def rank(
        self,
        prompt_rank: str,
        candidate_titles: Sequence[str],
        system_prompt: str | None = None,
    ) -> List[int]:
        ...


@dataclass(frozen=True)
class RankedResult:
    ranked_indices: List[int]

    def topk(self, k: int) -> List[int]:
        return self.ranked_indices[:k]
