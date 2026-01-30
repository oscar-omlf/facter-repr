"""Define ranking interfaces and lightweight ranking result containers.

This module specifies a protocol for ranker backends that order a provided
candidate set given a ranking prompt.
"""

from dataclasses import dataclass
from typing import List, Protocol, Sequence


class Ranker(Protocol):
    """Specify an interface for ranking a provided candidate set.

    Implementations return a permutation of candidate indices from best to
    worst.
    """

    def rank(
        self,
        prompt_rank: str,
        candidate_titles: Sequence[str],
        system_prompt: str | None = None,
    ) -> List[int]:
        """Rank candidates according to the provided prompt.

        Args:
            prompt_rank (str): Ranking prompt.
            candidate_titles (Sequence[str]): Candidate titles to rank.
            system_prompt (str | None): Optional system prompt passed to the
                underlying backend.

        Returns:
            List[int]: A permutation of indices ``0..len(candidate_titles)-1``
            ordered from best to worst.
        """
        ...


@dataclass(frozen=True)
class RankedResult:
    """Store a full ranking over a candidate set.

    Attributes:
        ranked_indices (List[int]): Candidate indices ordered from best to
            worst.
    """
    ranked_indices: List[int]

    def topk(self, k: int) -> List[int]:
        """Return the top-``k`` ranked candidate indices.

        Args:
            k (int): Number of top indices to return.

        Returns:
            List[int]: The first ``k`` indices of :attr:`ranked_indices`.
        """
        return self.ranked_indices[:k]
