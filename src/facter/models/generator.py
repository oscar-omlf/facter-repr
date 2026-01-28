"""Define the text generation interface used by FACTER components.

This module defines a small protocol that abstracts over a "generator" model.
Implementations are expected to take a batch of input prompts and return a
nested list of generated strings.
"""

from typing import List, Protocol, Sequence

class Generator(Protocol):
    """Specify a minimal interface for batched text generation.

    Implementations may wrap different backends (e.g., local LLMs, hosted
    inference APIs). This protocol defines only the method surface used by the
    rest of the codebase.
    """

    def generate_topk(self, prompts: Sequence[str], system_prompt: str) -> List[List[str]]:
        """Generate a list of candidate generations for each prompt.

        Args:
            prompts (Sequence[str]): Input prompts to generate from.
            system_prompt (str): System prompt or instruction prefix passed to
                the underlying generator backend.

        Returns:
            List[List[str]]: A nested list where each outer element corresponds
            to an input prompt, and each inner list contains that prompt's
            generated candidate strings.
        """
        ...
