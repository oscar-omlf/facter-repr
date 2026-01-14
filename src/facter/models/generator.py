from typing import List, Protocol, Sequence, Optional

class Generator(Protocol):
    def generate_topk(self, prompts: Sequence[str], system_prompt: str) -> List[List[str]]:
        ...
