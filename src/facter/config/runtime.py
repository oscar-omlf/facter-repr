from dataclasses import dataclass
from typing import Literal

TaskMode = Literal["open", "rank"]

@dataclass(frozen=True)
class RuntimeConfig:
    # TODO: we can add other parameters we hardcoded so far here, or in arg.flags, or both (?)
    task_mode: TaskMode = "open"   # this is the one in the paper
    top_k: int = 10
    min_sim_catalog: float = 0.65  # authors' default
