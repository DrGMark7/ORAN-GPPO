from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    benchmark: str = "small"
    train_selection_mode: str = "random_per_reset"
    eval_selection_mode: str = "fixed"
    train_topology_id: Optional[str] = None
    eval_topology_id: Optional[str] = None
    seed: int = 42
