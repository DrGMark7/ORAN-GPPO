from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .topologies import DEFAULT_BENCHMARK_TOPOLOGY_POOLS, DEFAULT_TOPOLOGY_REGISTRY, TopologySpec


@dataclass
class TopologySelection:
    topology_id: str
    topology_spec: TopologySpec
    pool_name: str
    selection_mode: str


class TopologyPool:
    def __init__(
        self,
        benchmark: str,
        pool_name: str = "train",
        topology_ids: Optional[Sequence[str]] = None,
        registry: Optional[Dict[str, TopologySpec]] = None,
    ):
        self.benchmark = benchmark
        self.pool_name = pool_name
        self.registry = registry or DEFAULT_TOPOLOGY_REGISTRY
        ids = topology_ids or DEFAULT_BENCHMARK_TOPOLOGY_POOLS[benchmark][pool_name]
        if not ids:
            raise ValueError(f"No topology IDs configured for benchmark={benchmark}, pool={pool_name}")
        self.topology_ids = tuple(ids)
        self._validate_group()

    def _validate_group(self) -> None:
        specs = [self.registry[topology_id] for topology_id in self.topology_ids]
        first = specs[0]
        expected = (first.num_rhs, first.num_ess, first.num_rcs, first.benchmark)
        for spec in specs[1:]:
            current = (spec.num_rhs, spec.num_ess, spec.num_rcs, spec.benchmark)
            if current != expected:
                raise ValueError(
                    "Topology pools must keep a stable benchmark shape. "
                    f"Expected {expected}, got {current} for topology_id={spec.topology_id}"
                )

    def select(
        self,
        *,
        selection_mode: str,
        rng: Optional[np.random.Generator] = None,
        requested_topology_id: Optional[str] = None,
    ) -> TopologySelection:
        if requested_topology_id is not None:
            if requested_topology_id not in self.topology_ids:
                raise ValueError(f"Topology {requested_topology_id} is not in pool {self.pool_name}")
            topology_id = requested_topology_id
        elif selection_mode == "fixed":
            topology_id = self.topology_ids[0]
        elif selection_mode == "random_per_reset":
            if rng is None:
                rng = np.random.default_rng()
            topology_id = self.topology_ids[int(rng.integers(0, len(self.topology_ids)))]
        else:
            raise ValueError(f"Unsupported topology selection mode: {selection_mode}")

        return TopologySelection(
            topology_id=topology_id,
            topology_spec=self.registry[topology_id],
            pool_name=self.pool_name,
            selection_mode=selection_mode,
        )
