from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    delay: float
    bandwidth: float


@dataclass(frozen=True)
class TopologySpec:
    topology_id: str
    benchmark: str
    num_rhs: int
    num_ess: int
    num_rcs: int
    edges: Tuple[EdgeSpec, ...]
    metadata: Dict[str, object] = field(default_factory=dict)

    def build_graph(self) -> nx.Graph:
        graph = nx.Graph()
        rh_nodes = [f"RH{i}" for i in range(self.num_rhs)]
        es_nodes = [f"ES{i}" for i in range(self.num_ess)]
        rc_nodes = [f"RC{i}" for i in range(self.num_rcs)]
        graph.add_nodes_from(rh_nodes + es_nodes + rc_nodes)

        for edge in self.edges:
            graph.add_edge(
                edge.source,
                edge.target,
                delay=float(edge.delay),
                bandwidth=float(edge.bandwidth),
                remaining_bandwidth=float(edge.bandwidth),
            )

        return graph


@dataclass(frozen=True)
class FamilyProfile:
    family: str
    difficulty: str
    rh_es_pattern: Tuple[int, ...]
    es_rc_pattern: Tuple[int, ...]
    direct_rc_stride: int
    direct_rc_offset: int
    rh_primary_bw: Tuple[float, float]
    rh_secondary_bw: Tuple[float, float]
    es_rc_primary_bw: Tuple[float, float]
    es_rc_secondary_bw: Tuple[float, float]
    rh_primary_delay: Tuple[float, float]
    rh_secondary_delay: Tuple[float, float]
    es_rc_primary_delay: Tuple[float, float]
    es_rc_secondary_delay: Tuple[float, float]
    direct_rc_delay: Tuple[float, float]
    direct_rc_bw: float
    description: str


def _edge(source: str, target: str, delay: float, bandwidth: float) -> EdgeSpec:
    return EdgeSpec(source=source, target=target, delay=round(float(delay), 4), bandwidth=round(float(bandwidth), 4))


FAMILY_PROFILES: Dict[str, FamilyProfile] = {
    "balanced": FamilyProfile(
        family="balanced",
        difficulty="moderate",
        rh_es_pattern=(0, 1),
        es_rc_pattern=(0, 1),
        direct_rc_stride=4,
        direct_rc_offset=1,
        rh_primary_bw=(24.0, 38.0),
        rh_secondary_bw=(18.0, 32.0),
        es_rc_primary_bw=(22.0, 36.0),
        es_rc_secondary_bw=(18.0, 32.0),
        rh_primary_delay=(0.3, 1.8),
        rh_secondary_delay=(0.8, 2.8),
        es_rc_primary_delay=(0.4, 1.9),
        es_rc_secondary_delay=(0.9, 2.8),
        direct_rc_delay=(0.08, 0.22),
        direct_rc_bw=160.0,
        description="Balanced project benchmark topology with moderate redundancy.",
    ),
    "clustered": FamilyProfile(
        family="clustered",
        difficulty="moderate",
        rh_es_pattern=(0, 0),
        es_rc_pattern=(0, 1),
        direct_rc_stride=4,
        direct_rc_offset=2,
        rh_primary_bw=(22.0, 34.0),
        rh_secondary_bw=(16.0, 28.0),
        es_rc_primary_bw=(20.0, 34.0),
        es_rc_secondary_bw=(16.0, 28.0),
        rh_primary_delay=(0.4, 1.9),
        rh_secondary_delay=(1.0, 3.0),
        es_rc_primary_delay=(0.5, 2.0),
        es_rc_secondary_delay=(1.0, 3.1),
        direct_rc_delay=(0.08, 0.23),
        direct_rc_bw=160.0,
        description="Clustered family with repeated RH-to-ES attachment patterns and shared bottlenecks.",
    ),
    "sparse_backhaul": FamilyProfile(
        family="sparse_backhaul",
        difficulty="hard",
        rh_es_pattern=(0, 1),
        es_rc_pattern=(0, 0),
        direct_rc_stride=4,
        direct_rc_offset=0,
        rh_primary_bw=(20.0, 30.0),
        rh_secondary_bw=(14.0, 24.0),
        es_rc_primary_bw=(12.0, 20.0),
        es_rc_secondary_bw=(10.0, 16.0),
        rh_primary_delay=(0.5, 2.2),
        rh_secondary_delay=(1.1, 3.2),
        es_rc_primary_delay=(0.8, 2.6),
        es_rc_secondary_delay=(1.4, 3.4),
        direct_rc_delay=(0.1, 0.24),
        direct_rc_bw=150.0,
        description="Harder project benchmark family with constrained ES-RC redundancy and lower backhaul capacity.",
    ),
    "direct_heavy": FamilyProfile(
        family="direct_heavy",
        difficulty="hard",
        rh_es_pattern=(0, 1),
        es_rc_pattern=(0, 1),
        direct_rc_stride=4,
        direct_rc_offset=3,
        rh_primary_bw=(18.0, 28.0),
        rh_secondary_bw=(12.0, 22.0),
        es_rc_primary_bw=(16.0, 24.0),
        es_rc_secondary_bw=(12.0, 20.0),
        rh_primary_delay=(0.4, 2.0),
        rh_secondary_delay=(1.0, 3.0),
        es_rc_primary_delay=(0.7, 2.5),
        es_rc_secondary_delay=(1.1, 3.1),
        direct_rc_delay=(0.08, 0.18),
        direct_rc_bw=170.0,
        description="Harder family with more policy pressure toward direct RH-RC routing and tighter ES/RC links.",
    ),
}


def _generate_topology_spec(
    *,
    topology_id: str,
    benchmark: str,
    num_rhs: int,
    num_ess: int,
    num_rcs: int,
    seed: int,
    family: str,
) -> TopologySpec:
    profile = FAMILY_PROFILES[family]
    rng = np.random.default_rng(seed)
    edges: List[EdgeSpec] = []

    num_direct_links = max(1, num_rhs // 4)
    direct_link_rhs: List[int] = []
    cursor = (seed + profile.direct_rc_offset) % num_rhs
    while len(direct_link_rhs) < num_direct_links:
        if cursor not in direct_link_rhs:
            direct_link_rhs.append(cursor)
        cursor = (cursor + profile.direct_rc_stride) % num_rhs

    for rh_idx in range(num_rhs):
        base_es = rh_idx % num_ess
        primary_es = (base_es + profile.rh_es_pattern[0]) % num_ess
        secondary_es = (base_es + profile.rh_es_pattern[1] + (seed % num_ess)) % num_ess
        if secondary_es == primary_es:
            secondary_es = (primary_es + 1) % num_ess

        edges.append(
            _edge(
                f"RH{rh_idx}",
                f"ES{primary_es}",
                delay=rng.uniform(*profile.rh_primary_delay),
                bandwidth=rng.uniform(*profile.rh_primary_bw),
            )
        )
        edges.append(
            _edge(
                f"RH{rh_idx}",
                f"ES{secondary_es}",
                delay=rng.uniform(*profile.rh_secondary_delay),
                bandwidth=rng.uniform(*profile.rh_secondary_bw),
            )
        )

        if rh_idx in direct_link_rhs:
            direct_rc = (rh_idx + seed + profile.direct_rc_offset) % num_rcs
            edges.append(
                _edge(
                    f"RH{rh_idx}",
                    f"RC{direct_rc}",
                    delay=rng.uniform(*profile.direct_rc_delay),
                    bandwidth=profile.direct_rc_bw,
                )
            )

    for es_idx in range(num_ess):
        base_rc = es_idx % num_rcs
        primary_rc = (base_rc + profile.es_rc_pattern[0]) % num_rcs
        secondary_rc = (base_rc + profile.es_rc_pattern[1] + (seed % num_rcs)) % num_rcs
        if secondary_rc == primary_rc:
            secondary_rc = (primary_rc + 1) % num_rcs

        edges.append(
            _edge(
                f"ES{es_idx}",
                f"RC{primary_rc}",
                delay=rng.uniform(*profile.es_rc_primary_delay),
                bandwidth=rng.uniform(*profile.es_rc_primary_bw),
            )
        )
        edges.append(
            _edge(
                f"ES{es_idx}",
                f"RC{secondary_rc}",
                delay=rng.uniform(*profile.es_rc_secondary_delay),
                bandwidth=rng.uniform(*profile.es_rc_secondary_bw),
            )
        )

    metadata = {
        "generator_seed": seed,
        "edge_count": len(edges),
        "family": profile.family,
        "difficulty": profile.difficulty,
        "description": profile.description,
        "benchmark_label": f"project_{benchmark}",
    }
    return TopologySpec(
        topology_id=topology_id,
        benchmark=benchmark,
        num_rhs=num_rhs,
        num_ess=num_ess,
        num_rcs=num_rcs,
        edges=tuple(edges),
        metadata=metadata,
    )


def build_default_topology_registry() -> Dict[str, TopologySpec]:
    specs = [
        _generate_topology_spec(
            topology_id="small_balanced_train_a",
            benchmark="small",
            num_rhs=8,
            num_ess=3,
            num_rcs=2,
            seed=101,
            family="balanced",
        ),
        _generate_topology_spec(
            topology_id="small_clustered_train_b",
            benchmark="small",
            num_rhs=8,
            num_ess=3,
            num_rcs=2,
            seed=202,
            family="clustered",
        ),
        _generate_topology_spec(
            topology_id="small_sparse_test_a",
            benchmark="small",
            num_rhs=8,
            num_ess=3,
            num_rcs=2,
            seed=303,
            family="sparse_backhaul",
        ),
        _generate_topology_spec(
            topology_id="small_direct_test_b",
            benchmark="small",
            num_rhs=8,
            num_ess=3,
            num_rcs=2,
            seed=404,
            family="direct_heavy",
        ),
        _generate_topology_spec(
            topology_id="large_balanced_train_a",
            benchmark="large",
            num_rhs=16,
            num_ess=5,
            num_rcs=3,
            seed=505,
            family="balanced",
        ),
        _generate_topology_spec(
            topology_id="large_clustered_train_b",
            benchmark="large",
            num_rhs=16,
            num_ess=5,
            num_rcs=3,
            seed=606,
            family="clustered",
        ),
        _generate_topology_spec(
            topology_id="large_sparse_test_a",
            benchmark="large",
            num_rhs=16,
            num_ess=5,
            num_rcs=3,
            seed=707,
            family="sparse_backhaul",
        ),
        _generate_topology_spec(
            topology_id="large_direct_test_b",
            benchmark="large",
            num_rhs=16,
            num_ess=5,
            num_rcs=3,
            seed=808,
            family="direct_heavy",
        ),
    ]
    return {spec.topology_id: spec for spec in specs}


DEFAULT_TOPOLOGY_REGISTRY = build_default_topology_registry()


DEFAULT_BENCHMARK_TOPOLOGY_POOLS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "small": {
        "train": ("small_balanced_train_a", "small_clustered_train_b"),
        "test": ("small_sparse_test_a", "small_direct_test_b"),
    },
    "large": {
        "train": ("large_balanced_train_a", "large_clustered_train_b"),
        "test": ("large_sparse_test_a", "large_direct_test_b"),
    },
}


def get_benchmark_dimensions(benchmark: str) -> Tuple[int, int, int]:
    pool = DEFAULT_BENCHMARK_TOPOLOGY_POOLS[benchmark]["train"]
    spec = DEFAULT_TOPOLOGY_REGISTRY[pool[0]]
    return spec.num_rhs, spec.num_ess, spec.num_rcs


def get_topology_spec(topology_id: str) -> TopologySpec:
    return DEFAULT_TOPOLOGY_REGISTRY[topology_id]


def iter_topology_specs(topology_ids: Iterable[str]) -> List[TopologySpec]:
    return [DEFAULT_TOPOLOGY_REGISTRY[topology_id] for topology_id in topology_ids]
