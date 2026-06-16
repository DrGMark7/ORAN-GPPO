from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from .topologies import DEFAULT_TOPOLOGY_REGISTRY, TopologySpec
from .topology_pool import TopologyPool


class SimplifiedORANEnv(gym.Env):
    """Paper-aligned GPPO baseline environment with explicit per-time-slot decisions."""

    metadata = {"render_modes": ["human"]}
    EXACT_FEASIBILITY_MAX_RHS = 10
    BOUNDED_FEASIBILITY_RHS = 8
    BOUNDED_FEASIBILITY_VISIT_LIMIT = 50_000

    def __init__(
        self,
        num_rhs: Optional[int] = None,
        num_ess: Optional[int] = None,
        num_rcs: Optional[int] = None,
        max_steps: int = 100,
        benchmark: str = "small",
        topology_pool_name: str = "train",
        topology_selection_mode: str = "random_per_reset",
        constraint_mode: str = "legacy",
        topology_id: Optional[str] = None,
        topology_ids: Optional[Tuple[str, ...]] = None,
        topology_registry: Optional[Dict[str, TopologySpec]] = None,
    ):
        self.max_steps = max_steps
        self.current_step = 0
        self.invalid_streak = 0

        self.benchmark = benchmark
        self.topology_pool_name = topology_pool_name
        self.topology_selection_mode = topology_selection_mode
        valid_constraint_modes = {
            "legacy",
            "strict",
            "strict_connectivity_only",
            "strict_connectivity_plus_capacity",
            "strict_connectivity_plus_capacity_plus_bandwidth",
            "strict_full",
        }
        if constraint_mode not in valid_constraint_modes:
            raise ValueError(f"constraint_mode must be one of {sorted(valid_constraint_modes)}")
        self.constraint_mode = constraint_mode
        self.requested_topology_id = topology_id
        self.topology_registry = topology_registry or DEFAULT_TOPOLOGY_REGISTRY
        self.topology_pool = TopologyPool(
            benchmark=benchmark,
            pool_name=topology_pool_name,
            topology_ids=topology_ids,
            registry=self.topology_registry,
        )
        initial_selection = self.topology_pool.select(
            selection_mode="fixed",
            requested_topology_id=topology_id,
        )
        initial_spec = initial_selection.topology_spec
        self.num_rhs = num_rhs if num_rhs is not None else initial_spec.num_rhs
        self.num_ess = num_ess if num_ess is not None else initial_spec.num_ess
        self.num_rcs = num_rcs if num_rcs is not None else initial_spec.num_rcs
        if (self.num_rhs, self.num_ess, self.num_rcs) != (initial_spec.num_rhs, initial_spec.num_ess, initial_spec.num_rcs):
            raise ValueError("num_rhs/num_ess/num_rcs must match the selected benchmark topology dimensions")

        self.es_capacity = 20.0
        self.rc_capacity = 100.0
        self.split_options = 4
        self.max_invalid_streak = 5
        self.phi_r = 1.0
        self.phi_l = 1.0

        self.du_costs = np.array([0.05, 0.04, 0.00325, 0.0], dtype=np.float32)
        self.cu_costs = np.array([0.0, 0.001, 0.00175, 0.05], dtype=np.float32)
        self.crosshaul_latency_limits = np.array([10.0, 1.0, 0.25, 0.25], dtype=np.float32)

        self.topology_spec: TopologySpec = initial_spec
        self.topology_id = initial_spec.topology_id
        self.topology = initial_spec.build_graph()
        self.node_order = list(self.topology.nodes())
        self.edge_order = list(self.topology.edges())

        self.es_remaining = np.ones(self.num_ess, dtype=np.float32) * self.es_capacity
        self.rc_remaining = np.ones(self.num_rcs, dtype=np.float32) * self.rc_capacity
        self.rh_demands = np.zeros(self.num_rhs, dtype=np.float32)
        self.rh_latencies = np.zeros(self.num_rhs, dtype=np.float32)
        self.edge_remaining_bandwidth = np.zeros(len(self.edge_order), dtype=np.float32)
        self.prev_action = None

        self.action_space = spaces.MultiDiscrete(
            [self.split_options] * self.num_rhs +
            [self.num_ess] * self.num_rhs +
            [self.num_rcs] * self.num_rhs
        )

        state_dim = (2 * self.num_rhs) + self.num_ess + self.num_rcs + len(self.edge_order)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state_dim,),
            dtype=np.float32,
        )

    def _load_topology_spec(self, topology_spec: TopologySpec) -> None:
        self.topology_spec = topology_spec
        self.topology_id = topology_spec.topology_id
        self.topology = topology_spec.build_graph()
        self.node_order = list(self.topology.nodes())
        self.edge_order = list(self.topology.edges())
        self.edge_remaining_bandwidth = np.zeros(len(self.edge_order), dtype=np.float32)

    def _sample_requests(self) -> None:
        slice_types = self.np_random.choice(["eMBB", "mMTC", "uRLLC"], size=self.num_rhs)
        demands: List[float] = []
        latencies: List[float] = []

        for slice_type in slice_types:
            if slice_type == "eMBB":
                demands.append(float(self.np_random.uniform(250.0, 300.0)))
                latencies.append(float(self.np_random.uniform(15.0, 20.0)))
            elif slice_type == "mMTC":
                demands.append(float(self.np_random.uniform(150.0, 200.0)))
                latencies.append(float(self.np_random.uniform(180.0, 200.0)))
            else:
                demands.append(float(self.np_random.uniform(20.0, 40.0)))
                latencies.append(float(self.np_random.uniform(2.0, 4.0)))

        self.rh_demands = np.array(demands, dtype=np.float32)
        self.rh_latencies = np.array(latencies, dtype=np.float32)

    def _refresh_edge_state(self, bandwidth_usage: Dict[Tuple[str, str], float]) -> None:
        edge_remaining: List[float] = []
        for u, v in self.edge_order:
            key = self._edge_key(u, v)
            capacity = float(self.topology.edges[u, v]["bandwidth"])
            remaining = max(capacity - bandwidth_usage.get(key, 0.0), 0.0)
            self.topology.edges[u, v]["remaining_bandwidth"] = remaining
            edge_remaining.append(remaining)
        self.edge_remaining_bandwidth = np.array(edge_remaining, dtype=np.float32)

    @staticmethod
    def _edge_key(u: str, v: str) -> Tuple[str, str]:
        return tuple(sorted((u, v)))

    def _get_adjacency_info(self) -> Tuple[np.ndarray, Dict[Tuple[int, int], Dict[str, float]], List[str]]:
        node_to_idx = {node: idx for idx, node in enumerate(self.node_order)}
        n = len(self.node_order)
        adjacency = np.zeros((n, n), dtype=np.float32)
        edge_features: Dict[Tuple[int, int], Dict[str, float]] = {}

        for u, v, data in self.topology.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            adjacency[u_idx, v_idx] = 1.0
            adjacency[v_idx, u_idx] = 1.0
            attrs = {
                "bandwidth": float(data.get("remaining_bandwidth", data["bandwidth"])),
                "delay": float(data["delay"]),
            }
            edge_features[(u_idx, v_idx)] = attrs
            edge_features[(v_idx, u_idx)] = attrs

        return adjacency, edge_features, self.node_order

    def reset(self, seed=None, options: Optional[Dict[str, object]] = None):
        super().reset(seed=seed)
        options = options or {}
        requested_topology_id = options.get("topology_id", self.requested_topology_id)
        pool_name = str(options.get("topology_pool_name", self.topology_pool_name))
        selection_mode = str(options.get("topology_selection_mode", self.topology_selection_mode))
        if pool_name != self.topology_pool_name:
            self.topology_pool_name = pool_name
            self.topology_pool = TopologyPool(
                benchmark=self.benchmark,
                pool_name=pool_name,
                registry=self.topology_registry,
            )

        selection = self.topology_pool.select(
            selection_mode=selection_mode,
            rng=self.np_random,
            requested_topology_id=requested_topology_id,
        )
        self._load_topology_spec(selection.topology_spec)

        self.current_step = 0
        self.invalid_streak = 0
        self.prev_action = None
        self.es_remaining = np.ones(self.num_ess, dtype=np.float32) * self.es_capacity
        self.rc_remaining = np.ones(self.num_rcs, dtype=np.float32) * self.rc_capacity
        self._sample_requests()
        feasible_action = self.find_feasible_action(constraint_mode="legacy")
        greedy_strict_feasible_action = self.find_feasible_action(constraint_mode="strict_full")
        exact_strict_feasible = self.has_exact_feasible_action(constraint_mode="strict_full")
        bounded_strict_probe = self.probe_bounded_feasible_action(constraint_mode="strict_full")
        strict_feasible_exists = exact_strict_feasible if exact_strict_feasible is not None else (greedy_strict_feasible_action is not None)
        self._refresh_edge_state({})
        info = {
            "topology_id": self.topology_id,
            "benchmark": self.benchmark,
            "constraint_mode": self.constraint_mode,
            "topology_pool": self.topology_pool_name,
            "topology_selection_mode": selection.selection_mode,
            "topology_metadata": dict(self.topology_spec.metadata),
            "time_slot": 0,
            "episode_length_time_slots": self.max_steps,
            "has_structurally_valid_action": feasible_action is not None,
            "has_strictly_valid_action": bool(strict_feasible_exists),
            "has_greedy_strictly_valid_action": greedy_strict_feasible_action is not None,
            "has_exact_strictly_valid_action": exact_strict_feasible,
            "bounded_strict_feasibility_probe": bounded_strict_probe,
        }
        return self._get_state(), info

    def _get_state(self) -> np.ndarray:
        return np.concatenate(
            [
                self.rh_demands / 300.0,
                np.clip(self.rh_latencies / 200.0, 0.0, 1.0),
                self.es_remaining / self.es_capacity,
                self.rc_remaining / self.rc_capacity,
                self.edge_remaining_bandwidth / 160.0,
            ]
        ).astype(np.float32)

    def _split_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=int)
        splits = action[:self.num_rhs]
        es_choices = action[self.num_rhs:2 * self.num_rhs]
        rc_choices = action[2 * self.num_rhs:]
        return splits, es_choices, rc_choices

    def _get_direct_rc_options(self, rh_idx: int) -> List[int]:
        rh_node = f"RH{rh_idx}"
        options = []
        for rc_idx in range(self.num_rcs):
            if self.topology.has_edge(rh_node, f"RC{rc_idx}"):
                options.append(rc_idx)
        return options

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        split_mask = np.ones((self.num_rhs, self.split_options), dtype=bool)
        es_mask = np.zeros((self.num_rhs, self.num_ess), dtype=bool)
        rc_mask = np.zeros((self.num_rhs, self.num_rcs), dtype=bool)

        for rh_idx in range(self.num_rhs):
            rh_node = f"RH{rh_idx}"

            for es_idx in range(self.num_ess):
                es_mask[rh_idx, es_idx] = self.topology.has_edge(rh_node, f"ES{es_idx}")

            for rc_idx in range(self.num_rcs):
                rc_node = f"RC{rc_idx}"
                has_path_via_es = any(
                    self.topology.has_edge(rh_node, f"ES{es_idx}") and self.topology.has_edge(f"ES{es_idx}", rc_node)
                    for es_idx in range(self.num_ess)
                )
                rc_mask[rh_idx, rc_idx] = has_path_via_es or self.topology.has_edge(rh_node, rc_node)

            if not self._get_direct_rc_options(rh_idx):
                split_mask[rh_idx, 3] = False

        return {"split": split_mask, "es": es_mask, "rc": rc_mask}

    def get_conditional_rc_mask(self, splits: np.ndarray, es_choices: np.ndarray) -> np.ndarray:
        rc_mask = np.zeros((self.num_rhs, self.num_rcs), dtype=bool)

        for rh_idx in range(self.num_rhs):
            split = int(splits[rh_idx])
            es_idx = int(es_choices[rh_idx])
            rh_node = f"RH{rh_idx}"
            es_node = f"ES{es_idx}"

            for rc_idx in range(self.num_rcs):
                rc_node = f"RC{rc_idx}"
                if split == 3:
                    rc_mask[rh_idx, rc_idx] = self.topology.has_edge(rh_node, rc_node)
                else:
                    rc_mask[rh_idx, rc_idx] = (
                        self.topology.has_edge(rh_node, es_node) and
                        self.topology.has_edge(es_node, rc_node)
                    )

        return rc_mask

    def find_feasible_action(self, constraint_mode: str = "legacy") -> Optional[np.ndarray]:
        action = np.zeros(3 * self.num_rhs, dtype=np.int64)
        es_load = np.zeros(self.num_ess, dtype=np.float32)
        rc_load = np.zeros(self.num_rcs, dtype=np.float32)
        bandwidth_usage: Dict[Tuple[str, str], float] = {}

        for rh_idx in range(self.num_rhs):
            demand = float(self.rh_demands[rh_idx])
            demand_gbps = demand / 1000.0
            candidates = []

            for split in range(self.split_options):
                if split == 3:
                    for rc_idx in self._get_direct_rc_options(rh_idx):
                        candidates.append((split, 0, rc_idx))
                    continue

                for es_idx in range(self.num_ess):
                    rh_node = f"RH{rh_idx}"
                    es_node = f"ES{es_idx}"
                    if not self.topology.has_edge(rh_node, es_node):
                        continue
                    for rc_idx in range(self.num_rcs):
                        rc_node = f"RC{rc_idx}"
                        if self.topology.has_edge(es_node, rc_node):
                            candidates.append((split, es_idx, rc_idx))

            candidates.sort(
                key=lambda item: (
                    self.du_costs[item[0]] * demand + self.cu_costs[item[0]] * demand,
                    item[0] == 3,
                )
            )

            chosen = None
            for split, es_idx, rc_idx in candidates:
                next_es_load = es_load.copy()
                next_rc_load = rc_load.copy()
                next_bandwidth_usage = dict(bandwidth_usage)

                if split != 3:
                    next_es_load[es_idx] += self.du_costs[split] * demand
                    next_rc_load[rc_idx] += self.cu_costs[split] * demand
                    if next_es_load[es_idx] > self.es_capacity or next_rc_load[rc_idx] > self.rc_capacity:
                        continue
                    edge_key = self._edge_key(f"ES{es_idx}", f"RC{rc_idx}")
                else:
                    next_rc_load[rc_idx] += self.cu_costs[split] * demand
                    if next_rc_load[rc_idx] > self.rc_capacity:
                        continue
                    edge_key = self._edge_key(f"RH{rh_idx}", f"RC{rc_idx}")

                edge_capacity = float(self.topology.edges[edge_key]["bandwidth"])
                next_bandwidth_usage[edge_key] = next_bandwidth_usage.get(edge_key, 0.0) + demand_gbps
                if next_bandwidth_usage[edge_key] > edge_capacity:
                    continue

                es_load = next_es_load
                rc_load = next_rc_load
                bandwidth_usage = next_bandwidth_usage
                chosen = (split, es_idx, rc_idx)
                break

            if chosen is None:
                return None

            split, es_idx, rc_idx = chosen
            action[rh_idx] = split
            action[self.num_rhs + rh_idx] = es_idx
            action[(2 * self.num_rhs) + rh_idx] = rc_idx

        metrics = self._evaluate_action(action, constraint_mode_override=constraint_mode)
        return action if metrics["valid"] else None

    def _constraint_mode_alias(self, constraint_mode: str) -> str:
        if constraint_mode == "strict":
            return "strict_full"
        return constraint_mode

    def _enforced_reason_set(self, constraint_mode: str) -> set[str]:
        active_mode = self._constraint_mode_alias(constraint_mode)
        if active_mode in {"legacy", "strict_connectivity_only"}:
            return set()
        if active_mode == "strict_connectivity_plus_capacity":
            return {"es_capacity_exceeded", "rc_capacity_exceeded"}
        if active_mode == "strict_connectivity_plus_capacity_plus_bandwidth":
            return {"es_capacity_exceeded", "rc_capacity_exceeded", "bandwidth_exceeded"}
        if active_mode == "strict_full":
            return {
                "es_capacity_exceeded",
                "rc_capacity_exceeded",
                "bandwidth_exceeded",
                "e2e_latency_exceeded",
                "crosshaul_latency_exceeded",
            }
        raise ValueError(f"Unsupported constraint_mode: {constraint_mode}")

    def _candidate_assignments_for_rh(self, rh_idx: int) -> List[Tuple[int, int, int]]:
        rh_node = f"RH{rh_idx}"
        candidates: List[Tuple[int, int, int]] = []
        for split in range(self.split_options):
            if split == 3:
                for rc_idx in self._get_direct_rc_options(rh_idx):
                    candidates.append((split, 0, rc_idx))
                continue
            for es_idx in range(self.num_ess):
                es_node = f"ES{es_idx}"
                if not self.topology.has_edge(rh_node, es_node):
                    continue
                for rc_idx in range(self.num_rcs):
                    rc_node = f"RC{rc_idx}"
                    if self.topology.has_edge(es_node, rc_node):
                        candidates.append((split, es_idx, rc_idx))
        return candidates

    def _search_feasible_action(
        self,
        *,
        constraint_mode: str,
        rh_indices: List[int],
        visit_limit: Optional[int] = None,
    ) -> Optional[bool]:
        enforced_reasons = self._enforced_reason_set(constraint_mode)
        candidate_map = {
            rh_idx: self._candidate_assignments_for_rh(rh_idx)
            for rh_idx in rh_indices
        }
        if any(not candidate_map[rh_idx] for rh_idx in rh_indices):
            return False

        order = sorted(rh_indices, key=lambda rh_idx: len(candidate_map[rh_idx]))
        edge_caps = []
        edge_index: Dict[Tuple[str, str], int] = {}
        for idx, (u, v) in enumerate(self.edge_order):
            edge_key = self._edge_key(u, v)
            edge_index[edge_key] = idx
            edge_caps.append(float(self.topology.edges[u, v]["bandwidth"]))
        visit_count = 0

        @lru_cache(maxsize=None)
        def dfs(
            pos: int,
            es_loads: Tuple[float, ...],
            rc_loads: Tuple[float, ...],
            edge_usage: Tuple[float, ...],
        ) -> bool:
            nonlocal visit_count
            visit_count += 1
            if visit_limit is not None and visit_count > visit_limit:
                raise RuntimeError("bounded_feasibility_visit_limit_reached")
            if pos == len(order):
                return True

            rh_idx = order[pos]
            demand = float(self.rh_demands[rh_idx])
            demand_gbps = demand / 1000.0
            rh_latency = float(self.rh_latencies[rh_idx])

            for split, es_idx, rc_idx in candidate_map[rh_idx]:
                next_es = list(es_loads)
                next_rc = list(rc_loads)
                next_edge = list(edge_usage)
                invalid_reasons = set()

                if split == 3:
                    next_rc[rc_idx] += float(self.cu_costs[split] * demand)
                    if next_rc[rc_idx] > self.rc_capacity + 1e-9:
                        invalid_reasons.add("rc_capacity_exceeded")
                    direct_edge = self.topology.edges[f"RH{rh_idx}", f"RC{rc_idx}"]
                    edge_key = self._edge_key(f"RH{rh_idx}", f"RC{rc_idx}")
                    edge_idx = edge_index[edge_key]
                    next_edge[edge_idx] += demand_gbps
                    if next_edge[edge_idx] > edge_caps[edge_idx] + 1e-9:
                        invalid_reasons.add("bandwidth_exceeded")
                    direct_delay = float(direct_edge["delay"])
                    if direct_delay > rh_latency + 1e-9:
                        invalid_reasons.add("e2e_latency_exceeded")
                    if direct_delay > float(self.crosshaul_latency_limits[split]) + 1e-9:
                        invalid_reasons.add("crosshaul_latency_exceeded")
                else:
                    next_es[es_idx] += float(self.du_costs[split] * demand)
                    next_rc[rc_idx] += float(self.cu_costs[split] * demand)
                    if next_es[es_idx] > self.es_capacity + 1e-9:
                        invalid_reasons.add("es_capacity_exceeded")
                    if next_rc[rc_idx] > self.rc_capacity + 1e-9:
                        invalid_reasons.add("rc_capacity_exceeded")
                    edge_key = self._edge_key(f"ES{es_idx}", f"RC{rc_idx}")
                    edge_idx = edge_index[edge_key]
                    next_edge[edge_idx] += demand_gbps
                    if next_edge[edge_idx] > edge_caps[edge_idx] + 1e-9:
                        invalid_reasons.add("bandwidth_exceeded")
                    rh_es_edge = self.topology.edges[f"RH{rh_idx}", f"ES{es_idx}"]
                    es_rc_edge = self.topology.edges[f"ES{es_idx}", f"RC{rc_idx}"]
                    e2e_delay = float(rh_es_edge["delay"] + es_rc_edge["delay"])
                    crosshaul_delay = float(es_rc_edge["delay"])
                    if e2e_delay > rh_latency + 1e-9:
                        invalid_reasons.add("e2e_latency_exceeded")
                    if crosshaul_delay > float(self.crosshaul_latency_limits[split]) + 1e-9:
                        invalid_reasons.add("crosshaul_latency_exceeded")

                if invalid_reasons & enforced_reasons:
                    continue

                rounded_es = tuple(round(value, 6) for value in next_es)
                rounded_rc = tuple(round(value, 6) for value in next_rc)
                rounded_edge = tuple(round(value, 6) for value in next_edge)
                if dfs(pos + 1, rounded_es, rounded_rc, rounded_edge):
                    return True
            return False

        try:
            return dfs(
                0,
                tuple(0.0 for _ in range(self.num_ess)),
                tuple(0.0 for _ in range(self.num_rcs)),
                tuple(0.0 for _ in range(len(self.edge_order))),
            )
        except RuntimeError as exc:
            if str(exc) == "bounded_feasibility_visit_limit_reached":
                return None
            raise

    def has_exact_feasible_action(self, constraint_mode: str = "strict_full") -> Optional[bool]:
        if self.num_rhs > self.EXACT_FEASIBILITY_MAX_RHS:
            return None
        return self._search_feasible_action(
            constraint_mode=constraint_mode,
            rh_indices=list(range(self.num_rhs)),
            visit_limit=None,
        )

    def probe_bounded_feasible_action(
        self,
        constraint_mode: str = "strict_full",
        max_rhs: Optional[int] = None,
        visit_limit: Optional[int] = None,
    ) -> Dict[str, object]:
        capped_rhs = min(self.num_rhs, max_rhs or self.BOUNDED_FEASIBILITY_RHS)
        capped_visit_limit = visit_limit or self.BOUNDED_FEASIBILITY_VISIT_LIMIT
        result = self._search_feasible_action(
            constraint_mode=constraint_mode,
            rh_indices=list(range(capped_rhs)),
            visit_limit=capped_visit_limit,
        )
        return {
            "result": result,
            "rhs_considered": int(capped_rhs),
            "visit_limit": int(capped_visit_limit),
            "is_full_instance": bool(capped_rhs == self.num_rhs),
            "is_exact": bool(capped_rhs == self.num_rhs and result is not None and self.num_rhs <= self.EXACT_FEASIBILITY_MAX_RHS),
            "status": (
                "exact"
                if capped_rhs == self.num_rhs and result is not None and self.num_rhs <= self.EXACT_FEASIBILITY_MAX_RHS
                else "bounded"
            ),
        }

    def _evaluate_action(self, action: np.ndarray, constraint_mode_override: Optional[str] = None) -> Dict[str, object]:
        active_constraint_mode = self._constraint_mode_alias(constraint_mode_override or self.constraint_mode)
        enforced_reasons = self._enforced_reason_set(active_constraint_mode)
        splits, es_choices, rc_choices = self._split_action(action)
        nfail = 0
        failure_counts = {
            "missing_direct_rc_link": 0,
            "missing_rh_es_link": 0,
            "missing_es_rc_link": 0,
            "es_capacity_exceeded": 0,
            "rc_capacity_exceeded": 0,
            "bandwidth_exceeded": 0,
            "e2e_latency_exceeded": 0,
            "crosshaul_latency_exceeded": 0,
        }
        du_load = np.zeros(self.num_ess, dtype=np.float32)
        cu_load = np.zeros(self.num_rcs, dtype=np.float32)
        bandwidth_usage: Dict[Tuple[str, str], float] = {}
        total_processing_cost = 0.0
        total_routing_cost = 0.0
        total_e2e_violation = 0.0
        total_cross_violation = 0.0
        valid = True
        split_usage = {f"S{i + 1}": 0 for i in range(self.split_options)}

        for rh_idx in range(self.num_rhs):
            split = int(splits[rh_idx])
            split_usage[f"S{split + 1}"] += 1
            es_idx = int(es_choices[rh_idx])
            rc_idx = int(rc_choices[rh_idx])
            demand_gbps = float(self.rh_demands[rh_idx] / 1000.0)
            rh_latency = float(self.rh_latencies[rh_idx])
            rh_node = f"RH{rh_idx}"
            es_node = f"ES{es_idx}"
            rc_node = f"RC{rc_idx}"
            rh_valid = True

            uses_direct_rc = split == 3

            if uses_direct_rc:
                if not self.topology.has_edge(rh_node, rc_node):
                    valid = False
                    nfail += 1
                    failure_counts["missing_direct_rc_link"] += 1
                    continue

                direct_edge = self.topology.edges[rh_node, rc_node]
                edge_key = self._edge_key(rh_node, rc_node)
                bandwidth_usage[edge_key] = bandwidth_usage.get(edge_key, 0.0) + demand_gbps
                total_routing_cost += self.phi_l * direct_edge["delay"] * demand_gbps
                total_e2e_violation += max(direct_edge["delay"] - rh_latency, 0.0)
                total_cross_violation += max(direct_edge["delay"] - self.crosshaul_latency_limits[split], 0.0)
            else:
                if not self.topology.has_edge(rh_node, es_node):
                    valid = False
                    rh_valid = False
                    nfail += 1
                    failure_counts["missing_rh_es_link"] += 1

                if not self.topology.has_edge(es_node, rc_node):
                    valid = False
                    rh_valid = False
                    nfail += 1
                    failure_counts["missing_es_rc_link"] += 1

                if not rh_valid:
                    continue

                rh_es_edge = self.topology.edges[rh_node, es_node]
                es_rc_edge = self.topology.edges[es_node, rc_node]
                crosshaul_delay = float(es_rc_edge["delay"])
                e2e_delay = float(rh_es_edge["delay"] + es_rc_edge["delay"])

                du_load[es_idx] += self.du_costs[split] * float(self.rh_demands[rh_idx])
                cu_load[rc_idx] += self.cu_costs[split] * float(self.rh_demands[rh_idx])

                edge_key = self._edge_key(es_node, rc_node)
                bandwidth_usage[edge_key] = bandwidth_usage.get(edge_key, 0.0) + demand_gbps
                total_routing_cost += self.phi_l * crosshaul_delay * demand_gbps
                total_e2e_violation += max(e2e_delay - rh_latency, 0.0)
                total_cross_violation += max(crosshaul_delay - self.crosshaul_latency_limits[split], 0.0)

            total_processing_cost += (
                self.du_costs[split] * float(self.rh_demands[rh_idx]) +
                self.cu_costs[split] * float(self.rh_demands[rh_idx])
            )

        split_changes = 0
        es_changes = 0
        rc_changes = 0
        total_reconfiguration_changes = 0
        reconfiguration_cost = 0.0
        if self.prev_action is not None:
            prev_splits, prev_es_choices, prev_rc_choices = self._split_action(self.prev_action)
            split_changes = int(np.count_nonzero(splits != prev_splits))
            es_changes = int(np.count_nonzero(es_choices != prev_es_choices))
            rc_changes = int(np.count_nonzero(rc_choices != prev_rc_choices))
            total_reconfiguration_changes = split_changes + es_changes + rc_changes
            reconfiguration_cost = self.phi_r * float(total_reconfiguration_changes)

        es_overuse = np.maximum(du_load - self.es_capacity, 0.0).sum()
        rc_overuse = np.maximum(cu_load - self.rc_capacity, 0.0).sum()
        bandwidth_overuse = 0.0
        for u, v in self.edge_order:
            key = self._edge_key(u, v)
            used = bandwidth_usage.get(key, 0.0)
            capacity = float(self.topology.edges[u, v]["bandwidth"])
            bandwidth_overuse += max(used - capacity, 0.0)

        slack_penalty = float(es_overuse + rc_overuse + bandwidth_overuse + total_e2e_violation + total_cross_violation)
        total_cost = float(total_processing_cost + total_routing_cost + reconfiguration_cost + slack_penalty)
        invalid_reasons = []

        if es_overuse > 0:
            failure_counts["es_capacity_exceeded"] = 1
            invalid_reasons.append("es_capacity_exceeded")
        if rc_overuse > 0:
            failure_counts["rc_capacity_exceeded"] = 1
            invalid_reasons.append("rc_capacity_exceeded")
        if bandwidth_overuse > 0:
            failure_counts["bandwidth_exceeded"] = 1
            invalid_reasons.append("bandwidth_exceeded")
        if total_e2e_violation > 0:
            failure_counts["e2e_latency_exceeded"] = 1
            invalid_reasons.append("e2e_latency_exceeded")
        if total_cross_violation > 0:
            failure_counts["crosshaul_latency_exceeded"] = 1
            invalid_reasons.append("crosshaul_latency_exceeded")

        if enforced_reasons.intersection(invalid_reasons):
            valid = False
            nfail += len(enforced_reasons.intersection(invalid_reasons))

        return {
            "valid": valid,
            "nfail": nfail,
            "du_load": du_load,
            "cu_load": cu_load,
            "bandwidth_usage": bandwidth_usage,
            "es_overuse": float(es_overuse),
            "rc_overuse": float(rc_overuse),
            "bandwidth_overuse": float(bandwidth_overuse),
            "e2e_violation": float(total_e2e_violation),
            "crosshaul_violation": float(total_cross_violation),
            "processing_cost": float(total_processing_cost),
            "routing_cost": float(total_routing_cost),
            "reconfiguration_cost": float(reconfiguration_cost),
            "split_changes": split_changes,
            "es_changes": es_changes,
            "rc_changes": rc_changes,
            "total_reconfiguration_changes": total_reconfiguration_changes,
            "sla_penalty": slack_penalty,
            "total_cost": total_cost,
            "failure_counts": failure_counts,
            "invalid_reasons": invalid_reasons,
            "constraint_mode": active_constraint_mode,
            "split_usage": split_usage,
        }

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # One environment step corresponds to one paper-style time slot.
        self.current_step += 1
        metrics = self._evaluate_action(np.asarray(action, dtype=int))

        if not metrics["valid"]:
            self.invalid_streak += 1
            early_terminated = self.invalid_streak >= self.max_invalid_streak
            reward = -1.0 if early_terminated else -(metrics["nfail"] / max(2 * self.num_rhs, 1))
        else:
            self.invalid_streak = 0
            reward = float((1.0 + np.log1p(metrics["total_cost"])) ** -1)

        self.es_remaining = np.maximum(self.es_capacity - metrics["du_load"], 0.0).astype(np.float32)
        self.rc_remaining = np.maximum(self.rc_capacity - metrics["cu_load"], 0.0).astype(np.float32)
        self._refresh_edge_state(metrics["bandwidth_usage"])
        self.prev_action = np.asarray(action, dtype=int).copy()

        terminated = self.current_step >= self.max_steps or self.invalid_streak >= self.max_invalid_streak
        truncated = False

        self._sample_requests()
        state = self._get_state()
        info = {
            "time_slot": self.current_step,
            "episode_length_time_slots": self.max_steps,
            "valid_deployment": bool(metrics["valid"]),
            "deployment_cost": metrics["total_cost"] if metrics["valid"] else float("inf"),
            "raw_total_cost": metrics["total_cost"],
            "processing_cost": metrics["processing_cost"],
            "routing_cost": metrics["routing_cost"],
            "reconfiguration_cost": metrics["reconfiguration_cost"],
            "split_changes": metrics["split_changes"],
            "es_changes": metrics["es_changes"],
            "rc_changes": metrics["rc_changes"],
            "total_reconfiguration_changes": metrics["total_reconfiguration_changes"],
            "sla_penalty": metrics["sla_penalty"],
            "es_overuse": metrics["es_overuse"],
            "rc_overuse": metrics["rc_overuse"],
            "bandwidth_overuse": metrics["bandwidth_overuse"],
            "e2e_violation": metrics["e2e_violation"],
            "crosshaul_violation": metrics["crosshaul_violation"],
            "split_usage": metrics["split_usage"],
            "failed_links": metrics["nfail"],
            "failure_counts": metrics["failure_counts"],
            "invalid_reasons": metrics["invalid_reasons"],
            "invalid_streak": self.invalid_streak,
            "topology_id": self.topology_id,
            "benchmark": self.benchmark,
            "constraint_mode": self.constraint_mode,
            "topology_pool": self.topology_pool_name,
            "topology_metadata": dict(self.topology_spec.metadata),
        }
        return state, reward, terminated, truncated, info
