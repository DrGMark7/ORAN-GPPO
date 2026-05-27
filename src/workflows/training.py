import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.common.paths import (
    DEFAULT_RUN_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_RESULTS_PATH,
    ensure_output_dirs,
    resolve_checkpoint_path,
    resolve_episode_traces_dir,
    resolve_results_path,
)
from src.core import (
    DEFAULT_BENCHMARK_TOPOLOGY_POOLS,
    ExperimentConfig,
    GNNFeatureExtractor,
    ORANGraphBuilder,
    PPOAgent,
    SimplifiedORANEnv,
    get_topology_spec,
    get_benchmark_dimensions,
)
from src.workflows.paper_training import _run_paper_mode_from_args
from src.workflows.training_constants import (
    PAPER_GNN_HIDDEN_DIM,
    PAPER_NUM_ENVS,
    PAPER_NUM_SEEDS,
    PAPER_TIMESTEPS,
)
from src.workflows.training_csv import _export_csv_artifacts, _write_episode_trace_csv


def _zero_cost_breakdown() -> Dict[str, float]:
    return {
        "processing_cost": 0.0,
        "routing_cost": 0.0,
        "reconfiguration_cost": 0.0,
        "sla_penalty": 0.0,
        "es_overuse": 0.0,
        "rc_overuse": 0.0,
        "bandwidth_overuse": 0.0,
        "e2e_violation": 0.0,
        "crosshaul_violation": 0.0,
    }


def _zero_reconfiguration_stats() -> Dict[str, float]:
    return {
        "split_changes": 0.0,
        "es_changes": 0.0,
        "rc_changes": 0.0,
        "total_reconfiguration_changes": 0.0,
    }


def _zero_timing_stats() -> Dict[str, float]:
    return {
        "total_seconds": 0.0,
        "reset_seconds": 0.0,
        "adjacency_seconds": 0.0,
        "graph_build_seconds": 0.0,
        "gnn_forward_seconds": 0.0,
        "action_selection_seconds": 0.0,
        "env_step_seconds": 0.0,
        "store_transition_seconds": 0.0,
        "ppo_update_seconds": 0.0,
        "results_write_seconds": 0.0,
        "checkpoint_save_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "comparison_seconds": 0.0,
    }


def _episode_trace_path(
    *,
    trace_output_dir: Path,
    topology_pool_name: str,
    topology_id: str,
    constraint_mode: str,
    episode_index: int,
) -> Path:
    safe_mode = constraint_mode.replace("/", "_")
    return trace_output_dir / f"{topology_pool_name}_{topology_id}_{safe_mode}_episode_{episode_index:03d}.json"


def _write_episode_trace(
    *,
    trace_path: Path,
    benchmark: str,
    topology_pool_name: str,
    topology_id: str,
    constraint_mode: str,
    episode_index: int,
    episode_length_time_slots: int,
    reset_info: Dict[str, object],
    slots: List[Dict[str, object]],
    episode_reward: float,
) -> None:
    split_changes = float(sum(slot["split_changes"] for slot in slots))
    es_changes = float(sum(slot["es_changes"] for slot in slots))
    rc_changes = float(sum(slot["rc_changes"] for slot in slots))
    total_reconfiguration_changes = float(sum(slot["total_reconfiguration_changes"] for slot in slots))
    reconfiguration_cost = float(sum(slot["reconfiguration_cost"] for slot in slots))
    raw_total_cost = float(sum(slot["raw_total_cost"] for slot in slots))
    valid_slots = int(sum(1 for slot in slots if slot["valid_deployment"]))
    summary = {
        "total_split_changes": split_changes,
        "total_es_changes": es_changes,
        "total_rc_changes": rc_changes,
        "total_reconfiguration_changes": total_reconfiguration_changes,
        "total_reconfiguration_cost": reconfiguration_cost,
        "average_cost_per_slot": float(raw_total_cost / max(len(slots), 1)),
        "average_reward_per_slot": float(episode_reward / max(len(slots), 1)),
        "valid_slot_percentage": float(100.0 * valid_slots / max(len(slots), 1)),
    }
    payload = {
        "benchmark": benchmark,
        "topology_pool_name": topology_pool_name,
        "topology_id": topology_id,
        "constraint_mode": constraint_mode,
        "episode_index": episode_index,
        "episode_length_time_slots": episode_length_time_slots,
        "reset_info": reset_info,
        "summary": summary,
        "slots": slots,
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def _zero_split_usage() -> Dict[str, int]:
    return {f"S{i}": 0 for i in range(1, 5)}


def _zero_failure_counts() -> Dict[str, int]:
    return {
        "missing_direct_rc_link": 0,
        "missing_rh_es_link": 0,
        "missing_es_rc_link": 0,
        "es_capacity_exceeded": 0,
        "rc_capacity_exceeded": 0,
        "bandwidth_exceeded": 0,
        "e2e_latency_exceeded": 0,
        "crosshaul_latency_exceeded": 0,
    }


def _zero_topology_debug() -> Dict[str, object]:
    return {
        "resets": 0,
        "strictly_valid_resets": 0,
        "exact_strictly_valid_resets": 0,
        "time_slots": 0,
        "valid_time_slots": 0,
        "invalid_reason_events": 0,
        "invalid_counts_by_reason": _zero_failure_counts(),
        "first_failing_constraint_counts": {},
        "bounded_probe_results": [],
    }


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _invalid_reason_percentages(
    invalid_counts: Dict[str, int],
    total_steps: int,
) -> Dict[str, float]:
    return {
        key: _safe_rate(value, total_steps)
        for key, value in invalid_counts.items()
    }


def _strict_feasibility_wording(
    exact_values: List[Optional[bool]],
) -> Dict[str, str]:
    has_exact = any(value is not None for value in exact_values)
    if has_exact:
        return {
            "rate_label": "Strict-Feasible Reset Rate",
            "no_action_label": "Episodes With No Strict-Valid Action",
            "status": "exact_supported",
        }
    return {
        "rate_label": "Strict-Feasible Reset Rate (Current Probe)",
        "no_action_label": "Episodes With No Strict-Valid Action Found By Current Probe",
        "status": "probe_only",
    }


def _bounded_probe_summary(probes: List[Dict[str, object]]) -> Dict[str, float]:
    if not probes:
        return {
            "positive_rate": 0.0,
            "negative_rate": 0.0,
            "inconclusive_rate": 0.0,
        }
    results = [probe.get("result") for probe in probes]
    total = len(results)
    return {
        "positive_rate": _safe_rate(sum(result is True for result in results), total),
        "negative_rate": _safe_rate(sum(result is False for result in results), total),
        "inconclusive_rate": _safe_rate(sum(result is None for result in results), total),
    }


def _collect_topology_latency_diagnostics(
    topology_id: str,
    crosshaul_latency_limits: np.ndarray,
) -> Dict[str, object]:
    spec = get_topology_spec(topology_id)
    graph = spec.build_graph()
    es_rc_delays: List[float] = []
    direct_rc_delays: List[float] = []

    for u, v, data in graph.edges(data=True):
        delay = float(data["delay"])
        if u.startswith("ES") and v.startswith("RC") or u.startswith("RC") and v.startswith("ES"):
            es_rc_delays.append(delay)
        if u.startswith("RH") and v.startswith("RC") or u.startswith("RC") and v.startswith("RH"):
            direct_rc_delays.append(delay)

    split_latency = {}
    split_structural = {}
    rh_direct_count = 0
    for rh_node in [node for node in graph.nodes if node.startswith("RH")]:
        if any(graph.has_edge(rh_node, rc_node) for rc_node in graph.nodes if rc_node.startswith("RC")):
            rh_direct_count += 1
    for split_idx, limit in enumerate(crosshaul_latency_limits):
        split_key = f"S{split_idx + 1}"
        if split_idx == 3:
            structural_available = rh_direct_count > 0
            compatible = [delay for delay in direct_rc_delays if delay <= float(limit)]
            total = len(direct_rc_delays)
            split_latency[split_key] = {
                "mode": "direct_rh_rc",
                "limit_ms": float(limit),
                "compatible_edge_count": len(compatible),
                "compatible_edge_fraction": _safe_rate(len(compatible), total),
                "min_delay_ms": float(min(direct_rc_delays)) if direct_rc_delays else None,
                "max_delay_ms": float(max(direct_rc_delays)) if direct_rc_delays else None,
                "strictly_possible_from_link_delay": bool(compatible),
            }
        else:
            structural_available = any(
                (
                    u.startswith("RH") and v.startswith("ES") or u.startswith("ES") and v.startswith("RH")
                )
                for u, v in graph.edges()
            ) and any(
                (
                    u.startswith("ES") and v.startswith("RC") or u.startswith("RC") and v.startswith("ES")
                )
                for u, v in graph.edges()
            )
            compatible = [delay for delay in es_rc_delays if delay <= float(limit)]
            total = len(es_rc_delays)
            split_latency[split_key] = {
                "mode": "es_rc",
                "limit_ms": float(limit),
                "compatible_edge_count": len(compatible),
                "compatible_edge_fraction": _safe_rate(len(compatible), total),
                "min_delay_ms": float(min(es_rc_delays)) if es_rc_delays else None,
                "max_delay_ms": float(max(es_rc_delays)) if es_rc_delays else None,
                "strictly_possible_from_link_delay": bool(compatible),
            }
        split_structural[split_key] = structural_available

    impossible_splits = [
        split_key
        for split_key, payload in split_latency.items()
        if not payload["strictly_possible_from_link_delay"]
    ]
    return {
        "topology_id": topology_id,
        "family": spec.metadata.get("family"),
        "difficulty": spec.metadata.get("difficulty"),
        "crosshaul_latency_limits_ms": [float(value) for value in crosshaul_latency_limits],
        "es_rc_delay_range_ms": {
            "min": float(min(es_rc_delays)) if es_rc_delays else None,
            "max": float(max(es_rc_delays)) if es_rc_delays else None,
        },
        "direct_rc_delay_range_ms": {
            "min": float(min(direct_rc_delays)) if direct_rc_delays else None,
            "max": float(max(direct_rc_delays)) if direct_rc_delays else None,
        },
        "min_feasible_crosshaul_delay_by_split_ms": {
            split_key: payload["min_delay_ms"]
            for split_key, payload in split_latency.items()
        },
        "direct_rh_rc_fraction": _safe_rate(rh_direct_count, spec.num_rhs),
        "split_structurally_available": split_structural,
        "split_latency_feasibility": split_latency,
        "splits_ruled_out_by_link_delay": impossible_splits,
    }


def _collect_benchmark_audit(benchmark: str, crosshaul_latency_limits: np.ndarray) -> Dict[str, Dict[str, object]]:
    audit = {}
    for pool_name, topology_ids in DEFAULT_BENCHMARK_TOPOLOGY_POOLS[benchmark].items():
        audit[pool_name] = {
            topology_id: _collect_topology_latency_diagnostics(topology_id, crosshaul_latency_limits)
            for topology_id in topology_ids
        }
    return audit


def _dict_l2_norm(state_dict: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in state_dict.values():
        total += float(torch.sum(tensor.detach().float() ** 2).item())
    return float(total ** 0.5)


def _build_env(
    *,
    benchmark: str,
    max_steps: int,
    topology_pool_name: str,
    topology_selection_mode: str,
    constraint_mode: str,
    topology_id: Optional[str] = None,
) -> SimplifiedORANEnv:
    num_rhs, num_ess, num_rcs = get_benchmark_dimensions(benchmark)
    return SimplifiedORANEnv(
        num_rhs=num_rhs,
        num_ess=num_ess,
        num_rcs=num_rcs,
        max_steps=max_steps,
        benchmark=benchmark,
        topology_pool_name=topology_pool_name,
        topology_selection_mode=topology_selection_mode,
        constraint_mode=constraint_mode,
        topology_id=topology_id,
    )


def train_gppo(
    num_episodes: int = 100,
    max_steps: int = 50,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 42,
    results_path: Path = DEFAULT_RESULTS_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    benchmark: str = "small",
    topology_selection_mode: str = "random_per_reset",
    constraint_mode: str = "legacy",
    train_topology_id: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    num_envs: int = 1,
    gnn_hidden_dim: int = 64,
    gnn_input_dim: int = 6,
    include_node_index: bool = False,
    paper_mode: bool = False,
) -> Tuple[PPOAgent, torch.nn.Module, dict]:
    if num_envs > 1 or total_timesteps is not None:
        from src.workflows.vectorized_training import _train_gppo_sync_vectorized

        return _train_gppo_sync_vectorized(
            total_timesteps=total_timesteps or num_episodes * max_steps * num_envs,
            max_steps=max_steps,
            batch_size=batch_size,
            device=device,
            seed=seed,
            results_path=results_path,
            checkpoint_path=checkpoint_path,
            benchmark=benchmark,
            topology_selection_mode=topology_selection_mode,
            constraint_mode=constraint_mode,
            train_topology_id=train_topology_id,
            num_envs=num_envs,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_input_dim=gnn_input_dim,
            include_node_index=include_node_index,
            paper_mode=paper_mode,
        )

    train_start_time = time.perf_counter()
    timing_stats = _zero_timing_stats()
    np.random.seed(seed)
    torch.manual_seed(seed)
    ensure_output_dirs()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        benchmark=benchmark,
        train_selection_mode=topology_selection_mode,
        eval_selection_mode="fixed",
        train_topology_id=train_topology_id,
        seed=seed,
    )

    env = _build_env(
        benchmark=benchmark,
        max_steps=max_steps,
        topology_pool_name="train",
        topology_selection_mode=topology_selection_mode,
        constraint_mode=constraint_mode,
        topology_id=train_topology_id,
    )
    gnn = GNNFeatureExtractor(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, output_dim=128).to(device)
    graph_builder = ORANGraphBuilder(env.num_rhs, env.num_ess, env.num_rcs, include_node_index=include_node_index)
    agent = PPOAgent(
        feature_dim=128,
        num_rhs=env.num_rhs,
        num_splits=4,
        num_ess=env.num_ess,
        num_rcs=env.num_rcs,
        lr=1e-4,
        gamma=0.98,
        gae_lambda=0.97,
        clip_ratio=0.3,
        device=device,
    )
    agent.attach_feature_extractor(gnn)
    initial_gnn_state = {key: value.detach().cpu().clone() for key, value in gnn.state_dict().items()}

    print("=" * 60)
    print("GPPO Training for O-RAN Resource Management")
    print("=" * 60)
    print(f"Project Benchmark: {benchmark} ({env.num_rhs} RH / {env.num_ess} ES / {env.num_rcs} RC)")
    print(f"Train topology pool: {DEFAULT_BENCHMARK_TOPOLOGY_POOLS[benchmark]['train']}")
    print(f"Selection mode: {topology_selection_mode}")
    print(f"Constraint mode: {constraint_mode}")
    print(f"Dimensions: {env.num_rhs} RHs, {env.num_ess} ESs, {env.num_rcs} RCs")
    print(f"Episodes: {num_episodes}, Time slots per episode: {max_steps}")
    print("Semantics: one environment step = one paper time slot with per-slot split/ES/RC decisions")
    print(f"Device: {device}")
    print(f"Results: {results_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    episode_rewards = []
    episode_costs = []
    valid_deployments = []
    valid_time_slot_rates = []
    episode_topology_ids = []
    episode_topology_pools = []
    episode_has_structurally_valid_action = []
    episode_has_strictly_valid_action = []
    episode_has_greedy_strictly_valid_action = []
    episode_has_exact_strictly_valid_action = []
    episode_bounded_strict_feasibility_probe = []
    episode_failure_counts = []
    episode_cost_breakdowns = []
    episode_reconfiguration_stats = []
    episode_split_usage = []
    training_topology_debug: Dict[str, Dict[str, object]] = {}
    first_failure_traces = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        reset_start = time.perf_counter()
        _, reset_info = env.reset(
            seed=seed + episode,
            options={
                "topology_pool_name": "train",
                "topology_selection_mode": topology_selection_mode,
                "topology_id": train_topology_id,
            },
        )
        timing_stats["reset_seconds"] += time.perf_counter() - reset_start
        episode_topology_ids.append(reset_info["topology_id"])
        episode_topology_pools.append(reset_info["topology_pool"])
        episode_has_structurally_valid_action.append(bool(reset_info.get("has_structurally_valid_action", False)))
        episode_has_strictly_valid_action.append(bool(reset_info.get("has_strictly_valid_action", False)))
        episode_has_greedy_strictly_valid_action.append(bool(reset_info.get("has_greedy_strictly_valid_action", False)))
        episode_has_exact_strictly_valid_action.append(reset_info.get("has_exact_strictly_valid_action"))
        episode_bounded_strict_feasibility_probe.append(reset_info.get("bounded_strict_feasibility_probe", {}))
        topology_debug = training_topology_debug.setdefault(reset_info["topology_id"], _zero_topology_debug())
        topology_debug["resets"] += 1
        topology_debug["strictly_valid_resets"] += int(bool(reset_info.get("has_strictly_valid_action", False)))
        exact_reset = reset_info.get("has_exact_strictly_valid_action")
        if exact_reset is True:
            topology_debug["exact_strictly_valid_resets"] += 1
        topology_debug["bounded_probe_results"].append(reset_info.get("bounded_strict_feasibility_probe", {}))
        episode_reward = 0.0
        valid_costs = []
        valid_steps = 0
        failure_counts = _zero_failure_counts()
        cost_breakdown = _zero_cost_breakdown()
        reconfiguration_stats = _zero_reconfiguration_stats()
        split_usage = _zero_split_usage()
        time_slot_count = 0

        for _ in range(max_steps):
            adjacency_start = time.perf_counter()
            adjacency, edge_features, _ = env._get_adjacency_info()
            timing_stats["adjacency_seconds"] += time.perf_counter() - adjacency_start

            graph_build_start = time.perf_counter()
            graph = graph_builder.build_graph(
                env.rh_demands,
                env.rh_latencies,
                env.es_remaining,
                env.rc_remaining,
                adjacency,
                edge_features,
            )
            timing_stats["graph_build_seconds"] += time.perf_counter() - graph_build_start

            # Rollout action selection does not need gradients because PPO
            # recomputes graph features during `agent.update()`.
            gnn_forward_start = time.perf_counter()
            with torch.no_grad():
                features = gnn(graph.to(device))
            timing_stats["gnn_forward_seconds"] += time.perf_counter() - gnn_forward_start

            action_selection_start = time.perf_counter()
            action_mask = env.get_action_mask()
            action, log_prob, value, used_action_mask = agent.select_action_sequential(
                features.squeeze(0),
                action_mask,
                env.get_conditional_rc_mask,
            )
            timing_stats["action_selection_seconds"] += time.perf_counter() - action_selection_start

            env_step_start = time.perf_counter()
            _, reward, terminated, truncated, info = env.step(action)
            timing_stats["env_step_seconds"] += time.perf_counter() - env_step_start
            topology_debug["time_slots"] += 1
            time_slot_count += 1

            store_transition_start = time.perf_counter()
            agent.store_transition(
                graph,
                action,
                reward,
                value,
                log_prob,
                terminated or truncated,
                used_action_mask,
            )
            timing_stats["store_transition_seconds"] += time.perf_counter() - store_transition_start

            episode_reward += reward
            if info["valid_deployment"]:
                valid_steps += 1
                valid_costs.append(info["deployment_cost"])
                topology_debug["valid_time_slots"] += 1
            else:
                for key, value in info["failure_counts"].items():
                    failure_counts[key] += int(value)
                    topology_debug["invalid_counts_by_reason"][key] += int(value)
                topology_debug["invalid_reason_events"] += len(info["invalid_reasons"])
                if info["invalid_reasons"]:
                    first_reason = info["invalid_reasons"][0]
                    reason_counts = topology_debug["first_failing_constraint_counts"]
                    reason_counts[first_reason] = int(reason_counts.get(first_reason, 0)) + 1
                if len(first_failure_traces) < 12:
                    split_vector = action[:env.num_rhs]
                    split_summary = {f"S{i + 1}": int(np.sum(split_vector == i)) for i in range(4)}
                    first_failure_traces.append(
                        {
                            "episode": episode,
                            "time_slot": int(info["time_slot"]),
                            "topology_id": info["topology_id"],
                            "constraint_mode": info["constraint_mode"],
                            "split_summary": split_summary,
                            "es_overuse": float(info["es_overuse"]),
                            "rc_overuse": float(info["rc_overuse"]),
                            "bandwidth_overuse": float(info["bandwidth_overuse"]),
                            "e2e_violation": float(info["e2e_violation"]),
                            "crosshaul_violation": float(info["crosshaul_violation"]),
                            "invalid_reasons": list(info["invalid_reasons"]),
                        }
                    )

            for key in cost_breakdown:
                cost_breakdown[key] += float(info[key])
            for key in reconfiguration_stats:
                reconfiguration_stats[key] += float(info[key])
            for key, value in info["split_usage"].items():
                split_usage[key] += int(value)

            if terminated or truncated:
                break

        ppo_update_start = time.perf_counter()
        agent.update(batch_size=batch_size, epochs=3)
        timing_stats["ppo_update_seconds"] += time.perf_counter() - ppo_update_start

        episode_rewards.append(float(episode_reward))
        episode_costs.append(float(np.mean(valid_costs)) if valid_costs else float("inf"))
        valid_rate = valid_steps / max(time_slot_count, 1)
        valid_deployments.append(valid_rate)
        valid_time_slot_rates.append(valid_rate)
        episode_failure_counts.append(failure_counts)
        episode_cost_breakdowns.append(cost_breakdown)
        episode_reconfiguration_stats.append(
            {
                **reconfiguration_stats,
                "time_slot_count": time_slot_count,
                "time_slot_transitions": max(time_slot_count - 1, 0),
                "avg_split_changes_per_transition": float(reconfiguration_stats["split_changes"] / max(time_slot_count - 1, 1)),
                "avg_es_changes_per_transition": float(reconfiguration_stats["es_changes"] / max(time_slot_count - 1, 1)),
                "avg_rc_changes_per_transition": float(reconfiguration_stats["rc_changes"] / max(time_slot_count - 1, 1)),
                "avg_total_reconfiguration_changes_per_transition": float(
                    reconfiguration_stats["total_reconfiguration_changes"] / max(time_slot_count - 1, 1)
                ),
                "reconfiguration_cost_pct_of_total_cost": float(
                    100.0 * cost_breakdown["reconfiguration_cost"] / max(sum(cost_breakdown.values()), 1e-9)
                ),
            }
        )
        episode_split_usage.append(split_usage)

        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_costs = [cost for cost in episode_costs[-10:] if np.isfinite(cost)]
            recent_valid = valid_deployments[-10:]
            recent_topologies = episode_topology_ids[-10:]
            recent_feasible = episode_has_structurally_valid_action[-10:]
            avg_cost = float(np.mean(recent_costs)) if recent_costs else float("inf")
            unique_topologies = ",".join(sorted(set(recent_topologies)))
            tqdm.write(
                f"Episode {episode + 1:3d} | "
                f"Reward: {np.mean(recent_rewards):7.3f} | "
                f"Cost: {avg_cost:7.3f} | "
                f"Valid: {np.mean(recent_valid):.1%} | "
                f"Structurally-valid-reset: {np.mean(recent_feasible):.1%} | "
                f"Topologies: {unique_topologies}"
            )

    results = {
        "config": {
            "benchmark": config.benchmark,
            "benchmark_label": f"project_{config.benchmark}",
            "train_selection_mode": config.train_selection_mode,
            "constraint_mode": constraint_mode,
            "episode_length_time_slots": max_steps,
            "paper_aligned_episode_length": max_steps == 288,
            "paper_mode": paper_mode,
            "paper_total_timesteps": total_timesteps,
            "paper_num_envs": num_envs,
            "seed": config.seed,
            "dimensions": {
                "num_rhs": env.num_rhs,
                "num_ess": env.num_ess,
                "num_rcs": env.num_rcs,
            },
            "gnn_hidden_dim": gnn_hidden_dim,
            "gnn_input_dim": gnn_input_dim,
            "include_node_index": include_node_index,
        },
        "benchmark_audit": _collect_benchmark_audit(benchmark, env.crosshaul_latency_limits),
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
        "valid_deployments": valid_deployments,
        "episode_valid_time_slot_rates": valid_time_slot_rates,
        "episode_topology_ids": episode_topology_ids,
        "episode_topology_pools": episode_topology_pools,
        "episode_has_structurally_valid_action": episode_has_structurally_valid_action,
        "episode_has_strictly_valid_action": episode_has_strictly_valid_action,
        "episode_has_greedy_strictly_valid_action": episode_has_greedy_strictly_valid_action,
        "episode_has_exact_strictly_valid_action": episode_has_exact_strictly_valid_action,
        "episode_bounded_strict_feasibility_probe": episode_bounded_strict_feasibility_probe,
        "episode_failure_counts": episode_failure_counts,
        "episode_cost_breakdowns": episode_cost_breakdowns,
        "episode_reconfiguration_stats": episode_reconfiguration_stats,
        "episode_split_usage": episode_split_usage,
        "training_summary": {
            "timing_profile_seconds": dict(timing_stats),
            "split_usage": {
                key: int(sum(split[key] for split in episode_split_usage))
                for key in _zero_split_usage()
            },
            "avg_cost_breakdown_per_episode": {
                key: float(np.mean([breakdown[key] for breakdown in episode_cost_breakdowns]))
                for key in _zero_cost_breakdown()
            },
            "avg_reconfiguration_stats_per_episode": {
                key: float(np.mean([stats[key] for stats in episode_reconfiguration_stats]))
                for key in [
                    "split_changes",
                    "es_changes",
                    "rc_changes",
                    "total_reconfiguration_changes",
                    "avg_split_changes_per_transition",
                    "avg_es_changes_per_transition",
                    "avg_rc_changes_per_transition",
                    "avg_total_reconfiguration_changes_per_transition",
                    "reconfiguration_cost_pct_of_total_cost",
                ]
            },
            "invalid_counts_by_reason": {
                key: int(sum(counts[key] for counts in episode_failure_counts))
                for key in _zero_failure_counts()
            },
            "strict_feasible_reset_rate": float(np.mean(episode_has_strictly_valid_action)) if episode_has_strictly_valid_action else 0.0,
            "exact_strict_feasible_reset_rate": float(
                np.mean([value for value in episode_has_exact_strictly_valid_action if value is not None])
            ) if any(value is not None for value in episode_has_exact_strictly_valid_action) else None,
            "strict_feasibility_reporting": _strict_feasibility_wording(episode_has_exact_strictly_valid_action),
            "bounded_strict_probe_summary": _bounded_probe_summary(episode_bounded_strict_feasibility_probe),
            "episodes_with_no_strict_valid_action_rate": 1.0 - (
                float(np.mean(episode_has_strictly_valid_action)) if episode_has_strictly_valid_action else 0.0
            ),
            "invalid_reason_percentages": _invalid_reason_percentages(
                {
                    key: int(sum(counts[key] for counts in episode_failure_counts))
                    for key in _zero_failure_counts()
                },
                sum(int(debug["time_slots"]) for debug in training_topology_debug.values()),
            ),
        },
    }
    results["strict_mode_debug"] = {
        "first_failure_traces": first_failure_traces,
        "per_topology": {},
        "topology_latency_diagnostics": {},
    }
    for topology_id, topology_debug in training_topology_debug.items():
        step_count = max(int(topology_debug["time_slots"]), 1)
        results["strict_mode_debug"]["per_topology"][topology_id] = {
            "strict_feasible_reset_rate": float(topology_debug["strictly_valid_resets"] / max(int(topology_debug["resets"]), 1)),
            "exact_strict_feasible_reset_count": int(topology_debug["exact_strictly_valid_resets"]),
            "valid_time_slot_rate": float(topology_debug["valid_time_slots"] / step_count),
            "avg_invalid_reasons_per_time_slot": float(topology_debug["invalid_reason_events"] / step_count),
            "invalid_counts_by_reason": topology_debug["invalid_counts_by_reason"],
            "invalid_reason_percentages": _invalid_reason_percentages(
                topology_debug["invalid_counts_by_reason"],
                step_count,
            ),
            "first_failing_constraint_counts": topology_debug["first_failing_constraint_counts"],
            "bounded_probe_results": topology_debug["bounded_probe_results"],
            "bounded_probe_summary": _bounded_probe_summary(topology_debug["bounded_probe_results"]),
        }
        results["strict_mode_debug"]["topology_latency_diagnostics"][topology_id] = _collect_topology_latency_diagnostics(
            topology_id,
            env.crosshaul_latency_limits,
        )

    final_gnn_state = {key: value.detach().cpu() for key, value in gnn.state_dict().items()}
    gnn_delta_sq = 0.0
    for key, initial_value in initial_gnn_state.items():
        diff = final_gnn_state[key] - initial_value
        gnn_delta_sq += float(torch.sum(diff.float() ** 2).item())
    results["gnn_training_verification"] = {
        "gnn_state_dict_saved": True,
        "initial_param_l2_norm": _dict_l2_norm(initial_gnn_state),
        "final_param_l2_norm": _dict_l2_norm(final_gnn_state),
        "parameter_delta_l2_norm": float(gnn_delta_sq ** 0.5),
    }

    timing_stats["total_seconds"] = time.perf_counter() - train_start_time
    results["training_summary"]["timing_profile_seconds"] = dict(timing_stats)
    results_write_start = time.perf_counter()
    with results_path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)
    timing_stats["results_write_seconds"] += time.perf_counter() - results_write_start
    results["training_summary"]["timing_profile_seconds"] = dict(timing_stats)

    checkpoint_save_start = time.perf_counter()
    agent.save(
        str(checkpoint_path),
        metadata={
            "benchmark": benchmark,
            "num_rhs": env.num_rhs,
            "num_ess": env.num_ess,
            "num_rcs": env.num_rcs,
            "train_topology_pool": list(DEFAULT_BENCHMARK_TOPOLOGY_POOLS[benchmark]["train"]),
            "train_selection_mode": topology_selection_mode,
            "constraint_mode": constraint_mode,
            "episode_length_time_slots": max_steps,
            "paper_mode": paper_mode,
            "paper_total_timesteps": total_timesteps,
            "paper_num_envs": num_envs,
            "gnn_hidden_dim": gnn_hidden_dim,
            "gnn_input_dim": gnn_input_dim,
            "include_node_index": include_node_index,
            "checkpoint_family": "paper_gppo" if paper_mode else "project_gppo",
        },
    )
    timing_stats["checkpoint_save_seconds"] += time.perf_counter() - checkpoint_save_start
    results["training_summary"]["timing_profile_seconds"] = dict(timing_stats)
    with results_path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Average Reward: {np.mean(episode_rewards[-10:]):.3f}")
    finite_costs = [cost for cost in episode_costs[-10:] if np.isfinite(cost)]
    print(f"Final Average Cost: {np.mean(finite_costs):.3f}" if finite_costs else "Final Average Cost: inf")
    print(f"Final Valid Deployment Rate: {np.mean(valid_deployments[-10:]):.1%}")
    print(f"Final Topologies Seen: {sorted(set(episode_topology_ids))}")
    split_text = ", ".join(
        f"{key}={value}"
        for key, value in results["training_summary"]["split_usage"].items()
    )
    print(f"Training Split Usage: {split_text}")
    print(
        "GNN Verification: "
        f"saved={results['gnn_training_verification']['gnn_state_dict_saved']} "
        f"delta_l2={results['gnn_training_verification']['parameter_delta_l2_norm']:.6f}"
    )
    reporting = results["training_summary"]["strict_feasibility_reporting"]
    print(f"{reporting['rate_label']}: {results['training_summary']['strict_feasible_reset_rate']:.1%}")
    exact_rate = results["training_summary"]["exact_strict_feasible_reset_rate"]
    if exact_rate is not None:
        print(f"Exact Strict-Feasible Reset Rate: {exact_rate:.1%}")
    print(
        f"{reporting['no_action_label']}: "
        f"{results['training_summary']['episodes_with_no_strict_valid_action_rate']:.1%}"
    )
    invalid_text = ", ".join(
        f"{key}={value}"
        for key, value in results["training_summary"]["invalid_counts_by_reason"].items()
        if value
    ) or "none"
    print(f"Invalid Counts By Reason: {invalid_text}")
    invalid_pct_text = ", ".join(
        f"{key}={value:.1%}"
        for key, value in results["training_summary"]["invalid_reason_percentages"].items()
        if value
    ) or "none"
    print(f"Invalid Reason Percent Of Time Slots: {invalid_pct_text}")
    reconfig_summary = results["training_summary"]["avg_reconfiguration_stats_per_episode"]
    print(
        "Average Consecutive-Slot Reconfiguration: "
        f"split={reconfig_summary['avg_split_changes_per_transition']:.3f}, "
        f"es={reconfig_summary['avg_es_changes_per_transition']:.3f}, "
        f"rc={reconfig_summary['avg_rc_changes_per_transition']:.3f}, "
        f"total={reconfig_summary['avg_total_reconfiguration_changes_per_transition']:.3f}"
    )
    print(
        "Reconfiguration Cost Share Of Total Cost: "
        f"{reconfig_summary['reconfiguration_cost_pct_of_total_cost']:.2f}%"
    )
    timing_summary = results["training_summary"]["timing_profile_seconds"]
    print(
        "Timing Profile (s): "
        f"total={timing_summary['total_seconds']:.3f}, "
        f"reset={timing_summary['reset_seconds']:.3f}, "
        f"graph={timing_summary['graph_build_seconds']:.3f}, "
        f"gnn={timing_summary['gnn_forward_seconds']:.3f}, "
        f"action={timing_summary['action_selection_seconds']:.3f}, "
        f"env_step={timing_summary['env_step_seconds']:.3f}, "
        f"ppo_update={timing_summary['ppo_update_seconds']:.3f}"
    )
    for topology_id, latency_debug in results["strict_mode_debug"]["topology_latency_diagnostics"].items():
        impossible_splits = latency_debug["splits_ruled_out_by_link_delay"]
        if impossible_splits:
            print(f"{topology_id} strict link-delay rules out splits: {', '.join(impossible_splits)}")
    print("=" * 60)

    return agent, gnn, results


def evaluate_gppo(
    agent: PPOAgent,
    gnn: torch.nn.Module,
    num_episodes: int = 10,
    benchmark: str = "small",
    device: str = "cpu",
    topology_pool_name: str = "test",
    topology_selection_mode: str = "fixed",
    constraint_mode: str = "legacy",
    topology_id: Optional[str] = None,
    max_steps: int = 50,
    export_episode_traces: bool = True,
    verbose: bool = True,
    csv_output_dir: Optional[Path] = None,
    trace_output_dir: Optional[Path] = None,
    include_node_index: bool = False,
) -> Dict[str, object]:
    env = _build_env(
        benchmark=benchmark,
        max_steps=max_steps,
        topology_pool_name=topology_pool_name,
        topology_selection_mode=topology_selection_mode,
        constraint_mode=constraint_mode,
        topology_id=topology_id,
    )
    graph_builder = ORANGraphBuilder(env.num_rhs, env.num_ess, env.num_rcs, include_node_index=include_node_index)

    rewards = []
    costs = []
    topology_ids = []
    has_structurally_valid_action = []
    has_strictly_valid_action = []
    has_greedy_strictly_valid_action = []
    has_exact_strictly_valid_action = []
    bounded_strict_feasibility_probe = []
    episode_reconfiguration_stats = []
    episode_trace_paths = []
    episode_trace_csv_paths = []
    per_topology: Dict[str, Dict[str, object]] = {}
    invalid_counts_by_reason = _zero_failure_counts()
    latency_diagnostics = _collect_topology_latency_diagnostics(env.topology_id, env.crosshaul_latency_limits)
    first_failing_constraint_counts: Dict[str, int] = {}

    for episode in range(num_episodes):
        _, reset_info = env.reset(
            seed=10_000 + episode,
            options={
                "topology_pool_name": topology_pool_name,
                "topology_selection_mode": topology_selection_mode,
                "topology_id": topology_id,
            },
        )
        episode_reward = 0.0
        episode_costs = []
        episode_cost_breakdown = _zero_cost_breakdown()
        topology_ids.append(reset_info["topology_id"])
        has_structurally_valid_action.append(bool(reset_info.get("has_structurally_valid_action", False)))
        has_strictly_valid_action.append(bool(reset_info.get("has_strictly_valid_action", False)))
        has_greedy_strictly_valid_action.append(bool(reset_info.get("has_greedy_strictly_valid_action", False)))
        has_exact_strictly_valid_action.append(reset_info.get("has_exact_strictly_valid_action"))
        bounded_strict_feasibility_probe.append(reset_info.get("bounded_strict_feasibility_probe", {}))
        topology_summary = per_topology.setdefault(
            reset_info["topology_id"],
            {
                "rewards": [],
                "costs": [],
                "valid_steps": 0,
                "total_time_slots": 0,
                "resets": 0,
                "structurally_valid_resets": 0,
                "strictly_valid_resets": 0,
                "exact_strictly_valid_resets": 0,
                "invalid_reason_events": 0,
                "split_usage": _zero_split_usage(),
                "cost_breakdown": _zero_cost_breakdown(),
                "reconfiguration_stats": _zero_reconfiguration_stats(),
                "invalid_counts_by_reason": _zero_failure_counts(),
                "first_failing_constraint_counts": {},
                "bounded_probe_results": [],
                "topology_metadata": reset_info.get("topology_metadata", {}),
                "latency_diagnostics": _collect_topology_latency_diagnostics(
                    reset_info["topology_id"],
                    env.crosshaul_latency_limits,
                ),
            },
        )
        topology_summary["resets"] += 1
        topology_summary["structurally_valid_resets"] += int(bool(reset_info.get("has_structurally_valid_action", False)))
        topology_summary["strictly_valid_resets"] += int(bool(reset_info.get("has_strictly_valid_action", False)))
        if reset_info.get("has_exact_strictly_valid_action") is True:
            topology_summary["exact_strictly_valid_resets"] += 1
        topology_summary["bounded_probe_results"].append(reset_info.get("bounded_strict_feasibility_probe", {}))

        slot_reconfiguration_stats = _zero_reconfiguration_stats()
        slot_count = 0
        episode_slots: List[Dict[str, object]] = []
        for _ in range(max_steps):
            adjacency, edge_features, _ = env._get_adjacency_info()
            graph = graph_builder.build_graph(
                env.rh_demands,
                env.rh_latencies,
                env.es_remaining,
                env.rc_remaining,
                adjacency,
                edge_features,
            ).to(device)

            with torch.no_grad():
                features = gnn(graph)

            action_mask = env.get_action_mask()
            action, _, _, _ = agent.select_action_sequential(
                features.squeeze(0),
                action_mask,
                env.get_conditional_rc_mask,
                deterministic=True,
            )
            _, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            topology_summary["total_time_slots"] += 1
            slot_count += 1
            if info["valid_deployment"]:
                topology_summary["valid_steps"] += 1

            if info["valid_deployment"]:
                episode_costs.append(info["deployment_cost"])
            else:
                for key, value in info["failure_counts"].items():
                    invalid_counts_by_reason[key] += int(value)
                    topology_summary["invalid_counts_by_reason"][key] += int(value)
                topology_summary["invalid_reason_events"] += len(info["invalid_reasons"])
                if info["invalid_reasons"]:
                    first_reason = info["invalid_reasons"][0]
                    first_failing_constraint_counts[first_reason] = int(first_failing_constraint_counts.get(first_reason, 0)) + 1
                    topology_counts = topology_summary["first_failing_constraint_counts"]
                    topology_counts[first_reason] = int(topology_counts.get(first_reason, 0)) + 1
            for key in topology_summary["cost_breakdown"]:
                topology_summary["cost_breakdown"][key] += float(info[key])
                episode_cost_breakdown[key] += float(info[key])
            for key in topology_summary["reconfiguration_stats"]:
                topology_summary["reconfiguration_stats"][key] += float(info[key])
                slot_reconfiguration_stats[key] += float(info[key])
            for key, value in info["split_usage"].items():
                topology_summary["split_usage"][key] += int(value)
            splits, es_choices, rc_choices = env._split_action(action)
            episode_slots.append(
                {
                    "time_slot": int(info["time_slot"]),
                    "topology_id": info["topology_id"],
                    "split_vector": [int(value) for value in splits.tolist()],
                    "es_choice_vector": [int(value) for value in es_choices.tolist()],
                    "rc_choice_vector": [int(value) for value in rc_choices.tolist()],
                    "valid_deployment": bool(info["valid_deployment"]),
                    "deployment_cost": float(info["deployment_cost"]) if np.isfinite(info["deployment_cost"]) else "inf",
                    "reward": float(reward),
                    "raw_total_cost": float(info["raw_total_cost"]),
                    "processing_cost": float(info["processing_cost"]),
                    "routing_cost": float(info["routing_cost"]),
                    "reconfiguration_cost": float(info["reconfiguration_cost"]),
                    "sla_penalty": float(info["sla_penalty"]),
                    "es_overuse": float(info["es_overuse"]),
                    "rc_overuse": float(info["rc_overuse"]),
                    "bandwidth_overuse": float(info["bandwidth_overuse"]),
                    "e2e_violation": float(info["e2e_violation"]),
                    "crosshaul_violation": float(info["crosshaul_violation"]),
                    "failed_links": int(info["failed_links"]),
                    "split_changes": int(info["split_changes"]),
                    "es_changes": int(info["es_changes"]),
                    "rc_changes": int(info["rc_changes"]),
                    "total_reconfiguration_changes": int(info["total_reconfiguration_changes"]),
                    "split_usage": {key: int(value) for key, value in info["split_usage"].items()},
                    "invalid_reasons": list(info["invalid_reasons"]),
                    "failure_counts": {key: int(value) for key, value in info["failure_counts"].items()},
                }
            )

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        costs.append(float(np.mean(episode_costs)) if episode_costs else float("inf"))
        topology_summary["rewards"].append(float(episode_reward))
        topology_summary["costs"].append(float(np.mean(episode_costs)) if episode_costs else float("inf"))
        episode_reconfiguration_stats.append(
            {
                **slot_reconfiguration_stats,
                "time_slot_count": slot_count,
                "time_slot_transitions": max(slot_count - 1, 0),
                "avg_split_changes_per_transition": float(slot_reconfiguration_stats["split_changes"] / max(slot_count - 1, 1)),
                "avg_es_changes_per_transition": float(slot_reconfiguration_stats["es_changes"] / max(slot_count - 1, 1)),
                "avg_rc_changes_per_transition": float(slot_reconfiguration_stats["rc_changes"] / max(slot_count - 1, 1)),
                "avg_total_reconfiguration_changes_per_transition": float(
                    slot_reconfiguration_stats["total_reconfiguration_changes"] / max(slot_count - 1, 1)
                ),
                "reconfiguration_cost_pct_of_total_cost": float(
                    100.0 * episode_cost_breakdown["reconfiguration_cost"] /
                    max(sum(episode_cost_breakdown.values()), 1e-9)
                ),
            }
        )
        if export_episode_traces:
            trace_path = _episode_trace_path(
                trace_output_dir=trace_output_dir or resolve_episode_traces_dir(DEFAULT_RESULTS_PATH),
                topology_pool_name=topology_pool_name,
                topology_id=reset_info["topology_id"],
                constraint_mode=constraint_mode,
                episode_index=episode,
            )
            _write_episode_trace(
                trace_path=trace_path,
                benchmark=benchmark,
                topology_pool_name=topology_pool_name,
                topology_id=reset_info["topology_id"],
                constraint_mode=constraint_mode,
                episode_index=episode,
                episode_length_time_slots=max_steps,
                reset_info={
                    "time_slot": int(reset_info.get("time_slot", 0)),
                    "episode_length_time_slots": int(reset_info.get("episode_length_time_slots", max_steps)),
                    "has_structurally_valid_action": bool(reset_info.get("has_structurally_valid_action", False)),
                    "has_strictly_valid_action": bool(reset_info.get("has_strictly_valid_action", False)),
                    "has_greedy_strictly_valid_action": bool(reset_info.get("has_greedy_strictly_valid_action", False)),
                    "has_exact_strictly_valid_action": reset_info.get("has_exact_strictly_valid_action"),
                    "bounded_strict_feasibility_probe": reset_info.get("bounded_strict_feasibility_probe", {}),
                    "topology_metadata": reset_info.get("topology_metadata", {}),
                },
                slots=episode_slots,
                episode_reward=float(episode_reward),
            )
            if csv_output_dir is not None:
                trace_csv_path = _write_episode_trace_csv(
                    trace_path=trace_path,
                    csv_output_dir=csv_output_dir,
                    benchmark=benchmark,
                    topology_pool_name=topology_pool_name,
                    constraint_mode=constraint_mode,
                    episode_index=episode,
                    slots=episode_slots,
                )
                episode_trace_csv_paths.append(str(trace_csv_path))
            episode_trace_paths.append(str(trace_path))

    finite_costs = [cost for cost in costs if np.isfinite(cost)]
    summary = {
        "benchmark": benchmark,
        "topology_pool_name": topology_pool_name,
        "selection_mode": topology_selection_mode,
        "constraint_mode": constraint_mode,
        "episode_length_time_slots": max_steps,
        "topology_ids": topology_ids,
        "has_structurally_valid_action": has_structurally_valid_action,
        "has_strictly_valid_action": has_strictly_valid_action,
        "has_greedy_strictly_valid_action": has_greedy_strictly_valid_action,
        "has_exact_strictly_valid_action": has_exact_strictly_valid_action,
        "bounded_strict_feasibility_probe": bounded_strict_feasibility_probe,
        "episode_trace_paths": episode_trace_paths,
        "episode_trace_csv_paths": episode_trace_csv_paths,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_cost": float(np.mean(finite_costs)) if finite_costs else float("inf"),
        "std_cost": float(np.std(finite_costs)) if finite_costs else float("inf"),
        "valid_rate": float(np.mean([row["valid_steps"] / max(row["total_time_slots"], 1) for row in per_topology.values()])) if per_topology else 0.0,
        "strict_feasible_reset_rate": float(np.mean(has_strictly_valid_action)) if has_strictly_valid_action else 0.0,
        "exact_strict_feasible_reset_rate": float(
            np.mean([value for value in has_exact_strictly_valid_action if value is not None])
        ) if any(value is not None for value in has_exact_strictly_valid_action) else None,
        "strict_feasibility_reporting": _strict_feasibility_wording(has_exact_strictly_valid_action),
        "bounded_strict_probe_summary": _bounded_probe_summary(bounded_strict_feasibility_probe),
        "invalid_counts_by_reason": invalid_counts_by_reason,
        "invalid_reason_percentages": _invalid_reason_percentages(
            invalid_counts_by_reason,
            sum(int(row["total_time_slots"]) for row in per_topology.values()),
        ),
        "avg_reconfiguration_stats_per_episode": {
            key: float(np.mean([stats[key] for stats in episode_reconfiguration_stats]))
            for key in [
                "split_changes",
                "es_changes",
                "rc_changes",
                "total_reconfiguration_changes",
                "avg_split_changes_per_transition",
                "avg_es_changes_per_transition",
                "avg_rc_changes_per_transition",
                "avg_total_reconfiguration_changes_per_transition",
                "reconfiguration_cost_pct_of_total_cost",
            ]
        },
        "first_failing_constraint_counts": first_failing_constraint_counts,
        "episodes_with_no_strict_valid_action_rate": 1.0 - (
            float(np.mean(has_strictly_valid_action)) if has_strictly_valid_action else 0.0
        ),
        "latency_diagnostics": latency_diagnostics,
        "per_topology": {},
    }

    for topology_id, topology_summary in per_topology.items():
        topology_costs = [cost for cost in topology_summary["costs"] if np.isfinite(cost)]
        total_steps = max(int(topology_summary["total_time_slots"]), 1)
        summary["per_topology"][topology_id] = {
            "mean_reward": float(np.mean(topology_summary["rewards"])),
            "std_reward": float(np.std(topology_summary["rewards"])),
            "mean_cost": float(np.mean(topology_costs)) if topology_costs else float("inf"),
            "std_cost": float(np.std(topology_costs)) if topology_costs else float("inf"),
            "validity_rate": float(topology_summary["valid_steps"] / total_steps),
            "structural_validity_rate_at_reset": float(
                topology_summary["structurally_valid_resets"] / max(int(topology_summary["resets"]), 1)
            ),
            "strict_feasible_reset_rate": float(topology_summary["strictly_valid_resets"] / max(int(topology_summary["resets"]), 1)),
            "exact_strict_feasible_reset_count": int(topology_summary["exact_strictly_valid_resets"]),
            "avg_invalid_reasons_per_time_slot": float(topology_summary["invalid_reason_events"] / total_steps),
            "sla_penalty": float(topology_summary["cost_breakdown"]["sla_penalty"] / total_steps),
            "invalid_counts_by_reason": topology_summary["invalid_counts_by_reason"],
            "invalid_reason_percentages": _invalid_reason_percentages(
                topology_summary["invalid_counts_by_reason"],
                total_steps,
            ),
            "first_failing_constraint_counts": topology_summary["first_failing_constraint_counts"],
            "bounded_probe_results": topology_summary["bounded_probe_results"],
            "bounded_probe_summary": _bounded_probe_summary(topology_summary["bounded_probe_results"]),
            "split_distribution": {
                key: float(value / max(sum(topology_summary["split_usage"].values()), 1))
                for key, value in topology_summary["split_usage"].items()
            },
            "cost_breakdown": {
                key: float(value / total_steps)
                for key, value in topology_summary["cost_breakdown"].items()
            },
            "reconfiguration_stats": {
                **{
                    key: float(value / max(int(topology_summary["resets"]), 1))
                    for key, value in topology_summary["reconfiguration_stats"].items()
                },
                "avg_split_changes_per_transition": float(
                    topology_summary["reconfiguration_stats"]["split_changes"] /
                    max(int(topology_summary["total_time_slots"]) - int(topology_summary["resets"]), 1)
                ),
                "avg_es_changes_per_transition": float(
                    topology_summary["reconfiguration_stats"]["es_changes"] /
                    max(int(topology_summary["total_time_slots"]) - int(topology_summary["resets"]), 1)
                ),
                "avg_rc_changes_per_transition": float(
                    topology_summary["reconfiguration_stats"]["rc_changes"] /
                    max(int(topology_summary["total_time_slots"]) - int(topology_summary["resets"]), 1)
                ),
                "avg_total_reconfiguration_changes_per_transition": float(
                    topology_summary["reconfiguration_stats"]["total_reconfiguration_changes"] /
                    max(int(topology_summary["total_time_slots"]) - int(topology_summary["resets"]), 1)
                ),
                "reconfiguration_cost_pct_of_total_cost": float(
                    100.0 * topology_summary["cost_breakdown"]["reconfiguration_cost"] /
                    max(sum(topology_summary["cost_breakdown"].values()), 1e-9)
                ),
            },
            "topology_metadata": topology_summary["topology_metadata"],
            "latency_diagnostics": topology_summary["latency_diagnostics"],
        }

    if verbose:
        print("\n" + "=" * 60)
        print(f"Evaluation Results ({topology_pool_name} pool)")
        print("=" * 60)
        print(f"Project Benchmark: {benchmark}")
        print(f"Constraint Mode: {constraint_mode}")
        print(f"Time Slots Per Episode: {max_steps}")
        print(f"Topologies: {sorted(set(topology_ids))}")
        print(f"Structural Validity Rate At Reset: {np.mean(has_structurally_valid_action):.1%}")
        print(f"{summary['strict_feasibility_reporting']['rate_label']}: {summary['strict_feasible_reset_rate']:.1%}")
        if summary["exact_strict_feasible_reset_rate"] is not None:
            print(f"Exact Strict-Feasible Rate At Reset: {summary['exact_strict_feasible_reset_rate']:.1%}")
        print(f"{summary['strict_feasibility_reporting']['no_action_label']}: {summary['episodes_with_no_strict_valid_action_rate']:.1%}")
        print(f"Valid Deployment Rate: {summary['valid_rate']:.1%}")
        print(f"Average Reward: {summary['avg_reward']:.3f} ± {summary['std_reward']:.3f}")
        if finite_costs:
            print(f"Average Cost: {summary['avg_cost']:.3f} ± {summary['std_cost']:.3f}")
        else:
            print("Average Cost: inf")
        reconfig_summary = summary["avg_reconfiguration_stats_per_episode"]
        print(
            "Average Consecutive-Slot Reconfiguration: "
            f"split={reconfig_summary['avg_split_changes_per_transition']:.3f}, "
            f"es={reconfig_summary['avg_es_changes_per_transition']:.3f}, "
            f"rc={reconfig_summary['avg_rc_changes_per_transition']:.3f}, "
            f"total={reconfig_summary['avg_total_reconfiguration_changes_per_transition']:.3f}"
        )
        print(
            "Reconfiguration Cost Share Of Total Cost: "
            f"{reconfig_summary['reconfiguration_cost_pct_of_total_cost']:.2f}%"
        )
        for topology_id, topology_summary in summary["per_topology"].items():
            split_text = ", ".join(f"{key}={value:.1%}" for key, value in topology_summary["split_distribution"].items())
            print(
                f"{topology_id} | reward={topology_summary['mean_reward']:.3f} "
                f"cost={topology_summary['mean_cost']:.3f} "
                f"valid={topology_summary['validity_rate']:.1%} "
                f"strict_reset={topology_summary['strict_feasible_reset_rate']:.1%} "
                f"sla={topology_summary['sla_penalty']:.3f} | {split_text}"
            )
            impossible_splits = topology_summary["latency_diagnostics"]["splits_ruled_out_by_link_delay"]
            if impossible_splits:
                print(f"  link-delay rules out strict splits: {', '.join(impossible_splits)}")
        print("=" * 60)
    return summary


def evaluate_pool_by_topology(
    agent: PPOAgent,
    gnn: torch.nn.Module,
    *,
    benchmark: str,
    topology_pool_name: str,
    num_episodes: int,
    device: str,
    constraint_mode: str,
    max_steps: int,
    csv_output_dir: Optional[Path] = None,
    trace_output_dir: Optional[Path] = None,
    include_node_index: bool = False,
) -> Dict[str, object]:
    topology_ids = DEFAULT_BENCHMARK_TOPOLOGY_POOLS[benchmark][topology_pool_name]
    summaries = {}
    for topology_id in topology_ids:
        summaries[topology_id] = evaluate_gppo(
            agent,
            gnn,
            num_episodes=num_episodes,
            benchmark=benchmark,
            device=device,
            topology_pool_name=topology_pool_name,
            topology_selection_mode="fixed",
            constraint_mode=constraint_mode,
            topology_id=topology_id,
            max_steps=max_steps,
            csv_output_dir=csv_output_dir,
            trace_output_dir=trace_output_dir,
            include_node_index=include_node_index,
        )
    return summaries


def compare_constraint_modes_on_pool(
    agent: PPOAgent,
    gnn: torch.nn.Module,
    *,
    benchmark: str,
    topology_pool_name: str,
    num_episodes: int,
    device: str,
    max_steps: int,
    include_node_index: bool = False,
) -> Dict[str, object]:
    staged_modes = [
        "legacy",
        "strict_connectivity_only",
        "strict_connectivity_plus_capacity",
        "strict_connectivity_plus_capacity_plus_bandwidth",
        "strict_full",
    ]
    comparisons = {}
    for topology_id in DEFAULT_BENCHMARK_TOPOLOGY_POOLS[benchmark][topology_pool_name]:
        mode_summaries = {}
        for mode in staged_modes:
            mode_summaries[mode] = evaluate_gppo(
                agent,
                gnn,
                num_episodes=num_episodes,
                benchmark=benchmark,
                device=device,
                topology_pool_name=topology_pool_name,
                topology_selection_mode="fixed",
                constraint_mode=mode,
                topology_id=topology_id,
                max_steps=max_steps,
                export_episode_traces=False,
                verbose=False,
                include_node_index=include_node_index,
            )
        legacy = mode_summaries["legacy"]
        strict = mode_summaries["strict_full"]
        first_failing_constraint_class = None
        for mode in staged_modes[1:]:
            if mode_summaries[mode]["valid_rate"] < 1.0:
                first_failing_constraint_class = mode
                break
        comparisons[topology_id] = {
            "mode_summaries": {
                mode: {
                    "valid_rate": summary["valid_rate"],
                    "avg_cost": summary["avg_cost"],
                    "avg_reward": summary["avg_reward"],
                    "strict_feasible_reset_rate": summary["strict_feasible_reset_rate"],
                    "exact_strict_feasible_reset_rate": summary["exact_strict_feasible_reset_rate"],
                    "invalid_reason_percentages": summary["invalid_reason_percentages"],
                    "first_failing_constraint_counts": summary["first_failing_constraint_counts"],
                    "strict_feasibility_reporting": summary["strict_feasibility_reporting"],
                }
                for mode, summary in mode_summaries.items()
            },
            "legacy": {
                "valid_rate": legacy["valid_rate"],
                "avg_cost": legacy["avg_cost"],
                "avg_reward": legacy["avg_reward"],
            },
            "strict": {
                "valid_rate": strict["valid_rate"],
                "avg_cost": strict["avg_cost"],
                "avg_reward": strict["avg_reward"],
                "strict_feasible_reset_rate": strict["strict_feasible_reset_rate"],
                "exact_strict_feasible_reset_rate": strict["exact_strict_feasible_reset_rate"],
                "episodes_with_no_strict_valid_action_rate": strict["episodes_with_no_strict_valid_action_rate"],
                "first_failing_constraint_counts": strict["first_failing_constraint_counts"],
                "invalid_reason_percentages": strict["invalid_reason_percentages"],
                "strict_feasibility_reporting": strict["strict_feasibility_reporting"],
            },
            "first_failing_constraint_class": first_failing_constraint_class,
            "no_strict_valid_reset_count": int(
                sum(1 for value in strict["has_strictly_valid_action"] if not value)
            ),
            "valid_rate_gap": legacy["valid_rate"] - strict["valid_rate"],
        }
    return comparisons


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GPPO for O-RAN")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--benchmark", choices=sorted(DEFAULT_BENCHMARK_TOPOLOGY_POOLS.keys()), default="small", help="Benchmark group")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=50, help="Time slots per episode")
    parser.add_argument("--paper-episode-length", action="store_true", help="Use the paper-aligned 288 time-slot episode horizon")
    parser.add_argument("--paper-mode", action="store_true", help="Run the paper-faithful preset without removing project/debug modes")
    parser.add_argument("--paper-timesteps", type=int, default=PAPER_TIMESTEPS, help="Paper-mode timesteps per seed")
    parser.add_argument("--paper-num-envs", type=int, default=PAPER_NUM_ENVS, help="Paper-mode synchronous parallel environments")
    parser.add_argument("--paper-num-seeds", type=int, default=PAPER_NUM_SEEDS, help="Paper-mode random seeds for aggregate reporting")
    parser.add_argument("--paper-gnn-hidden-dim", type=int, default=PAPER_GNN_HIDDEN_DIM, help="Paper-mode GNN hidden size")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO batch size")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RUN_DIR, help="Run directory or output JSON path")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Checkpoint file path or directory")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training evaluation")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Number of evaluation episodes per pool")
    parser.add_argument(
        "--topology-selection-mode",
        choices=["fixed", "random_per_reset"],
        default="random_per_reset",
        help="Train topology selection mode",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=[
            "legacy",
            "strict",
            "strict_connectivity_only",
            "strict_connectivity_plus_capacity",
            "strict_connectivity_plus_capacity_plus_bandwidth",
            "strict_full",
        ],
        default="legacy",
        help="Constraint handling mode",
    )
    parser.add_argument("--train-topology-id", type=str, default=None, help="Force a specific train-pool topology")
    return parser


def run_training_from_args(args: argparse.Namespace) -> None:
    resolved_results_path = resolve_results_path(args.results_path)
    resolved_checkpoint_path = resolve_checkpoint_path(args.results_path, args.checkpoint_path)
    resolved_trace_output_dir = resolve_episode_traces_dir(args.results_path)

    if args.paper_mode:
        args.results_path = resolved_results_path
        args.checkpoint_path = resolved_checkpoint_path
        _run_paper_mode_from_args(args)
        return

    effective_max_steps = 288 if args.paper_episode_length else args.max_steps
    agent, gnn, results = train_gppo(
        num_episodes=args.episodes,
        max_steps=effective_max_steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        results_path=resolved_results_path,
        checkpoint_path=resolved_checkpoint_path,
        benchmark=args.benchmark,
        topology_selection_mode=args.topology_selection_mode,
        constraint_mode=args.constraint_mode,
        train_topology_id=args.train_topology_id,
    )
    if not args.skip_eval:
        evaluation_start = time.perf_counter()
        train_eval = evaluate_pool_by_topology(
            agent,
            gnn,
            benchmark=args.benchmark,
            topology_pool_name="train",
            num_episodes=args.eval_episodes,
            device=args.device,
            constraint_mode=args.constraint_mode,
            max_steps=effective_max_steps,
            csv_output_dir=resolved_results_path.parent,
            trace_output_dir=resolved_trace_output_dir,
        )
        train_eval_done = time.perf_counter()
        test_eval = evaluate_pool_by_topology(
            agent,
            gnn,
            benchmark=args.benchmark,
            topology_pool_name="test",
            num_episodes=args.eval_episodes,
            device=args.device,
            constraint_mode=args.constraint_mode,
            max_steps=effective_max_steps,
            csv_output_dir=resolved_results_path.parent,
            trace_output_dir=resolved_trace_output_dir,
        )
        eval_done = time.perf_counter()
        results["evaluation"] = {
            "train_pool": train_eval,
            "test_pool": test_eval,
        }
        results["training_summary"]["timing_profile_seconds"]["evaluation_seconds"] = eval_done - evaluation_start
        results["constraint_mode_comparison"] = {
            "train_pool": compare_constraint_modes_on_pool(
                agent,
                gnn,
                benchmark=args.benchmark,
                topology_pool_name="train",
                num_episodes=args.eval_episodes,
                device=args.device,
                max_steps=effective_max_steps,
            ),
            "test_pool": compare_constraint_modes_on_pool(
                agent,
                gnn,
                benchmark=args.benchmark,
                topology_pool_name="test",
                num_episodes=args.eval_episodes,
                device=args.device,
                max_steps=effective_max_steps,
            ),
        }
        comparison_done = time.perf_counter()
        results["training_summary"]["timing_profile_seconds"]["comparison_seconds"] = comparison_done - eval_done
        results["training_summary"]["timing_profile_seconds"]["evaluation_train_pool_seconds"] = train_eval_done - evaluation_start
        results["training_summary"]["timing_profile_seconds"]["evaluation_test_pool_seconds"] = eval_done - train_eval_done
        print("\n" + "=" * 60)
        print("Staged Constraint Comparison")
        print("=" * 60)
        for pool_name, pool_results in results["constraint_mode_comparison"].items():
            print(f"{pool_name}:")
            for topology_id, comparison in pool_results.items():
                print(
                    f"  {topology_id} | "
                    f"legacy_valid={comparison['legacy']['valid_rate']:.1%} "
                    f"strict_valid={comparison['strict']['valid_rate']:.1%} "
                    f"gap={comparison['valid_rate_gap']:.1%} "
                    f"strict_no_action={comparison['strict']['episodes_with_no_strict_valid_action_rate']:.1%} "
                    f"first_fail={comparison['first_failing_constraint_class'] or 'none'}"
                )
                if args.benchmark == "large":
                    for mode_name, mode_summary in comparison["mode_summaries"].items():
                        invalid_text = ", ".join(
                            f"{key}={value:.0%}"
                            for key, value in mode_summary["invalid_reason_percentages"].items()
                            if value
                        ) or "none"
                        print(
                            f"    {mode_name}: "
                            f"valid={mode_summary['valid_rate']:.1%} "
                            f"reward={mode_summary['avg_reward']:.3f} "
                            f"cost={mode_summary['avg_cost']:.3f} "
                            f"invalid={invalid_text}"
                        )
    csv_paths = _export_csv_artifacts(results, resolved_results_path)
    trace_csv_paths = []
    for pool_results in results.get("evaluation", {}).values():
        for topology_eval in pool_results.values():
            trace_csv_paths.extend(topology_eval.get("episode_trace_csv_paths", []))
    if trace_csv_paths:
        csv_paths["episode_trace_csvs"] = trace_csv_paths
    results["csv_exports"] = csv_paths

    with resolved_results_path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)

    print("\nCSV Exports")
    for name, value in results["csv_exports"].items():
        if isinstance(value, list):
            print(f"{name}:")
            for item in value:
                print(f"  {item}")
        else:
            print(f"{name}: {value}")


def main(argv=None) -> None:
    parser = build_train_parser()
    args = parser.parse_args(argv)
    run_training_from_args(args)
