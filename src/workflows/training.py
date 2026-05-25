import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.common.paths import DEFAULT_CHECKPOINT_PATH, DEFAULT_RESULTS_PATH, ensure_output_dirs
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
        "steps": 0,
        "valid_steps": 0,
        "invalid_reason_events": 0,
        "invalid_counts_by_reason": _zero_failure_counts(),
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
    for split_idx, limit in enumerate(crosshaul_latency_limits):
        split_key = f"S{split_idx + 1}"
        if split_idx == 3:
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
        "split_latency_feasibility": split_latency,
        "splits_ruled_out_by_link_delay": impossible_splits,
    }


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
) -> Tuple[PPOAgent, torch.nn.Module, dict]:
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
    gnn = GNNFeatureExtractor(input_dim=6, hidden_dim=64, output_dim=128).to(device)
    graph_builder = ORANGraphBuilder(env.num_rhs, env.num_ess, env.num_rcs)
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
    print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
    print(f"Device: {device}")
    print(f"Results: {results_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    episode_rewards = []
    episode_costs = []
    valid_deployments = []
    episode_topology_ids = []
    episode_topology_pools = []
    episode_has_structurally_valid_action = []
    episode_has_strictly_valid_action = []
    episode_failure_counts = []
    episode_cost_breakdowns = []
    episode_split_usage = []
    training_topology_debug: Dict[str, Dict[str, object]] = {}
    first_failure_traces = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        _, reset_info = env.reset(
            seed=seed + episode,
            options={
                "topology_pool_name": "train",
                "topology_selection_mode": topology_selection_mode,
                "topology_id": train_topology_id,
            },
        )
        episode_topology_ids.append(reset_info["topology_id"])
        episode_topology_pools.append(reset_info["topology_pool"])
        episode_has_structurally_valid_action.append(bool(reset_info.get("has_structurally_valid_action", False)))
        episode_has_strictly_valid_action.append(bool(reset_info.get("has_strictly_valid_action", False)))
        topology_debug = training_topology_debug.setdefault(reset_info["topology_id"], _zero_topology_debug())
        topology_debug["resets"] += 1
        topology_debug["strictly_valid_resets"] += int(bool(reset_info.get("has_strictly_valid_action", False)))
        episode_reward = 0.0
        valid_costs = []
        valid_steps = 0
        failure_counts = _zero_failure_counts()
        cost_breakdown = _zero_cost_breakdown()
        split_usage = _zero_split_usage()

        for _ in range(max_steps):
            adjacency, edge_features, _ = env._get_adjacency_info()
            graph = graph_builder.build_graph(
                env.rh_demands,
                env.rh_latencies,
                env.es_remaining,
                env.rc_remaining,
                adjacency,
                edge_features,
            )

            features = gnn(graph.to(device))

            action_mask = env.get_action_mask()
            action, log_prob, value, used_action_mask = agent.select_action_sequential(
                features.squeeze(0),
                action_mask,
                env.get_conditional_rc_mask,
            )
            _, reward, terminated, truncated, info = env.step(action)
            topology_debug["steps"] += 1

            agent.store_transition(
                graph,
                action,
                reward,
                value,
                log_prob,
                terminated or truncated,
                used_action_mask,
            )

            episode_reward += reward
            if info["valid_deployment"]:
                valid_steps += 1
                valid_costs.append(info["deployment_cost"])
                topology_debug["valid_steps"] += 1
            else:
                for key, value in info["failure_counts"].items():
                    failure_counts[key] += int(value)
                    topology_debug["invalid_counts_by_reason"][key] += int(value)
                topology_debug["invalid_reason_events"] += len(info["invalid_reasons"])
                if len(first_failure_traces) < 12:
                    split_vector = action[:env.num_rhs]
                    split_summary = {f"S{i + 1}": int(np.sum(split_vector == i)) for i in range(4)}
                    first_failure_traces.append(
                        {
                            "episode": episode,
                            "step": int(topology_debug["steps"]),
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
            for key, value in info["split_usage"].items():
                split_usage[key] += int(value)

            if terminated or truncated:
                break

        agent.update(batch_size=batch_size, epochs=3)

        episode_rewards.append(float(episode_reward))
        episode_costs.append(float(np.mean(valid_costs)) if valid_costs else float("inf"))
        valid_deployments.append(valid_steps / max_steps)
        episode_failure_counts.append(failure_counts)
        episode_cost_breakdowns.append(cost_breakdown)
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
            "seed": config.seed,
            "dimensions": {
                "num_rhs": env.num_rhs,
                "num_ess": env.num_ess,
                "num_rcs": env.num_rcs,
            },
        },
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
        "valid_deployments": valid_deployments,
        "episode_topology_ids": episode_topology_ids,
        "episode_topology_pools": episode_topology_pools,
        "episode_has_structurally_valid_action": episode_has_structurally_valid_action,
        "episode_has_strictly_valid_action": episode_has_strictly_valid_action,
        "episode_failure_counts": episode_failure_counts,
        "episode_cost_breakdowns": episode_cost_breakdowns,
        "episode_split_usage": episode_split_usage,
        "training_summary": {
            "split_usage": {
                key: int(sum(split[key] for split in episode_split_usage))
                for key in _zero_split_usage()
            },
            "avg_cost_breakdown_per_episode": {
                key: float(np.mean([breakdown[key] for breakdown in episode_cost_breakdowns]))
                for key in _zero_cost_breakdown()
            },
            "invalid_counts_by_reason": {
                key: int(sum(counts[key] for counts in episode_failure_counts))
                for key in _zero_failure_counts()
            },
            "strict_feasible_reset_rate": float(np.mean(episode_has_strictly_valid_action)) if episode_has_strictly_valid_action else 0.0,
            "episodes_with_no_strict_valid_action_rate": 1.0 - (
                float(np.mean(episode_has_strictly_valid_action)) if episode_has_strictly_valid_action else 0.0
            ),
            "invalid_reason_percentages": _invalid_reason_percentages(
                {
                    key: int(sum(counts[key] for counts in episode_failure_counts))
                    for key in _zero_failure_counts()
                },
                sum(int(debug["steps"]) for debug in training_topology_debug.values()),
            ),
        },
    }
    results["strict_mode_debug"] = {
        "first_failure_traces": first_failure_traces,
        "per_topology": {},
        "topology_latency_diagnostics": {},
    }
    for topology_id, topology_debug in training_topology_debug.items():
        step_count = max(int(topology_debug["steps"]), 1)
        results["strict_mode_debug"]["per_topology"][topology_id] = {
            "strict_feasible_reset_rate": float(topology_debug["strictly_valid_resets"] / max(int(topology_debug["resets"]), 1)),
            "valid_step_rate": float(topology_debug["valid_steps"] / step_count),
            "avg_invalid_reasons_per_step": float(topology_debug["invalid_reason_events"] / step_count),
            "invalid_counts_by_reason": topology_debug["invalid_counts_by_reason"],
            "invalid_reason_percentages": _invalid_reason_percentages(
                topology_debug["invalid_counts_by_reason"],
                step_count,
            ),
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

    with results_path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)

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
        },
    )

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
    print(f"Strict-Feasible Reset Rate: {results['training_summary']['strict_feasible_reset_rate']:.1%}")
    print(
        "Episodes With No Strict-Valid Action: "
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
    print(f"Invalid Reason Percent Of Steps: {invalid_pct_text}")
    for topology_id, latency_debug in results["strict_mode_debug"]["topology_latency_diagnostics"].items():
        impossible_splits = latency_debug["splits_ruled_out_by_link_delay"]
        if impossible_splits:
            print(f"{topology_id} strict link-delay impossibility: {', '.join(impossible_splits)}")
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
    verbose: bool = True,
) -> Dict[str, object]:
    env = _build_env(
        benchmark=benchmark,
        max_steps=50,
        topology_pool_name=topology_pool_name,
        topology_selection_mode=topology_selection_mode,
        constraint_mode=constraint_mode,
        topology_id=topology_id,
    )
    graph_builder = ORANGraphBuilder(env.num_rhs, env.num_ess, env.num_rcs)

    rewards = []
    costs = []
    topology_ids = []
    has_structurally_valid_action = []
    has_strictly_valid_action = []
    per_topology: Dict[str, Dict[str, object]] = {}
    invalid_counts_by_reason = _zero_failure_counts()
    latency_diagnostics = _collect_topology_latency_diagnostics(env.topology_id, env.crosshaul_latency_limits)

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
        topology_ids.append(reset_info["topology_id"])
        has_structurally_valid_action.append(bool(reset_info.get("has_structurally_valid_action", False)))
        has_strictly_valid_action.append(bool(reset_info.get("has_strictly_valid_action", False)))
        topology_summary = per_topology.setdefault(
            reset_info["topology_id"],
            {
                "rewards": [],
                "costs": [],
                "valid_steps": 0,
                "total_steps": 0,
                "resets": 0,
                "strictly_valid_resets": 0,
                "invalid_reason_events": 0,
                "split_usage": _zero_split_usage(),
                "cost_breakdown": _zero_cost_breakdown(),
                "invalid_counts_by_reason": _zero_failure_counts(),
                "topology_metadata": reset_info.get("topology_metadata", {}),
                "latency_diagnostics": _collect_topology_latency_diagnostics(
                    reset_info["topology_id"],
                    env.crosshaul_latency_limits,
                ),
            },
        )
        topology_summary["resets"] += 1
        topology_summary["strictly_valid_resets"] += int(bool(reset_info.get("has_strictly_valid_action", False)))

        for _ in range(50):
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
            topology_summary["total_steps"] += 1
            if info["valid_deployment"]:
                topology_summary["valid_steps"] += 1

            if info["valid_deployment"]:
                episode_costs.append(info["deployment_cost"])
            else:
                for key, value in info["failure_counts"].items():
                    invalid_counts_by_reason[key] += int(value)
                    topology_summary["invalid_counts_by_reason"][key] += int(value)
                topology_summary["invalid_reason_events"] += len(info["invalid_reasons"])
            for key in topology_summary["cost_breakdown"]:
                topology_summary["cost_breakdown"][key] += float(info[key])
            for key, value in info["split_usage"].items():
                topology_summary["split_usage"][key] += int(value)

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        costs.append(float(np.mean(episode_costs)) if episode_costs else float("inf"))
        topology_summary["rewards"].append(float(episode_reward))
        topology_summary["costs"].append(float(np.mean(episode_costs)) if episode_costs else float("inf"))

    finite_costs = [cost for cost in costs if np.isfinite(cost)]
    summary = {
        "benchmark": benchmark,
        "topology_pool_name": topology_pool_name,
        "selection_mode": topology_selection_mode,
        "constraint_mode": constraint_mode,
        "topology_ids": topology_ids,
        "has_structurally_valid_action": has_structurally_valid_action,
        "has_strictly_valid_action": has_strictly_valid_action,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_cost": float(np.mean(finite_costs)) if finite_costs else float("inf"),
        "std_cost": float(np.std(finite_costs)) if finite_costs else float("inf"),
        "valid_rate": float(np.mean([row["valid_steps"] / max(row["total_steps"], 1) for row in per_topology.values()])) if per_topology else 0.0,
        "strict_feasible_reset_rate": float(np.mean(has_strictly_valid_action)) if has_strictly_valid_action else 0.0,
        "invalid_counts_by_reason": invalid_counts_by_reason,
        "invalid_reason_percentages": _invalid_reason_percentages(
            invalid_counts_by_reason,
            sum(int(row["total_steps"]) for row in per_topology.values()),
        ),
        "episodes_with_no_strict_valid_action_rate": 1.0 - (
            float(np.mean(has_strictly_valid_action)) if has_strictly_valid_action else 0.0
        ),
        "latency_diagnostics": latency_diagnostics,
        "per_topology": {},
    }

    for topology_id, topology_summary in per_topology.items():
        topology_costs = [cost for cost in topology_summary["costs"] if np.isfinite(cost)]
        total_steps = max(int(topology_summary["total_steps"]), 1)
        summary["per_topology"][topology_id] = {
            "mean_reward": float(np.mean(topology_summary["rewards"])),
            "mean_cost": float(np.mean(topology_costs)) if topology_costs else float("inf"),
            "validity_rate": float(topology_summary["valid_steps"] / total_steps),
            "strict_feasible_reset_rate": float(topology_summary["strictly_valid_resets"] / max(int(topology_summary["resets"]), 1)),
            "avg_invalid_reasons_per_step": float(topology_summary["invalid_reason_events"] / total_steps),
            "sla_penalty": float(topology_summary["cost_breakdown"]["sla_penalty"] / total_steps),
            "invalid_counts_by_reason": topology_summary["invalid_counts_by_reason"],
            "invalid_reason_percentages": _invalid_reason_percentages(
                topology_summary["invalid_counts_by_reason"],
                total_steps,
            ),
            "split_distribution": {
                key: float(value / max(sum(topology_summary["split_usage"].values()), 1))
                for key, value in topology_summary["split_usage"].items()
            },
            "cost_breakdown": {
                key: float(value / total_steps)
                for key, value in topology_summary["cost_breakdown"].items()
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
        print(f"Topologies: {sorted(set(topology_ids))}")
        print(f"Structural Validity Rate At Reset: {np.mean(has_structurally_valid_action):.1%}")
        print(f"Strict-Feasible Rate At Reset: {summary['strict_feasible_reset_rate']:.1%}")
        print(f"Episodes With No Strict-Valid Action: {summary['episodes_with_no_strict_valid_action_rate']:.1%}")
        print(f"Valid Deployment Rate: {summary['valid_rate']:.1%}")
        print(f"Average Reward: {summary['avg_reward']:.3f} ± {summary['std_reward']:.3f}")
        if finite_costs:
            print(f"Average Cost: {summary['avg_cost']:.3f} ± {summary['std_cost']:.3f}")
        else:
            print("Average Cost: inf")
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
                print(f"  link-delay ruled out strict splits: {', '.join(impossible_splits)}")
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
) -> Dict[str, object]:
    comparisons = {}
    for topology_id in DEFAULT_BENCHMARK_TOPOLOGY_POOLS[benchmark][topology_pool_name]:
        legacy = evaluate_gppo(
            agent,
            gnn,
            num_episodes=num_episodes,
            benchmark=benchmark,
            device=device,
            topology_pool_name=topology_pool_name,
            topology_selection_mode="fixed",
            constraint_mode="legacy",
            topology_id=topology_id,
            verbose=False,
        )
        strict = evaluate_gppo(
            agent,
            gnn,
            num_episodes=num_episodes,
            benchmark=benchmark,
            device=device,
            topology_pool_name=topology_pool_name,
            topology_selection_mode="fixed",
            constraint_mode="strict",
            topology_id=topology_id,
            verbose=False,
        )
        comparisons[topology_id] = {
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
                "episodes_with_no_strict_valid_action_rate": strict["episodes_with_no_strict_valid_action_rate"],
            },
            "valid_rate_gap": legacy["valid_rate"] - strict["valid_rate"],
        }
    return comparisons


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GPPO for O-RAN")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--benchmark", choices=sorted(DEFAULT_BENCHMARK_TOPOLOGY_POOLS.keys()), default="small", help="Benchmark group")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO batch size")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH, help="Output JSON path")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Output checkpoint path")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training evaluation")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes per pool")
    parser.add_argument(
        "--topology-selection-mode",
        choices=["fixed", "random_per_reset"],
        default="random_per_reset",
        help="Train topology selection mode",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=["legacy", "strict"],
        default="legacy",
        help="Constraint handling mode",
    )
    parser.add_argument("--train-topology-id", type=str, default=None, help="Force a specific train-pool topology")
    return parser


def run_training_from_args(args: argparse.Namespace) -> None:
    agent, gnn, results = train_gppo(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        results_path=args.results_path,
        checkpoint_path=args.checkpoint_path,
        benchmark=args.benchmark,
        topology_selection_mode=args.topology_selection_mode,
        constraint_mode=args.constraint_mode,
        train_topology_id=args.train_topology_id,
    )
    if not args.skip_eval:
        train_eval = evaluate_pool_by_topology(
            agent,
            gnn,
            benchmark=args.benchmark,
            topology_pool_name="train",
            num_episodes=args.eval_episodes,
            device=args.device,
            constraint_mode=args.constraint_mode,
        )
        test_eval = evaluate_pool_by_topology(
            agent,
            gnn,
            benchmark=args.benchmark,
            topology_pool_name="test",
            num_episodes=args.eval_episodes,
            device=args.device,
            constraint_mode=args.constraint_mode,
        )
        results["evaluation"] = {
            "train_pool": train_eval,
            "test_pool": test_eval,
        }
        results["constraint_mode_comparison"] = {
            "train_pool": compare_constraint_modes_on_pool(
                agent,
                gnn,
                benchmark=args.benchmark,
                topology_pool_name="train",
                num_episodes=args.eval_episodes,
                device=args.device,
            ),
            "test_pool": compare_constraint_modes_on_pool(
                agent,
                gnn,
                benchmark=args.benchmark,
                topology_pool_name="test",
                num_episodes=args.eval_episodes,
                device=args.device,
            ),
        }
        print("\n" + "=" * 60)
        print("Legacy vs Strict Comparison")
        print("=" * 60)
        for pool_name, pool_results in results["constraint_mode_comparison"].items():
            print(f"{pool_name}:")
            for topology_id, comparison in pool_results.items():
                print(
                    f"  {topology_id} | "
                    f"legacy_valid={comparison['legacy']['valid_rate']:.1%} "
                    f"strict_valid={comparison['strict']['valid_rate']:.1%} "
                    f"gap={comparison['valid_rate_gap']:.1%} "
                    f"strict_no_action={comparison['strict']['episodes_with_no_strict_valid_action_rate']:.1%}"
                )
        with args.results_path.open("w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, indent=2)


def main(argv=None) -> None:
    parser = build_train_parser()
    args = parser.parse_args(argv)
    run_training_from_args(args)
