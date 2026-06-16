import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.common.paths import ensure_output_dirs
from src.core import DEFAULT_BENCHMARK_TOPOLOGY_POOLS, GNNFeatureExtractor, ORANGraphBuilder, PPOAgent
from src.workflows.training_constants import PAPER_TIME_SLOTS_PER_EPISODE


def _train_gppo_sync_vectorized(
    *,
    total_timesteps: int,
    max_steps: int,
    batch_size: int,
    device: str,
    seed: int,
    results_path: Path,
    checkpoint_path: Path,
    benchmark: str,
    topology_selection_mode: str,
    constraint_mode: str,
    train_topology_id: Optional[str],
    num_envs: int,
    gnn_hidden_dim: int,
    gnn_input_dim: int,
    include_node_index: bool,
    paper_mode: bool,
) -> Tuple[PPOAgent, torch.nn.Module, dict]:
    from src.workflows.training import (
        _bounded_probe_summary,
        _build_env,
        _collect_benchmark_audit,
        _collect_topology_latency_diagnostics,
        _dict_l2_norm,
        _invalid_reason_percentages,
        _strict_feasibility_wording,
        _zero_cost_breakdown,
        _zero_failure_counts,
        _zero_reconfiguration_stats,
        _zero_split_usage,
        _zero_timing_stats,
        _zero_topology_debug,
    )

    train_start_time = time.perf_counter()
    timing_stats = _zero_timing_stats()
    np.random.seed(seed)
    torch.manual_seed(seed)
    ensure_output_dirs()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    envs = [
        _build_env(
            benchmark=benchmark,
            max_steps=max_steps,
            topology_pool_name="train",
            topology_selection_mode=topology_selection_mode,
            constraint_mode=constraint_mode,
            topology_id=train_topology_id,
        )
        for _ in range(num_envs)
    ]
    env0 = envs[0]
    graph_builders = [
        ORANGraphBuilder(env.num_rhs, env.num_ess, env.num_rcs, include_node_index=include_node_index)
        for env in envs
    ]
    gnn = GNNFeatureExtractor(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, output_dim=128).to(device)
    agent = PPOAgent(
        feature_dim=128,
        num_rhs=env0.num_rhs,
        num_splits=4,
        num_ess=env0.num_ess,
        num_rcs=env0.num_rcs,
        lr=1e-4,
        gamma=0.98,
        gae_lambda=0.97,
        clip_ratio=0.3,
        device=device,
    )
    agent.attach_feature_extractor(gnn)
    initial_gnn_state = {key: value.detach().cpu().clone() for key, value in gnn.state_dict().items()}

    print("=" * 60)
    print("GPPO Paper-Mode Training")
    print("=" * 60)
    print(f"Benchmark: {benchmark} ({env0.num_rhs} RH / {env0.num_ess} ES / {env0.num_rcs} RC)")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Parallel environments: {num_envs}")
    print(f"Time slots per episode: {max_steps}")
    print(f"GNN hidden dim: {gnn_hidden_dim}")
    print(f"Results: {results_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    episode_rewards: List[float] = []
    episode_costs: List[float] = []
    valid_deployments: List[float] = []
    valid_time_slot_rates: List[float] = []
    episode_topology_ids: List[str] = []
    episode_topology_pools: List[str] = []
    episode_has_structurally_valid_action: List[bool] = []
    episode_has_strictly_valid_action: List[bool] = []
    episode_has_greedy_strictly_valid_action: List[bool] = []
    episode_has_exact_strictly_valid_action: List[Optional[bool]] = []
    episode_bounded_strict_feasibility_probe: List[Dict[str, object]] = []
    episode_failure_counts: List[Dict[str, int]] = []
    episode_cost_breakdowns: List[Dict[str, float]] = []
    episode_reconfiguration_stats: List[Dict[str, float]] = []
    episode_split_usage: List[Dict[str, int]] = []
    training_topology_debug: Dict[str, Dict[str, object]] = {}
    first_failure_traces: List[Dict[str, object]] = []

    active: List[Dict[str, object]] = []

    def reset_worker(env_idx: int, episode_idx: int) -> None:
        reset_start = time.perf_counter()
        _, reset_info = envs[env_idx].reset(
            seed=seed + (episode_idx * num_envs) + env_idx,
            options={
                "topology_pool_name": "train",
                "topology_selection_mode": topology_selection_mode,
                "topology_id": train_topology_id,
            },
        )
        timing_stats["reset_seconds"] += time.perf_counter() - reset_start
        active[env_idx] = {
            "reset_info": reset_info,
            "episode_reward": 0.0,
            "valid_costs": [],
            "valid_steps": 0,
            "failure_counts": _zero_failure_counts(),
            "cost_breakdown": _zero_cost_breakdown(),
            "reconfiguration_stats": _zero_reconfiguration_stats(),
            "split_usage": _zero_split_usage(),
            "time_slot_count": 0,
        }
        topology_debug = training_topology_debug.setdefault(reset_info["topology_id"], _zero_topology_debug())
        topology_debug["resets"] += 1
        topology_debug["strictly_valid_resets"] += int(bool(reset_info.get("has_strictly_valid_action", False)))
        if reset_info.get("has_exact_strictly_valid_action") is True:
            topology_debug["exact_strictly_valid_resets"] += 1
        topology_debug["bounded_probe_results"].append(reset_info.get("bounded_strict_feasibility_probe", {}))

    def finish_worker_episode(env_idx: int) -> None:
        item = active[env_idx]
        reset_info = item["reset_info"]
        time_slot_count = int(item["time_slot_count"])
        if time_slot_count == 0:
            return
        valid_costs = item["valid_costs"]
        reconfiguration_stats = item["reconfiguration_stats"]
        cost_breakdown = item["cost_breakdown"]

        episode_rewards.append(float(item["episode_reward"]))
        episode_costs.append(float(np.mean(valid_costs)) if valid_costs else float("inf"))
        valid_rate = float(item["valid_steps"] / max(time_slot_count, 1))
        valid_deployments.append(valid_rate)
        valid_time_slot_rates.append(valid_rate)
        episode_topology_ids.append(reset_info["topology_id"])
        episode_topology_pools.append(reset_info["topology_pool"])
        episode_has_structurally_valid_action.append(bool(reset_info.get("has_structurally_valid_action", False)))
        episode_has_strictly_valid_action.append(bool(reset_info.get("has_strictly_valid_action", False)))
        episode_has_greedy_strictly_valid_action.append(bool(reset_info.get("has_greedy_strictly_valid_action", False)))
        episode_has_exact_strictly_valid_action.append(reset_info.get("has_exact_strictly_valid_action"))
        episode_bounded_strict_feasibility_probe.append(reset_info.get("bounded_strict_feasibility_probe", {}))
        episode_failure_counts.append(item["failure_counts"])
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
        episode_split_usage.append(item["split_usage"])

    active = [{} for _ in range(num_envs)]
    episode_counters = [0 for _ in range(num_envs)]
    for env_idx in range(num_envs):
        reset_worker(env_idx, episode_counters[env_idx])

    total_steps = 0
    progress = tqdm(total=total_timesteps, desc="Paper-mode timesteps")
    while total_steps < total_timesteps:
        for env_idx, env in enumerate(envs):
            if total_steps >= total_timesteps:
                break

            adjacency_start = time.perf_counter()
            adjacency, edge_features, _ = env._get_adjacency_info()
            timing_stats["adjacency_seconds"] += time.perf_counter() - adjacency_start

            graph_build_start = time.perf_counter()
            graph = graph_builders[env_idx].build_graph(
                env.rh_demands,
                env.rh_latencies,
                env.es_remaining,
                env.rc_remaining,
                adjacency,
                edge_features,
            )
            timing_stats["graph_build_seconds"] += time.perf_counter() - graph_build_start

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

            item = active[env_idx]
            item["episode_reward"] += float(reward)
            item["time_slot_count"] += 1
            topology_debug = training_topology_debug.setdefault(info["topology_id"], _zero_topology_debug())
            topology_debug["time_slots"] += 1
            if info["valid_deployment"]:
                item["valid_steps"] += 1
                item["valid_costs"].append(info["deployment_cost"])
                topology_debug["valid_time_slots"] += 1
            else:
                for key, value in info["failure_counts"].items():
                    item["failure_counts"][key] += int(value)
                    topology_debug["invalid_counts_by_reason"][key] += int(value)
                topology_debug["invalid_reason_events"] += len(info["invalid_reasons"])
                if info["invalid_reasons"] and len(first_failure_traces) < 12:
                    first_failure_traces.append(
                        {
                            "episode": len(episode_rewards),
                            "time_slot": int(info["time_slot"]),
                            "topology_id": info["topology_id"],
                            "constraint_mode": info["constraint_mode"],
                            "invalid_reasons": list(info["invalid_reasons"]),
                        }
                    )

            for key in item["cost_breakdown"]:
                item["cost_breakdown"][key] += float(info[key])
            for key in item["reconfiguration_stats"]:
                item["reconfiguration_stats"][key] += float(info[key])
            for key, split_count in info["split_usage"].items():
                item["split_usage"][key] += int(split_count)

            total_steps += 1
            progress.update(1)
            if len(agent.states) >= batch_size:
                ppo_update_start = time.perf_counter()
                agent.update(batch_size=batch_size, epochs=3)
                timing_stats["ppo_update_seconds"] += time.perf_counter() - ppo_update_start

            if terminated or truncated:
                finish_worker_episode(env_idx)
                episode_counters[env_idx] += 1
                reset_worker(env_idx, episode_counters[env_idx])
    progress.close()

    if agent.states:
        ppo_update_start = time.perf_counter()
        agent.update(batch_size=batch_size, epochs=3)
        timing_stats["ppo_update_seconds"] += time.perf_counter() - ppo_update_start
    for env_idx in range(num_envs):
        finish_worker_episode(env_idx)

    results = {
        "config": {
            "benchmark": benchmark,
            "benchmark_label": benchmark if paper_mode else f"project_{benchmark}",
            "train_selection_mode": topology_selection_mode,
            "constraint_mode": constraint_mode,
            "episode_length_time_slots": max_steps,
            "paper_aligned_episode_length": max_steps == PAPER_TIME_SLOTS_PER_EPISODE,
            "paper_mode": paper_mode,
            "paper_total_timesteps": total_timesteps,
            "paper_num_envs": num_envs,
            "seed": seed,
            "dimensions": {
                "num_rhs": env0.num_rhs,
                "num_ess": env0.num_ess,
                "num_rcs": env0.num_rcs,
            },
            "gnn_hidden_dim": gnn_hidden_dim,
            "gnn_input_dim": gnn_input_dim,
            "include_node_index": include_node_index,
        },
        "benchmark_audit": _collect_benchmark_audit(benchmark, env0.crosshaul_latency_limits),
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
            "total_timesteps_collected": total_steps,
            "parallel_envs": num_envs,
            "split_usage": {
                key: int(sum(split[key] for split in episode_split_usage))
                for key in _zero_split_usage()
            },
            "avg_cost_breakdown_per_episode": {
                key: float(np.mean([breakdown[key] for breakdown in episode_cost_breakdowns])) if episode_cost_breakdowns else 0.0
                for key in _zero_cost_breakdown()
            },
            "avg_reconfiguration_stats_per_episode": {
                key: float(np.mean([stats[key] for stats in episode_reconfiguration_stats])) if episode_reconfiguration_stats else 0.0
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
        "strict_mode_debug": {
            "first_failure_traces": first_failure_traces,
            "per_topology": {},
            "topology_latency_diagnostics": {},
        },
    }
    for topology_id, topology_debug in training_topology_debug.items():
        step_count = max(int(topology_debug["time_slots"]), 1)
        results["strict_mode_debug"]["per_topology"][topology_id] = {
            "strict_feasible_reset_rate": float(topology_debug["strictly_valid_resets"] / max(int(topology_debug["resets"]), 1)),
            "exact_strict_feasible_reset_count": int(topology_debug["exact_strictly_valid_resets"]),
            "valid_time_slot_rate": float(topology_debug["valid_time_slots"] / step_count),
            "avg_invalid_reasons_per_time_slot": float(topology_debug["invalid_reason_events"] / step_count),
            "invalid_counts_by_reason": topology_debug["invalid_counts_by_reason"],
            "invalid_reason_percentages": _invalid_reason_percentages(topology_debug["invalid_counts_by_reason"], step_count),
            "first_failing_constraint_counts": topology_debug["first_failing_constraint_counts"],
            "bounded_probe_results": topology_debug["bounded_probe_results"],
            "bounded_probe_summary": _bounded_probe_summary(topology_debug["bounded_probe_results"]),
        }
        results["strict_mode_debug"]["topology_latency_diagnostics"][topology_id] = _collect_topology_latency_diagnostics(
            topology_id,
            env0.crosshaul_latency_limits,
        )

    final_gnn_state = {key: value.detach().cpu() for key, value in gnn.state_dict().items()}
    gnn_delta_sq = 0.0
    for key, initial_value in initial_gnn_state.items():
        diff = final_gnn_state[key] - initial_value
        gnn_delta_sq += float(torch.sum(diff.float() ** 2).item())
    timing_stats["total_seconds"] = time.perf_counter() - train_start_time
    results["training_summary"]["timing_profile_seconds"] = dict(timing_stats)
    results["gnn_training_verification"] = {
        "gnn_state_dict_saved": True,
        "initial_param_l2_norm": _dict_l2_norm(initial_gnn_state),
        "final_param_l2_norm": _dict_l2_norm(final_gnn_state),
        "parameter_delta_l2_norm": float(gnn_delta_sq ** 0.5),
    }

    agent.save(
        str(checkpoint_path),
        metadata={
            "benchmark": benchmark,
            "num_rhs": env0.num_rhs,
            "num_ess": env0.num_ess,
            "num_rcs": env0.num_rcs,
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
    with results_path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)

    finite_costs = [cost for cost in episode_costs if np.isfinite(cost)]
    print("\n" + "=" * 60)
    print("Paper-Mode Training Complete")
    print(f"Collected timesteps: {total_steps}")
    print(f"Episodes recorded: {len(episode_rewards)}")
    print(f"Average Reward: {np.mean(episode_rewards):.3f}" if episode_rewards else "Average Reward: n/a")
    print(f"Average Cost: {np.mean(finite_costs):.3f}" if finite_costs else "Average Cost: inf")
    print(f"Valid Deployment Rate: {np.mean(valid_deployments):.1%}" if valid_deployments else "Valid Deployment Rate: n/a")
    print("=" * 60)
    return agent, gnn, results
