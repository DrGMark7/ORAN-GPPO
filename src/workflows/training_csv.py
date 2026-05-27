import csv
from pathlib import Path
from typing import Dict, List

def _write_flat_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _encode_vector(values: List[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def _trace_csv_path(trace_path: Path, csv_output_dir: Path) -> Path:
    return csv_output_dir / f"{trace_path.stem}.csv"


def _build_episode_trace_csv_rows(
    *,
    benchmark: str,
    topology_pool_name: str,
    constraint_mode: str,
    episode_index: int,
    slots: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows = []
    for slot in slots:
        rows.append(
            {
                "episode_index": int(episode_index),
                "benchmark": benchmark,
                "pool": topology_pool_name,
                "constraint_mode": constraint_mode,
                "slot": int(slot["time_slot"]),
                "topology_id": slot["topology_id"],
                "valid_deployment": bool(slot["valid_deployment"]),
                "deployment_cost": slot["deployment_cost"],
                "reward": float(slot["reward"]),
                "processing_cost": float(slot["processing_cost"]),
                "routing_cost": float(slot["routing_cost"]),
                "reconfiguration_cost": float(slot["reconfiguration_cost"]),
                "sla_penalty": float(slot["sla_penalty"]),
                "es_overuse": float(slot["es_overuse"]),
                "rc_overuse": float(slot["rc_overuse"]),
                "bandwidth_overuse": float(slot["bandwidth_overuse"]),
                "e2e_violation": float(slot["e2e_violation"]),
                "crosshaul_violation": float(slot["crosshaul_violation"]),
                "failed_links": int(slot["failed_links"]),
                "split_changes": int(slot["split_changes"]),
                "es_changes": int(slot["es_changes"]),
                "rc_changes": int(slot["rc_changes"]),
                "invalid_reasons": "|".join(slot["invalid_reasons"]),
                "split_vector": _encode_vector(slot["split_vector"]),
                "es_choice_vector": _encode_vector(slot["es_choice_vector"]),
                "rc_choice_vector": _encode_vector(slot["rc_choice_vector"]),
            }
        )
    return rows


def _write_episode_trace_csv(
    *,
    trace_path: Path,
    csv_output_dir: Path,
    benchmark: str,
    topology_pool_name: str,
    constraint_mode: str,
    episode_index: int,
    slots: List[Dict[str, object]],
) -> Path:
    csv_path = _trace_csv_path(trace_path, csv_output_dir)
    _write_flat_csv(
        csv_path,
        [
            "episode_index",
            "benchmark",
            "pool",
            "constraint_mode",
            "slot",
            "topology_id",
            "valid_deployment",
            "deployment_cost",
            "reward",
            "processing_cost",
            "routing_cost",
            "reconfiguration_cost",
            "sla_penalty",
            "es_overuse",
            "rc_overuse",
            "bandwidth_overuse",
            "e2e_violation",
            "crosshaul_violation",
            "failed_links",
            "split_changes",
            "es_changes",
            "rc_changes",
            "invalid_reasons",
            "split_vector",
            "es_choice_vector",
            "rc_choice_vector",
        ],
        _build_episode_trace_csv_rows(
            benchmark=benchmark,
            topology_pool_name=topology_pool_name,
            constraint_mode=constraint_mode,
            episode_index=episode_index,
            slots=slots,
        ),
    )
    return csv_path


def _run_id_from_results_path(results_path: Path) -> str:
    return results_path.stem


def _build_training_episode_csv_rows(results: Dict[str, object]) -> List[Dict[str, object]]:
    config = results["config"]
    rows = []
    for episode_idx, reward in enumerate(results["episode_rewards"]):
        cost_breakdown = results["episode_cost_breakdowns"][episode_idx]
        reconfig = results["episode_reconfiguration_stats"][episode_idx]
        split_usage = results["episode_split_usage"][episode_idx]
        time_slot_count = max(int(reconfig["time_slot_count"]), 1)
        rows.append(
            {
                "episode": int(episode_idx),
                "benchmark": config["benchmark"],
                "topology_id": results["episode_topology_ids"][episode_idx],
                "pool": results["episode_topology_pools"][episode_idx],
                "constraint_mode": config["constraint_mode"],
                "reward": float(reward),
                "cost": results["episode_costs"][episode_idx],
                "valid_rate": float(results["episode_valid_time_slot_rates"][episode_idx]),
                "average_reward_per_slot": float(reward / time_slot_count),
                "average_cost_per_slot": float(sum(cost_breakdown.values()) / time_slot_count),
                "processing_cost": float(cost_breakdown["processing_cost"]),
                "routing_cost": float(cost_breakdown["routing_cost"]),
                "reconfiguration_cost": float(cost_breakdown["reconfiguration_cost"]),
                "sla_penalty": float(cost_breakdown["sla_penalty"]),
                "split_s1_count": int(split_usage["S1"]),
                "split_s2_count": int(split_usage["S2"]),
                "split_s3_count": int(split_usage["S3"]),
                "split_s4_count": int(split_usage["S4"]),
                "split_changes": float(reconfig["split_changes"]),
                "es_changes": float(reconfig["es_changes"]),
                "rc_changes": float(reconfig["rc_changes"]),
                "total_reconfig_changes": float(reconfig["total_reconfiguration_changes"]),
                "strict_feasible_reset_rate": float(results["episode_has_strictly_valid_action"][episode_idx]),
                "has_structurally_valid_action": bool(results["episode_has_structurally_valid_action"][episode_idx]),
                "has_strictly_valid_action": bool(results["episode_has_strictly_valid_action"][episode_idx]),
            }
        )
    return rows


def _build_evaluation_topology_csv_rows(results: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for pool_results in results.get("evaluation", {}).values():
        for topology_eval in pool_results.values():
            for topology_id, summary in topology_eval["per_topology"].items():
                cost_breakdown = summary["cost_breakdown"]
                split_distribution = summary["split_distribution"]
                rows.append(
                    {
                        "topology_id": topology_id,
                        "benchmark": topology_eval["benchmark"],
                        "pool_type": topology_eval["topology_pool_name"],
                        "constraint_mode": topology_eval["constraint_mode"],
                        "avg_reward": float(summary["mean_reward"]),
                        "std_reward": float(summary["std_reward"]),
                        "avg_cost": summary["mean_cost"],
                        "std_cost": summary["std_cost"],
                        "valid_rate": float(summary["validity_rate"]),
                        "structural_validity_rate_at_reset": float(summary["structural_validity_rate_at_reset"]),
                        "strict_feasible_rate_at_reset": float(summary["strict_feasible_reset_rate"]),
                        "avg_processing_cost": float(cost_breakdown["processing_cost"]),
                        "avg_routing_cost": float(cost_breakdown["routing_cost"]),
                        "avg_reconfiguration_cost": float(cost_breakdown["reconfiguration_cost"]),
                        "avg_sla_penalty": float(cost_breakdown["sla_penalty"]),
                        "split_s1_pct": float(split_distribution["S1"]),
                        "split_s2_pct": float(split_distribution["S2"]),
                        "split_s3_pct": float(split_distribution["S3"]),
                        "split_s4_pct": float(split_distribution["S4"]),
                    }
                )
    return rows


def _build_invalid_reason_csv_rows(results: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for topology_id, summary in results.get("strict_mode_debug", {}).get("per_topology", {}).items():
        row = {
            "scope": "train",
            "pool_type": "train",
            "topology_id": topology_id,
            "constraint_mode": results["config"]["constraint_mode"],
        }
        row.update({key: int(value) for key, value in summary["invalid_counts_by_reason"].items()})
        rows.append(row)

    for pool_results in results.get("evaluation", {}).values():
        for topology_eval in pool_results.values():
            for topology_id, summary in topology_eval["per_topology"].items():
                row = {
                    "scope": "eval",
                    "pool_type": topology_eval["topology_pool_name"],
                    "topology_id": topology_id,
                    "constraint_mode": topology_eval["constraint_mode"],
                }
                row.update({key: int(value) for key, value in summary["invalid_counts_by_reason"].items()})
                rows.append(row)
    return rows


def _build_timing_profile_csv_rows(results: Dict[str, object], results_path: Path) -> List[Dict[str, object]]:
    timing = results.get("training_summary", {}).get("timing_profile_seconds", {})
    if not timing:
        return []
    return [
        {
            "run_id": _run_id_from_results_path(results_path),
            "total_time": float(timing.get("total_seconds", 0.0)),
            "reset_time": float(timing.get("reset_seconds", 0.0)),
            "graph_time": float(timing.get("graph_build_seconds", 0.0)),
            "gnn_time": float(timing.get("gnn_forward_seconds", 0.0)),
            "action_time": float(timing.get("action_selection_seconds", 0.0)),
            "env_step_time": float(timing.get("env_step_seconds", 0.0)),
            "ppo_update_time": float(timing.get("ppo_update_seconds", 0.0)),
        }
    ]


def _export_csv_artifacts(results: Dict[str, object], results_path: Path) -> Dict[str, object]:
    output_dir = results_path.parent
    csv_paths: Dict[str, object] = {}

    training_episode_path = output_dir / "training_episode_metrics.csv"
    _write_flat_csv(
        training_episode_path,
        [
            "episode",
            "benchmark",
            "topology_id",
            "pool",
            "constraint_mode",
            "reward",
            "cost",
            "valid_rate",
            "average_reward_per_slot",
            "average_cost_per_slot",
            "processing_cost",
            "routing_cost",
            "reconfiguration_cost",
            "sla_penalty",
            "split_s1_count",
            "split_s2_count",
            "split_s3_count",
            "split_s4_count",
            "split_changes",
            "es_changes",
            "rc_changes",
            "total_reconfig_changes",
            "strict_feasible_reset_rate",
            "has_structurally_valid_action",
            "has_strictly_valid_action",
        ],
        _build_training_episode_csv_rows(results),
    )
    csv_paths["training_episode_metrics"] = str(training_episode_path)

    evaluation_rows = _build_evaluation_topology_csv_rows(results)
    if evaluation_rows:
        evaluation_path = output_dir / "evaluation_topology_summary.csv"
        _write_flat_csv(
            evaluation_path,
            [
                "topology_id",
                "benchmark",
                "pool_type",
                "constraint_mode",
                "avg_reward",
                "std_reward",
                "avg_cost",
                "std_cost",
                "valid_rate",
                "structural_validity_rate_at_reset",
                "strict_feasible_rate_at_reset",
                "avg_processing_cost",
                "avg_routing_cost",
                "avg_reconfiguration_cost",
                "avg_sla_penalty",
                "split_s1_pct",
                "split_s2_pct",
                "split_s3_pct",
                "split_s4_pct",
            ],
            evaluation_rows,
        )
        csv_paths["evaluation_topology_summary"] = str(evaluation_path)

    invalid_reason_path = output_dir / "invalid_reason_summary.csv"
    _write_flat_csv(
        invalid_reason_path,
        [
            "scope",
            "pool_type",
            "topology_id",
            "constraint_mode",
            "missing_direct_rc_link",
            "missing_rh_es_link",
            "missing_es_rc_link",
            "es_capacity_exceeded",
            "rc_capacity_exceeded",
            "bandwidth_exceeded",
            "e2e_latency_exceeded",
            "crosshaul_latency_exceeded",
        ],
        _build_invalid_reason_csv_rows(results),
    )
    csv_paths["invalid_reason_summary"] = str(invalid_reason_path)

    timing_rows = _build_timing_profile_csv_rows(results, results_path)
    if timing_rows:
        timing_path = output_dir / "timing_profile.csv"
        _write_flat_csv(
            timing_path,
            [
                "run_id",
                "total_time",
                "reset_time",
                "graph_time",
                "gnn_time",
                "action_time",
                "env_step_time",
                "ppo_update_time",
            ],
            timing_rows,
        )
        csv_paths["timing_profile"] = str(timing_path)

    return csv_paths
