import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common.paths import resolve_checkpoint_path, resolve_results_path
from src.workflows.training_constants import PAPER_GNN_INPUT_DIM, PAPER_TIME_SLOTS_PER_EPISODE
from src.workflows.training_csv import _export_csv_artifacts


def _seed_summary_rows(seed_results: List[Dict[str, object]]) -> Dict[str, object]:
    def mean_std(values: List[float]) -> Dict[str, float]:
        finite = [float(value) for value in values if np.isfinite(value)]
        if not finite:
            return {"mean": float("inf"), "std": float("inf")}
        return {"mean": float(np.mean(finite)), "std": float(np.std(finite))}

    rows = []
    per_topology: Dict[str, Dict[str, List[float]]] = {}
    for result in seed_results:
        summary = result.get("training_summary", {})
        rewards = result.get("episode_rewards", [])
        costs = result.get("episode_costs", [])
        valid = result.get("valid_deployments", [])
        row = {
            "seed": result.get("config", {}).get("seed"),
            "reward": float(np.mean(rewards)) if rewards else 0.0,
            "cost": float(np.mean([cost for cost in costs if np.isfinite(cost)])) if any(np.isfinite(cost) for cost in costs) else float("inf"),
            "valid_rate": float(np.mean(valid)) if valid else 0.0,
        }
        rows.append(row)
        for pool_summary in result.get("evaluation", {}).values():
            for topology_eval in pool_summary.values():
                for topology_id, topology_summary in topology_eval.get("per_topology", {}).items():
                    metrics = per_topology.setdefault(
                        topology_id,
                        {"reward": [], "cost": [], "valid_rate": []},
                    )
                    metrics["reward"].append(float(topology_summary.get("mean_reward", 0.0)))
                    metrics["cost"].append(float(topology_summary.get("mean_cost", float("inf"))))
                    metrics["valid_rate"].append(float(topology_summary.get("validity_rate", 0.0)))

    return {
        "num_seeds": len(seed_results),
        "reward": mean_std([row["reward"] for row in rows]),
        "deployment_cost": mean_std([row["cost"] for row in rows]),
        "valid_deployment_rate": mean_std([row["valid_rate"] for row in rows]),
        "per_seed": rows,
        "per_topology": {
            topology_id: {
                metric: mean_std(values)
                for metric, values in metrics.items()
            }
            for topology_id, metrics in per_topology.items()
        },
    }


def _paper_benchmark_name(requested_benchmark: str) -> str:
    if requested_benchmark in {"paper_small", "paper_large"}:
        return requested_benchmark
    if requested_benchmark == "large":
        return "paper_large"
    return "paper_small"


def _write_paper_alignment_report(path: Path) -> None:
    content = """# GPPO Paper-Alignment Report

## A. now matches paper exactly

- Paper-mode topology labels are explicit: `paper_small` and `paper_large`.
- `paper_small` uses `8 RH / 3 ES / 2 RC`.
- `paper_large` uses `64 RH / 4 ES / 2 RC`.
- Paper-mode non-direct links use bandwidth `U(10, 40)` Gbps and latency `U(0, 3.6)` ms.
- Paper-mode direct RH-RC links use per-RH probability `0.10`, bandwidth `160` Gbps, and latency `U(0.1, 0.25)` ms.
- ES and RC capacities are `20` and `100`.
- Request distributions match the paper's eMBB, mMTC, and uRLLC ranges.
- `--paper-mode` uses `288` slots per episode, `600000` timesteps, `32` synchronous environments, and `6` seeds by default.
- GPPO keeps `GINEConv` with two graph convolution layers.
- Paper-mode GNN hidden size is `1024`.
- Policy/value MLP remains two hidden layers of width `256`.
- Xavier uniform initialization is applied to linear layers.

## B. still approximates paper

- Parallel collection is implemented as an in-repo synchronous vectorized rollout path, not Stable-Baselines3 `SubprocVecEnv`.
- Paper-mode reporting aggregates six seeds and per-topology summaries, but exact plotting/table formatting may still differ from the paper.
- Node features include the restored normalized node index in paper mode, alongside the existing semantic node features.

## C. still does not match paper and why

- The resource releasing ratio described in the paper is still not explicitly implemented; requests are still sampled per slot by the current environment.
- The current code still preserves project/debug benchmarks and workflows, so paper-faithful behavior requires using `--paper-mode` or explicit `paper_*` benchmarks.
- Existing old checkpoints are not architecture-compatible with paper-mode GNN input/hidden dimensions; paper-mode writes checkpoint metadata with `checkpoint_family=paper_gppo`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_paper_mode_from_args(args: argparse.Namespace) -> None:
    from src.workflows.training import evaluate_pool_by_topology, train_gppo

    benchmark = _paper_benchmark_name(args.benchmark)
    base_results_path = resolve_results_path(args.results_path)
    base_checkpoint_path = resolve_checkpoint_path(args.results_path, args.checkpoint_path)
    seed_results: List[Dict[str, object]] = []
    seed_output_dir = base_results_path.parent / "paper_runs"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    for seed_offset in range(args.paper_num_seeds):
        run_seed = args.seed + seed_offset
        seed_results_path = seed_output_dir / f"{base_results_path.stem}_seed_{run_seed}.json"
        seed_checkpoint_path = seed_output_dir / f"{base_checkpoint_path.stem}_seed_{run_seed}.pt"
        agent, gnn, results = train_gppo(
            num_episodes=max(1, math.ceil(args.paper_timesteps / (args.paper_num_envs * PAPER_TIME_SLOTS_PER_EPISODE))),
            max_steps=PAPER_TIME_SLOTS_PER_EPISODE,
            batch_size=args.batch_size,
            device=args.device,
            seed=run_seed,
            results_path=seed_results_path,
            checkpoint_path=seed_checkpoint_path,
            benchmark=benchmark,
            topology_selection_mode=args.topology_selection_mode,
            constraint_mode=args.constraint_mode,
            train_topology_id=args.train_topology_id,
            total_timesteps=args.paper_timesteps,
            num_envs=args.paper_num_envs,
            gnn_hidden_dim=args.paper_gnn_hidden_dim,
            gnn_input_dim=PAPER_GNN_INPUT_DIM,
            include_node_index=True,
            paper_mode=True,
        )

        if not args.skip_eval:
            train_eval = evaluate_pool_by_topology(
                agent,
                gnn,
                benchmark=benchmark,
                topology_pool_name="train",
                num_episodes=args.eval_episodes,
                device=args.device,
                constraint_mode=args.constraint_mode,
                max_steps=PAPER_TIME_SLOTS_PER_EPISODE,
                csv_output_dir=seed_output_dir,
                trace_output_dir=seed_output_dir / "episode_traces",
                include_node_index=True,
            )
            test_eval = evaluate_pool_by_topology(
                agent,
                gnn,
                benchmark=benchmark,
                topology_pool_name="test",
                num_episodes=args.eval_episodes,
                device=args.device,
                constraint_mode=args.constraint_mode,
                max_steps=PAPER_TIME_SLOTS_PER_EPISODE,
                csv_output_dir=seed_output_dir,
                trace_output_dir=seed_output_dir / "episode_traces",
                include_node_index=True,
            )
            results["evaluation"] = {
                "train_pool": train_eval,
                "test_pool": test_eval,
            }
        results["paper_mode_seed_index"] = seed_offset
        csv_paths = _export_csv_artifacts(results, seed_results_path)
        results["csv_exports"] = csv_paths
        with seed_results_path.open("w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, indent=2)
        seed_results.append(results)

    aggregate = {
        "paper_mode": True,
        "benchmark": benchmark,
        "timesteps_per_seed": args.paper_timesteps,
        "time_slots_per_episode": PAPER_TIME_SLOTS_PER_EPISODE,
        "parallel_envs": args.paper_num_envs,
        "num_seeds": args.paper_num_seeds,
        "seeds": [args.seed + idx for idx in range(args.paper_num_seeds)],
        "summary": _seed_summary_rows(seed_results),
        "seed_result_paths": [
            str(seed_output_dir / f"{base_results_path.stem}_seed_{args.seed + idx}.json")
            for idx in range(args.paper_num_seeds)
        ],
    }
    with base_results_path.open("w", encoding="utf-8") as file_obj:
        json.dump(aggregate, file_obj, indent=2)

    report_path = base_results_path.parent / "paper_alignment_report.json"
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(aggregate, file_obj, indent=2)
    _write_paper_alignment_report(Path("Reports.md"))

    print("\n" + "=" * 60)
    print("Paper-Mode Aggregate Report")
    print("=" * 60)
    print(f"Benchmark: {benchmark}")
    print(f"Seeds: {aggregate['seeds']}")
    print(f"Reward mean/std: {aggregate['summary']['reward']}")
    print(f"Deployment cost mean/std: {aggregate['summary']['deployment_cost']}")
    print(f"Valid deployment rate mean/std: {aggregate['summary']['valid_deployment_rate']}")
    print(f"Aggregate JSON: {base_results_path}")
    print(f"Paper report JSON: {report_path}")
    print("Markdown alignment report: Reports.md")
    print("=" * 60)
