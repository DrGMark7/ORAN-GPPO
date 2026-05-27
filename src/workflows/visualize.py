import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.common.paths import DEFAULT_CHECKPOINT_PATH, DEFAULT_RESULTS_PATH, EPISODE_TRACES_DIR, VISUALIZATIONS_DIR, ensure_output_dirs
from src.core import DEFAULT_BENCHMARK_TOPOLOGY_POOLS
from src.visualization import (
    ActionSpaceVisualizer,
    CostBreakdownVisualizer,
    NetworkTopologyVisualizer,
    TrainingVisualization,
)


def _csv_path(results_path: Path, filename: str) -> Path:
    return results_path.parent / filename


def _discover_episode_trace_csvs(results_path: Path) -> list[Path]:
    roots = [results_path.parent, EPISODE_TRACES_DIR]
    found = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*_episode_*.csv")):
            if not (path.name.startswith("train_") or path.name.startswith("test_")):
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def _checkpoint_benchmark(checkpoint_path: Path) -> str | None:
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        benchmark = metadata.get("benchmark")
        if isinstance(benchmark, str):
            return benchmark
    return None


def _topology_targets_for_benchmark(benchmark: str) -> list[tuple[str, str, str]]:
    targets = []
    for pool_name in ["train", "test"]:
        for idx, topology_id in enumerate(DEFAULT_BENCHMARK_TOPOLOGY_POOLS[benchmark][pool_name], start=1):
            file_index = len(targets) + 1
            targets.append((topology_id, pool_name, f"01_{file_index:02d}_topology_{topology_id}.png"))
    return targets


def generate_all_visualizations(
    results_path: Path = DEFAULT_RESULTS_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> None:
    ensure_output_dirs()
    training_csv = _csv_path(results_path, "training_episode_metrics.csv")
    evaluation_csv = _csv_path(results_path, "evaluation_topology_summary.csv")
    invalid_csv = _csv_path(results_path, "invalid_reason_summary.csv")
    timing_csv = _csv_path(results_path, "timing_profile.csv")
    episode_trace_csvs = _discover_episode_trace_csvs(results_path)

    print("\n" + "=" * 70)
    print("GPPO Visualization Suite - Generating All Charts")
    print("=" * 70 + "\n")

    print("[1/20] Generating Topology Visualizations...")
    checkpoint_str = str(checkpoint_path) if checkpoint_path.exists() else None
    checkpoint_benchmark = _checkpoint_benchmark(checkpoint_path) or "large"
    topology_targets = _topology_targets_for_benchmark(checkpoint_benchmark)
    inferred_dims = (
        NetworkTopologyVisualizer.infer_topology_from_checkpoint(checkpoint_str)
        if checkpoint_str else None
    )
    default_dims = (64, 4, 2) if checkpoint_benchmark == "paper_large" else (16, 5, 3)
    num_rhs, num_ess, num_rcs = inferred_dims if inferred_dims else default_dims
    viz_topology = NetworkTopologyVisualizer(num_rhs=num_rhs, num_ess=num_ess, num_rcs=num_rcs, seed=42, benchmark=checkpoint_benchmark)
    for topology_id, topology_pool_name, filename in topology_targets:
        output_path = VISUALIZATIONS_DIR / filename
        viz_topology.draw_topology(
            save_path=str(output_path),
            checkpoint_path=checkpoint_str,
            topology_pool_name=topology_pool_name,
            topology_id=topology_id,
        )
        print(f"     ✓ Saved: visualizations/{filename} ({topology_id})")
    print()

    print("[2/20] Generating Action Space Visualization...")
    ActionSpaceVisualizer.plot_action_space(
        num_rhs=num_rhs,
        num_ess=num_ess,
        num_rcs=num_rcs,
        save_path=str(VISUALIZATIONS_DIR / "05_action_space.png"),
    )
    print("     ✓ Saved: visualizations/05_action_space.png\n")

    if results_path.exists():
        print("[3/20] Generating Policy-Derived Split Usage...")
        TrainingVisualization.plot_split_usage(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "06_split_usage.png"),
        )
        print("     ✓ Saved: visualizations/06_split_usage.png\n")
    else:
        print("[3/20] Training results not found. Skipping policy-derived split usage.\n")

    if results_path.exists():
        print("[4/20] Generating Training Curves...")
        TrainingVisualization.plot_training_curves(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "07_training_curves.png"),
        )
        print("     ✓ Saved: visualizations/07_training_curves.png\n")

        print("[5/20] Generating Training Phase Analysis...")
        TrainingVisualization.plot_statistics(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "08_phase_analysis.png"),
        )
        print("     ✓ Saved: visualizations/08_phase_analysis.png\n")

        print("[6/20] Generating Cost Breakdown Analysis...")
        CostBreakdownVisualizer.plot_cost_components(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "09_cost_breakdown.png"),
        )
        print("     ✓ Saved: visualizations/09_cost_breakdown.png\n")

        print("[7/20] Generating Cost Breakdown by Phase...")
        TrainingVisualization.plot_cost_breakdown_by_phase(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "10_cost_breakdown_by_phase.png"),
        )
        print("     ✓ Saved: visualizations/10_cost_breakdown_by_phase.png\n")

        print("[8/20] Generating Evaluation Topology Summary...")
        TrainingVisualization.plot_evaluation_topology_summary(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "11_evaluation_topology_summary.png"),
        )
        print("     ✓ Saved: visualizations/11_evaluation_topology_summary.png\n")
    else:
        print("[4/20] Training results not found. Run training first!")
        print("     Skipping training curves visualization\n")
        print("[5/20] Skipping phase analysis...\n")
        print("[6/20] Skipping cost breakdown...\n")
        print("[7/20] Skipping phase cost breakdown...\n")
        print("[8/20] Skipping evaluation topology summary...\n")

    print("[9/20] Generating CSV Training Metrics...")
    TrainingVisualization.plot_training_metrics_from_csv(
        str(training_csv),
        save_path=str(VISUALIZATIONS_DIR / "12_training_metrics_from_csv.png"),
    )
    print()

    print("[10/20] Generating CSV Cost Component Trends...")
    TrainingVisualization.plot_cost_components_from_csv(
        str(training_csv),
        save_path=str(VISUALIZATIONS_DIR / "13_cost_components_from_csv.png"),
    )
    print()

    print("[11/20] Generating CSV Evaluation Topology Summary...")
    TrainingVisualization.plot_evaluation_topology_summary_from_csv(
        str(evaluation_csv),
        save_path=str(VISUALIZATIONS_DIR / "15_evaluation_topology_summary_from_csv.png"),
    )
    print()

    print("[12/20] Generating CSV Evaluation Split Distribution...")
    TrainingVisualization.plot_eval_split_distribution_by_topology_from_csv(
        str(evaluation_csv),
        save_path=str(VISUALIZATIONS_DIR / "16_eval_split_distribution_by_topology.png"),
    )
    print()

    print("[13/20] Generating CSV Timing Profile...")
    TrainingVisualization.plot_timing_profile_from_csv(
        str(timing_csv),
        save_path=str(VISUALIZATIONS_DIR / "18_timing_profile_from_csv.png"),
    )
    print()

    print("[14/20] Generating CSV Episode Trace Plots...")
    if episode_trace_csvs:
        for trace_csv in episode_trace_csvs:
            output_name = f"19_episode_trace_{trace_csv.stem}.png"
            fig = TrainingVisualization.plot_episode_trace_from_csv(
                str(trace_csv),
                save_path=str(VISUALIZATIONS_DIR / output_name),
            )
            if fig is not None:
                plt.close("all")
    else:
        print("Skipping episode trace plots: no episode trace CSV files found.")
    print()

    print("[15/20] Generating CSV Invalid Reason Summary...")
    TrainingVisualization.plot_invalid_reason_summary_from_csv(
        str(invalid_csv),
        save_path=str(VISUALIZATIONS_DIR / "17_invalid_reason_summary.png"),
    )
    print()

    print("[16/20] Generating Stationarity Summary...")
    TrainingVisualization.plot_stationarity_summary_from_trace_csvs(
        [str(path) for path in episode_trace_csvs],
        save_path=str(VISUALIZATIONS_DIR / "20_stationarity_summary.png"),
    )
    print()

    print("[17/20] Generating CSV Split Usage Over Training...")
    TrainingVisualization.plot_split_usage_over_training_from_csv(
        str(training_csv),
        save_path=str(VISUALIZATIONS_DIR / "14_split_usage_over_training.png"),
    )
    print()

    plt.close("all")

    print("=" * 70)
    print("✓ All Visualizations Generated Successfully!")
    print("=" * 70)


def build_visualize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GPPO visualizations")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH, help="Training results JSON path")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Policy checkpoint path")
    return parser


def main(argv=None) -> None:
    parser = build_visualize_parser()
    args = parser.parse_args(argv)
    generate_all_visualizations(results_path=args.results_path, checkpoint_path=args.checkpoint_path)
