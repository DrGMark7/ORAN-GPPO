import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.common.paths import DEFAULT_CHECKPOINT_PATH, DEFAULT_RESULTS_PATH, VISUALIZATIONS_DIR, ensure_output_dirs
from src.visualization import (
    ActionSpaceVisualizer,
    CostBreakdownVisualizer,
    NetworkTopologyVisualizer,
    TrainingVisualization,
)


def generate_all_visualizations(
    results_path: Path = DEFAULT_RESULTS_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> None:
    ensure_output_dirs()
    topology_targets = [
        ("large_balanced_train_a", "train", "01_01_topology_large_balanced_train_a.png"),
        ("large_clustered_train_b", "train", "01_02_topology_large_clustered_train_b.png"),
        ("large_sparse_test_a", "test", "01_03_topology_large_sparse_test_a.png"),
        ("large_direct_test_b", "test", "01_04_topology_large_direct_test_b.png"),
    ]

    print("\n" + "=" * 70)
    print("GPPO Visualization Suite - Generating All Charts")
    print("=" * 70 + "\n")

    print("[1/11] Generating Topology Visualizations...")
    checkpoint_str = str(checkpoint_path) if checkpoint_path.exists() else None
    inferred_dims = (
        NetworkTopologyVisualizer.infer_topology_from_checkpoint(checkpoint_str)
        if checkpoint_str else None
    )
    num_rhs, num_ess, num_rcs = inferred_dims if inferred_dims else (16, 5, 3)
    viz_topology = NetworkTopologyVisualizer(num_rhs=num_rhs, num_ess=num_ess, num_rcs=num_rcs, seed=42, benchmark="large")
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

    print("[2/11] Generating Action Space Visualization...")
    ActionSpaceVisualizer.plot_action_space(
        num_rhs=num_rhs,
        num_ess=num_ess,
        num_rcs=num_rcs,
        save_path=str(VISUALIZATIONS_DIR / "05_action_space.png"),
    )
    print("     ✓ Saved: visualizations/05_action_space.png\n")

    if results_path.exists():
        print("[3/11] Generating Policy-Derived Split Usage...")
        TrainingVisualization.plot_split_usage(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "06_split_usage.png"),
        )
        print("     ✓ Saved: visualizations/06_split_usage.png\n")
    else:
        print("[3/11] Training results not found. Skipping policy-derived split usage.\n")

    if results_path.exists():
        print("[4/11] Generating Training Curves...")
        TrainingVisualization.plot_training_curves(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "07_training_curves.png"),
        )
        print("     ✓ Saved: visualizations/07_training_curves.png\n")

        print("[5/11] Generating Training Phase Analysis...")
        TrainingVisualization.plot_statistics(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "08_phase_analysis.png"),
        )
        print("     ✓ Saved: visualizations/08_phase_analysis.png\n")

        print("[6/11] Generating Cost Breakdown Analysis...")
        CostBreakdownVisualizer.plot_cost_components(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "09_cost_breakdown.png"),
        )
        print("     ✓ Saved: visualizations/09_cost_breakdown.png\n")

        print("[7/11] Generating Cost Breakdown by Phase...")
        TrainingVisualization.plot_cost_breakdown_by_phase(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "10_cost_breakdown_by_phase.png"),
        )
        print("     ✓ Saved: visualizations/10_cost_breakdown_by_phase.png\n")

        print("[8/11] Generating Evaluation Topology Summary...")
        TrainingVisualization.plot_evaluation_topology_summary(
            str(results_path),
            save_path=str(VISUALIZATIONS_DIR / "11_evaluation_topology_summary.png"),
        )
        print("     ✓ Saved: visualizations/11_evaluation_topology_summary.png\n")
    else:
        print("[4/11] Training results not found. Run training first!")
        print("     Skipping training curves visualization\n")
        print("[5/11] Skipping phase analysis...\n")
        print("[6/11] Skipping cost breakdown...\n")
        print("[7/11] Skipping phase cost breakdown...\n")
        print("[8/11] Skipping evaluation topology summary...\n")

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
