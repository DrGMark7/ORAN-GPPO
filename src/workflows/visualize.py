import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.common.paths import (
    DEFAULT_RUN_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_RESULTS_PATH,
    EPISODE_TRACES_DIR,
    VISUALIZATIONS_DIR,
    ensure_output_dirs,
    resolve_checkpoint_path,
    resolve_episode_traces_dir,
    resolve_results_path,
    resolve_run_dir,
)
from src.core import DEFAULT_BENCHMARK_TOPOLOGY_POOLS
from src.visualization import (
    CostBreakdownVisualizer,
    NetworkTopologyVisualizer,
    TrainingVisualization,
)
from src.visualization.plots import clear_skip_plot_messages, consume_skip_plot_messages


def _csv_path(results_path: Path, filename: str) -> Path:
    return resolve_run_dir(results_path) / filename


def _discover_episode_trace_csvs(results_path: Path) -> list[Path]:
    local_trace_dir = resolve_episode_traces_dir(results_path)
    local_run_dir = resolve_run_dir(results_path)
    roots = [local_trace_dir, local_run_dir, EPISODE_TRACES_DIR]
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


def _print_saved(saved_paths: list[Path]) -> None:
    if not saved_paths:
        return
    print("\nSaved visualizations:")
    for path in saved_paths:
        print(f"- {path}")


def _record_saved(saved_paths: list[Path], output_path: Path, fig) -> None:
    if fig is not None and output_path.exists():
        saved_paths.append(output_path)


def _prepare_output_path(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()


def _print_skip_summary() -> None:
    skipped = consume_skip_plot_messages()
    if not skipped:
        return
    names = ", ".join(name for name, _ in skipped)
    print(f"\nSkipped {len(skipped)} plot(s): {names}")


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
    mode: str = "compact",
) -> None:
    ensure_output_dirs()
    results_path = resolve_results_path(results_path)
    checkpoint_path = resolve_checkpoint_path(results_path, checkpoint_path)
    clear_skip_plot_messages()
    training_csv = _csv_path(results_path, "training_episode_metrics.csv")
    evaluation_csv = _csv_path(results_path, "evaluation_topology_summary.csv")
    invalid_csv = _csv_path(results_path, "invalid_reason_summary.csv")
    timing_csv = _csv_path(results_path, "timing_profile.csv")
    episode_trace_csvs = _discover_episode_trace_csvs(results_path)
    saved_paths: list[Path] = []

    print("\n" + "=" * 70)
    print(f"GPPO Visualization Suite - Mode: {mode}")
    print("=" * 70 + "\n")

    checkpoint_str = str(checkpoint_path) if checkpoint_path.exists() else None

    if mode == "topology":
        checkpoint_benchmark = _checkpoint_benchmark(checkpoint_path) or "large"
        topology_targets = _topology_targets_for_benchmark(checkpoint_benchmark)
        inferred_dims = (
            NetworkTopologyVisualizer.infer_topology_from_checkpoint(checkpoint_str)
            if checkpoint_str else None
        )
        default_dims = (64, 4, 2) if checkpoint_benchmark == "paper_large" else (16, 5, 3)
        num_rhs, num_ess, num_rcs = inferred_dims if inferred_dims else default_dims
        viz_topology = NetworkTopologyVisualizer(
            num_rhs=num_rhs,
            num_ess=num_ess,
            num_rcs=num_rcs,
            seed=42,
            benchmark=checkpoint_benchmark,
        )
        print("Generating topology snapshots...")
        for topology_id, topology_pool_name, filename in topology_targets:
            output_path = VISUALIZATIONS_DIR / filename
            viz_topology.draw_topology(
                save_path=str(output_path),
                checkpoint_path=checkpoint_str,
                topology_pool_name=topology_pool_name,
                topology_id=topology_id,
            )
            saved_paths.append(output_path)
    elif mode == "csv":
        print("Generating compact CSV-backed visualizations...")
        output_path = VISUALIZATIONS_DIR / "training_curves.png"
        _prepare_output_path(output_path)
        fig = TrainingVisualization.plot_training_metrics_from_csv(
            str(training_csv),
            save_path=str(output_path),
        )
        if fig is None and results_path.exists():
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_training_curves(
                str(results_path),
                save_path=str(output_path),
            )
        _record_saved(saved_paths, output_path, fig)
        output_path = VISUALIZATIONS_DIR / "evaluation_topology_summary.png"
        _prepare_output_path(output_path)
        fig = TrainingVisualization.plot_evaluation_topology_summary_from_csv(
            str(evaluation_csv),
            save_path=str(output_path),
        )
        if fig is None and results_path.exists():
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_evaluation_topology_summary(
                str(results_path),
                save_path=str(output_path),
            )
        _record_saved(saved_paths, output_path, fig)
        output_path = VISUALIZATIONS_DIR / "cost_breakdown.png"
        _prepare_output_path(output_path)
        fig = TrainingVisualization.plot_cost_components_from_csv(
            str(training_csv),
            save_path=str(output_path),
        )
        if fig is None and results_path.exists():
            _prepare_output_path(output_path)
            fig = CostBreakdownVisualizer.plot_cost_components(
                str(results_path),
                save_path=str(output_path),
            )
        _record_saved(saved_paths, output_path, fig)
        output_path = VISUALIZATIONS_DIR / "split_usage.png"
        _prepare_output_path(output_path)
        fig = TrainingVisualization.plot_split_usage_over_training_from_csv(
            str(training_csv),
            save_path=str(output_path),
        )
        if fig is None and results_path.exists():
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_split_usage(
                str(results_path),
                save_path=str(output_path),
            )
        _record_saved(saved_paths, output_path, fig)
        if episode_trace_csvs:
            output_path = VISUALIZATIONS_DIR / "episode_trace.png"
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_episode_trace_from_csv(
                str(episode_trace_csvs[0]),
                save_path=str(output_path),
            )
            _record_saved(saved_paths, output_path, fig)
    elif mode == "full":
        print("Generating full visualization suite...")
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
            saved_paths.append(output_path)
        if results_path.exists():
            for output_name, func in [
                ("split_usage.png", lambda: TrainingVisualization.plot_split_usage(str(results_path), save_path=str(VISUALIZATIONS_DIR / "split_usage.png"))),
                ("training_curves.png", lambda: TrainingVisualization.plot_training_curves(str(results_path), save_path=str(VISUALIZATIONS_DIR / "training_curves.png"))),
                ("cost_breakdown.png", lambda: CostBreakdownVisualizer.plot_cost_components(str(results_path), save_path=str(VISUALIZATIONS_DIR / "cost_breakdown.png"))),
                ("phase_analysis.png", lambda: TrainingVisualization.plot_statistics(str(results_path), save_path=str(VISUALIZATIONS_DIR / "phase_analysis.png"))),
                ("cost_breakdown_by_phase.png", lambda: TrainingVisualization.plot_cost_breakdown_by_phase(str(results_path), save_path=str(VISUALIZATIONS_DIR / "cost_breakdown_by_phase.png"))),
                ("evaluation_topology_summary_legacy.png", lambda: TrainingVisualization.plot_evaluation_topology_summary(str(results_path), save_path=str(VISUALIZATIONS_DIR / "evaluation_topology_summary_legacy.png"))),
            ]:
                fig = func()
                if fig is not None:
                    saved_paths.append(VISUALIZATIONS_DIR / output_name)
        for output_name, func in [
            ("training_metrics_from_csv.png", lambda: TrainingVisualization.plot_training_metrics_from_csv(str(training_csv), save_path=str(VISUALIZATIONS_DIR / "training_metrics_from_csv.png"))),
            ("cost_components_from_csv.png", lambda: TrainingVisualization.plot_cost_components_from_csv(str(training_csv), save_path=str(VISUALIZATIONS_DIR / "cost_components_from_csv.png"))),
            ("evaluation_topology_summary.png", lambda: TrainingVisualization.plot_evaluation_topology_summary_from_csv(str(evaluation_csv), save_path=str(VISUALIZATIONS_DIR / "evaluation_topology_summary.png"))),
            ("eval_split_distribution.png", lambda: TrainingVisualization.plot_eval_split_distribution_by_topology_from_csv(str(evaluation_csv), save_path=str(VISUALIZATIONS_DIR / "eval_split_distribution.png"))),
            ("timing_profile.png", lambda: TrainingVisualization.plot_timing_profile_from_csv(str(timing_csv), save_path=str(VISUALIZATIONS_DIR / "timing_profile.png"))),
            ("invalid_reason_summary.png", lambda: TrainingVisualization.plot_invalid_reason_summary_from_csv(str(invalid_csv), save_path=str(VISUALIZATIONS_DIR / "invalid_reason_summary.png"))),
            ("split_usage_over_training.png", lambda: TrainingVisualization.plot_split_usage_over_training_from_csv(str(training_csv), save_path=str(VISUALIZATIONS_DIR / "split_usage_over_training.png"))),
            ("stationarity_summary.png", lambda: TrainingVisualization.plot_stationarity_summary_from_trace_csvs([str(path) for path in episode_trace_csvs], save_path=str(VISUALIZATIONS_DIR / "stationarity_summary.png"))),
        ]:
            fig = func()
            if fig is not None:
                saved_paths.append(VISUALIZATIONS_DIR / output_name)
        if episode_trace_csvs:
            for trace_csv in episode_trace_csvs:
                output_name = f"episode_trace_{trace_csv.stem}.png"
                fig = TrainingVisualization.plot_episode_trace_from_csv(
                    str(trace_csv),
                    save_path=str(VISUALIZATIONS_DIR / output_name),
                )
                if fig is not None:
                    saved_paths.append(VISUALIZATIONS_DIR / output_name)
    else:
        print("Generating compact default visualizations...")
        if training_csv.exists():
            output_path = VISUALIZATIONS_DIR / "training_curves.png"
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_training_metrics_from_csv(
                str(training_csv),
                save_path=str(output_path),
            )
            if fig is None and results_path.exists():
                _prepare_output_path(output_path)
                fig = TrainingVisualization.plot_training_curves(
                    str(results_path),
                    save_path=str(output_path),
                )
            _record_saved(saved_paths, output_path, fig)
            output_path = VISUALIZATIONS_DIR / "cost_breakdown.png"
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_cost_components_from_csv(
                str(training_csv),
                save_path=str(output_path),
            )
            if fig is None and results_path.exists():
                _prepare_output_path(output_path)
                fig = CostBreakdownVisualizer.plot_cost_components(
                    str(results_path),
                    save_path=str(output_path),
                )
            _record_saved(saved_paths, output_path, fig)
        elif results_path.exists():
            output_path = VISUALIZATIONS_DIR / "training_curves.png"
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_training_curves(
                str(results_path),
                save_path=str(output_path),
            )
            _record_saved(saved_paths, output_path, fig)
            output_path = VISUALIZATIONS_DIR / "cost_breakdown.png"
            _prepare_output_path(output_path)
            fig = CostBreakdownVisualizer.plot_cost_components(
                str(results_path),
                save_path=str(output_path),
            )
            _record_saved(saved_paths, output_path, fig)

        if evaluation_csv.exists():
            output_path = VISUALIZATIONS_DIR / "evaluation_topology_summary.png"
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_evaluation_topology_summary_from_csv(
                str(evaluation_csv),
                save_path=str(output_path),
            )
            if fig is None and results_path.exists():
                _prepare_output_path(output_path)
                fig = TrainingVisualization.plot_evaluation_topology_summary(
                    str(results_path),
                    save_path=str(output_path),
                )
            _record_saved(saved_paths, output_path, fig)
        elif results_path.exists():
            output_path = VISUALIZATIONS_DIR / "evaluation_topology_summary.png"
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_evaluation_topology_summary(
                str(results_path),
                save_path=str(output_path),
            )
            _record_saved(saved_paths, output_path, fig)

        if results_path.exists():
            output_path = VISUALIZATIONS_DIR / "split_usage.png"
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_split_usage(
                str(results_path),
                save_path=str(output_path),
            )
            _record_saved(saved_paths, output_path, fig)

        if episode_trace_csvs:
            output_path = VISUALIZATIONS_DIR / "episode_trace.png"
            _prepare_output_path(output_path)
            fig = TrainingVisualization.plot_episode_trace_from_csv(
                str(episode_trace_csvs[0]),
                save_path=str(output_path),
            )
            _record_saved(saved_paths, output_path, fig)

    plt.close("all")
    _print_saved(saved_paths)
    _print_skip_summary()

    print("=" * 70)
    print("✓ Visualization Run Complete")
    print("=" * 70)


def build_visualize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GPPO visualizations")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RUN_DIR, help="Run directory or training results JSON path")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Policy checkpoint file path or directory")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--compact", action="store_true", help="Generate the compact default visualization set")
    mode_group.add_argument("--full-visualization", action="store_true", help="Generate the larger legacy visualization suite")
    mode_group.add_argument("--topology-only", action="store_true", help="Generate topology snapshot visualizations only")
    mode_group.add_argument("--csv-only", action="store_true", help="Generate only CSV-backed visualizations")
    return parser


def main(argv=None) -> None:
    parser = build_visualize_parser()
    args = parser.parse_args(argv)
    mode = "compact"
    if args.full_visualization:
        mode = "full"
    elif args.topology_only:
        mode = "topology"
    elif args.csv_only:
        mode = "csv"
    generate_all_visualizations(results_path=args.results_path, checkpoint_path=args.checkpoint_path, mode=mode)
