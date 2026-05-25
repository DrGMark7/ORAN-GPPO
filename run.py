import argparse
from pathlib import Path

from src.visualization.animation import create_all_animations
from src.workflows.demo import run_demo
from src.workflows.training import build_train_parser, run_training_from_args
from src.workflows.visualize import generate_all_visualizations


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Convenience runner for GPPO workflows")
    parser.add_argument(
        "task",
        choices=["demo", "train", "visualize", "animate"],
        help="Workflow to run",
    )
    parser.add_argument("--results-path", type=Path, default=None, help="Optional training results path for animate")
    parser.add_argument("--gif-workers", type=int, default=None, help="Worker threads for GIF frame rendering")
    args, remaining = parser.parse_known_args(argv)

    if args.task == "demo":
        run_demo()
        return
    if args.task == "train":
        train_args = build_train_parser().parse_args(remaining)
        run_training_from_args(train_args)
        return
    if args.task == "visualize":
        generate_all_visualizations()
        return
    results_path = args.results_path if args.results_path is not None else None
    create_all_animations(results_path=results_path, gif_workers=args.gif_workers) if results_path else create_all_animations(gif_workers=args.gif_workers)


if __name__ == "__main__":
    main()
