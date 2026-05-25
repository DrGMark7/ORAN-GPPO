import argparse
from pathlib import Path

from src.visualization.animation import create_all_animations
from src.workflows.demo import run_demo
from src.workflows.training import build_train_parser, run_training_from_args
from src.workflows.visualize import build_visualize_parser, generate_all_visualizations


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="GPPO project entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="Run the demo workflow")

    train_parser = subparsers.add_parser("train", help="Train the GPPO agent")
    for action in build_train_parser()._actions[1:]:
        train_parser._add_action(action)

    visualize_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    for action in build_visualize_parser()._actions[1:]:
        visualize_parser._add_action(action)

    animate_parser = subparsers.add_parser("animate", help="Generate animations")
    animate_parser.add_argument("--results-path", type=Path, default=None, help="Optional training results path")
    animate_parser.add_argument("--gif-workers", type=int, default=None, help="Worker threads for GIF frame rendering")

    args = parser.parse_args(argv)

    if args.command == "demo":
        run_demo()
    elif args.command == "train":
        run_training_from_args(args)
    elif args.command == "visualize":
        generate_all_visualizations(results_path=args.results_path, checkpoint_path=args.checkpoint_path)
    elif args.command == "animate":
        create_all_animations(args.results_path, gif_workers=args.gif_workers) if args.results_path else create_all_animations(gif_workers=args.gif_workers)


if __name__ == "__main__":
    main()
