"""Filesystem paths for outputs and generated assets."""

from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
EPISODE_TRACES_DIR = OUTPUTS_DIR / "episode_traces"
VISUALIZATIONS_DIR = REPO_ROOT / "visualizations"
ANIMATIONS_DIR = REPO_ROOT / "animations"

DEFAULT_RUN_DIR = OUTPUTS_DIR
TRAINING_RESULTS_FILENAME = "training_results.json"
CHECKPOINT_FILENAME = "gppo_policy.pt"
EPISODE_TRACES_SUBDIR = "episode_traces"
DEFAULT_RESULTS_PATH = OUTPUTS_DIR / "training_results.json"
DEFAULT_CHECKPOINT_PATH = OUTPUTS_DIR / "gppo_policy.pt"


def _is_directory_like(path: Path) -> bool:
    return path.suffix == ""


def resolve_results_path(path: Optional[Path]) -> Path:
    candidate = path if path is not None else DEFAULT_RUN_DIR
    if candidate.exists() and candidate.is_dir():
        return candidate / TRAINING_RESULTS_FILENAME
    if _is_directory_like(candidate):
        return candidate / TRAINING_RESULTS_FILENAME
    return candidate


def resolve_run_dir(path: Optional[Path]) -> Path:
    return resolve_results_path(path).parent


def resolve_checkpoint_path(results_path: Optional[Path], checkpoint_path: Optional[Path]) -> Path:
    run_dir = resolve_run_dir(results_path)
    if checkpoint_path is None:
        return run_dir / CHECKPOINT_FILENAME
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        return checkpoint_path / CHECKPOINT_FILENAME
    if _is_directory_like(checkpoint_path):
        return checkpoint_path / CHECKPOINT_FILENAME
    if checkpoint_path == DEFAULT_CHECKPOINT_PATH and run_dir != DEFAULT_CHECKPOINT_PATH.parent:
        return run_dir / CHECKPOINT_FILENAME
    return checkpoint_path


def resolve_episode_traces_dir(results_path: Optional[Path]) -> Path:
    return resolve_run_dir(results_path) / EPISODE_TRACES_SUBDIR


def ensure_output_dirs() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    EPISODE_TRACES_DIR.mkdir(exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    ANIMATIONS_DIR.mkdir(exist_ok=True)
