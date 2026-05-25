"""Filesystem paths for outputs and generated assets."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
VISUALIZATIONS_DIR = REPO_ROOT / "visualizations"
ANIMATIONS_DIR = REPO_ROOT / "animations"

DEFAULT_RESULTS_PATH = OUTPUTS_DIR / "training_results.json"
DEFAULT_CHECKPOINT_PATH = OUTPUTS_DIR / "gppo_policy.pt"


def ensure_output_dirs() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    ANIMATIONS_DIR.mkdir(exist_ok=True)
