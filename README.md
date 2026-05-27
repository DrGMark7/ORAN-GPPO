# GPPO for O-RAN Resource Management

Simplified implementation of graph-augmented PPO for O-RAN resource management, using PyTorch and PyTorch Geometric.

## Project Layout

```text
intern-research/
├── main.py                    # Primary CLI entrypoint
├── run.py                     # Convenience launcher
├── train.py                   # Compatibility wrapper
├── demo.py                    # Compatibility wrapper
├── train_gppo.py              # Compatibility wrapper
├── visualize_results.py       # Compatibility wrapper
├── animate_training.py        # Compatibility wrapper
├── src/
│   ├── common/
│   │   └── paths.py           # Shared output paths
│   ├── core/
│   │   ├── agent.py           # PPO policy and agent
│   │   ├── environment.py     # O-RAN environment
│   │   ├── experiment_config.py # Benchmark experiment config
│   │   ├── gnn.py             # Graph encoder and graph builder
│   │   ├── topologies.py      # Named topology definitions and pools
│   │   └── topology_pool.py   # Topology selection logic
│   ├── visualization/
│   │   ├── animation.py       # Animation generation
│   │   └── plots.py           # Plotting utilities
│   └── workflows/
│       ├── demo.py            # Demo workflow
│       ├── paper_training.py  # Paper-mode orchestration
│       ├── training.py        # Train/evaluate workflow
│       ├── training_constants.py
│       ├── training_csv.py
│       ├── vectorized_training.py
│       └── visualize.py       # Visualization workflow
├── outputs/                   # Generated checkpoints and metrics
├── visualizations/            # Generated plots
└── animations/                # Generated animations
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run the demo suite:

```bash
python run.py demo
```

Train with project defaults:

```bash
python run.py train --episodes 100 --device cpu --benchmark small
python run.py train --episodes 100 --device cpu --benchmark large
```

Train with the paper-aligned preset:

```bash
python run.py train --paper-mode --device cpu --benchmark small
python run.py train --paper-mode --device cpu --benchmark large
```

Generate plots:

```bash
python run.py visualize
```

Generate animations:

```bash
python run.py animate
python run.py animate --episode-trace-path outputs/episode_traces/train_small_balanced_train_a_legacy_episode_000.json
```

## Benchmarks

Available benchmark labels:

- `small`: project benchmark, `8 RH / 3 ES / 2 RC`
- `large`: project benchmark, `16 RH / 5 ES / 3 RC`
- `paper_small`: paper-aligned benchmark, `8 RH / 3 ES / 2 RC`
- `paper_large`: paper-aligned benchmark, `64 RH / 4 ES / 2 RC`

## CLI Overview

The top-level convenience entrypoint is:

```bash
python run.py <task> [flags]
```

Available tasks:

- `demo`
- `train`
- `visualize`
- `animate`

`run.py` only owns a small set of wrapper flags. The full training flags are handled by `src/workflows/training.py`.

## `run.py` Flags

Syntax:

```bash
python run.py <task> [--results-path PATH] [--gif-workers N] [--episode-trace-path PATH]
```

Flags:

- `task`: selects the workflow. Valid values are `demo`, `train`, `visualize`, and `animate`.
- `--results-path PATH`: top-level results file path. For `train`, it is forwarded into the training parser. For `animate`, it selects which training results JSON to animate. It is not currently forwarded by `run.py visualize`.
- `--gif-workers N`: number of worker threads used during learning-animation frame rendering in the animate workflow. The output is now MP4, but the flag name is still `gif-workers`.
- `--episode-trace-path PATH`: explicit evaluated episode trace JSON for the episode animation.

Examples:

```bash
python run.py train --results-path outputs/my_run.json --episodes 50 --benchmark small
python run.py animate --results-path outputs/my_run.json
python run.py animate --episode-trace-path outputs/episode_traces/train_small_balanced_train_a_legacy_episode_000.json
```

## Training Flags

Preferred command:

```bash
python run.py train [training flags]
```

Equivalent direct workflow command:

```bash
python -m src.workflows.training [training flags]
```

### Core Run Control

- `--episodes INT`
  Number of training episodes in normal project mode. Default: `100`.
- `--benchmark {large,paper_large,paper_small,small}`
  Benchmark group to load. Default: `small`.
- `--device STR`
  Torch device such as `cpu` or `cuda`. Default: `cpu`.
- `--seed INT`
  Random seed for the run. Default: `42`.
- `--max-steps INT`
  Time slots per episode in normal project mode. Default: `50`.
- `--batch-size INT`
  PPO batch size used during updates. Default: `128`.

### Paper-Length and Paper-Mode Flags

- `--paper-episode-length`
  Forces the episode horizon to `288` time slots while staying in the normal project training flow.
- `--paper-mode`
  Switches to the paper-oriented preset path. This uses the paper-mode training orchestration rather than the normal episode-count loop.
- `--paper-timesteps INT`
  Total timesteps collected per seed in paper mode. Default: `600000`.
- `--paper-num-envs INT`
  Number of synchronous parallel environments in paper mode. Default: `32`.
- `--paper-num-seeds INT`
  Number of sequential seeds used for aggregate paper-mode reporting. Default: `6`.
- `--paper-gnn-hidden-dim INT`
  Hidden size for the paper-mode GNN. Default: `1024`.

Important behavior:

- `--paper-mode` short-circuits the normal `--episodes` flow.
- `--paper-mode --benchmark large` is mapped to the paper-aligned `paper_large` benchmark.
- `--paper-mode --benchmark small` is mapped to the paper-aligned `paper_small` benchmark.

### Output and Evaluation Flags

- `--results-path PATH`
  Output JSON path for run metrics. Default: `outputs/training_results.json`.
- `--checkpoint-path PATH`
  Output checkpoint path. Default: `outputs/gppo_policy.pt`.
- `--skip-eval`
  Skips post-training evaluation on the train/test topology pools.
- `--eval-episodes INT`
  Number of evaluation episodes per pool after training. Default: `1`.

### Topology and Constraint Flags

- `--topology-selection-mode {fixed,random_per_reset}`
  How the train topology is chosen during resets. Default: `random_per_reset`.
- `--constraint-mode {legacy,strict,strict_connectivity_only,strict_connectivity_plus_capacity,strict_connectivity_plus_bandwidth,strict_full}`
  Selects how aggressively feasibility constraints are enforced and audited. Default: `legacy`.
- `--train-topology-id STR`
  Forces a specific topology from the train pool instead of sampling by pool mode. Default: unset.

### Training Examples

Project-mode training:

```bash
python run.py train --episodes 100 --device cuda --benchmark small
python run.py train --episodes 200 --device cuda --benchmark large --constraint-mode legacy
python run.py train --episodes 100 --device cuda --benchmark small --skip-eval
```

Fixed topology debugging:

```bash
python run.py train --episodes 20 --benchmark small --topology-selection-mode fixed --train-topology-id small_balanced_train_a
```

Paper-length without full paper mode:

```bash
python run.py train --episodes 100 --benchmark small --paper-episode-length
```

Paper-mode aggregate run:

```bash
python run.py train --paper-mode --benchmark large --device cuda --paper-num-seeds 6 --paper-num-envs 32
```

## Visualization Flags

Direct workflow command:

```bash
python -m src.workflows.visualize [flags]
```

Supported flags:

- `--results-path PATH`
  Training results JSON to visualize. Default: `outputs/training_results.json`.
- `--checkpoint-path PATH`
  Policy checkpoint used for topology-aware visualizations. Default: `outputs/gppo_policy.pt`.

Example:

```bash
python -m src.workflows.visualize --results-path outputs/my_run.json --checkpoint-path outputs/my_policy.pt
```

Note:

- `python run.py visualize` uses default paths only.
- If you need custom paths, use `python -m src.workflows.visualize ...`.

## Animation Flags

Top-level command:

```bash
python run.py animate [flags]
```

Supported flags:

- `--results-path PATH`
  Training results JSON used for learning-progress and reward-landscape animations. Default fallback: `outputs/training_results.json`.
- `--gif-workers INT`
  Worker threads for learning-animation frame rendering. The flag name is historical; the rendered animation is MP4.
- `--episode-trace-path PATH`
  Explicit episode trace JSON for the paper-aligned episode animation. If omitted, the animation workflow tries to auto-discover a default trace under `outputs/episode_traces/`.

Outputs:

- `animations/01_learning_progress.mp4`
- `animations/02_reward_landscape.png`
- `animations/03_episode_evolution.mp4`
- `animations/04_network_state.mp4`

## Compatibility Commands

Legacy root-level wrappers still exist:

```bash
python demo.py
python train.py --episodes 100
python train_gppo.py --episodes 100
python visualize_results.py
python animate_training.py
```

New code should prefer `run.py` or `src.workflows.*`.

## Outputs

Default output files:

- `outputs/training_results.json`
- `outputs/gppo_policy.pt`
- `outputs/training_episode_metrics.csv`
- `outputs/evaluation_topology_summary.csv`
- `outputs/invalid_reason_summary.csv`
- `outputs/timing_profile.csv`
- `outputs/episode_traces/*.json`
- `outputs/episode_traces/*.csv`

`training_results.json` includes:

- per-episode rewards and costs
- validity rates
- split usage
- reconfiguration statistics
- per-topology evaluation summaries
- timing profile
- GNN training verification
- CSV export paths

Episode trace JSON and CSV files contain slot-by-slot decisions and diagnostics, including:

- split / ES / RC choices
- reward and deployment cost
- cost components
- reconfiguration changes
- invalid reasons and violation counters

Visualization outputs are written under `visualizations/`.
Animation outputs are written under `animations/`.

## Paper Time-Slot Semantics

- One environment step corresponds to one paper time slot.
- At every time slot, each RH re-selects exactly one split, one ES, and one RC for that slot.
- Reconfiguration cost is computed from decision changes between consecutive time slots.
- `--paper-episode-length` gives the `288`-slot horizon inside the normal workflow.
- `--paper-mode` switches to the paper-oriented multi-seed, multi-environment training path.

## Notes

- The repository still contains legacy root-level module names for compatibility, but new code should import from `src.*`.
- Training and visualization paths are centralized in [paths.py](/home/hpcnc/intern-research/src/common/paths.py).
- Matplotlib may warn about a non-writable config directory in restricted environments; generated files still work.

## Architecture

For a code-mapped architecture reference with Mermaid diagrams covering entrypoints, workflows, training flow, visualization flow, and compatibility wrappers, see [ARCHITECTURE.md](/home/hpcnc/intern-research/ARCHITECTURE.md).
