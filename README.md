# GPPO for O-RAN Resource Management

Simplified implementation of graph-augmented PPO for O-RAN resource management, using PyTorch and PyTorch Geometric.

## Project Layout

```text
intern-research/
├── main.py                    # Primary CLI entrypoint
├── run.py                     # Convenience launcher
├── train.py                   # Training entrypoint
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
│       ├── training.py        # Train/evaluate workflow
│       └── visualize.py       # Visualization workflow
├── outputs/                   # Generated checkpoints and metrics
└── visualizations/            # Generated plots
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Default Commands

Run the demo suite:

```bash
python main.py demo
```

Train the agent:

```bash
python train.py --episodes 100 --device cpu --benchmark small
python train.py --episodes 100 --device cpu --benchmark large
```

These are project benchmark settings:

- `small`: `8 RH / 3 ES / 2 RC`
- `large`: `16 RH / 5 ES / 3 RC`

Generate visualizations from the latest outputs:

```bash
python main.py visualize
```

Generate animations:

```bash
python main.py animate
```

Use the convenience launcher:

```bash
python run.py demo
python run.py train --episodes 50 --benchmark small --skip-eval
python run.py visualize
```

## Compatibility Commands

The previous script entrypoints still work and now delegate into `src/`:

```bash
python demo.py
python train_gppo.py --episodes 100
python visualize_results.py
python animate_training.py
```

## Outputs

Training writes artifacts to:

- `outputs/training_results.json`
- `outputs/gppo_policy.pt`

`training_results.json` includes policy-derived diagnostics such as split usage, per-episode cost breakdowns, per-topology evaluation summaries, and GNN training verification.

Visualization commands write plots to:

- `visualizations/01_network_topology.png`
- `visualizations/02_action_space.png`
- `visualizations/03_cost_breakdown.png`  # policy-derived split usage
- `visualizations/04_training_curves.png`
- `visualizations/05_phase_analysis.png`
- `visualizations/06_baseline_comparison.png`  # policy-derived cost breakdown
- `visualizations/07_cost_breakdown_by_phase.png`
- `visualizations/08_evaluation_topology_summary.png`

Animation commands write assets to:

- `animations/01_learning_progress.mp4`
- `animations/02_reward_landscape.png`
- `animations/03_network_state.mp4`

## Core Components

- `src/core/environment.py`: O-RAN simulator with topology-pool selection, capacity, and masking logic
- `src/core/topologies.py`: project benchmark topology families including balanced, clustered, sparse-backhaul, and direct-heavy variants
- `src/core/topology_pool.py`: train/test topology pools with `fixed` and `random_per_reset` selection
- `src/core/gnn.py`: graph builder plus GINE-based graph encoder
- `src/core/agent.py`: masked PPO policy, rollout storage, and PPO update step
- `src/workflows/training.py`: benchmark-aware training loop plus per-topology train/test diagnostics
- `src/visualization/plots.py`: topology, training, cost, and comparison charts

## Notes

- The repository still contains legacy root-level module names for compatibility, but new code should import from `src.*`.
- Training and visualization paths are centralized in `src/common/paths.py`.
- Matplotlib may warn about a non-writable config directory in restricted environments; generated files still work.

## Architecture

For a code-mapped architecture reference with Mermaid diagrams covering entrypoints, workflows, training flow, visualization flow, and compatibility wrappers, see [ARCHITECTURE.md](/home/hpcnc/intern-research/ARCHITECTURE.md).
