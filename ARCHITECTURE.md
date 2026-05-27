# GPPO Project Architecture

This document describes the current software architecture as it exists in the repository now.

It focuses on:

1. CLI entrypoints and wrappers
2. core training and evaluation flow
3. paper-mode orchestration
4. run-directory path resolution
5. visualization and animation flows
6. artifact layout

## 1. Top-Level Entry Points

```mermaid
flowchart TD
    User[User CLI Invocation]

    User --> MainPy[main.py]
    User --> RunPy[run.py]
    User --> TrainPy[train.py]
    User --> TrainGppoPy[train_gppo.py]
    User --> DemoPy[demo.py]
    User --> VizPy[visualize_results.py]
    User --> AnimatePy[animate_training.py]

    MainPy -->|demo| DemoWorkflow[src/workflows/demo.py::run_demo]
    MainPy -->|train| TrainWorkflow[src/workflows/training.py::run_training_from_args]
    MainPy -->|visualize| VisualizeWorkflow[src/workflows/visualize.py::generate_all_visualizations]
    MainPy -->|animate| AnimateWorkflow[src/visualization/animation.py::create_all_animations]

    RunPy -->|demo| DemoWorkflow
    RunPy -->|train| TrainParser[src/workflows/training.py::build_train_parser]
    TrainParser --> TrainWorkflow
    RunPy -->|visualize| VisualizeParser[src/workflows/visualize.py::build_visualize_parser]
    VisualizeParser --> VisualizeWorkflow
    RunPy -->|animate| AnimateWorkflow

    TrainPy --> TrainingMain[src/workflows/training.py::main]
    TrainGppoPy --> TrainingMain
    DemoPy --> DemoWorkflow
    VizPy --> VisualizeMain[src/workflows/visualize.py::main]
    VisualizeMain --> VisualizeWorkflow
    AnimatePy --> AnimateWorkflow
```

## 2. Module Architecture

```mermaid
flowchart LR
    subgraph Common[src/common]
        Paths[src/common/paths.py]
    end

    subgraph Core[src/core]
        Env[src/core/environment.py\nSimplifiedORANEnv]
        GNN[src/core/gnn.py\nORANGraphBuilder\nGNNFeatureExtractor]
        Agent[src/core/agent.py\nMaskedPPOPolicy\nPPOAgent]
        Topologies[src/core/topologies.py]
        Pools[src/core/topology_pool.py]
        Config[src/core/experiment_config.py]
    end

    subgraph Workflows[src/workflows]
        Demo[src/workflows/demo.py]
        Training[src/workflows/training.py]
        Vectorized[src/workflows/vectorized_training.py]
        Paper[src/workflows/paper_training.py]
        Csv[src/workflows/training_csv.py]
        Constants[src/workflows/training_constants.py]
        Visualize[src/workflows/visualize.py]
    end

    subgraph Visualization[src/visualization]
        Plots[src/visualization/plots.py]
        Animation[src/visualization/animation.py]
    end

    Training --> Env
    Training --> GNN
    Training --> Agent
    Training --> Topologies
    Training --> Pools
    Training --> Config
    Training --> Paths
    Training --> Csv
    Training --> Constants
    Training --> Paper
    Training --> Vectorized

    Vectorized --> Env
    Vectorized --> GNN
    Vectorized --> Agent
    Vectorized --> Paths

    Paper --> Training
    Paper --> Csv
    Paper --> Constants
    Paper --> Paths

    Demo --> Env
    Demo --> GNN
    Demo --> Agent

    Visualize --> Paths
    Visualize --> Plots

    Plots --> Env
    Plots --> GNN
    Plots --> Agent
    Plots --> Paths

    Animation --> Paths
```

## 3. Run-Directory Path Resolution

The software now treats `--results-path` as either:

- a run directory such as `outputs/my_run`
- or a concrete JSON file such as `outputs/my_run/training_results.json`

This resolution is centralized in `src/common/paths.py`.

```mermaid
flowchart TD
    Input[User path argument]
    Input --> ResolveResults[resolve_results_path]
    Input --> ResolveRunDir[resolve_run_dir]
    Input --> ResolveTraces[resolve_episode_traces_dir]

    ResolveResults --> ResultsJson[training_results.json]
    ResolveRunDir --> RunDir[run directory]
    ResolveTraces --> TraceDir[run_dir/episode_traces]

    ResolveRunDir --> ResolveCheckpoint[resolve_checkpoint_path]
    ResolveCheckpoint --> CheckpointPt[gppo_policy.pt]
```

### Current path rules

| Function | File | Purpose |
| --- | --- | --- |
| `resolve_results_path()` | [paths.py](/home/hpcnc/intern-research/src/common/paths.py:25) | Normalizes a run directory or JSON path into the actual results JSON path |
| `resolve_run_dir()` | [paths.py](/home/hpcnc/intern-research/src/common/paths.py:35) | Returns the owning run directory |
| `resolve_checkpoint_path()` | [paths.py](/home/hpcnc/intern-research/src/common/paths.py:38) | Places the checkpoint in the same run directory unless an explicit file path is given |
| `resolve_episode_traces_dir()` | [paths.py](/home/hpcnc/intern-research/src/common/paths.py:51) | Places trace JSON files under `run_dir/episode_traces/` |

## 4. Training Workflow

`run_training_from_args()` is the main orchestrator for project-mode training.

```mermaid
flowchart TD
    CLI[CLI train args] --> ResolvePaths[Resolve run paths]
    ResolvePaths --> PaperMode{paper_mode?}

    PaperMode -->|Yes| PaperRunner[src/workflows/paper_training.py::_run_paper_mode_from_args]
    PaperMode -->|No| TrainLoop[src/workflows/training.py::train_gppo]

    TrainLoop --> EnvCtor[Construct SimplifiedORANEnv]
    TrainLoop --> GnnCtor[Construct GNNFeatureExtractor]
    TrainLoop --> BuilderCtor[Construct ORANGraphBuilder]
    TrainLoop --> AgentCtor[Construct PPOAgent]

    AgentCtor --> EpisodeLoop{Per episode}
    EpisodeLoop --> Reset[env.reset]
    Reset --> StepLoop{Per time slot}
    StepLoop --> Adj[env._get_adjacency_info]
    Adj --> BuildGraph[graph_builder.build_graph]
    BuildGraph --> Encode[gnn forward]
    Encode --> ActionMask[env.get_action_mask]
    ActionMask --> Select[agent.select_action_sequential]
    Select --> Step[env.step]
    Step --> Store[agent.store_transition]
    Store --> DoneCheck{done?}
    DoneCheck -->|No| StepLoop
    DoneCheck -->|Yes| Update[agent.update]
    Update --> EpisodeMetrics[Collect reward/cost/validity/reconfig stats]
    EpisodeMetrics --> EpisodeLoop

    EpisodeLoop --> Eval{skip_eval?}
    Eval -->|No| TrainEval[evaluate_pool_by_topology train]
    TrainEval --> TestEval[evaluate_pool_by_topology test]
    Eval -->|Yes| CsvExport
    TestEval --> CsvExport[training_csv._export_csv_artifacts]
    CsvExport --> SaveJson[Write training_results.json]
    SaveJson --> SavePt[Write gppo_policy.pt]
```

## 5. Vectorized Paper-Mode Training

When `num_envs > 1` or `total_timesteps` is set, `train_gppo()` delegates to the synchronous vectorized path in `src/workflows/vectorized_training.py`.

```mermaid
flowchart TD
    TrainGPPO[train_gppo] --> Branch{num_envs > 1 or total_timesteps set?}
    Branch -->|No| StandardLoop[Standard episode loop in training.py]
    Branch -->|Yes| SyncVectorized[_train_gppo_sync_vectorized]

    SyncVectorized --> EnvGroup[Build N environments]
    EnvGroup --> ResetWorkers[Reset all workers]
    ResetWorkers --> TimestepLoop{Until total_timesteps}
    TimestepLoop --> ForEachEnv[Iterate each active env]
    ForEachEnv --> GraphBuild[Build graph]
    GraphBuild --> GnnForward[GNN forward]
    GnnForward --> SelectAction[Masked sequential action selection]
    SelectAction --> EnvStep[env.step]
    EnvStep --> StoreTransition[agent.store_transition]
    StoreTransition --> BatchReady{batch_size reached?}
    BatchReady -->|Yes| PPOUpdate[agent.update]
    BatchReady -->|No| ContinueLoop
    PPOUpdate --> ContinueLoop
    ContinueLoop --> EpisodeDone{worker episode done?}
    EpisodeDone -->|Yes| FinalizeWorkerEpisode[append metrics and reset worker]
    EpisodeDone -->|No| TimestepLoop
```

## 6. Paper-Mode Orchestration

Paper mode is a higher-level wrapper around training and evaluation.

```mermaid
flowchart TD
    PaperCLI[--paper-mode] --> Resolve[resolve_results_path + resolve_checkpoint_path]
    Resolve --> BenchmarkMap[_paper_benchmark_name]
    BenchmarkMap --> SeedLoop{for each seed}

    SeedLoop --> SeedTrain[train_gppo with paper params]
    SeedTrain --> SeedEvalTrain[evaluate_pool_by_topology train]
    SeedEvalTrain --> SeedEvalTest[evaluate_pool_by_topology test]
    SeedEvalTest --> SeedCsv[_export_csv_artifacts]
    SeedCsv --> SeedJson[write seed JSON]
    SeedJson --> SeedLoop

    SeedLoop --> Aggregate[_seed_summary_rows]
    Aggregate --> AggregateJson[write aggregate JSON]
    AggregateJson --> ReportJson[write paper_alignment_report.json]
    ReportJson --> ReportMd[write Reports.md]
```

## 7. Environment and Decision Pipeline

The environment executes one paper-style time slot per `step()`.

```mermaid
flowchart LR
    Requests[RH demands and latency budgets]
    Topology[Topology graph]
    PrevAction[Previous action]

    Requests --> State[Environment state]
    Topology --> State
    PrevAction --> Eval

    State --> Mask[get_action_mask]
    State --> Adj[_get_adjacency_info]
    Adj --> GraphBuilder[ORANGraphBuilder.build_graph]
    GraphBuilder --> PYGGraph[PyG graph]
    PYGGraph --> GNNForward[GNNFeatureExtractor.forward]
    GNNForward --> Features[Graph embedding]
    Mask --> Policy[PPOAgent.select_action_sequential]
    Features --> Policy
    Policy --> Action[flat action vector]

    Action --> Decode[_split_action]
    Decode --> Eval[_evaluate_action]
    Eval --> Reward[step reward]
    Eval --> Info[per-slot metrics and reasons]
    Eval --> NextResources[ES/RC remaining + edge remaining]
```

### Core environment methods

| File | Method | Role |
| --- | --- | --- |
| [environment.py](/home/hpcnc/intern-research/src/core/environment.py:173) | `reset()` | Select topology, sample requests, report feasibility diagnostics |
| [environment.py](/home/hpcnc/intern-research/src/core/environment.py:234) | `_split_action()` | Decode flat action into split / ES / RC vectors |
| [environment.py](/home/hpcnc/intern-research/src/core/environment.py:249) | `get_action_mask()` | Structural action masking |
| [environment.py](/home/hpcnc/intern-research/src/core/environment.py:273) | `get_conditional_rc_mask()` | RC mask conditioned on split and ES |
| [environment.py](/home/hpcnc/intern-research/src/core/environment.py:551) | `_evaluate_action()` | Compute validity, costs, penalties, invalid reasons |
| [environment.py](/home/hpcnc/intern-research/src/core/environment.py:710) | `step()` | Convert action metrics into reward, termination, next state, and `info` |

## 8. Evaluation and Trace Export

Evaluation runs the policy deterministically and records traces plus aggregated summaries.

```mermaid
flowchart TD
    EvaluatePool[evaluate_pool_by_topology] --> ForEachTopology{topology in pool}
    ForEachTopology --> EvaluateGPPO[evaluate_gppo]

    EvaluateGPPO --> Reset[env.reset]
    Reset --> SlotLoop{per slot}
    SlotLoop --> GraphBuild[build graph]
    GraphBuild --> PolicyStep[deterministic policy action]
    PolicyStep --> EnvStep[env.step]
    EnvStep --> TraceRow[append slot trace]
    TraceRow --> SlotLoop

    SlotLoop --> EpisodeTraceJson[_write_episode_trace]
    EpisodeTraceJson --> EpisodeTraceCsv[_write_episode_trace_csv]
    EpisodeTraceCsv --> TopologySummary[aggregate per-topology metrics]
    TopologySummary --> PoolSummary[return pool summary]
```

### Trace and CSV responsibilities

| File | Function | Output |
| --- | --- | --- |
| [training.py](/home/hpcnc/intern-research/src/workflows/training.py:79) | `_write_episode_trace()` | Per-episode trace JSON |
| [training_csv.py](/home/hpcnc/intern-research/src/workflows/training_csv.py:66) | `_write_episode_trace_csv()` | Per-episode trace CSV |
| [training_csv.py](/home/hpcnc/intern-research/src/workflows/training_csv.py:238) | `_export_csv_artifacts()` | Run-level CSV summaries |

## 9. Visualization Architecture

The visualization workflow is now compact by default.

```mermaid
flowchart TD
    VisualizeEntry[generate_all_visualizations] --> Resolve[resolve_results_path + resolve_checkpoint_path]
    Resolve --> Mode{mode}

    Mode -->|compact| Compact[compact default set]
    Mode -->|full| Full[full visualization suite]
    Mode -->|topology| TopologyOnly[topology snapshots only]
    Mode -->|csv| CsvOnly[CSV-backed plots only]

    Compact --> PlotTrain[training_curves.png]
    Compact --> PlotEval[evaluation_topology_summary.png]
    Compact --> PlotSplit[split_usage.png]
    Compact --> PlotCost[cost_breakdown.png]
    Compact --> PlotTrace[episode_trace.png if available]

    Full --> LegacyPlots[extra diagnostics and topology plots]
    TopologyOnly --> TopologyPlots[per-topology snapshot PNGs]
    CsvOnly --> CsvPlots[CSV-first compact charts]
```

### Default compact output set

| Output | Source |
| --- | --- |
| `training_curves.png` | CSV-backed if available, otherwise JSON-backed |
| `evaluation_topology_summary.png` | CSV-backed if available, otherwise JSON-backed |
| `split_usage.png` | JSON-backed split-usage visualization |
| `cost_breakdown.png` | CSV-backed if available, otherwise JSON-backed |
| `episode_trace.png` | First available episode trace CSV |

### Key visualization methods

| File | Method | Purpose |
| --- | --- | --- |
| [visualize.py](/home/hpcnc/intern-research/src/workflows/visualize.py:99) | `generate_all_visualizations()` | Main visualization orchestrator |
| [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:363) | `plot_training_curves()` | JSON-backed training overview |
| [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:643) | `plot_training_metrics_from_csv()` | CSV-backed training overview |
| [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:586) | `plot_evaluation_topology_summary()` | JSON-backed evaluation summary |
| [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:731) | `plot_evaluation_topology_summary_from_csv()` | CSV-backed evaluation summary |
| [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:416) | `plot_split_usage()` | Policy interpretation via split usage |
| [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:679) | `plot_cost_components_from_csv()` | Cost trend summary |
| [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:856) | `plot_episode_trace_from_csv()` | Time-slot behavior plot |
| [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:259) | `NetworkTopologyVisualizer.draw_topology()` | Topology snapshot with optional checkpoint inference |

## 10. Animation Architecture

```mermaid
flowchart TD
    AnimateEntry[create_all_animations] --> ResolveResults[resolve_results_path]
    ResolveResults --> FindTrace[_find_default_trace_path]

    ResolveResults --> Learning[TrainingAnimator.create_learning_animation]
    ResolveResults --> RewardLandscape[TrainingAnimator.create_reward_landscape]
    FindTrace --> EpisodeAnim[EpisodeTraceAnimator.create_episode_animation]
    AnimateEntry --> NetworkAnim[NetworkStateAnimator.create_network_state_animation]

    Learning --> A1[animations/01_learning_progress.mp4]
    RewardLandscape --> A2[animations/02_reward_landscape.png]
    EpisodeAnim --> A3[animations/03_episode_evolution.mp4]
    NetworkAnim --> A4[animations/04_network_state.mp4]
```

The animation entrypoint now prefers trace JSON files under the same run directory as `--results-path`, then falls back to the global `outputs/episode_traces/` directory.

## 11. Artifact Layout

Current run-directory layout:

```mermaid
flowchart LR
    RunDir[outputs/my_run]
    RunDir --> Results[training_results.json]
    RunDir --> Checkpoint[gppo_policy.pt]
    RunDir --> Csv1[training_episode_metrics.csv]
    RunDir --> Csv2[evaluation_topology_summary.csv]
    RunDir --> Csv3[invalid_reason_summary.csv]
    RunDir --> Csv4[timing_profile.csv]
    RunDir --> TraceCsvs[*_episode_*.csv]
    RunDir --> TraceDir[episode_traces/]
    TraceDir --> TraceJsons[*_episode_*.json]
```

Paper mode adds a nested seed directory structure:

```mermaid
flowchart LR
    BaseRun[outputs/my_paper_run]
    BaseRun --> AggregateJson[training_results.json]
    BaseRun --> AggregateReport[paper_alignment_report.json]
    BaseRun --> PaperRuns[paper_runs/]

    PaperRuns --> SeedJsons[*_seed_*.json]
    PaperRuns --> SeedPts[*_seed_*.pt]
    PaperRuns --> SeedCsvs[run-level CSV summaries]
    PaperRuns --> SeedTraceCsvs[*_episode_*.csv]
    PaperRuns --> SeedTraceDir[episode_traces/]
```

## 12. Current Key Function Map

| File | Function / Class | Calls / Uses | Main output |
| --- | --- | --- | --- |
| `run.py` | `main()` | training, visualize, animate workflows | Convenience task router |
| `main.py` | `main()` | same workflows via subcommands | Alternative CLI entrypoint |
| `src/workflows/training.py` | `run_training_from_args()` | path resolution, `train_gppo()`, evaluation, CSV export | Full train/eval workflow |
| `src/workflows/training.py` | `train_gppo()` | env, GNN, graph builder, PPO agent | Standard training path |
| `src/workflows/vectorized_training.py` | `_train_gppo_sync_vectorized()` | multi-env synchronous rollout | Paper/vectorized training path |
| `src/workflows/paper_training.py` | `_run_paper_mode_from_args()` | multi-seed orchestration | Aggregate paper-mode run |
| `src/workflows/training.py` | `evaluate_gppo()` | deterministic policy rollout, trace export | Per-topology evaluation summary |
| `src/workflows/training.py` | `evaluate_pool_by_topology()` | repeated `evaluate_gppo()` | Pool-wide evaluation summary |
| `src/workflows/visualize.py` | `generate_all_visualizations()` | mode-based plot orchestration | Compact/full visualization output |
| `src/visualization/animation.py` | `create_all_animations()` | learning animation, reward landscape, episode animation, network animation | MP4/PNG animation assets |
| `src/common/paths.py` | path resolvers | run-dir normalization | Consistent artifact placement |

