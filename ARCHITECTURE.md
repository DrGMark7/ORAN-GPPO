# GPPO Project Architecture

This document now follows the requirements in `INSTRUCTION.md`.

It covers two views:

1. the current code-mapped architecture that exists in the repository today
2. the required revised architecture for GPPO baseline reproduction, centered on controlled topology management

The key requirement from `INSTRUCTION.md` is that the environment must move from a single implicit topology to an explicit experiment structure with fixed benchmark topologies, train/test topology pools, and reset-time topology selection.

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
    MainPy -->|visualize| VizWorkflow[src/workflows/visualize.py::generate_all_visualizations]
    MainPy -->|animate| AnimateWorkflow[src/visualization/animation.py::create_all_animations]

    RunPy -->|demo| DemoWorkflow
    RunPy -->|train| TrainParser[src/workflows/training.py::build_train_parser]
    TrainParser --> TrainWorkflow
    RunPy -->|visualize| VizWorkflow
    RunPy -->|animate| AnimateWorkflow

    TrainPy --> TrainingMain[src/workflows/training.py::main]
    TrainGppoPy --> TrainingMain
    TrainingMain --> TrainParser
    TrainingMain --> TrainWorkflow

    DemoPy --> DemoWorkflow
    VizPy --> VisualizeMain[src/workflows/visualize.py::main]
    VisualizeMain --> VizWorkflow
    AnimatePy --> AnimateWorkflow
```

## 2. Source Module Architecture

```mermaid
flowchart LR
    subgraph Core[src/core]
        Env[src/core/environment.py\nSimplifiedORANEnv]
        GNN[src/core/gnn.py\nORANGraphBuilder\nGNNFeatureExtractor]
        Agent[src/core/agent.py\nMaskedPPOPolicy\nPPOAgent]
    end

    subgraph Workflows[src/workflows]
        Demo[src/workflows/demo.py]
        Training[src/workflows/training.py]
        Visualize[src/workflows/visualize.py]
    end

    subgraph Visualization[src/visualization]
        Plots[src/visualization/plots.py]
        Animation[src/visualization/animation.py]
        VizInit[src/visualization/__init__.py]
    end

    Paths[src/common/paths.py]
    Readme[README.md]

    Demo --> Env
    Demo --> GNN
    Demo --> Agent

    Training --> Env
    Training --> GNN
    Training --> Agent
    Training --> Paths

    Visualize --> VizInit
    Visualize --> Paths
    VizInit --> Plots
    VizInit --> Animation

    Plots --> Env
    Plots --> GNN
    Plots --> Agent
    Plots --> Paths

    Animation --> Paths

    Readme --> Training
    Readme --> Visualize
```

## 3. Current vs Required Environment Design

```mermaid
flowchart LR
    subgraph Current[Current prototype]
        C1[Topology created once in __init__]
        C2[reset only refreshes capacities and requests]
        C3[Single environment instance holds one graph]
        C4[Observation length depends on edge count]
        C5[Good for one fixed benchmark only]
    end

    subgraph Required[Required baseline reproduction design]
        R1[Topology is explicit experiment object]
        R2[Named benchmark topologies]
        R3[Train topology pool and test topology pool]
        R4[reset can select and rebuild topology]
        R5[Stable observation shape inside each benchmark group]
        R6[Topology metadata exposed in info]
    end

    C1 --> R1
    C2 --> R4
    C3 --> R3
    C4 --> R5
    C5 --> R2
```

## 4. Current Training Pipeline

```mermaid
flowchart TD
    Start[train_gppo] --> Seed[Set numpy and torch seeds]
    Seed --> EnsureDirs[ensure_output_dirs]
    EnsureDirs --> EnvCtor[Construct SimplifiedORANEnv]
    EnvCtor --> GnnCtor[Construct GNNFeatureExtractor]
    GnnCtor --> BuilderCtor[Construct ORANGraphBuilder]
    BuilderCtor --> AgentCtor[Construct PPOAgent]
    AgentCtor --> EpisodeLoop{For each episode}

    EpisodeLoop --> Reset[env.reset]
    Reset --> StepLoop{For each step}

    StepLoop --> Adj[env._get_adjacency_info]
    Adj --> BuildGraph[graph_builder.build_graph]
    BuildGraph --> Encode[gnn graph forward pass]
    Encode --> Mask[env.get_action_mask]
    Mask --> Select[agent.select_action]
    Select --> EnvStep[env.step]
    EnvStep --> Store[agent.store_transition]
    Store --> Metrics[Accumulate reward, cost, validity]
    Metrics --> DoneCheck{terminated or truncated?}
    DoneCheck -->|No| StepLoop
    DoneCheck -->|Yes| Update[agent.update]
    Update --> EpisodeStats[Append episode metrics]
    EpisodeStats --> EpisodeLoop

    EpisodeLoop -->|finished| WriteJson[Write outputs/training_results.json]
    WriteJson --> SavePolicy[agent.save outputs/gppo_policy.pt]
    SavePolicy --> Return[Return agent, gnn, results]
```

## 5. Current Training Runtime Data Flow

```mermaid
flowchart LR
    Requests[Sampled RH demands and latencies] --> EnvState[Environment state]
    Topology[NetworkX topology with bandwidth and delay] --> EnvState

    EnvState --> Adjacency[Adjacency plus edge feature dict]
    EnvState --> Mask[Action mask]

    Adjacency --> GraphBuilder[ORANGraphBuilder.build_graph]
    GraphBuilder --> PYGGraph[torch_geometric.data.Data]
    PYGGraph --> GNNForward[GNNFeatureExtractor.forward]
    GNNForward --> Embedding[128-d graph embedding]

    Mask --> SelectAction[PPOAgent.select_action]
    Embedding --> SelectAction
    SelectAction --> Action[3N action vector\nsplit + ES + RC]

    Action --> Step[env.step]
    Step --> Reward[reward]
    Step --> NextState[next state]
    Step --> Info[deployment cost, penalties, validity]

    Embedding --> StoreTransition[PPOAgent.store_transition]
    Action --> StoreTransition
    Reward --> StoreTransition
    Mask --> StoreTransition
    StoreTransition --> RolloutBuffer[states, actions, rewards, values,\nlog_probs, dones, action_masks]
    RolloutBuffer --> Update[PPOAgent.update]
```

## 6. Required Paper-Aligned Topology Architecture

This is the target architecture implied by `INSTRUCTION.md`. It is the design the project should follow before adding GPU-aware or mobility-aware extensions.

```mermaid
flowchart TD
    subgraph Config[Experiment configuration]
        Bench[Benchmark name\nsmall or large]
        SelectMode[topology_selection_mode\nfixed or random_per_reset]
        SeedCfg[reproducible seeds]
    end

    subgraph Registry[Topology registry layer]
        NamedTopo[Named topology definitions\nsmall_fixed_topology\nlarge_fixed_topology]
        TrainPool[train_topology_pool]
        TestPool[test_topology_pool]
    end

    subgraph Env[Revised environment]
        EnvInit[SimplifiedORANEnv init]
        Reset[reset]
        ChooseTopo[select or load topology]
        BuildGraphState[rebuild topology-dependent state]
        SampleReq[sample traffic requests]
        GetObs[get fixed-group observation]
        Step[step]
        Info[info with topology_id and metadata]
    end

    Bench --> NamedTopo
    Bench --> TrainPool
    Bench --> TestPool
    SelectMode --> Reset
    SeedCfg --> Reset

    NamedTopo --> EnvInit
    TrainPool --> ChooseTopo
    TestPool --> ChooseTopo

    EnvInit --> Reset
    Reset --> ChooseTopo
    ChooseTopo --> BuildGraphState
    BuildGraphState --> SampleReq
    SampleReq --> GetObs
    GetObs --> Step
    Step --> Info
```

## 7. Required Reset-Time Topology Reload Flow

This is the main environment change requested by `INSTRUCTION.md`.

```mermaid
sequenceDiagram
    participant Trainer as Training loop
    participant Env as SimplifiedORANEnv
    participant Pool as Topology pool
    participant Topo as Topology object

    Trainer->>Env: reset(seed, mode)
    Env->>Pool: choose topology_id
    Pool-->>Env: topology definition
    Env->>Topo: build or load graph
    Topo-->>Env: topology, node_order, edge_order
    Env->>Env: rebuild adjacency-dependent arrays
    Env->>Env: reset ES and RC capacities
    Env->>Env: sample slice requests
    Env->>Env: refresh edge state
    Env-->>Trainer: observation, info{topology_id, benchmark, metadata}
```

## 8. Required Topology-Pool Training Pipeline

```mermaid
flowchart TD
    TrainStart[train_gppo] --> ExpCfg[Load experiment config]
    ExpCfg --> BenchGroup[Choose benchmark group\nsmall or large]
    BenchGroup --> EnvCtor[Construct env with topology pools]
    EnvCtor --> EpisodeLoop{For each episode}

    EpisodeLoop --> Reset[env.reset]
    Reset --> TopoChoice[Select topology from train_topology_pool]
    TopoChoice --> Rebuild[Rebuild topology-dependent structures]
    Rebuild --> RequestSample[Sample requests]
    RequestSample --> StepLoop{For each step}

    StepLoop --> Adj[env._get_adjacency_info]
    Adj --> BuildGraph[graph_builder.build_graph]
    BuildGraph --> Encode[gnn forward]
    Encode --> Mask[env.get_action_mask]
    Mask --> Select[agent.select_action]
    Select --> EnvStep[env.step]
    EnvStep --> Store[agent.store_transition]
    Store --> DoneCheck{terminated or truncated}
    DoneCheck -->|No| StepLoop
    DoneCheck -->|Yes| Update[agent.update]
    Update --> EpisodeLog[Log reward plus topology_id]
    EpisodeLog --> EpisodeLoop

    EpisodeLoop --> Eval[Evaluate on seen and held-out topologies]
```

## 9. Benchmark Group Constraint

`INSTRUCTION.md` recommends keeping observations stable within each benchmark group instead of supporting arbitrary graph sizes immediately.

```mermaid
flowchart LR
    SmallGroup[Small benchmark group]
    SmallTopo1[small topology A]
    SmallTopo2[small topology B]
    SmallTopo3[small topology C]
    SmallObs[Same observation contract]
    SmallAct[Same action contract]

    LargeGroup[Large benchmark group]
    LargeTopo1[large topology A]
    LargeTopo2[large topology B]
    LargeObs[Same observation contract]
    LargeAct[Same action contract]

    SmallGroup --> SmallTopo1
    SmallGroup --> SmallTopo2
    SmallGroup --> SmallTopo3
    SmallTopo1 --> SmallObs
    SmallTopo2 --> SmallObs
    SmallTopo3 --> SmallObs
    SmallTopo1 --> SmallAct
    SmallTopo2 --> SmallAct
    SmallTopo3 --> SmallAct

    LargeGroup --> LargeTopo1
    LargeGroup --> LargeTopo2
    LargeTopo1 --> LargeObs
    LargeTopo2 --> LargeObs
    LargeTopo1 --> LargeAct
    LargeTopo2 --> LargeAct
```

## 10. Visualization and Animation Pipeline

```mermaid
flowchart TD
    Results[outputs/training_results.json]
    Checkpoint[outputs/gppo_policy.pt]

    VisualizeEntrypoint[src/workflows/visualize.py::generate_all_visualizations]
    AnimateEntrypoint[src/visualization/animation.py::create_all_animations]

    VisualizeEntrypoint --> EnsureVizDirs[ensure_output_dirs]
    AnimateEntrypoint --> EnsureAnimDirs[ensure_output_dirs]

    Checkpoint --> TopologyViz[NetworkTopologyVisualizer.draw_topology]
    Checkpoint --> InferDims[NetworkTopologyVisualizer.infer_topology_from_checkpoint]
    InferDims --> TopologyViz

    VisualizeEntrypoint --> ActionViz[ActionSpaceVisualizer.plot_action_space]
    VisualizeEntrypoint --> CostViz[CostBreakdownVisualizer.plot_cost_components]
    Results --> CurvesViz[TrainingVisualization.plot_training_curves]
    Results --> StatsViz[TrainingVisualization.plot_statistics]
    VisualizeEntrypoint --> BaselineViz[PerformanceComparison.plot_baseline_comparison]

    TopologyViz --> Png1[visualizations/01_network_topology.png]
    ActionViz --> Png2[visualizations/02_action_space.png]
    CostViz --> Png3[visualizations/03_cost_breakdown.png]
    CurvesViz --> Png4[visualizations/04_training_curves.png]
    StatsViz --> Png5[visualizations/05_phase_analysis.png]
    BaselineViz --> Png6[visualizations/06_baseline_comparison.png]

    Results --> LearnAnim[TrainingAnimator.create_learning_animation]
    Results --> RewardLandscape[TrainingAnimator.create_reward_landscape]
    AnimateEntrypoint --> NetAnim[NetworkStateAnimator.create_network_state_animation]

    LearnAnim --> Gif1[animations/01_learning_progress.gif]
    RewardLandscape --> Img2[animations/02_reward_landscape.png]
    NetAnim --> Gif3[animations/03_network_state.gif]
```

## 11. Key Function Call Map

| File | Function / Class | Calls / Uses | Output |
|---|---|---|---|
| `main.py` | `main()` | `run_demo()`, `run_training_from_args()`, `generate_all_visualizations()`, `create_all_animations()` | Dispatches subcommands |
| `run.py` | `main()` | `build_train_parser()`, same workflow functions as `main.py` | Convenience task router |
| `src/workflows/training.py` | `train_gppo()` | `SimplifiedORANEnv`, `GNNFeatureExtractor`, `ORANGraphBuilder`, `PPOAgent` | Trained agent, GNN, metrics dict, checkpoint, JSON results |
| `src/workflows/training.py` | `evaluate_gppo()` | `env.reset()`, `build_graph()`, `gnn()`, `agent.select_action()`, `env.step()` | Printed evaluation metrics |
| `src/workflows/training.py` | `run_training_from_args()` | `train_gppo()`, optionally `evaluate_gppo()` | Full train workflow from CLI args |
| `src/workflows/demo.py` | `run_demo()` | `demo_environment()`, `demo_gnn()`, `demo_ppo_agent()`, `demo_integration()` | Printed sanity-check demo output |
| `src/workflows/visualize.py` | `generate_all_visualizations()` | Visualization classes from `src.visualization` and `ensure_output_dirs()` | PNG charts in `visualizations/` |
| `src/core/environment.py` | `reset()` | `_sample_requests()`, `_refresh_edge_state()`, `_get_state()` | Current behavior: initial observation and empty info dict |
| `src/core/environment.py` | `get_action_mask()` | topology queries | Boolean masks for split, ES, and RC actions |
| `src/core/environment.py` | `step()` | `_evaluate_action()`, `_refresh_edge_state()`, `_sample_requests()`, `_get_state()` | Next state, reward, terminated, truncated, info |
| `src/core/environment.py` | `_get_adjacency_info()` | topology edge scan | Adjacency matrix, edge feature dict, node order |
| `src/core/gnn.py` | `ORANGraphBuilder.build_graph()` | current environment arrays plus adjacency/edge features | `torch_geometric.data.Data` graph |
| `src/core/gnn.py` | `GNNFeatureExtractor.forward()` | `GINEConv`, `global_mean_pool` | Graph embedding tensor |
| `src/core/agent.py` | `PPOAgent.select_action()` | `MaskedPPOPolicy.get_distributions()` | Action vector, log-prob, value |
| `src/core/agent.py` | `PPOAgent.store_transition()` | rollout buffer append | In-memory trajectory data |
| `src/core/agent.py` | `PPOAgent.compute_advantages()` | stored rewards, values, dones | GAE advantages and returns |
| `src/core/agent.py` | `PPOAgent.update()` | `compute_advantages()`, `policy.get_distributions()`, optimizer step | Updated policy parameters |
| `src/visualization/animation.py` | `create_all_animations()` | `TrainingAnimator`, `NetworkStateAnimator`, `ensure_output_dirs()` | GIF and PNG animation assets |
| `src/common/paths.py` | `ensure_output_dirs()` | `Path.mkdir()` | Creates `outputs/`, `visualizations/`, `animations/` |

## 12. Required Environment Refactor Map

This section translates `INSTRUCTION.md` into concrete code responsibilities.

| Required capability | Current status | Needed code change |
|---|---|---|
| Fixed named topologies | Missing | Add named topology definitions instead of one internal random generator |
| Train/test topology pools | Missing | Add topology pool objects or config lists with topology IDs |
| Reset-time topology selection | Missing | Move topology selection and rebuild logic into `reset()` |
| Reproducible benchmark configs | Partial | Add experiment-level config for benchmark, seed, topology mode |
| Stable observation inside benchmark group | Partial | Keep edge and node layout fixed within a benchmark pool |
| Topology metadata in `info` | Missing | Return `topology_id`, benchmark name, and topology metadata |
| Cross-topology evaluation | Missing | Separate seen-topology and held-out-topology evaluation paths |

## 13. Compatibility Layer

These files are wrappers or re-exports, not the main implementation:

```mermaid
flowchart LR
    TrainShim[train.py] --> TrainImpl[src/workflows/training.py]
    TrainCompat[train_gppo.py] --> TrainImpl
    DemoShim[demo.py] --> DemoImpl[src/workflows/demo.py]
    VizShim[visualize_results.py] --> VizImpl[src/workflows/visualize.py]
    AnimShim[animate_training.py] --> AnimImpl[src/visualization/animation.py]

    PPOShim[ppo_agent.py] --> PPOImpl[src/core/agent.py]
    EnvShim[oran_environment.py] --> EnvImpl[src/core/environment.py]
    GNNShim[gnn_feature_extractor.py] --> GNNImpl[src/core/gnn.py]
    VizExports[visualization.py] --> PlotImpl[src/visualization/plots.py]
```

## 14. Proposed File-Level Revision Plan

This is the cleanest file-level architecture that follows the instruction while preserving the current baseline action and reward logic.

```mermaid
flowchart LR
    New1[src/core/topologies.py\nNamed topology definitions]
    New2[src/core/topology_pool.py\nTrain and test pool logic]
    New3[src/core/experiment_config.py\nBenchmark and selection config]

    Env[src/core/environment.py]
    Train[src/workflows/training.py]
    Viz[src/workflows/visualize.py]

    New1 --> New2
    New3 --> New2
    New2 --> Env
    New3 --> Train
    Env --> Train
    Train --> Viz
```

## 15. Artifact Summary

```mermaid
flowchart LR
    Train[Training workflow] --> ResultsJson[outputs/training_results.json]
    Train --> PolicyPt[outputs/gppo_policy.pt]

    ResultsJson --> VisualCharts[Training plots]
    ResultsJson --> AnimAssets[Learning animation and reward landscape]
    PolicyPt --> TopologyInference[Topology inference for visualization]

    VisualCharts --> VizDir[visualizations/*.png]
    AnimAssets --> AnimDir[animations/*]
```
