# GPPO O-RAN Implementation Report

This report explains the implementation pipeline in the current codebase, using the `src/` implementation as the primary path. The top-level files such as `train.py`, `ppo_agent.py`, and `oran_environment.py` are compatibility wrappers; the actual logic used by the current workflows lives in [src/core](</home/hpcnc/intern-research/src/core>) and [src/workflows](</home/hpcnc/intern-research/src/workflows>).

## 1. High-level system overview

The implemented pipeline is:

1. A benchmark topology is selected from a named registry and pool. The registry defines fixed benchmark instances such as `small_balanced_train_a` and `paper_large_direct_test_b`, and the pool logic decides whether training resets use a fixed topology or sample one from the train/test pool. See [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:306) and [topology_pool.py](/home/hpcnc/intern-research/src/core/topology_pool.py:19).
2. The environment builds a NetworkX resource graph from that topology, samples one demand and latency request per RH for the current time slot, initializes ES/RC remaining capacities, and exposes both a flat normalized state and topology-aware graph inputs. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:85), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:120), and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:223).
3. The resource network is converted into a PyTorch Geometric graph with node features for RH/ES/RC roles and dynamic state, and edge features for remaining bandwidth and link delay. See [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:63).
4. A two-layer GINE-based GNN converts the graph into a single graph embedding, which is then passed to a factorized PPO policy/value network. The policy separately outputs logits for split choice, ES choice, and RC choice for every RH. See [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:9) and [agent.py](/home/hpcnc/intern-research/src/core/agent.py:10).
5. The action is a flat `3 * num_rhs` vector. It is decoded into three per-RH decision arrays: split assignment, ES placement, and RC placement. The actor uses structural masks before sampling, and then a conditional RC mask after split and ES choices are known. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:234), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:249), and [agent.py](/home/hpcnc/intern-research/src/core/agent.py:162).
6. The environment evaluates the decoded action against structural connectivity, server capacities, backhaul bandwidth, end-to-end latency, and split-specific crosshaul latency. It computes processing cost, routing cost, reconfiguration cost, SLA penalty, and total cost. Whether a violation is treated as soft or hard depends on `constraint_mode`. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:373) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:551).
7. PPO training collects graph/action/reward transitions one time slot at a time, stores action masks, recomputes graph embeddings during the PPO update, and optimizes the policy and value heads with clipped PPO. In the normal workflow, the update happens once per episode; in paper-mode vectorized training, it happens whenever the rollout buffer reaches the minibatch threshold. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:348), [training.py](/home/hpcnc/intern-research/src/workflows/training.py:497), and [agent.py](/home/hpcnc/intern-research/src/core/agent.py:252).
8. Evaluation runs the trained policy deterministically on fixed topologies, writes per-slot episode traces, aggregates per-topology summaries, exports CSV artifacts, and produces plots and topology snapshots. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:861), [training_csv.py](/home/hpcnc/intern-research/src/workflows/training_csv.py:238), and [visualize.py](/home/hpcnc/intern-research/src/workflows/visualize.py:99).

In short: the code represents each time slot as a graph-structured orchestration problem, the GNN encodes the current topology-plus-load state, the PPO policy emits one split/ES/RC configuration for every RH, the environment converts that into cost and feasibility outcomes, and training optimizes the policy against those outcomes.

## 2. Problem representation

The software represents the O-RAN orchestration problem using four entity types:

- `RH`: radio heads, indexed as `RH0`, `RH1`, ..., created in [TopologySpec.build_graph](/home/hpcnc/intern-research/src/core/topologies.py:28).
- `ES`: edge servers, indexed as `ES0`, `ES1`, ...
- `RC`: regional clouds, indexed as `RC0`, `RC1`, ...
- `splits`: four split options, represented internally as integer indices `0,1,2,3` and reported as `S1..S4` in metrics. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:76) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:574).

The model must make three decisions for each RH in every time slot:

- which split to use
- which ES hosts the DU-side processing if the split uses an ES
- which RC hosts the CU-side processing

Conceptually, this maps to the paper variables as:

- split assignment: `split[r]`
- vDU placement: `es_choice[r]`
- vCU placement: `rc_choice[r]`

The code stores those decisions in one flat action vector and decodes them with:

```python
def _split_action(self, action):
    splits = action[:self.num_rhs]
    es_choices = action[self.num_rhs:2 * self.num_rhs]
    rc_choices = action[2 * self.num_rhs:]
    return splits, es_choices, rc_choices
```

Source: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:234)

One environment step means one paper-style time slot. The code says this explicitly in `step()` and increments `current_step` once per slot. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:710).

One episode means a sequence of `max_steps` time slots on one selected topology. At each slot:

- the current requests are encoded
- the policy chooses a full system-wide action for all RHs
- the environment evaluates it
- new RH requests are sampled for the next slot

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:173) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:731).

Paper time slots are therefore represented directly by:

- `current_step`: current slot index in the episode
- `max_steps`: total slots per episode
- `time_slot` in `info`: slot number reported outward

In project mode the default is `50` slots; in paper mode it becomes `288` slots per episode. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:349), [paper_training.py](/home/hpcnc/intern-research/src/workflows/paper_training.py:119), and [training_constants.py](/home/hpcnc/intern-research/src/workflows/training_constants.py).

## 3. Topology generation

### Where topology is created

The topology registry and generator are implemented in [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:182). Each topology is stored as a `TopologySpec` containing:

- benchmark label
- numbers of RH/ES/RC nodes
- an immutable tuple of `EdgeSpec` links
- metadata

The actual NetworkX graph is created by `TopologySpec.build_graph()`. See [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:18).

### How `small`, `large`, `paper_small`, and `paper_large` are defined

The dimensions are:

- `small`: `8 RH / 3 ES / 2 RC`
- `large`: `16 RH / 5 ES / 3 RC`
- `paper_small`: `8 RH / 3 ES / 2 RC`
- `paper_large`: `64 RH / 4 ES / 2 RC`

Those are instantiated in the registry in [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:306).

### How nodes and links are sampled

Topology generation is family-based. A `FamilyProfile` defines structural patterns and sampling ranges for:

- RH-to-ES primary and secondary links
- ES-to-RC primary and secondary links
- direct RH-to-RC links
- delay and bandwidth sampling ranges

See [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:47) and [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:87).

The generator does the following:

1. For each RH, create two RH-ES edges: a primary and secondary ES attachment.
2. For some RHs, create a direct RH-RC edge.
3. For each ES, create two ES-RC edges: a primary and secondary RC attachment.

This logic is implemented in `_generate_topology_spec()`: [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:182).

Short snippet:

```python
for rh_idx in range(num_rhs):
    ...
    edges.append(_edge(f"RH{rh_idx}", f"ES{primary_es}", ...))
    edges.append(_edge(f"RH{rh_idx}", f"ES{secondary_es}", ...))
    if rh_idx in direct_link_rhs:
        edges.append(_edge(f"RH{rh_idx}", f"RC{direct_rc}", ...))

for es_idx in range(num_ess):
    ...
    edges.append(_edge(f"ES{es_idx}", f"RC{primary_rc}", ...))
    edges.append(_edge(f"ES{es_idx}", f"RC{secondary_rc}", ...))
```

Source: [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:225) and [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:260)

### Capacities, bandwidth, and latencies

Link bandwidth and delay are sampled uniformly from family-specific ranges. Example family ranges are defined in [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:87). Paper-mode overrides those ranges with the `PAPER_SECTION_VI_LINKS` profile:

```python
PAPER_SECTION_VI_LINKS = LinkSamplingOverride(
    rh_primary_bw=(10.0, 40.0),
    ...
    direct_rc_delay=(0.1, 0.25),
    direct_rc_bw=160.0,
    direct_rc_probability=0.10,
)
```

Source: [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:167)

Server capacities are not stored in the topology. They are fixed environment parameters:

- `ES capacity = 20`
- `RC capacity = 100`

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:74).

### Direct RH-RC links

Direct RH-RC links are handled explicitly during topology generation. In project benchmarks, the generator picks about `num_rhs // 4` RHs using a deterministic stride rule; in paper-mode topologies, direct links are sampled independently with probability `0.10`. See [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:210).

These links are what make split 4 structurally possible later in the environment. If an RH has no direct RH-RC edge, split 4 is masked out. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:268).

### Train pool and test pool separation

The benchmark-to-pool split is defined in `DEFAULT_BENCHMARK_TOPOLOGY_POOLS`: [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:477).

- `small` and `large` use disjoint train/test pools.
- `paper_large` also uses disjoint train/test pools.
- `paper_small` currently uses the same five named topologies in both train and test pools.

Pool validation and selection are implemented in [topology_pool.py](/home/hpcnc/intern-research/src/core/topology_pool.py:19).

## 4. Graph construction and graph features

This is the main state-to-model conversion path.

### How the resource network becomes a graph

The environment first exposes:

- an adjacency matrix
- a per-edge feature dictionary keyed by node indices
- the current node ordering

This happens in `_get_adjacency_info()`: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:153).

Then `ORANGraphBuilder.build_graph()` converts those arrays into a PyTorch Geometric `Data` object with:

- `x`: node feature matrix
- `edge_index`: directed COO edge list
- `edge_attr`: edge feature matrix

See [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:73).

### What nodes represent

Nodes are the physical/logical resource entities:

- RH nodes represent traffic sources with current demand and latency budget.
- ES nodes represent candidate DU hosts with current remaining ES capacity.
- RC nodes represent candidate CU hosts with current remaining RC capacity.

### What edges represent

Edges are the topology links created in the benchmark:

- RH-ES fronthaul-like connectivity
- ES-RC crosshaul/backhaul connectivity
- RH-RC direct links for split 4

Each edge carries:

- remaining bandwidth
- delay

### Node features

The node feature template is implemented directly in [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:90).

Conceptually, the 6-dimensional node feature vector is:

- feature 1: `is_rh`
- feature 2: `is_es`
- feature 3: `is_rc`
- feature 4: node-local scalar state
  For RH this is `0.0`
  For ES this is `es_remaining / 20`
  For RC this is `rc_remaining / 100`
- feature 5: `demand_norm`
  For RH: `rh_demands[i] / 300`
  For ES/RC: `0`
- feature 6: `latency_norm`
  For RH: `rh_latencies[i] / 200`
  For ES/RC: `0`

In paper mode, a 7th feature is appended: normalized node index. See [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:84) and [paper_training.py](/home/hpcnc/intern-research/src/workflows/paper_training.py:133).

Short snippet:

```python
# RH node
[1.0, 0.0, 0.0, 0.0, rh_demands[i] / 300.0, rh_latencies[i] / 200.0]

# ES node
[0.0, 1.0, 0.0, es_remaining[i] / 20.0, 0.0, 0.0]

# RC node
[0.0, 0.0, 1.0, rc_remaining[i] / 100.0, 0.0, 0.0]
```

Source: [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:90)

### Edge features

The edge feature vector is 2-dimensional:

- `remaining_bandwidth / 160`
- `delay / 3.6`

See [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:133).

So conceptually:

- node feature shape: `(num_nodes, 6)` in normal mode, `(num_nodes, 7)` in paper mode
- edge feature shape: `(num_directed_edges, 2)`

### How RH / ES / RC are distinguished

They are distinguished only by the first three one-hot type indicators in the node feature vector. There is no separate learned embedding per node class and no explicit edge-type feature for RH-ES versus ES-RC versus RH-RC.

### How current demand, latency, remaining capacity, and bandwidth are encoded

- RH demand and latency are encoded on RH nodes.
- ES remaining capacity is encoded on ES nodes.
- RC remaining capacity is encoded on RC nodes.
- Remaining bandwidth is encoded on edges through `remaining_bandwidth`.
- Delay is encoded on edges through `delay`.

The environment refreshes dynamic edge bandwidth after every step with `_refresh_edge_state()`: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:139).

### How the graph is packaged for PyTorch Geometric

The final object is:

```python
return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

Source: [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:143)

During PPO updates, multiple `Data` objects are batched with `Batch.from_data_list(...)`. See [agent.py](/home/hpcnc/intern-research/src/core/agent.py:286).

Important implementation detail: the environment also defines a flat observation vector in `_get_state()`, but the training loop does not feed that flat vector directly to the policy. It builds a graph from the same state and uses the GNN output instead. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:223) and [training.py](/home/hpcnc/intern-research/src/workflows/training.py:502).

## 5. Action representation

### Action vector shape

The action space is:

```python
spaces.MultiDiscrete(
    [self.split_options] * self.num_rhs +
    [self.num_ess] * self.num_rhs +
    [self.num_rcs] * self.num_rhs
)
```

Source: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:98)

So the action length is:

- `3 * num_rhs`

and the conceptual layout is:

- positions `0 : num_rhs` = split choices
- positions `num_rhs : 2*num_rhs` = ES choices
- positions `2*num_rhs : 3*num_rhs` = RC choices

### How split / ES / RC decisions are packed

The packing is grouped by decision type, not by RH. So for `N` RHs:

- `action[0] ... action[N-1]` are all split decisions
- `action[N] ... action[2N-1]` are all ES decisions
- `action[2N] ... action[3N-1]` are all RC decisions

This is decoded by `_split_action()`: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:234).

### How many decisions are made per RH

Each RH receives exactly three discrete choices per slot:

- one split
- one ES index
- one RC index

For split 4, the ES component is still present in the action vector, but the evaluator ignores it because split 4 uses the direct RH-RC path. In the environment logic, split index `3` is the direct mode. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:588).

### How the action is decoded

The policy produces logits separately for split, ES, and RC:

```python
split_logits = self.split_head(hidden).view(-1, self.num_rhs, self.num_splits)
es_logits = self.es_head(hidden).view(-1, self.num_rhs, self.num_ess)
rc_logits = self.rc_head(hidden).view(-1, self.num_rhs, self.num_rcs)
```

Source: [agent.py](/home/hpcnc/intern-research/src/core/agent.py:47)

The training/evaluation code uses `select_action_sequential()`, not the simpler `select_action()`. That matters because RC validity depends on the chosen split and ES. See [agent.py](/home/hpcnc/intern-research/src/core/agent.py:162).

### How action masking works

There are two masking stages.

1. Base structural masks from `get_action_mask()`:
   - `split_mask[r, s]`
   - `es_mask[r, e]`
   - `rc_mask[r, c]`

   These only encode coarse topology structure. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:249).

2. Conditional RC mask from `get_conditional_rc_mask(splits, es_choices)`:
   - if split 4, RC must be directly connected to RH
   - otherwise RC must be reachable from the chosen ES

   See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:273).

The sequential policy method applies this exactly:

```python
split_action = split_dist.sample()
es_action = es_dist.sample()
rc_mask_np = rc_mask_fn(split_action..., es_action...)
rc_dist = Categorical(logits=rc_logits.masked_fill(~rc_mask, -inf))
rc_action = rc_dist.sample()
```

Source: [agent.py](/home/hpcnc/intern-research/src/core/agent.py:173)

### How invalid structural choices are prevented or handled

They are partially prevented by masks, but not fully by capacity/latency checks.

- Prevented before sampling:
  - choosing split 4 when RH has no direct RH-RC link
  - choosing ES with no RH-ES link
  - choosing RC that is structurally unreachable
- Still possible after sampling:
  - capacity overload
  - bandwidth overload
  - latency violations

Those second-stage violations are handled in `_evaluate_action()` rather than the mask. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:551).

## 6. Environment design

### Observation/state generation

The environment defines a flat observation vector:

```python
[
    rh_demands / 300,
    rh_latencies / 200,
    es_remaining / es_capacity,
    rc_remaining / rc_capacity,
    edge_remaining_bandwidth / 160,
]
```

Source: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:223)

However, the training path immediately converts the same environment state into a graph, so the GNN path is the actual model input path. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:497).

### `reset()`

`reset()` does an implementation-specific sequence:

1. select topology from the requested pool and selection mode
2. rebuild the NetworkX topology
3. reset `current_step`, `invalid_streak`, `prev_action`
4. reset ES and RC remaining capacities
5. sample the initial RH requests for slot 0
6. run feasibility probes for diagnostics
7. set all edge remaining bandwidth to full capacity
8. return the normalized flat state and a rich `info` dict

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:173).

The reset `info` already includes topology-level feasibility diagnostics such as:

- `has_structurally_valid_action`
- `has_strictly_valid_action`
- `has_exact_strictly_valid_action`
- `bounded_strict_feasibility_probe`

### `step()`

`step()` performs:

1. increment `current_step`
2. evaluate the current action with `_evaluate_action()`
3. compute reward from validity and cost
4. update `es_remaining`, `rc_remaining`, and edge remaining bandwidth
5. store `prev_action`
6. terminate if episode horizon or invalid streak is reached
7. sample next-slot RH requests
8. return next state and a detailed `info` dict

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:710).

### `current_step` and time-slot semantics

`current_step` is the slot counter. Slot `0` is reported at reset, then each call to `step()` produces `time_slot = 1, 2, ...`. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:213) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:734).

### How requests are sampled each slot

Requests are sampled independently for each RH by first choosing a slice type from `{eMBB, mMTC, uRLLC}` and then sampling:

- eMBB: demand `250-300`, latency `15-20`
- mMTC: demand `150-200`, latency `180-200`
- uRLLC: demand `20-40`, latency `2-4`

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:120).

This means the current implementation represents time variation by re-sampling fresh RH requests every slot, not by explicit paper-style request carryover/release dynamics.

### How `prev_action` is stored

`prev_action` starts as `None` at reset and becomes the last applied full action vector after each step. It is used only for reconfiguration accounting. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:96) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:726).

### How reconfiguration is tracked

Reconfiguration is measured by comparing current and previous action slices:

- `split_changes`
- `es_changes`
- `rc_changes`
- `total_reconfiguration_changes`

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:638).

### What goes into `info`

The `info` dict contains:

- time slot and episode length
- validity flag
- total and component costs
- all violation magnitudes
- split usage counts
- failure counters and invalid reason labels
- invalid streak
- topology metadata

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:733).

The training and evaluation workflows treat this `info` dict as the source of truth for logging. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:547) and [training.py](/home/hpcnc/intern-research/src/workflows/training.py:978).

## 7. Feasibility checking and constraints

This is the critical section.

### Where feasibility is checked

There are two different mechanisms:

- constructive/reset-time feasibility probes:
  - `find_feasible_action()`
  - `_search_feasible_action()`
  - `has_exact_feasible_action()`
  - `probe_bounded_feasible_action()`
- actual post-action evaluation:
  - `_evaluate_action()`

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:294), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:409), and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:551).

### A. Structural feasibility checks

These are the topology-only checks.

For split 4, the action is structurally valid only if a direct `RH -> RC` link exists:

```python
if uses_direct_rc:
    if not self.topology.has_edge(rh_node, rc_node):
        valid = False
        failure_counts["missing_direct_rc_link"] += 1
```

Source: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:590)

For split 1/2/3, the action is structurally valid only if:

- `RH -> ES` exists
- `ES -> RC` exists

Source: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:603)

These structural failures are always hard-invalid, regardless of `constraint_mode`.

### B. Post-action resource/performance checks

After structural validity is established, the code checks:

- ES capacity
- RC capacity
- selected backhaul/direct bandwidth
- end-to-end latency against the sampled RH latency budget
- crosshaul latency against `crosshaul_latency_limits[split]`

The exact logic is in `_evaluate_action()` and `_search_feasible_action()`. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:458) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:624).

### How split 4 is treated differently

Internally, split index `3` is split 4.

For split 4:

- no ES load is added
- ES choice is effectively ignored during evaluation
- RC load uses `cu_costs[3]`
- bandwidth is checked on the direct `RH-RC` edge
- routing cost uses direct-link delay
- end-to-end latency uses direct-link delay
- crosshaul latency also uses direct-link delay compared against `crosshaul_latency_limits[3]`

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:458) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:590).

For split 1/2/3:

- ES load uses `du_costs[split]`
- RC load uses `cu_costs[split]`
- bandwidth is checked only on the chosen `ES-RC` edge
- routing cost uses only crosshaul delay
- end-to-end latency uses `RH-ES delay + ES-RC delay`
- crosshaul latency uses only `ES-RC delay`

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:473) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:619).

Important code-grounded observation: the current bandwidth accounting does not charge the RH-ES edge for non-direct splits. It only charges `ES-RC` bandwidth. That is a real implementation simplification and matters when comparing to the paper.

### Capacity, bandwidth, and latency checks

- ES capacity: `du_load[es] <= es_capacity`
- RC capacity: `cu_load[rc] <= rc_capacity`
- bandwidth: aggregated used Gbps on a chosen edge must not exceed that edge's bandwidth
- end-to-end latency: actual path delay must not exceed the sampled `rh_latency`
- crosshaul latency: chosen direct or ES-RC link delay must not exceed `crosshaul_latency_limits[split]`

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:651), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:660), and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:673).

### How the code maps conceptually to paper constraints

Conceptually:

- topology connectivity constraints are implemented through masks and structural checks
- DU placement capacity constraints are implemented through `du_load` vs `es_capacity`
- CU placement capacity constraints are implemented through `cu_load` vs `rc_capacity`
- link capacity constraints are approximated through per-edge aggregated bandwidth usage
- end-to-end and crosshaul latency constraints are implemented explicitly

But the exact enforcement is mode-dependent, which means the code can run in a softer-than-paper regime (`legacy`) or a stricter regime (`strict_full`).

## 8. Penalty and reward design

### Where `total_cost` is computed

`total_cost` is computed in `_evaluate_action()`:

```python
slack_penalty = es_overuse + rc_overuse + bandwidth_overuse + total_e2e_violation + total_cross_violation
total_cost = total_processing_cost + total_routing_cost + reconfiguration_cost + slack_penalty
```

Source: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:660)

### Meaning of each cost component

- `processing_cost`: DU plus CU processing burden induced by the chosen split, using `du_costs` and `cu_costs`. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:633).
- `routing_cost`: delay-weighted traffic cost on the selected ES-RC or direct RH-RC edge, scaled by `phi_l`. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:600) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:629).
- `reconfiguration_cost`: number of per-RH split/ES/RC changes relative to `prev_action`, scaled by `phi_r`. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:643).
- `sla_penalty`: additive soft violation term composed of ES overuse, RC overuse, bandwidth overuse, end-to-end latency violation, and crosshaul latency violation. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:660).

### What soft violations are allowed in legacy mode

In `legacy`, the code still computes these violations and adds them to `sla_penalty`, but they do not by themselves set `valid=False`:

- `es_capacity_exceeded`
- `rc_capacity_exceeded`
- `bandwidth_exceeded`
- `e2e_latency_exceeded`
- `crosshaul_latency_exceeded`

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:373) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:680).

### What becomes hard invalid in strict modes

Structural failures are always hard-invalid. In addition:

- `strict_connectivity_plus_capacity` hard-enforces ES/RC capacity
- `strict_connectivity_plus_capacity_plus_bandwidth` also hard-enforces bandwidth
- `strict_full` also hard-enforces both latency constraints

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:373).

### Exact reward formula

If `valid=True`:

```python
reward = float((1.0 + np.log1p(metrics["total_cost"])) ** -1)
```

If `valid=False`:

```python
reward = -1.0 if early_terminated else -(metrics["nfail"] / max(2 * self.num_rhs, 1))
```

Source: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:715)

Plain-language interpretation:

- valid deployments get a positive reward in `(0, 1]`, smaller when total cost is larger
- invalid deployments get a negative reward proportional to the number of failed constraints
- repeated invalid actions trigger a harsher terminal penalty

### Early termination

The environment tracks `invalid_streak`. If invalid actions happen for `max_invalid_streak = 5` consecutive slots, the episode terminates early. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:77) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:728).

## 9. Legacy vs strict constraint modes

The mode switch lives in `_enforced_reason_set()` and `_constraint_mode_alias()`: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:368).

Important naming note: the current code does not implement a mode literally named `strict_connectivity_plus_bandwidth`. The actual implemented name is `strict_connectivity_plus_capacity_plus_bandwidth`. I use the exact code name below and mention the requested label in parentheses.

| Report label | Actual code mode | Structural connectivity hard-enforced | ES/RC capacity hard-enforced | Bandwidth hard-enforced | E2E latency hard-enforced | Crosshaul latency hard-enforced |
| --- | --- | --- | --- | --- | --- | --- |
| legacy | `legacy` | Yes | No | No | No | No |
| strict_connectivity_only | `strict_connectivity_only` | Yes | No | No | No | No |
| strict_connectivity_plus_capacity | `strict_connectivity_plus_capacity` | Yes | Yes | No | No | No |
| strict_connectivity_plus_bandwidth | `strict_connectivity_plus_capacity_plus_bandwidth` | Yes | Yes | Yes | No | No |
| strict_full | `strict_full` | Yes | Yes | Yes | Yes | Yes |
| strict alias | `strict -> strict_full` | Yes | Yes | Yes | Yes | Yes |

Two implementation consequences matter:

- `legacy` and `strict_connectivity_only` are effectively identical with respect to soft-violation enforcement, because both return an empty enforced-reason set.
- The code stages constraint hardness by promoting already-computed soft violations into hard invalidity, rather than changing the cost formulas themselves.

## 10. Model architecture

### GNN feature extractor

The graph encoder is `GNNFeatureExtractor` in [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:9).

Architecture:

- input node projection: `Linear(input_dim, hidden_dim)`
- `GINEConv` layer 1
- `GINEConv` layer 2
- global mean pooling over nodes
- output projection to `128`-dimensional graph embedding

Short snippet:

```python
x = F.relu(self.node_proj(x))
x = F.relu(self.gine1(x, edge_index, edge_attr))
x = F.relu(self.gine2(x, edge_index, edge_attr))
graph_embed = global_mean_pool(x, batch)
return self.output_proj(graph_embed)
```

Source: [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:45)

### GINEConv usage

Both graph convolution layers are `GINEConv` with `edge_dim=2`, so edge attributes are explicitly used during message passing. See [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:19).

### Number of layers and hidden dimensions

Normal project mode defaults:

- GNN input dim: `6`
- GNN hidden dim: `64`
- graph embedding dim: `128`
- policy/value hidden dim: `256`

Paper mode changes:

- GNN input dim: `7`
- GNN hidden dim: `1024`

See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:362), [paper_training.py](/home/hpcnc/intern-research/src/workflows/paper_training.py:132), and [training_constants.py](/home/hpcnc/intern-research/src/workflows/training_constants.py).

### Policy network

The policy/value model is `MaskedPPOPolicy` in [agent.py](/home/hpcnc/intern-research/src/core/agent.py:10).

It uses:

- a two-layer MLP backbone
- a split head producing `num_rhs * 4` logits
- an ES head producing `num_rhs * num_ess` logits
- an RC head producing `num_rhs * num_rcs` logits
- a scalar value head

### Value network

The value network is just `self.value_net = nn.Linear(hidden_dim, 1)` on top of the shared backbone. See [agent.py](/home/hpcnc/intern-research/src/core/agent.py:37).

### Initialization

All `Linear` layers in both the GNN and policy use Xavier uniform initialization with zero bias. See [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:38) and [agent.py](/home/hpcnc/intern-research/src/core/agent.py:40).

### How graph embeddings are combined with action selection

The combination is simple:

1. the GNN outputs one graph embedding
2. the PPO policy backbone maps that embedding to a hidden vector
3. the three action heads produce per-RH logits for split/ES/RC
4. masking is applied
5. categorical samples are drawn

There is no autoregressive neural dependency between RHs. The only sequential dependency is the conditional RC mask applied after split and ES are sampled. See [agent.py](/home/hpcnc/intern-research/src/core/agent.py:173).

## 11. Training pipeline

### Normal training workflow

The main training loop is `train_gppo()` in [training.py](/home/hpcnc/intern-research/src/workflows/training.py:348).

The normal single-environment workflow is:

1. build one environment with the chosen benchmark, train pool, constraint mode, and topology selection mode
2. build one graph builder
3. build the GNN
4. build the PPO agent and attach the GNN so the optimizer updates both policy and GNN parameters
5. repeat for each episode:
   - reset environment
   - for each time slot:
     - read adjacency and edge features from the environment
     - build a `Data` graph
     - run the GNN forward pass
     - get structural masks
     - sample a sequential masked action
     - step the environment
     - store `(graph, action, reward, value, log_prob, done, masks)` in the rollout buffer
   - run PPO update over the collected transitions
6. write results JSON and checkpoint

The core inner loop is in [training.py](/home/hpcnc/intern-research/src/workflows/training.py:497).

### How rollouts are collected

Rollout collection stores graph objects directly, not flat state vectors. Each stored transition includes:

- the `Data` graph
- the chosen flat action vector
- scalar reward
- predicted value
- action log-probability
- done flag
- the masks actually used during sampling

See [agent.py](/home/hpcnc/intern-research/src/core/agent.py:220).

### How state graphs are built each step

Each step uses:

```python
adjacency, edge_features, _ = env._get_adjacency_info()
graph = graph_builder.build_graph(
    env.rh_demands,
    env.rh_latencies,
    env.es_remaining,
    env.rc_remaining,
    adjacency,
    edge_features,
)
```

Source: [training.py](/home/hpcnc/intern-research/src/workflows/training.py:498)

### How actions are selected

Training uses stochastic sampling:

```python
action, log_prob, value, used_action_mask = agent.select_action_sequential(
    features.squeeze(0),
    action_mask,
    env.get_conditional_rc_mask,
)
```

Source: [training.py](/home/hpcnc/intern-research/src/workflows/training.py:520)

### How transitions are stored

Transitions are appended to in-memory lists in `PPOAgent`. See [agent.py](/home/hpcnc/intern-research/src/core/agent.py:109).

### How PPO update is run

`agent.update()` does:

1. compute GAE-style advantages and returns
2. normalize advantages
3. stack all actions and masks
4. for each PPO epoch:
   - shuffle indices
   - take minibatches of transition indices
   - batch graphs with `Batch.from_data_list`
   - recompute graph embeddings with the GNN
   - recompute masked action distributions
   - compute new log-probs, entropy, and values
   - apply PPO clipped objective plus value loss
   - backpropagate and clip gradients
5. clear the rollout buffer

See [agent.py](/home/hpcnc/intern-research/src/core/agent.py:238) and [agent.py](/home/hpcnc/intern-research/src/core/agent.py:252).

Important implementation detail: graph features are recomputed during update. The rollout phase uses `torch.no_grad()` only to avoid unnecessary memory growth. The code comment says this explicitly. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:513).

### What batch/minibatch means in this codebase

- In normal mode, one PPO update happens after one episode.
- Inside that update, `batch_size` means the number of stored transitions per minibatch, not the number of episodes.
- A minibatch can therefore contain time slots from different parts of the same episode.

See [agent.py](/home/hpcnc/intern-research/src/core/agent.py:283).

### How timesteps, episodes, and `max_steps` relate

In normal mode:

- one episode can last up to `max_steps` slots
- training runs for `num_episodes`
- total collected transitions are roughly `num_episodes * realized_episode_length`

In paper mode:

- training is driven by `total_timesteps`
- environments are synchronized in a manual vectorized loop
- `max_steps` is usually `288`
- the run ends when total collected slots reach the requested budget

### What paper-mode changes

Paper mode is launched by `_run_paper_mode_from_args()` in [paper_training.py](/home/hpcnc/intern-research/src/workflows/paper_training.py:104).

It changes:

- benchmark mapping to `paper_small` or `paper_large`
- episode length to `288`
- training budget to a fixed total timestep count
- number of parallel environments
- GNN input dim to `7`
- GNN hidden dim to `1024`
- `include_node_index=True`
- multi-seed aggregate reporting

The vectorized implementation is `_train_gppo_sync_vectorized()` in [vectorized_training.py](/home/hpcnc/intern-research/src/workflows/vectorized_training.py:15).

In that path:

- multiple environments are stepped round-robin
- each environment maintains its own active episode statistics
- PPO updates happen whenever `len(agent.states) >= batch_size`

See [vectorized_training.py](/home/hpcnc/intern-research/src/workflows/vectorized_training.py:198) and [vectorized_training.py](/home/hpcnc/intern-research/src/workflows/vectorized_training.py:284).

## 12. Evaluation pipeline

### How train-pool evaluation works

After training, the workflow can evaluate the learned policy separately on the train pool. This is done through `evaluate_pool_by_topology(..., topology_pool_name="train")`. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:1265).

### How test-pool evaluation works

The exact same path is used for the test pool, but with `topology_pool_name="test"`. Each topology in the pool is evaluated separately under fixed topology selection. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:1279).

### How topology-level summaries are computed

`evaluate_gppo()` aggregates:

- episode rewards and costs
- valid step counts
- reset-time structural and strict feasibility rates
- invalid reason counts
- split distribution
- average cost breakdown per slot
- reconfiguration statistics

See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:922) and [training.py](/home/hpcnc/intern-research/src/workflows/training.py:1156).

### How per-slot episode traces are recorded

During evaluation, every slot is logged into an `episode_slots` list containing:

- split vector
- ES vector
- RC vector
- reward
- cost components
- violation magnitudes
- invalid reasons
- reconfiguration counts

See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:1005).

Those traces are then written as JSON via `_write_episode_trace()` and optionally flattened to CSV with `_write_episode_trace_csv()`. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:1061) and [training_csv.py](/home/hpcnc/intern-research/src/workflows/training_csv.py:65).

### How deterministic evaluation differs from training

Training samples from the masked categorical distributions.

Evaluation uses:

```python
deterministic=True
```

inside `select_action_sequential()`, so it takes argmax actions instead of stochastic samples. See [training.py](/home/hpcnc/intern-research/src/workflows/training.py:972).

This means evaluation measures the current greedy policy induced by the trained logits, not exploration behavior.

## 13. Visualization pipeline

### What the main plots are

The main plots come from [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py) and are orchestrated by [visualize.py](/home/hpcnc/intern-research/src/workflows/visualize.py:99).

Key plots are:

- training curves
- cost breakdown
- split usage
- evaluation topology summary
- invalid reason summary
- timing profile
- episode trace plots
- topology snapshots with model decisions overlaid

### What data sources they use

The visualization workflow prefers CSV-backed plots first, then falls back to `training_results.json` where needed. See [visualize.py](/home/hpcnc/intern-research/src/workflows/visualize.py:147).

The CSV artifacts are produced by `_export_csv_artifacts()`:

- `training_episode_metrics.csv`
- `evaluation_topology_summary.csv`
- `invalid_reason_summary.csv`
- `timing_profile.csv`

See [training_csv.py](/home/hpcnc/intern-research/src/workflows/training_csv.py:238).

Episode trace CSVs are discovered automatically from the run directory and trace directory. See [visualize.py](/home/hpcnc/intern-research/src/workflows/visualize.py:32).

### How CSV and `training_results.json` are used

- CSV files drive compact plots such as training metrics, evaluation summaries, timing profiles, and episode trace plots.
- `training_results.json` is used for legacy/fallback plotting and for richer aggregate information that may not exist in CSV-only form.

This hybrid logic is implemented in [visualize.py](/home/hpcnc/intern-research/src/workflows/visualize.py:149).

### What topology snapshots show

Topology snapshots are created by `NetworkTopologyVisualizer.draw_topology()`, which:

1. creates a one-step environment for a chosen topology
2. optionally loads the trained checkpoint
3. runs deterministic inference for a single time slot
4. overlays chosen paths in split-specific colors
5. annotates node demands/capacities and link delay/bandwidth

See [visualize.py](/home/hpcnc/intern-research/src/workflows/visualize.py:121) and [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:360).

### What episode trace plots show

Episode trace plots visualize per-slot quantities across one full evaluated episode, such as:

- deployment cost per slot
- reward per slot
- reconfiguration cost per slot
- latency violation magnitudes
- invalid slots

The plot is driven from per-slot trace CSV rows written during evaluation. See [training_csv.py](/home/hpcnc/intern-research/src/workflows/training_csv.py:22) and [plots.py](/home/hpcnc/intern-research/src/visualization/plots.py:896).

### What the professor should and should not conclude from a single-slot snapshot

A topology snapshot is useful for explaining:

- the benchmark structure
- which split/path choices the trained policy favors on one sampled slot
- whether the chosen decision is direct or ES-mediated

But a single snapshot should not be used to conclude:

- long-run policy stability
- average reward/cost quality
- reconfiguration behavior across time
- strict feasibility rates

Those require full-episode traces and pooled evaluation summaries, because requests are re-sampled every slot and the reconfiguration penalty depends on consecutive actions.

## 14. End-to-end example

This example is schematic but follows the actual code structure.

Assume the benchmark is `small`, so there are `8 RH / 3 ES / 2 RC`. Suppose the environment has selected topology `small_balanced_train_a` from the train pool. See [topologies.py](/home/hpcnc/intern-research/src/core/topologies.py:308).

### Step 1: requests are sampled

At reset or after a step, `_sample_requests()` assigns each RH a slice type and samples:

- demand in Mbps
- latency budget in ms

Suppose for `RH0` the sampled values are:

- demand = `280 Mbps`
- latency budget = `18 ms`

### Step 2: graph is built

For `RH0`, the graph builder will encode:

```text
RH0 node feature = [1, 0, 0, 0, 280/300, 18/200]
                 = [1, 0, 0, 0, 0.933, 0.090]
```

If `ES1` currently has full remaining capacity, its node feature is:

```text
ES1 node feature = [0, 1, 0, 20/20, 0, 0]
                 = [0, 1, 0, 1.0, 0, 0]
```

If `RC0` currently has full remaining capacity:

```text
RC0 node feature = [0, 0, 1, 100/100, 0, 0]
                 = [0, 0, 1, 1.0, 0, 0]
```

If edge `ES1-RC0` has remaining bandwidth `30 Gbps` and delay `1.2 ms`, its edge feature is:

```text
[30/160, 1.2/3.6] = [0.1875, 0.3333]
```

This `Data(x, edge_index, edge_attr)` object becomes the GNN input. See [gnn.py](/home/hpcnc/intern-research/src/core/gnn.py:122).

### Step 3: model outputs action logits

The GNN produces one graph embedding of size `128`. The PPO policy turns that into:

- split logits of shape `(8, 4)`
- ES logits of shape `(8, 3)`
- RC logits of shape `(8, 2)`

See [agent.py](/home/hpcnc/intern-research/src/core/agent.py:47).

Suppose for `RH0` the sampled decision is:

- split index `1` which corresponds to `S2`
- ES index `1`
- RC index `0`

Inside the full action vector this means:

- `action[0] = 1`
- `action[8 + 0] = 1`
- `action[16 + 0] = 0`

### Step 4: action decoding

`_split_action()` decodes the flat vector into:

- `splits[0] = 1`
- `es_choices[0] = 1`
- `rc_choices[0] = 0`

See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:234).

### Step 5: feasibility checks

Because split index `1` is not split 4, the environment checks:

- does `RH0 -> ES1` exist?
- does `ES1 -> RC0` exist?

If yes, it then adds:

- ES load increment `du_costs[1] * 280 = 0.04 * 280 = 11.2`
- RC load increment `cu_costs[1] * 280 = 0.001 * 280 = 0.28`
- bandwidth usage increment on edge `ES1-RC0` of `280 / 1000 = 0.28 Gbps`

It also computes:

- routing cost increment `phi_l * crosshaul_delay * demand_gbps`
- end-to-end latency = `RH0-ES1 delay + ES1-RC0 delay`
- crosshaul latency = `ES1-RC0 delay`

All of this matches [environment.py](/home/hpcnc/intern-research/src/core/environment.py:619).

### Step 6: cost components

For this one RH, the processing cost contribution is:

```text
du_costs[1] * 280 + cu_costs[1] * 280
= 0.04 * 280 + 0.001 * 280
= 11.48
```

If `ES1-RC0 delay = 1.2 ms`, routing contribution is:

```text
1.0 * 1.2 * 0.28 = 0.336
```

If this action differs from the previous slot for RH0, it may also contribute to reconfiguration cost through split/ES/RC change counts.

### Step 7: reward

If the full system-wide action remains valid under the current `constraint_mode`, the reward is:

```text
reward = (1 + log1p(total_cost))^-1
```

If the full action is invalid, the reward becomes negative according to the number of failed constraints and the invalid streak. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:715).

This example shows the full implemented path:

topology -> sampled requests -> graph features -> GNN embedding -> split/ES/RC action -> structural/resource checks -> cost breakdown -> reward.

## 15. Final implementation summary

### What is already implemented correctly

- The project has a complete end-to-end pipeline from named benchmark topology selection to graph construction, PPO training, deterministic evaluation, CSV export, and visualization.
- The action representation cleanly matches the intended orchestration decisions: split, ES placement, and RC placement for each RH.
- The graph pipeline is consistent and uses edge-aware message passing with `GINEConv`.
- The environment already measures the main categories of constraints the paper cares about: connectivity, capacity, bandwidth, end-to-end latency, and crosshaul latency.
- The code supports both project-scale and paper-style benchmark families, including multi-topology train/test pools.

### What is simplified or approximate

- The environment samples fresh RH requests every slot rather than implementing a richer request arrival/release process.
- Non-direct bandwidth accounting is applied only to the chosen `ES-RC` edge, not the `RH-ES` edge.
- `legacy` mode treats most feasibility violations as soft penalties rather than hard infeasibility.
- The policy is factorized over RHs and does not model inter-RH coupling explicitly except through the shared graph embedding and shared validity checks.
- The paper-mode parallel training path is a custom synchronous vectorized loop, not a standard external RL framework implementation.

### What still differs from the paper

- The code supports multiple staged constraint modes, whereas the paper’s optimization constraints are conceptually hard feasibility conditions.
- The benchmark topology pools and debug-friendly project modes go beyond a minimal paper reproduction.
- `paper_small` currently uses the same topology set for train and test pools.
- The direct split uses the same action-vector layout as other splits, which means an ES slot is still carried even though it is ignored in evaluation.

### Current strongest implementation limitations

- The biggest modeling limitation is that feasibility and reward are partially decoupled in `legacy`, so a structurally connected but resource/latency-violating decision can still receive positive reward.
- Bandwidth modeling is asymmetric because `RH-ES` bandwidth is not charged in the current evaluator.
- The node feature design is intentionally compact; it does not include richer per-node or per-edge context beyond role, capacity, demand, latency, bandwidth, and delay.
- Exact strict feasibility search is only attempted for small enough instances, and larger instances rely on bounded probes or greedy construction at reset. See [environment.py](/home/hpcnc/intern-research/src/core/environment.py:19) and [environment.py](/home/hpcnc/intern-research/src/core/environment.py:516).

Overall, the project already implements a real graph-based PPO orchestration pipeline, not just a conceptual prototype. The strongest parts are the topology management, graph conversion, masked action generation, and reproducible evaluation tooling. The main caveats are the softened feasibility behavior in legacy mode and several modeling simplifications that should be explained clearly when comparing the implementation to the paper.
