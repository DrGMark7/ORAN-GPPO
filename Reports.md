# Soft-Penalty Tracing Report

## 1. Entry points

| Stage | File | Method | What it does |
| --- | --- | --- | --- |
| Environment step | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:710) | `SimplifiedORANEnv.step` | Advances one time slot, calls `_evaluate_action`, converts metrics into reward, termination, next sampled requests, and `info`. |
| Action decoding | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:234) | `SimplifiedORANEnv._split_action` | Decodes the flat action vector into per-RH `splits`, `es_choices`, and `rc_choices`. |
| Action masking | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:249) | `SimplifiedORANEnv.get_action_mask` | Builds structural masks for split, ES, and RC choices before the policy samples an action. |
| Conditional RC masking | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:273) | `SimplifiedORANEnv.get_conditional_rc_mask` | Refines RC options after split and ES have been chosen. |
| Action evaluation | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:551) | `SimplifiedORANEnv._evaluate_action` | Computes structural validity, resource loads, routing delay, soft violations, total cost, invalid reasons, and final `metrics["valid"]`. |
| Constraint-mode selection | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:368) | `SimplifiedORANEnv._constraint_mode_alias` | Normalizes `strict` to `strict_full`. |
| Constraint enforcement set | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:373) | `SimplifiedORANEnv._enforced_reason_set` | Defines which violation reasons are hard-invalid under each `constraint_mode`. |
| Feasibility probing at reset | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:294) | `SimplifiedORANEnv.find_feasible_action` | Greedy feasibility probe used by `reset()` for `has_structurally_valid_action` and strict-feasible diagnostics. |
| Exact / bounded feasibility search | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:409) | `SimplifiedORANEnv._search_feasible_action` | DFS-based feasibility checker used only for reset-time diagnostics, not for training reward. |
| Invalid-reason recording | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:662) | `_evaluate_action` invalid-reason block | Converts violation magnitudes into named reasons and failure counters. |
| Training/eval logging | [training.py](/home/hpcnc/intern-research/src/workflows/training.py:968) | `evaluate_gppo` | Consumes `env.step()` info and persists `valid_deployment`, costs, and `invalid_reasons` into traces/CSVs. |

## 2. Soft-penalty pipeline

One action flows through the current legacy path like this:

1. The policy emits a flat `3 * num_rhs` vector.
   Code: [_split_action`](/home/hpcnc/intern-research/src/core/environment.py:234)
2. `_split_action()` slices it into split, ES, and RC assignments per RH.
3. `_evaluate_action()` loops over RHs and first checks structural connectivity:
   - split `3` requires a direct `RH -> RC` link
   - splits `0/1/2` require both `RH -> ES` and `ES -> RC`
   Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:576)
4. If structural links exist, `_evaluate_action()` accumulates:
   - DU load on ES
   - CU load on RC
   - bandwidth usage on the chosen backhaul edge
   - routing cost
   - end-to-end latency violation
   - crosshaul latency violation
   Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:597), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:624)
5. After all RHs are processed, `_evaluate_action()` computes:
   - reconfiguration deltas vs `prev_action`
   - capacity overuse
   - bandwidth overuse
   - aggregate soft penalty `sla_penalty`
   - final `total_cost`
   Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:638)
6. `_evaluate_action()` converts positive violation magnitudes into named `invalid_reasons`.
7. `constraint_mode` decides whether those reasons only raise cost or also force `valid=False`.
8. `step()` converts `metrics["valid"]` and `metrics["nfail"]` into reward.

### What only increases cost in `legacy`

In `legacy`, `_enforced_reason_set()` returns an empty set, so these violations do **not** flip `valid=False`; they only increase `sla_penalty` and therefore `total_cost`:

- `es_capacity_exceeded`
- `rc_capacity_exceeded`
- `bandwidth_exceeded`
- `e2e_latency_exceeded`
- `crosshaul_latency_exceeded`

Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:373), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:660)

### What always flips `valid=False`

These are hard-invalid regardless of `constraint_mode`, because they set `valid = False` immediately inside the per-RH loop:

- `missing_direct_rc_link`
- `missing_rh_es_link`
- `missing_es_rc_link`

Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:590), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:603)

### Where `constraint_mode` changes behavior

`_enforced_reason_set()` is the switch:

- `legacy` -> enforce nothing
- `strict_connectivity_only` -> enforce nothing
- `strict_connectivity_plus_capacity` -> enforce ES/RC capacity only
- `strict_connectivity_plus_capacity_plus_bandwidth` -> enforce ES/RC capacity and bandwidth
- `strict_full` and `strict` -> enforce ES/RC capacity, bandwidth, e2e latency, and crosshaul latency

Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:373)

## 3. Method-by-method table

| Method | File | Purpose | Inputs | Outputs | Relevant penalty / constraint logic | Legacy behavior | Strict behavior |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reset` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:173) | Initializes a new episode/time-slot sequence | seed, topology selection options | state, reset `info` | Calls `find_feasible_action()` and strict probes for diagnostics only | Reports whether a greedy legacy-feasible action exists | Also reports strict feasibility probes |
| `_split_action` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:234) | Decodes flat action vector | action array | splits, ES choices, RC choices | No penalty logic; pure decoding | Same | Same |
| `get_action_mask` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:249) | Removes structurally impossible choices before sampling | current topology | boolean masks | Masks missing links and disables split 3 when no direct RC exists | Same | Same |
| `get_conditional_rc_mask` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:273) | RC mask conditioned on chosen split/ES | splits, ES choices | RC mask | Structural filtering only | Same | Same |
| `find_feasible_action` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:294) | Greedy constructive feasibility probe | `constraint_mode` | action or `None` | Greedily avoids capacity/bandwidth overflow, then rechecks through `_evaluate_action()` | In `legacy`, success means no structural invalidity after evaluation | In strict modes, returned action must also avoid enforced reasons |
| `_constraint_mode_alias` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:368) | Normalizes aliases | mode string | canonical mode string | `strict` becomes `strict_full` | Same | Same |
| `_enforced_reason_set` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:373) | Maps mode to hard-invalid reasons | mode string | set of reasons | Core switch for soft vs hard treatment | Empty set, so soft violations stay soft | Non-empty set in staged/full strict modes |
| `_search_feasible_action` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:409) | DFS feasibility checker for diagnostics | mode, RH subset, visit limit | `True`, `False`, or `None` | Skips branches when candidate reasons intersect enforced reasons | Ignores soft violations because enforced set is empty | Prunes any branch with enforced violations |
| `_evaluate_action` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:551) | Main action evaluator | action, optional mode override | metrics dict | Computes loads, costs, violations, invalid reasons, `valid`, `nfail` | Structural link failures invalidate; capacity/bw/latency only add cost and reasons | Structural failures invalidate; enforced reasons also invalidate |
| `step` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:710) | Applies one time slot transition | action | next state, reward, terminated, truncated, info | Uses `metrics["valid"]` and `metrics["nfail"]` to compute reward and invalid streak | Positive reward possible with capacity/bw/latency violations if structural links are valid | Any enforced violation yields negative reward path |
| `evaluate_gppo` | [training.py](/home/hpcnc/intern-research/src/workflows/training.py:860) | Logs per-slot outputs during evaluation | agent, gnn, env settings | summary dict, traces | Treats `info["valid_deployment"]`, costs, and reasons as the truth source | Invalidity mostly means structural/link issues under legacy | Invalidity expands according to strict mode |

## 4. Penalty components

| Component | Computed in | Plain-English logic | Contributes to `total_cost` | Contributes to `valid` |
| --- | --- | --- | --- | --- |
| `processing_cost` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:633) | Sum, for each RH, of `du_costs[split] * demand + cu_costs[split] * demand` | Yes | No directly |
| `routing_cost` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:600), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:629) | Delay-weighted traffic cost on the selected direct or ES-RC path: `phi_l * delay * demand_gbps` | Yes | No directly |
| `reconfiguration_cost` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:638) | `phi_r * (split_changes + es_changes + rc_changes)` against `prev_action` | Yes | No directly |
| `sla_penalty` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:660) | Aggregate soft slack: `es_overuse + rc_overuse + bandwidth_overuse + e2e_violation + crosshaul_violation` | Yes | Indirectly, only if enforced by mode |
| `es_overuse` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:651) | Sum of positive overflow above ES capacity | Via `sla_penalty` | Yes in capacity/full strict modes |
| `rc_overuse` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:652) | Sum of positive overflow above RC capacity | Via `sla_penalty` | Yes in capacity/full strict modes |
| `bandwidth_overuse` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:653) | Sum over edges of positive overflow above edge bandwidth | Via `sla_penalty` | Yes in bandwidth/full strict modes |
| `e2e_violation` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:601), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:630) | Sum of `max(actual_e2e_delay - RH_latency_budget, 0)` | Via `sla_penalty` | Yes only in `strict_full` |
| `crosshaul_violation` | [environment.py](/home/hpcnc/intern-research/src/core/environment.py:602), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:631) | Sum of `max(actual_crosshaul_delay - split_crosshaul_limit, 0)` | Via `sla_penalty` | Yes only in `strict_full` |

## 5. Reward logic

Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:715)

- Reward becomes negative whenever `metrics["valid"]` is `False`.
- If invalid but the invalid streak has not yet hit `max_invalid_streak`, reward is `-(nfail / (2 * num_rhs))`.
- If invalid and the streak reaches `max_invalid_streak`, reward is clamped to `-1.0`.
- Reward uses the positive log-based formula only when `metrics["valid"]` is `True`:
  `reward = (1 + log1p(total_cost))^-1`.

`metrics["valid"]` in practice:

- `legacy`: means the action has no structural link failures. Capacity, bandwidth, e2e latency, and crosshaul latency can all be violated while `valid=True`.
- `strict_connectivity_only`: same practical meaning as `legacy`, because `_enforced_reason_set()` is still empty.
- `strict_connectivity_plus_capacity`: structural validity plus no ES/RC overuse.
- `strict_connectivity_plus_capacity_plus_bandwidth`: structural validity plus no ES/RC overuse and no bandwidth overuse.
- `strict_full` / `strict`: structural validity plus no ES/RC overuse, no bandwidth overuse, no e2e latency violation, and no crosshaul latency violation.

## 6. Constraint-by-constraint mapping

### A. Current software soft-penalty constraints

These are always measured in `_evaluate_action()` and always add to `sla_penalty`, but only become hard-invalid if the current mode enforces them:

- ES capacity overflow
  Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:651)
- RC capacity overflow
  Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:652)
- Backhaul bandwidth overflow
  Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:653)
- End-to-end latency excess over the sampled RH latency budget
  Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:601), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:630)
- Crosshaul latency excess over `crosshaul_latency_limits[split]`
  Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:602), [environment.py](/home/hpcnc/intern-research/src/core/environment.py:631)

### B. Current software hard-invalid constraints

Always hard-invalid:

- Missing direct `RH -> RC` link for split 3
  Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:590)
- Missing `RH -> ES` link for non-direct splits
  Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:603)
- Missing `ES -> RC` link for non-direct splits
  Code: [environment.py](/home/hpcnc/intern-research/src/core/environment.py:610)

Mode-dependent hard-invalid:

- `es_capacity_exceeded`
- `rc_capacity_exceeded`
- `bandwidth_exceeded`
- `e2e_latency_exceeded`
- `crosshaul_latency_exceeded`

Enforcement switch:
[environment.py](/home/hpcnc/intern-research/src/core/environment.py:373)

## 7. Final summary

### Current legacy soft-penalty design

`legacy` computes all resource and latency violations, adds them into `sla_penalty`, and includes that in `total_cost`, but does not mark the deployment invalid for those violations. In practice, `valid=True` mainly means “structurally connected action,” not “mathematically feasible action.”

### Current strict design

Strictness is staged by `_enforced_reason_set()`. Structural connectivity is always hard. Capacity, bandwidth, and latency violations are first measured as soft quantities, then promoted to hard-invalid depending on mode. `strict_full` is the only mode that hard-enforces all currently measured soft violations.

### Main differences from paper-style mathematical constraints

- The current software separates “violation magnitude” from “validity” through `constraint_mode`, instead of treating all feasibility constraints as uniformly hard.
- In `legacy`, infeasible resource/latency assignments can still receive positive rewards if structural links exist.
- `sla_penalty` is a plain additive slack sum, not a separate constrained optimization layer.
- `strict_connectivity_only` is effectively equivalent to `legacy` for validity, because both enforce an empty soft-violation reason set.
