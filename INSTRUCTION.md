Please debug why strict constraint mode produces 0.0% valid deployments and infinite average cost across all train/test topologies.

Likely cause:
- In environment.py, deployment_cost is set to inf whenever metrics["valid"] is False in step().
- In strict mode, _evaluate_action() marks the deployment invalid if any of these occur:
  es_capacity_exceeded, rc_capacity_exceeded, bandwidth_exceeded,
  e2e_latency_exceeded, crosshaul_latency_exceeded.
- Current logs show that strict mode is dominated by:
  es_capacity_exceeded, e2e_latency_exceeded, crosshaul_latency_exceeded.
- Therefore, inf is a consequence of zero valid steps, not a numeric overflow bug.

Please implement the following debugging support:

1. Add strict-feasible reset check
- Right now find_feasible_action() temporarily forces constraint_mode="legacy".
- Add a separate strict-mode feasibility checker, or at least a reset-time field:
  has_strictly_valid_action
- This will tell us whether each sampled scenario has any action that can satisfy strict mode at all.

2. Add per-topology strict feasibility diagnostics
For each topology_id, report:
- fraction of resets with at least one strict-valid action
- average number of invalid reasons per step
- counts for each strict invalid reason

3. Add first-failure tracing
For a small number of episodes/steps, print or save:
- chosen split vector summary
- selected topology_id
- es_overuse
- rc_overuse
- bandwidth_overuse
- e2e_violation
- crosshaul_violation
- invalid_reasons
This should help identify which constraint is killing validity first.

4. Check whether strict mode is practically impossible under current benchmark settings
- Verify if the current topology delays and latency limits make strict-valid actions too rare or nonexistent.
- In particular inspect crosshaul latency:
  crosshaul_latency_limits = [10.0, 1.0, 0.25, 0.25]
- Compare these against the actual ES-RC / RH-RC delays in the topology specs.

5. Add an optional debug summary:
- percentage of steps invalid due to each reason
- percentage of episodes with no strict-valid action at all
- average valid rate under legacy vs strict on the same topology and seed

Goal:
Determine whether strict mode is failing because:
(a) the benchmark/topology setup makes strict-feasible actions nearly impossible,
(b) the policy never reaches the strict-feasible region,
or (c) there is a logic issue in constraint evaluation.