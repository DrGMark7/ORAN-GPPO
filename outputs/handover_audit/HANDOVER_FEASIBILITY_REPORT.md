# Handover Feasibility Report

This audit checks whether the local Colosseum/ColO-RAN CSV dataset can support handover-aware RH-level inputs for orchestration. It does not assume handovers exist.

## Evidence Summary

- Dataset groups: `2`
- Scenarios: `4`
- Base-station folders: `2144`
- Slice metric files: `21612`
- IMSI/context groups: `21183`
- IMSI/context groups observed in multiple BS folders: `420`
- IMSI/context groups with same-timestamp multi-BS ambiguity: `0`
- IMSI/context groups with sequential non-overlapping BS intervals: `364`
- Candidate handover transitions: `38837`
- Valid handover transitions counted for H_rr_prime_t: `38837`

## A. Can this dataset reconstruct UE -> RH -> time?

Answer: **PARTIALLY**.

The slice metrics expose stable IMSI values and base-station folder metadata, so a candidate UE-to-RH time trace can be reconstructed as an audit artifact. The trace should be treated as RH assignment evidence, not as ground-truth handover labels. Multi-BS same-timestamp observations are preserved in `ambiguous_ue_rh_assignments.csv`.

## B. Can this dataset directly produce N_r^t?

Answer: **BOTH, and whether they match is reported in `rh_timeseries_validation.md`**.

`bs*.csv` provides `nof_ue`; the reconstructed IMSI trace also supports `N_r_t` by counting unique IMSIs per RH and timestamp.

## C. Can this dataset directly produce lambda_r^t?

Answer: **BOTH, and whether they match is reported in `rh_timeseries_validation.md`**.

`bs*.csv` provides BS-level bitrate fields. Slice metrics provide per-IMSI downlink and uplink bitrate fields that are aggregated into `lambda_dl_mbps`, `lambda_ul_mbps`, and `lambda_total_mbps`.

## D. Can this dataset directly produce H_rr'^t?

Answer: **YES, from valid IMSI transitions across RHs**.

Valid candidate RH transitions were found and used to populate `handover_matrix_timeseries.csv`. These are still inferred from logs rather than labeled handover events.

## E. Can this dataset estimate S_PDCP,r^t and S_RLC,r^t?

- PDCP forwarding volume: **proxy**. It is a bitrate-window approximation only; direct PDCP forwarding logs are not present.
- RLC buffer state: **proxy**. The dataset exposes downlink and uplink buffer fields, so RLC-like buffer pressure can be estimated, but it is not a direct protocol state dump.

## F. Is Colosseum enough for the handover part of this proposal?

Conclusion: **Partial support**.

The dataset contains inferred RH transitions, but the orchestration study should still distinguish inferred handover evidence from explicit handover-event labels.

## Research-State Design Note

The dataset audit may track per-UE identity using IMSI to reconstruct mobility. The RL environment should expose aggregated RH-level features only: N_r^t, lambda_r^t, H_rr_prime_t when valid, and state proxies. This avoids O(N_RH x N_UE) state growth.

## Generated Artifacts

- `dataset_inventory.md` and `dataset_inventory.json`
- `imsi_trace_summary.csv` and `imsi_trace_summary.md`
- `ue_rh_trace.csv`
- `ambiguous_ue_rh_assignments.csv`
- `handover_events_candidate.csv`
- `rh_timeseries.csv`
- `rh_timeseries_validation.md` and `rh_timeseries_validation.csv`
- `handover_matrix_timeseries.csv`
- `handover_statistics.md` and `handover_statistics.json`
- `state_proxy_timeseries.csv` and `state_proxy_summary.md`

