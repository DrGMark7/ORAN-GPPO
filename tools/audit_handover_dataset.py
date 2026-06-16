#!/usr/bin/env python3
"""Audit Colosseum/ColO-RAN data for handover-aware orchestration inputs.

The script intentionally audits the dataset before assuming mobility exists. It
discovers Colosseum-style CSV trees under --data-root, reconstructs candidate
UE-to-RH traces from slice metrics, preserves ambiguous multi-BS observations,
and writes evidence-backed feasibility reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_GROUPS = {"slice_mixed", "slice_traffic"}

BS_COLUMNS = ["time", "nof_ue", "dl_brate", "ul_brate"]
UE_COLUMNS = [
    "time",
    "cc",
    "pci",
    "earfcn",
    "rsrp",
    "pl",
    "cfo",
    "dl_mcs",
    "dl_snr",
    "dl_turbo",
    "dl_brate",
    "dl_bler",
    "ul_ta",
    "ul_mcs",
    "ul_buff",
    "ul_brate",
    "ul_bler",
    "rf_o",
    "rf_u",
    "rf_l",
    "is_attached",
]
SLICE_RAW_COLUMNS = [
    "Timestamp",
    "num_ues",
    "IMSI",
    "RNTI",
    "slicing_enabled",
    "slice_id",
    "slice_prb",
    "power_multiplier",
    "scheduling_policy",
    "dl_mcs",
    "dl_n_samples",
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
    "tx_errors downlink (%)",
    "dl_cqi",
    "ul_mcs",
    "ul_n_samples",
    "ul_buffer [bytes]",
    "rx_brate uplink [Mbps]",
    "rx_pkts uplink",
    "rx_errors uplink (%)",
    "ul_rssi",
    "ul_sinr",
    "phr",
    "sum_requested_prbs",
    "sum_granted_prbs",
    "dl_pmi",
    "dl_ri",
    "ul_n",
    "ul_turbo_iters",
]
SLICE_RENAME = {
    "Timestamp": "time",
    "num_ues": "num_ues",
    "IMSI": "imsi",
    "RNTI": "rnti",
    "slicing_enabled": "slicing_enabled",
    "slice_id": "slice_id",
    "slice_prb": "slice_prb",
    "power_multiplier": "power_multiplier",
    "scheduling_policy": "scheduling_policy",
    "dl_mcs": "dl_mcs",
    "dl_n_samples": "dl_n_samples",
    "dl_buffer [bytes]": "dl_buffer_bytes",
    "tx_brate downlink [Mbps]": "tx_brate_dl_mbps",
    "tx_pkts downlink": "tx_pkts_downlink",
    "tx_errors downlink (%)": "tx_errors_downlink_pct",
    "dl_cqi": "dl_cqi",
    "ul_mcs": "ul_mcs",
    "ul_n_samples": "ul_n_samples",
    "ul_buffer [bytes]": "ul_buffer_bytes",
    "rx_brate uplink [Mbps]": "rx_brate_ul_mbps",
    "rx_pkts uplink": "rx_pkts_uplink",
    "rx_errors uplink (%)": "rx_errors_uplink_pct",
    "ul_rssi": "ul_rssi",
    "ul_sinr": "ul_sinr",
    "phr": "phr",
    "sum_requested_prbs": "sum_requested_prbs",
    "sum_granted_prbs": "sum_granted_prbs",
    "dl_pmi": "dl_pmi",
    "dl_ri": "dl_ri",
    "ul_n": "ul_n",
    "ul_turbo_iters": "ul_turbo_iters",
}

TRACE_COLUMNS = [
    "dataset_group",
    "scenario",
    "training_id",
    "experiment_id",
    "time",
    "imsi",
    "rh_id",
    "base_station_id",
    "rnti",
    "slice_id",
    "dl_rate_mbps",
    "ul_rate_mbps",
    "dl_buffer_bytes",
    "ul_buffer_bytes",
    "dl_cqi",
    "ul_sinr",
    "sum_requested_prbs",
    "sum_granted_prbs",
    "source_file",
    "trace_confidence",
]
CONTEXT_COLUMNS = ["dataset_group", "scenario", "training_id", "experiment_id"]
BS_META_COLUMNS = CONTEXT_COLUMNS + ["base_station_id", "source_file"]


@dataclass(frozen=True)
class PathMeta:
    dataset_group: str
    scenario: str
    training_id: str
    experiment_id: str
    base_station_id: str
    source_file: str

    @property
    def context(self) -> tuple[str, str, str, str]:
        return (self.dataset_group, self.scenario, self.training_id, self.experiment_id)


@dataclass
class ImsiSummary:
    records: int = 0
    min_time: float | None = None
    max_time: float | None = None
    base_stations: set[str] = field(default_factory=set)
    rntis: set[str] = field(default_factory=set)
    slice_ids: set[str] = field(default_factory=set)
    rnti_changes: int = 0
    intervals: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))
    same_timestamp_multi_bs_count: int = 0
    sequential_multi_bs: bool = False
    ambiguous_multi_bs_observation: bool = False

    def update_times(self, min_time: float | None, max_time: float | None) -> None:
        if min_time is None or max_time is None:
            return
        self.min_time = min_time if self.min_time is None else min(self.min_time, min_time)
        self.max_time = max_time if self.max_time is None else max(self.max_time, max_time)


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.sum_abs = 0.0

    def update(self, x: pd.Series, y: pd.Series) -> None:
        vals = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1)
        vals = vals.dropna()
        if vals.empty:
            return
        xv = vals.iloc[:, 0].astype(float)
        yv = vals.iloc[:, 1].astype(float)
        self.n += int(len(vals))
        self.sum_x += float(xv.sum())
        self.sum_y += float(yv.sum())
        self.sum_x2 += float((xv * xv).sum())
        self.sum_y2 += float((yv * yv).sum())
        self.sum_xy += float((xv * yv).sum())
        self.sum_abs += float((xv - yv).abs().sum())

    def as_dict(self, prefix: str) -> dict[str, float | int]:
        if self.n == 0:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_correlation": math.nan,
                f"{prefix}_mae": math.nan,
            }
        denom_x = self.n * self.sum_x2 - self.sum_x * self.sum_x
        denom_y = self.n * self.sum_y2 - self.sum_y * self.sum_y
        corr = math.nan
        if denom_x > 0 and denom_y > 0:
            corr = (self.n * self.sum_xy - self.sum_x * self.sum_y) / math.sqrt(denom_x * denom_y)
        return {
            f"{prefix}_n": self.n,
            f"{prefix}_correlation": corr,
            f"{prefix}_mae": self.sum_abs / self.n,
        }


class ValidationAcc:
    def __init__(self) -> None:
        self.n_stats = RunningStats()
        self.lambda_stats = RunningStats()

    def update(self, reconstructed: pd.DataFrame, bs_df: pd.DataFrame) -> None:
        if reconstructed.empty or bs_df.empty:
            return
        joined = reconstructed.merge(bs_df, on="time", how="inner", suffixes=("_recon", "_bs"))
        if joined.empty:
            return
        self.n_stats.update(joined["N_r_t"], joined["nof_ue"])
        self.lambda_stats.update(joined["lambda_total_mbps"], joined["bs_lambda_total"])

    def as_dict(self, key: str) -> dict[str, Any]:
        out: dict[str, Any] = {"group": key}
        out.update(self.n_stats.as_dict("N_r_t_vs_bs_nof_ue"))
        out.update(self.lambda_stats.as_dict("lambda_total_vs_bs_total_brate"))
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--output-root", default=Path("outputs/handover_audit"), type=Path)
    parser.add_argument("--sample-limit", default=5000, type=int)
    parser.add_argument("--window-sec", default=[1, 5, 10], nargs="+", type=int)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def read_header(path: Path) -> list[str] | None:
    try:
        with path.open(newline="") as fh:
            return next(csv.reader(fh))
    except Exception:
        return None


def parse_colosseum_meta(path: Path) -> PathMeta | None:
    parts = path.parts
    group_idx = next((i for i, part in enumerate(parts) if part in DATASET_GROUPS), None)
    if group_idx is None or len(parts) <= group_idx + 4:
        return None
    dataset_group, scenario, training_id, experiment_id, base_station_id = parts[group_idx : group_idx + 5]
    if not re.fullmatch(r"tr\d+", training_id) or not re.fullmatch(r"exp\d+", experiment_id):
        return None
    if not re.fullmatch(r"bs\d+", base_station_id):
        return None
    return PathMeta(
        dataset_group=dataset_group,
        scenario=scenario,
        training_id=training_id,
        experiment_id=experiment_id,
        base_station_id=base_station_id,
        source_file=str(path),
    )


def append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        if not path.exists():
            df.to_csv(path, index=False)
        return
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def normalize_slice_df(path: Path, meta: PathMeta) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:") or str(c).strip() == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    missing = [c for c in SLICE_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing slice columns {missing}")
    df = df[SLICE_RAW_COLUMNS].rename(columns=SLICE_RENAME)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, value in {
        "dataset_group": meta.dataset_group,
        "scenario": meta.scenario,
        "training_id": meta.training_id,
        "experiment_id": meta.experiment_id,
        "base_station_id": meta.base_station_id,
        "source_file": meta.source_file,
    }.items():
        df[col] = value
    return df


def read_bs_df(path: Path, meta: PathMeta) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in BS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing bs columns {missing}")
    df = df[BS_COLUMNS].copy()
    for col in BS_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, value in {
        "dataset_group": meta.dataset_group,
        "scenario": meta.scenario,
        "training_id": meta.training_id,
        "experiment_id": meta.experiment_id,
        "base_station_id": meta.base_station_id,
        "source_file": meta.source_file,
    }.items():
        df[col] = value
    df["bs_lambda_total"] = df["dl_brate"].fillna(0) + df["ul_brate"].fillna(0)
    return df


def read_ue_sample(path: Path, meta: PathMeta, sample_limit: int) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=sample_limit)
    missing = [c for c in UE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing ue columns {missing}")
    df = df[UE_COLUMNS].copy()
    for col in UE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    match = re.match(r"ue(\d+)\.csv$", path.name)
    df["ue_file_id"] = int(match.group(1)) if match else pd.NA
    for col, value in {
        "dataset_group": meta.dataset_group,
        "scenario": meta.scenario,
        "training_id": meta.training_id,
        "experiment_id": meta.experiment_id,
        "base_station_id": meta.base_station_id,
        "source_file": meta.source_file,
    }.items():
        df[col] = value
    return df


def build_inventory(data_root: Path, output_root: Path) -> dict[str, Any]:
    malformed_folders: list[str] = []
    malformed_csv_files: list[dict[str, Any]] = []
    unexpected_schemas: dict[str, list[str]] = defaultdict(list)
    duplicate_schemas: Counter[str] = Counter()
    schema_examples: dict[str, str] = {}

    groups, scenarios, trainings, experiments, bs_folders = set(), set(), set(), set(), set()
    counts = Counter()

    for path in data_root.rglob("*.csv"):
        meta = parse_colosseum_meta(path)
        if meta is None:
            malformed_folders.append(str(path))
            continue
        groups.add(meta.dataset_group)
        scenarios.add(meta.scenario)
        trainings.add(meta.training_id)
        experiments.add(meta.experiment_id)
        bs_folders.add(str(Path(meta.dataset_group, meta.scenario, meta.training_id, meta.experiment_id, meta.base_station_id)))

        header = read_header(path)
        if header is None:
            malformed_csv_files.append({"file": str(path), "reason": "could not read header"})
            continue
        schema_key = ",".join(header)
        duplicate_schemas[schema_key] += 1
        schema_examples.setdefault(schema_key, str(path))

        if re.fullmatch(r"bs\d+\.csv", path.name):
            counts["bs_csv_files"] += 1
            if header != BS_COLUMNS:
                unexpected_schemas["bs_metrics"].append(str(path))
        elif re.fullmatch(r"ue\d+\.csv", path.name):
            counts["ue_csv_files"] += 1
            if header != UE_COLUMNS:
                unexpected_schemas["ue_metrics"].append(str(path))
        elif path.name.endswith("_metrics.csv") and path.parent.name.startswith("slices_bs"):
            counts["slice_metrics_csv_files"] += 1
            meaningful = [h for h in header if h.strip()]
            if meaningful != SLICE_RAW_COLUMNS:
                unexpected_schemas["slice_metrics"].append(str(path))
        else:
            counts["other_csv_files"] += 1
            unexpected_schemas["other"].append(str(path))

    inventory = {
        "dataset_groups": sorted(groups),
        "num_dataset_groups": len(groups),
        "scenarios": sorted(scenarios),
        "num_scenarios": len(scenarios),
        "training_folders": sorted(trainings, key=lambda s: int(s[2:]) if s[2:].isdigit() else s),
        "num_training_folders": len(trainings),
        "experiment_folders": sorted(experiments, key=lambda s: int(s[3:]) if s[3:].isdigit() else s),
        "num_experiment_folders": len(experiments),
        "num_base_station_folders": len(bs_folders),
        "counts": dict(counts),
        "missing_or_malformed_folders": malformed_folders[:1000],
        "num_missing_or_malformed_folders": len(malformed_folders),
        "malformed_csv_files": malformed_csv_files,
        "unexpected_schemas": {k: v[:1000] for k, v in unexpected_schemas.items()},
        "num_unexpected_schema_files": {k: len(v) for k, v in unexpected_schemas.items()},
        "duplicate_or_observed_schemas": [
            {"count": count, "example": schema_examples[schema], "header": schema}
            for schema, count in duplicate_schemas.most_common()
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2))
    write_inventory_md(inventory, output_root / "dataset_inventory.md")
    return inventory


def write_inventory_md(inventory: dict[str, Any], path: Path) -> None:
    rows = [
        ("Dataset groups", inventory["num_dataset_groups"]),
        ("Scenarios", inventory["num_scenarios"]),
        ("Training folders", inventory["num_training_folders"]),
        ("Experiment folders", inventory["num_experiment_folders"]),
        ("Base station folders", inventory["num_base_station_folders"]),
        ("bs*.csv files", inventory["counts"].get("bs_csv_files", 0)),
        ("ue*.csv files", inventory["counts"].get("ue_csv_files", 0)),
        ("*_metrics.csv files", inventory["counts"].get("slice_metrics_csv_files", 0)),
        ("Malformed folders/files", inventory["num_missing_or_malformed_folders"]),
    ]
    lines = [
        "# Dataset Inventory",
        "",
        "| Item | Count |",
        "| --- | ---: |",
    ]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += [
        "",
        "## Dataset Groups",
        "",
        ", ".join(inventory["dataset_groups"]) or "None",
        "",
        "## Scenarios",
        "",
        ", ".join(inventory["scenarios"]) or "None",
        "",
        "## Observed Schemas",
        "",
        "| Count | Example | Header |",
        "| ---: | --- | --- |",
    ]
    for item in inventory["duplicate_or_observed_schemas"]:
        lines.append(f"| {item['count']} | `{item['example']}` | `{item['header']}` |")
    lines += [
        "",
        "## Malformed CSV Files",
        "",
        "None found." if not inventory["malformed_csv_files"] else "",
    ]
    for item in inventory["malformed_csv_files"][:50]:
        lines.append(f"- `{item['file']}`: {item['reason']}")
    path.write_text("\n".join(lines) + "\n")


def trace_from_slice_df(df: pd.DataFrame, confidence: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "dataset_group": df["dataset_group"],
            "scenario": df["scenario"],
            "training_id": df["training_id"],
            "experiment_id": df["experiment_id"],
            "time": df["time"],
            "imsi": df["imsi"].astype("Int64").astype(str),
            "rh_id": df["base_station_id"],
            "base_station_id": df["base_station_id"],
            "rnti": df["rnti"],
            "slice_id": df["slice_id"],
            "dl_rate_mbps": df["tx_brate_dl_mbps"],
            "ul_rate_mbps": df["rx_brate_ul_mbps"],
            "dl_buffer_bytes": df["dl_buffer_bytes"],
            "ul_buffer_bytes": df["ul_buffer_bytes"],
            "dl_cqi": df["dl_cqi"],
            "ul_sinr": df["ul_sinr"],
            "sum_requested_prbs": df["sum_requested_prbs"],
            "sum_granted_prbs": df["sum_granted_prbs"],
            "source_file": df["source_file"],
            "trace_confidence": confidence,
        }
    )
    return out[TRACE_COLUMNS]


def aggregate_rh_timeseries(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame()
    df = trace_df.copy()
    numeric_cols = [
        "dl_rate_mbps",
        "ul_rate_mbps",
        "dl_buffer_bytes",
        "ul_buffer_bytes",
        "dl_cqi",
        "ul_sinr",
        "sum_requested_prbs",
        "sum_granted_prbs",
    ]
    for col in numeric_cols + ["slice_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    group_cols = CONTEXT_COLUMNS + ["time", "rh_id"]
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.agg(
        N_r_t=("imsi", "nunique"),
        lambda_dl_mbps=("dl_rate_mbps", "sum"),
        lambda_ul_mbps=("ul_rate_mbps", "sum"),
        avg_dl_cqi=("dl_cqi", "mean"),
        avg_ul_sinr=("ul_sinr", "mean"),
        avg_dl_buffer_bytes=("dl_buffer_bytes", "mean"),
        avg_ul_buffer_bytes=("ul_buffer_bytes", "mean"),
        sum_requested_prbs=("sum_requested_prbs", "sum"),
        sum_granted_prbs=("sum_granted_prbs", "sum"),
    ).reset_index()
    out["lambda_total_mbps"] = out["lambda_dl_mbps"] + out["lambda_ul_mbps"]
    for slice_id in [0, 1, 2]:
        counts = (
            df[df["slice_id"] == slice_id]
            .groupby(group_cols, dropna=False)["imsi"]
            .nunique()
            .rename(f"num_slice_{slice_id}")
            .reset_index()
        )
        out = out.merge(counts, on=group_cols, how="left")
        out[f"num_slice_{slice_id}"] = out[f"num_slice_{slice_id}"].fillna(0).astype(int)
    total_buffer = out["avg_dl_buffer_bytes"].fillna(0) + out["avg_ul_buffer_bytes"].fillna(0)
    high_buffer = (
        (total_buffer > total_buffer.quantile(0.90)).astype(float)
        if len(total_buffer) > 10
        else pd.Series(0.0, index=out.index)
    )
    out["rho_proxy"] = (
        ((out["avg_ul_sinr"] < 5).fillna(False)).astype(float)
        + ((out["avg_dl_cqi"] < 4).fillna(False)).astype(float)
        + high_buffer
    ) / 3.0
    return out[
        CONTEXT_COLUMNS
        + [
            "time",
            "rh_id",
            "N_r_t",
            "lambda_dl_mbps",
            "lambda_ul_mbps",
            "lambda_total_mbps",
            "avg_dl_cqi",
            "avg_ul_sinr",
            "avg_dl_buffer_bytes",
            "avg_ul_buffer_bytes",
            "sum_requested_prbs",
            "sum_granted_prbs",
            "num_slice_0",
            "num_slice_1",
            "num_slice_2",
            "rho_proxy",
        ]
    ]


def meaningful_score(df: pd.DataFrame) -> pd.Series:
    cols = [
        "dl_rate_mbps",
        "ul_rate_mbps",
        "dl_buffer_bytes",
        "ul_buffer_bytes",
        "sum_requested_prbs",
        "sum_granted_prbs",
    ]
    score = pd.Series(0.0, index=df.index)
    for col in cols:
        score += pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    return score


def summarize_file_for_imsi(df: pd.DataFrame, summary: ImsiSummary) -> None:
    if df.empty:
        return
    times = pd.to_numeric(df["time"], errors="coerce").dropna()
    summary.records += len(df)
    if not times.empty:
        min_t, max_t = float(times.min()), float(times.max())
        summary.update_times(min_t, max_t)
        summary.intervals[str(df["base_station_id"].iloc[0])].append((min_t, max_t))
    summary.base_stations.add(str(df["base_station_id"].iloc[0]))
    summary.rntis.update(str(v) for v in pd.to_numeric(df["rnti"], errors="coerce").dropna().unique())
    summary.slice_ids.update(str(v) for v in pd.to_numeric(df["slice_id"], errors="coerce").dropna().unique())
    ordered = df[["time", "rnti"]].dropna().sort_values("time")
    if len(ordered) > 1:
        summary.rnti_changes += int((ordered["rnti"].shift() != ordered["rnti"]).sum() - 1)


def detect_multi_timestamp_overlap(frames: list[pd.DataFrame]) -> int:
    by_bs: dict[str, set[Any]] = {}
    for frame in frames:
        if frame.empty:
            continue
        bs = str(frame["base_station_id"].iloc[0])
        by_bs.setdefault(bs, set()).update(frame["time"].dropna().tolist())
    overlap_count = 0
    seen: set[Any] = set()
    for bs, times in by_bs.items():
        overlap_count += len(seen.intersection(times))
        seen.update(times)
    return overlap_count


def intervals_are_sequential(intervals: dict[str, list[tuple[float, float]]]) -> bool:
    flat: list[tuple[float, float, str]] = []
    for bs, ranges in intervals.items():
        for start, end in ranges:
            flat.append((start, end, bs))
    flat.sort()
    for (_, prev_end, prev_bs), (curr_start, _, curr_bs) in zip(flat, flat[1:]):
        if prev_bs != curr_bs and prev_end < curr_start:
            return True
    return False


def resolve_multi_imsi_trace(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not frames:
        return pd.DataFrame(columns=TRACE_COLUMNS), pd.DataFrame(columns=TRACE_COLUMNS + ["ambiguity_reason"])
    df = pd.concat(frames, ignore_index=True)
    group_cols = CONTEXT_COLUMNS + ["time", "imsi"]
    resolved_parts: list[pd.DataFrame] = []
    ambiguity_parts: list[pd.DataFrame] = []
    for _, group in df.groupby(group_cols, dropna=False):
        if len(group) == 1:
            resolved_parts.append(group.assign(trace_confidence="high"))
            continue
        scores = meaningful_score(group)
        meaningful = group[scores > 0]
        if len(meaningful) == 1:
            resolved_parts.append(meaningful.assign(trace_confidence="high"))
        elif len(meaningful) == 0:
            ranked = group.copy()
            ranked["ambiguity_reason"] = "multiple_bs_same_time_no_meaningful_metric"
            ambiguity_parts.append(ranked)
        else:
            ranked = group.copy()
            ranked["ambiguity_reason"] = "multiple_bs_same_time_multiple_meaningful_metrics"
            ambiguity_parts.append(ranked)
    resolved = pd.concat(resolved_parts, ignore_index=True) if resolved_parts else pd.DataFrame(columns=TRACE_COLUMNS)
    ambiguous = (
        pd.concat(ambiguity_parts, ignore_index=True)
        if ambiguity_parts
        else pd.DataFrame(columns=TRACE_COLUMNS + ["ambiguity_reason"])
    )
    return resolved[TRACE_COLUMNS], ambiguous


def infer_handover_events(trace_df: pd.DataFrame, ambiguous_keys: set[tuple[Any, ...]]) -> pd.DataFrame:
    columns = [
        "dataset_group",
        "scenario",
        "training_id",
        "experiment_id",
        "imsi",
        "time_prev",
        "time_curr",
        "src_rh",
        "dst_rh",
        "src_rnti",
        "dst_rnti",
        "src_slice_id",
        "dst_slice_id",
        "dt_ms",
        "dl_rate_before_mbps",
        "ul_rate_before_mbps",
        "dl_buffer_before_bytes",
        "ul_buffer_before_bytes",
        "dl_cqi_before",
        "ul_sinr_before",
        "handover_confidence",
        "reason",
    ]
    if trace_df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    sort_cols = CONTEXT_COLUMNS + ["imsi", "time"]
    trace_df = trace_df.sort_values(sort_cols)
    for _, group in trace_df.groupby(CONTEXT_COLUMNS + ["imsi"], dropna=False):
        group = group.sort_values("time")
        prev = None
        for row in group.itertuples(index=False):
            curr = row._asdict()
            if prev is not None and curr["rh_id"] != prev["rh_id"]:
                dt_ms = pd.to_numeric(pd.Series([curr["time"]]), errors="coerce").iloc[0] - pd.to_numeric(
                    pd.Series([prev["time"]]), errors="coerce"
                ).iloc[0]
                confidence = "high"
                reason = "clear_consecutive_rh_change"
                prev_key = tuple(prev[c] for c in CONTEXT_COLUMNS + ["time", "imsi"])
                curr_key = tuple(curr[c] for c in CONTEXT_COLUMNS + ["time", "imsi"])
                if prev_key in ambiguous_keys or curr_key in ambiguous_keys:
                    confidence = "low"
                    reason = "transition_adjacent_to_same_time_multi_bs_ambiguity"
                elif pd.isna(dt_ms) or dt_ms <= 0:
                    confidence = "invalid"
                    reason = "non_positive_or_missing_dt"
                elif dt_ms > 10_000:
                    confidence = "medium"
                    reason = "large_timestamp_gap"
                rows.append(
                    {
                        "dataset_group": curr["dataset_group"],
                        "scenario": curr["scenario"],
                        "training_id": curr["training_id"],
                        "experiment_id": curr["experiment_id"],
                        "imsi": curr["imsi"],
                        "time_prev": prev["time"],
                        "time_curr": curr["time"],
                        "src_rh": prev["rh_id"],
                        "dst_rh": curr["rh_id"],
                        "src_rnti": prev["rnti"],
                        "dst_rnti": curr["rnti"],
                        "src_slice_id": prev["slice_id"],
                        "dst_slice_id": curr["slice_id"],
                        "dt_ms": dt_ms,
                        "dl_rate_before_mbps": prev["dl_rate_mbps"],
                        "ul_rate_before_mbps": prev["ul_rate_mbps"],
                        "dl_buffer_before_bytes": prev["dl_buffer_bytes"],
                        "ul_buffer_before_bytes": prev["ul_buffer_bytes"],
                        "dl_cqi_before": prev["dl_cqi"],
                        "ul_sinr_before": prev["ul_sinr"],
                        "handover_confidence": confidence,
                        "reason": reason,
                    }
                )
            prev = curr
    return pd.DataFrame(rows, columns=columns)


def write_empty_csv(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def run_audit(args: argparse.Namespace) -> None:
    data_root = args.data_root
    output_root = args.output_root
    sample_dir = output_root / "samples"
    figure_dir = output_root / "figures"
    sample_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Building dataset inventory")
    inventory = build_inventory(data_root, output_root)

    slice_files = sorted(
        p
        for p in data_root.rglob("*_metrics.csv")
        if p.parent.name.startswith("slices_bs") and parse_colosseum_meta(p) is not None
    )
    bs_files = sorted(p for p in data_root.rglob("bs*.csv") if parse_colosseum_meta(p) is not None)
    ue_files = sorted(p for p in data_root.rglob("ue*.csv") if parse_colosseum_meta(p) is not None)

    logging.info("Precomputing IMSI/base-station file multiplicity")
    imsi_file_bases: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    for path in slice_files:
        meta = parse_colosseum_meta(path)
        assert meta is not None
        imsi = path.name.split("_")[0]
        imsi_file_bases[meta.context + (imsi,)].add(meta.base_station_id)
    multi_imsi_keys = {key for key, bases in imsi_file_bases.items() if len(bases) > 1}

    logging.info("Writing samples")
    write_samples(bs_files, ue_files, slice_files, sample_dir, args.sample_limit)

    trace_path = output_root / "ue_rh_trace.csv"
    ambiguous_path = output_root / "ambiguous_ue_rh_assignments.csv"
    rh_path = output_root / "rh_timeseries.csv"
    for path in [trace_path, ambiguous_path, rh_path]:
        if path.exists():
            path.unlink()

    imsi_summaries: dict[tuple[str, str, str, str, str], ImsiSummary] = defaultdict(ImsiSummary)
    multi_frames: dict[tuple[str, str, str, str, str], list[pd.DataFrame]] = defaultdict(list)
    validation: dict[str, ValidationAcc] = defaultdict(ValidationAcc)
    validation["ALL"] = ValidationAcc()
    malformed_csv_files: list[dict[str, str]] = []

    logging.info("Streaming slice metrics by base-station folder")
    by_bs_dir: dict[Path, list[Path]] = defaultdict(list)
    for path in slice_files:
        by_bs_dir[path.parents[1]].append(path)

    for idx, (bs_dir, files) in enumerate(sorted(by_bs_dir.items()), start=1):
        if idx % 100 == 0:
            logging.info("Processed %s/%s base-station folders", idx, len(by_bs_dir))
        folder_trace_parts: list[pd.DataFrame] = []
        bs_meta = parse_colosseum_meta(files[0])
        if bs_meta is None:
            continue
        for path in files:
            meta = parse_colosseum_meta(path)
            if meta is None:
                continue
            imsi = path.name.split("_")[0]
            key = meta.context + (imsi,)
            try:
                slice_df = normalize_slice_df(path, meta)
            except Exception as exc:
                malformed_csv_files.append({"file": str(path), "reason": str(exc)})
                continue
            summarize_file_for_imsi(slice_df, imsi_summaries[key])
            trace = trace_from_slice_df(slice_df, confidence="high")
            if key in multi_imsi_keys:
                multi_frames[key].append(trace)
            else:
                folder_trace_parts.append(trace)

        if folder_trace_parts:
            folder_trace = pd.concat(folder_trace_parts, ignore_index=True)
            append_csv(folder_trace, trace_path)
            rh_df = aggregate_rh_timeseries(folder_trace)
            append_csv(rh_df, rh_path)
            update_validation_for_folder(bs_dir, rh_df, validation)

    logging.info("Resolving %s multi-BS IMSI groups", len(multi_frames))
    resolved_multi_parts: list[pd.DataFrame] = []
    ambiguous_parts: list[pd.DataFrame] = []
    ambiguous_keys: set[tuple[Any, ...]] = set()
    for key, frames in multi_frames.items():
        overlap_count = detect_multi_timestamp_overlap(frames)
        summary = imsi_summaries[key]
        summary.same_timestamp_multi_bs_count = overlap_count
        summary.ambiguous_multi_bs_observation = overlap_count > 0
        summary.sequential_multi_bs = intervals_are_sequential(summary.intervals)
        resolved, ambiguous = resolve_multi_imsi_trace(frames)
        if not resolved.empty:
            resolved_multi_parts.append(resolved)
        if not ambiguous.empty:
            ambiguous_parts.append(ambiguous)
            for row in ambiguous[CONTEXT_COLUMNS + ["time", "imsi"]].itertuples(index=False):
                ambiguous_keys.add(tuple(row))

    if resolved_multi_parts:
        resolved_multi = pd.concat(resolved_multi_parts, ignore_index=True)
        append_csv(resolved_multi, trace_path)
        append_csv(aggregate_rh_timeseries(resolved_multi), rh_path)
    else:
        resolved_multi = pd.DataFrame(columns=TRACE_COLUMNS)

    if ambiguous_parts:
        ambiguous_df = pd.concat(ambiguous_parts, ignore_index=True)
        append_csv(ambiguous_df, ambiguous_path)
    else:
        write_empty_csv(ambiguous_path, TRACE_COLUMNS + ["ambiguity_reason"])

    if not trace_path.exists():
        write_empty_csv(trace_path, TRACE_COLUMNS)
    if not rh_path.exists():
        write_empty_csv(
            rh_path,
            CONTEXT_COLUMNS
            + [
                "time",
                "rh_id",
                "N_r_t",
                "lambda_dl_mbps",
                "lambda_ul_mbps",
                "lambda_total_mbps",
                "avg_dl_cqi",
                "avg_ul_sinr",
                "avg_dl_buffer_bytes",
                "avg_ul_buffer_bytes",
                "sum_requested_prbs",
                "sum_granted_prbs",
                "num_slice_0",
                "num_slice_1",
                "num_slice_2",
                "rho_proxy",
            ],
        )

    logging.info("Writing IMSI trace summaries")
    imsi_summary_df = write_imsi_summary(imsi_summaries, output_root)

    logging.info("Inferring candidate handovers")
    candidate_traces = []
    if resolved_multi_parts:
        candidate_traces.append(pd.concat(resolved_multi_parts, ignore_index=True))
    handover_df = infer_handover_events(
        pd.concat(candidate_traces, ignore_index=True) if candidate_traces else pd.DataFrame(columns=TRACE_COLUMNS),
        ambiguous_keys,
    )
    handover_path = output_root / "handover_events_candidate.csv"
    handover_df.to_csv(handover_path, index=False)

    logging.info("Computing handover matrices and statistics")
    matrix_df, handover_stats = compute_handover_outputs(handover_df, rh_path, args.window_sec, output_root)

    logging.info("Writing validation outputs")
    write_validation_outputs(validation, output_root)

    logging.info("Writing state proxy outputs")
    write_state_proxy_outputs(rh_path, output_root)

    logging.info("Generating plots")
    plot_notes = generate_plots(imsi_summary_df, handover_df, matrix_df, rh_path, figure_dir)

    logging.info("Writing final feasibility report")
    write_final_report(
        output_root=output_root,
        inventory=inventory,
        imsi_summary=imsi_summary_df,
        handover_df=handover_df,
        matrix_df=matrix_df,
        handover_stats=handover_stats,
        plot_notes=plot_notes,
        malformed_csv_files=malformed_csv_files,
    )
    logging.info("Audit complete: %s", output_root)


def write_samples(
    bs_files: list[Path],
    ue_files: list[Path],
    slice_files: list[Path],
    sample_dir: Path,
    sample_limit: int,
) -> None:
    sample_specs = [
        ("bs_metrics_sample.csv", bs_files, read_bs_df),
        ("ue_metrics_sample.csv", ue_files, None),
        ("slice_metrics_sample.csv", slice_files, normalize_slice_df),
    ]
    for filename, files, reader in sample_specs:
        out_path = sample_dir / filename
        if out_path.exists():
            out_path.unlink()
        remaining = sample_limit
        for path in files:
            if remaining <= 0:
                break
            meta = parse_colosseum_meta(path)
            if meta is None:
                continue
            try:
                if filename.startswith("ue_"):
                    df = read_ue_sample(path, meta, remaining)
                else:
                    assert reader is not None
                    df = reader(path, meta)
            except Exception as exc:
                logging.warning("Skipping sample from %s: %s", path, exc)
                continue
            df.head(remaining).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            remaining -= min(len(df), remaining)
        if not out_path.exists():
            pd.DataFrame().to_csv(out_path, index=False)


def update_validation_for_folder(bs_dir: Path, rh_df: pd.DataFrame, validation: dict[str, ValidationAcc]) -> None:
    if rh_df.empty:
        return
    bs_files = list(bs_dir.glob("bs*.csv"))
    if not bs_files:
        return
    meta = parse_colosseum_meta(bs_files[0])
    if meta is None:
        return
    try:
        bs_df = read_bs_df(bs_files[0], meta)
    except Exception:
        return
    local_rh = rh_df[rh_df["rh_id"] == meta.base_station_id]
    validation["ALL"].update(local_rh, bs_df)
    validation[meta.scenario].update(local_rh, bs_df)


def write_imsi_summary(
    imsi_summaries: dict[tuple[str, str, str, str, str], ImsiSummary], output_root: Path
) -> pd.DataFrame:
    rows = []
    for key, summary in imsi_summaries.items():
        dataset_group, scenario, training_id, experiment_id, imsi = key
        if len(summary.base_stations) > 1 and not summary.sequential_multi_bs:
            summary.sequential_multi_bs = intervals_are_sequential(summary.intervals)
        rows.append(
            {
                "dataset_group": dataset_group,
                "scenario": scenario,
                "training_id": training_id,
                "experiment_id": experiment_id,
                "imsi": imsi,
                "num_records": summary.records,
                "min_time": summary.min_time,
                "max_time": summary.max_time,
                "unique_base_station_count": len(summary.base_stations),
                "base_stations_observed": ";".join(sorted(summary.base_stations)),
                "unique_rnti_count": len(summary.rntis),
                "rnti_values": ";".join(sorted(summary.rntis)[:20]),
                "rnti_change_count": summary.rnti_changes,
                "unique_slice_id_count": len(summary.slice_ids),
                "slice_ids_observed": ";".join(sorted(summary.slice_ids)),
                "appears_in_more_than_one_base_station": len(summary.base_stations) > 1,
                "sequential_multi_bs_timestamps": summary.sequential_multi_bs,
                "same_timestamp_multi_bs_count": summary.same_timestamp_multi_bs_count,
                "ambiguous_multi_bs_observation": summary.ambiguous_multi_bs_observation,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "imsi_trace_summary.csv", index=False)
    multi = int((df["unique_base_station_count"] > 1).sum()) if not df.empty else 0
    mostly_single = float((df["unique_base_station_count"] == 1).mean()) if not df.empty else math.nan
    lines = [
        "# IMSI Trace Summary",
        "",
        f"- IMSI/context groups: `{len(df)}`",
        f"- Groups observed in one base station: `{int((df['unique_base_station_count'] == 1).sum()) if not df.empty else 0}`",
        f"- Groups observed in multiple base stations: `{multi}`",
        f"- Fraction single-BS: `{mostly_single:.4f}`" if not math.isnan(mostly_single) else "- Fraction single-BS: `nan`",
        f"- Groups with same-timestamp multi-BS ambiguity: `{int((df['ambiguous_multi_bs_observation']).sum()) if not df.empty else 0}`",
        f"- Groups with sequential non-overlapping BS intervals: `{int((df['sequential_multi_bs_timestamps']).sum()) if not df.empty else 0}`",
        "",
    ]
    if multi:
        lines += [
            "## Multi-BS Examples",
            "",
            "| dataset_group | scenario | training_id | experiment_id | imsi | base_stations | same_timestamp_multi_bs_count | sequential |",
            "| --- | --- | --- | --- | --- | --- | ---: | --- |",
        ]
        cols = [
            "dataset_group",
            "scenario",
            "training_id",
            "experiment_id",
            "imsi",
            "base_stations_observed",
            "same_timestamp_multi_bs_count",
            "sequential_multi_bs_timestamps",
        ]
        for row in df[df["unique_base_station_count"] > 1].head(25)[cols].itertuples(index=False):
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
    (output_root / "imsi_trace_summary.md").write_text("\n".join(lines) + "\n")
    return df


def compute_handover_outputs(
    handover_df: pd.DataFrame, rh_path: Path, window_secs: list[int], output_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix_rows: list[dict[str, Any]] = []
    valid = handover_df[handover_df["handover_confidence"].isin(["high", "medium"])] if not handover_df.empty else handover_df
    if not valid.empty:
        for window_sec in window_secs:
            window_ms = window_sec * 1000
            temp = valid.copy()
            temp["window_start"] = (pd.to_numeric(temp["time_curr"], errors="coerce") // window_ms) * window_ms
            temp["window_end"] = temp["window_start"] + window_ms
            counts = (
                temp.groupby(CONTEXT_COLUMNS + ["window_start", "window_end", "src_rh", "dst_rh"], dropna=False)
                .size()
                .reset_index(name="num_handover_events")
            )
            for row in counts.itertuples(index=False):
                matrix_rows.append(
                    {
                        **{c: getattr(row, c) for c in CONTEXT_COLUMNS},
                        "window_sec": window_sec,
                        "window_start": row.window_start,
                        "window_end": row.window_end,
                        "src_rh": row.src_rh,
                        "dst_rh": row.dst_rh,
                        "num_handover_events": row.num_handover_events,
                        "num_ues_src_rh": math.nan,
                        "H_rr_prime_t": math.nan,
                    }
                )
    matrix_df = pd.DataFrame(
        matrix_rows,
        columns=CONTEXT_COLUMNS
        + [
            "window_sec",
            "window_start",
            "window_end",
            "src_rh",
            "dst_rh",
            "num_handover_events",
            "num_ues_src_rh",
            "H_rr_prime_t",
        ],
    )
    if not matrix_df.empty and rh_path.exists():
        matrix_df = fill_handover_denominators(matrix_df, rh_path)
    matrix_df.to_csv(output_root / "handover_matrix_timeseries.csv", index=False)

    stats = {
        "total_candidate_handovers": int(len(handover_df)),
        "valid_candidate_handovers": int(len(valid)),
        "handover_count_by_scenario": valid.groupby("scenario").size().to_dict() if not valid.empty else {},
        "handover_count_by_confidence": handover_df.groupby("handover_confidence").size().to_dict()
        if not handover_df.empty
        else {},
        "dominant_rh_pairs": (
            valid.groupby(["src_rh", "dst_rh"]).size().sort_values(ascending=False).head(20).to_dict()
            if not valid.empty
            else {}
        ),
    }
    (output_root / "handover_statistics.json").write_text(json.dumps(stringify_keys(stats), indent=2))
    lines = [
        "# Handover Statistics",
        "",
        f"- Candidate handover transitions: `{len(handover_df)}`",
        f"- Valid transitions counted for H_rr_prime_t: `{len(valid)}`",
        "",
    ]
    if valid.empty:
        lines.append("No valid candidate handovers were found, so empirical transition matrices are empty.")
    else:
        lines += ["## Scenario Counts", "", "| Scenario | Valid handovers |", "| --- | ---: |"]
        for scenario, count in stats["handover_count_by_scenario"].items():
            lines.append(f"| {scenario} | {count} |")
    (output_root / "handover_statistics.md").write_text("\n".join(lines) + "\n")
    return matrix_df, stats


def fill_handover_denominators(matrix_df: pd.DataFrame, rh_path: Path) -> pd.DataFrame:
    rh = pd.read_csv(rh_path, usecols=CONTEXT_COLUMNS + ["time", "rh_id", "N_r_t"])
    out = matrix_df.copy()
    merged_parts: list[pd.DataFrame] = []
    for window_sec, matrix_part in out.groupby("window_sec", dropna=False):
        window_ms = int(window_sec) * 1000
        rh_part = rh.copy()
        rh_part["window_start"] = (pd.to_numeric(rh_part["time"], errors="coerce") // window_ms) * window_ms
        denom = (
            rh_part.groupby(CONTEXT_COLUMNS + ["window_start", "rh_id"], dropna=False)["N_r_t"]
            .mean()
            .reset_index()
            .rename(columns={"rh_id": "src_rh", "N_r_t": "num_ues_src_rh"})
        )
        matrix_part = matrix_part.drop(columns=["num_ues_src_rh"], errors="ignore").merge(
            denom, on=CONTEXT_COLUMNS + ["window_start", "src_rh"], how="left"
        )
        merged_parts.append(matrix_part)
    if not merged_parts:
        return out
    out = pd.concat(merged_parts, ignore_index=True)
    out["H_rr_prime_t"] = out["num_handover_events"] / out["num_ues_src_rh"]
    return out


def stringify_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [stringify_keys(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def write_validation_outputs(validation: dict[str, ValidationAcc], output_root: Path) -> None:
    rows = [acc.as_dict(key) for key, acc in validation.items()]
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "rh_timeseries_validation.csv", index=False)
    lines = [
        "# RH Time-Series Validation",
        "",
        "The validation joins reconstructed RH time series with `bs*.csv` on exact timestamps within the same base-station folder.",
        "",
    ]
    if df.empty:
        lines.append("No exact timestamp matches were available for validation.")
    else:
        lines += ["| Group | N corr | N MAE | Lambda corr | Lambda MAE |", "| --- | ---: | ---: | ---: | ---: |"]
        for row in df.itertuples(index=False):
            lines.append(
                f"| {row.group} | {row.N_r_t_vs_bs_nof_ue_correlation:.4g} | "
                f"{row.N_r_t_vs_bs_nof_ue_mae:.4g} | "
                f"{row.lambda_total_vs_bs_total_brate_correlation:.4g} | "
                f"{row.lambda_total_vs_bs_total_brate_mae:.4g} |"
            )
    (output_root / "rh_timeseries_validation.md").write_text("\n".join(lines) + "\n")


def write_state_proxy_outputs(rh_path: Path, output_root: Path) -> None:
    out_path = output_root / "state_proxy_timeseries.csv"
    if out_path.exists():
        out_path.unlink()
    if not rh_path.exists():
        write_empty_csv(out_path, [])
        return
    summary_rows = []
    for chunk in pd.read_csv(rh_path, chunksize=250_000):
        chunk["rate_per_ue_mbps"] = np.where(
            chunk["N_r_t"] > 0, chunk["lambda_total_mbps"] / chunk["N_r_t"], np.nan
        )
        for t_fwd in [0.01, 0.05, 0.1]:
            chunk[f"pdcp_proxy_mb_t_fwd_{str(t_fwd).replace('.', '_')}"] = (
                chunk["rate_per_ue_mbps"] * t_fwd / 8.0
            )
        for t_drain in [0.01, 0.02, 0.05]:
            chunk[f"rlc_window_proxy_mb_t_drain_{str(t_drain).replace('.', '_')}"] = (
                chunk["rate_per_ue_mbps"] * t_drain / 8.0
            )
        chunk["rlc_buffer_proxy_mb"] = (
            chunk["avg_dl_buffer_bytes"].fillna(0) + chunk["avg_ul_buffer_bytes"].fillna(0)
        ) / (1024 * 1024)
        append_csv(chunk, out_path)
        summary_rows.append(
            {
                "rows": len(chunk),
                "mean_rate_per_ue_mbps": float(chunk["rate_per_ue_mbps"].mean()),
                "mean_rlc_buffer_proxy_mb": float(chunk["rlc_buffer_proxy_mb"].mean()),
            }
        )
    total_rows = sum(r["rows"] for r in summary_rows)
    mean_rate = np.average(
        [r["mean_rate_per_ue_mbps"] for r in summary_rows],
        weights=[r["rows"] for r in summary_rows],
    ) if summary_rows else math.nan
    mean_buffer = np.average(
        [r["mean_rlc_buffer_proxy_mb"] for r in summary_rows],
        weights=[r["rows"] for r in summary_rows],
    ) if summary_rows else math.nan
    lines = [
        "# State Proxy Summary",
        "",
        f"- RH time-series rows processed: `{total_rows}`",
        f"- Mean per-UE rate proxy: `{mean_rate:.6g}` Mbps",
        f"- Mean RLC buffer proxy: `{mean_buffer:.6g}` MB",
        "",
        "PDCP forwarding volume is only a bitrate-window approximation; the dataset does not contain direct PDCP forwarding logs.",
        "RLC buffer state has a stronger proxy because slice metrics expose downlink and uplink buffer fields, but it is still not a protocol-state trace.",
    ]
    (output_root / "state_proxy_summary.md").write_text("\n".join(lines) + "\n")


def generate_plots(
    imsi_summary: pd.DataFrame,
    handover_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    rh_path: Path,
    figure_dir: Path,
) -> list[str]:
    notes: list[str] = []
    if not imsi_summary.empty:
        plt.figure(figsize=(7, 4))
        imsi_summary["unique_base_station_count"].hist(bins=range(1, int(imsi_summary["unique_base_station_count"].max()) + 3))
        plt.xlabel("Unique RHs per IMSI/context")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figure_dir / "unique_rhs_per_imsi_histogram.png")
        plt.close()
    else:
        notes.append("unique RHs per IMSI histogram was not generated because IMSI summary is empty.")

    if not handover_df.empty:
        plt.figure(figsize=(8, 4))
        handover_df.groupby("scenario").size().sort_values().plot(kind="barh")
        plt.xlabel("Candidate handovers")
        plt.tight_layout()
        plt.savefig(figure_dir / "candidate_handover_count_by_scenario.png")
        plt.close()

        valid = handover_df[handover_df["handover_confidence"].isin(["high", "medium"])]
        if not valid.empty:
            plt.figure(figsize=(7, 4))
            valid.groupby(["src_rh", "dst_rh"]).size().sort_values(ascending=False).head(20).plot(kind="bar")
            plt.ylabel("Valid handovers")
            plt.tight_layout()
            plt.savefig(figure_dir / "valid_handover_count_by_rh_pair.png")
            plt.close()
        else:
            notes.append("valid handover count by RH pair was not generated because no valid handovers were found.")
    else:
        notes.append("candidate handover count by scenario was not generated because no candidate handovers were found.")
        notes.append("valid handover count by RH pair was not generated because no candidate handovers were found.")

    if rh_path.exists():
        rh = pd.read_csv(rh_path, nrows=200_000)
        if not rh.empty:
            context = rh[CONTEXT_COLUMNS].drop_duplicates().iloc[0].to_dict()
            rep = rh.copy()
            for col, val in context.items():
                rep = rep[rep[col] == val]
            plt.figure(figsize=(9, 4))
            for rh_id, group in rep.groupby("rh_id"):
                plt.plot(group["time"], group["N_r_t"], label=rh_id, linewidth=1)
            plt.xlabel("Time")
            plt.ylabel("N_r_t")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figure_dir / "representative_N_r_t_over_time.png")
            plt.close()

            plt.figure(figsize=(9, 4))
            for rh_id, group in rep.groupby("rh_id"):
                plt.plot(group["time"], group["lambda_total_mbps"], label=rh_id, linewidth=1)
            plt.xlabel("Time")
            plt.ylabel("lambda total Mbps")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figure_dir / "representative_lambda_over_time.png")
            plt.close()
        else:
            notes.append("N_r_t and lambda time-series plots were not generated because rh_timeseries is empty.")

    if not matrix_df.empty:
        aggregate = matrix_df.groupby(["scenario", "src_rh", "dst_rh"])["H_rr_prime_t"].sum().reset_index()
        for scenario, group in aggregate.groupby("scenario"):
            pivot = group.pivot(index="src_rh", columns="dst_rh", values="H_rr_prime_t").fillna(0)
            plt.figure(figsize=(5, 4))
            plt.imshow(pivot.values, aspect="auto")
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.colorbar(label="Aggregate H")
            plt.title(scenario)
            plt.tight_layout()
            plt.savefig(figure_dir / f"aggregate_handover_heatmap_{scenario}.png")
            plt.close()
    else:
        notes.append("aggregate H_rr_prime heatmap was not generated because handover matrices are empty.")

    if not handover_df.empty:
        valid = handover_df[handover_df["handover_confidence"].isin(["high", "medium"])].copy()
        if not valid.empty:
            valid["dt_ms"] = pd.to_numeric(valid["dt_ms"], errors="coerce")
            plt.figure(figsize=(7, 4))
            valid["dt_ms"].dropna().hist(bins=50)
            plt.xlabel("Dwell/change interval proxy (ms)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(figure_dir / "dwell_time_distribution.png")
            plt.close()
        else:
            notes.append("dwell time distribution was not generated because no valid handovers were found.")
    return notes


def write_final_report(
    output_root: Path,
    inventory: dict[str, Any],
    imsi_summary: pd.DataFrame,
    handover_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    handover_stats: dict[str, Any],
    plot_notes: list[str],
    malformed_csv_files: list[dict[str, str]],
) -> None:
    total_imsi = len(imsi_summary)
    multi_imsi = int((imsi_summary["unique_base_station_count"] > 1).sum()) if total_imsi else 0
    ambiguous_imsi = int(imsi_summary["ambiguous_multi_bs_observation"].sum()) if total_imsi else 0
    sequential_imsi = int(imsi_summary["sequential_multi_bs_timestamps"].sum()) if total_imsi else 0
    valid_handovers = int(
        handover_df["handover_confidence"].isin(["high", "medium"]).sum()
    ) if not handover_df.empty else 0

    can_trace = "PARTIALLY" if total_imsi and multi_imsi else "PARTIALLY" if total_imsi else "NO"
    if total_imsi and multi_imsi == 0:
        can_trace = "PARTIALLY"
    h_answer = (
        "YES, from valid IMSI transitions across RHs"
        if valid_handovers > 0
        else "NO, because the audited trace did not produce valid sequential IMSI transitions across RHs"
    )
    support = (
        "Strong support"
        if valid_handovers > 0 and multi_imsi / max(total_imsi, 1) > 0.05
        else "Partial support"
        if total_imsi > 0
        else "Weak support"
    )
    if valid_handovers == 0:
        support = "Partial support"

    lines = [
        "# Handover Feasibility Report",
        "",
        "This audit checks whether the local Colosseum/ColO-RAN CSV dataset can support handover-aware RH-level inputs for orchestration. It does not assume handovers exist.",
        "",
        "## Evidence Summary",
        "",
        f"- Dataset groups: `{inventory['num_dataset_groups']}`",
        f"- Scenarios: `{inventory['num_scenarios']}`",
        f"- Base-station folders: `{inventory['num_base_station_folders']}`",
        f"- Slice metric files: `{inventory['counts'].get('slice_metrics_csv_files', 0)}`",
        f"- IMSI/context groups: `{total_imsi}`",
        f"- IMSI/context groups observed in multiple BS folders: `{multi_imsi}`",
        f"- IMSI/context groups with same-timestamp multi-BS ambiguity: `{ambiguous_imsi}`",
        f"- IMSI/context groups with sequential non-overlapping BS intervals: `{sequential_imsi}`",
        f"- Candidate handover transitions: `{len(handover_df)}`",
        f"- Valid handover transitions counted for H_rr_prime_t: `{valid_handovers}`",
        "",
        "## A. Can this dataset reconstruct UE -> RH -> time?",
        "",
        f"Answer: **{can_trace}**.",
        "",
        "The slice metrics expose stable IMSI values and base-station folder metadata, so a candidate UE-to-RH time trace can be reconstructed as an audit artifact. The trace should be treated as RH assignment evidence, not as ground-truth handover labels. Multi-BS same-timestamp observations are preserved in `ambiguous_ue_rh_assignments.csv`.",
        "",
        "## B. Can this dataset directly produce N_r^t?",
        "",
        "Answer: **BOTH, and whether they match is reported in `rh_timeseries_validation.md`**.",
        "",
        "`bs*.csv` provides `nof_ue`; the reconstructed IMSI trace also supports `N_r_t` by counting unique IMSIs per RH and timestamp.",
        "",
        "## C. Can this dataset directly produce lambda_r^t?",
        "",
        "Answer: **BOTH, and whether they match is reported in `rh_timeseries_validation.md`**.",
        "",
        "`bs*.csv` provides BS-level bitrate fields. Slice metrics provide per-IMSI downlink and uplink bitrate fields that are aggregated into `lambda_dl_mbps`, `lambda_ul_mbps`, and `lambda_total_mbps`.",
        "",
        "## D. Can this dataset directly produce H_rr'^t?",
        "",
        f"Answer: **{h_answer}**.",
        "",
    ]
    if valid_handovers == 0:
        lines.append(
            "The audit did not find valid sequential RH changes after resolving same-time multi-BS ambiguity. Therefore direct empirical H_rr_prime_t should not be claimed from this Colosseum dataset."
        )
    else:
        lines.append(
            "Valid candidate RH transitions were found and used to populate `handover_matrix_timeseries.csv`. These are still inferred from logs rather than labeled handover events."
        )
    lines += [
        "",
        "## E. Can this dataset estimate S_PDCP,r^t and S_RLC,r^t?",
        "",
        "- PDCP forwarding volume: **proxy**. It is a bitrate-window approximation only; direct PDCP forwarding logs are not present.",
        "- RLC buffer state: **proxy**. The dataset exposes downlink and uplink buffer fields, so RLC-like buffer pressure can be estimated, but it is not a direct protocol state dump.",
        "",
        "## F. Is Colosseum enough for the handover part of this proposal?",
        "",
        f"Conclusion: **{support}**.",
        "",
    ]
    if valid_handovers == 0:
        lines.append(
            "The dataset supports N_r^t, lambda_r^t, traffic, slice, buffer, and radio calibration, but it does not provide enough valid handover transitions for direct handover modeling. Use it as a real-data calibration source and combine it with a handover mobility dataset or simulator for H_rr_prime_t."
        )
    else:
        lines.append(
            "The dataset contains inferred RH transitions, but the orchestration study should still distinguish inferred handover evidence from explicit handover-event labels."
        )
    lines += [
        "",
        "## Research-State Design Note",
        "",
        "The dataset audit may track per-UE identity using IMSI to reconstruct mobility. The RL environment should expose aggregated RH-level features only: N_r^t, lambda_r^t, H_rr_prime_t when valid, and state proxies. This avoids O(N_RH x N_UE) state growth.",
        "",
        "## Generated Artifacts",
        "",
        "- `dataset_inventory.md` and `dataset_inventory.json`",
        "- `imsi_trace_summary.csv` and `imsi_trace_summary.md`",
        "- `ue_rh_trace.csv`",
        "- `ambiguous_ue_rh_assignments.csv`",
        "- `handover_events_candidate.csv`",
        "- `rh_timeseries.csv`",
        "- `rh_timeseries_validation.md` and `rh_timeseries_validation.csv`",
        "- `handover_matrix_timeseries.csv`",
        "- `handover_statistics.md` and `handover_statistics.json`",
        "- `state_proxy_timeseries.csv` and `state_proxy_summary.md`",
        "",
    ]
    if malformed_csv_files:
        lines += ["## Malformed CSV Files Seen During Full Parse", ""]
        for item in malformed_csv_files[:50]:
            lines.append(f"- `{item['file']}`: {item['reason']}")
        lines.append("")
    if plot_notes:
        lines += ["## Plot Notes", ""]
        for note in plot_notes:
            lines.append(f"- {note}")
        lines.append("")
    (output_root / "HANDOVER_FEASIBILITY_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parsed_args = parse_args()
    setup_logging(parsed_args.log_level)
    run_audit(parsed_args)
