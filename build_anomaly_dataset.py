
#!/usr/bin/env python
"""Leakage-free anomaly-detection dataset from validated RAW telemetry only.

Reads a pull_all_data.py versioned run (telemetry/*.parquet + coverage
ledger), NEVER the rule-derived exports (ml_training_timeseries.jsonl, daily
snapshot CSVs, dashboards). Only device+key+windows with proven-complete
coverage enter; partial/failed windows are excluded and reported.

Feature governance (feature_policy.json, default-deny):
  * denylist patterns checked BEFORE feature generation,
  * allowlist required for any key (unknown keys excluded, fail-closed),
  * prohibited-output-column patterns checked AGAIN before writing,
  * the build fails if a prohibited column, sensitive key, or rule-derived
    field would reach the ML artifact.

Time safety: per-device backward windows only, capped forward-fill, strict
max_source_ts <= anchor_ts assertion, scalers fit on TRAIN only.

Splits: chronological, grouped (customer -> device fallback), purged.

Outputs in <out-dir>/ (gitignored by default):
  anomaly_features.parquet, scaler.json, feature_dictionary.csv,
  data_card.md, exclusion_report.json, split_report.json, build_report.json

Exit codes: 0 ok; 2 bad config/usage; 4 no usable data (nothing complete or
no allowlisted keys — lead must review keys first).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tb_resilient import CoverageLedger, atomic_write_bytes

UTC = timezone.utc

META_COLUMNS = (
    "device_id", "group_key", "anchor_ts", "split",
    "max_source_ts", "n_rows_used", "feature_ready",
)

FORBIDDEN_INPUT_FILES = ("ml_training_timeseries", "ts_daily_snapshots", "dashboard_data")


class PolicyViolation(Exception):
    pass


class NoUsableData(Exception):
    pass


# ─── policy ──────────────────────────────────────────────────────────────────


def load_policy(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg["_deny_res"] = [(re.compile(p, re.I), reason) for p, reason in cfg.get("denylist_patterns", [])]
    cfg["_allow"] = set(cfg.get("allowlist", []))
    cfg["_prohibited_res"] = [re.compile(p, re.I) for p in cfg.get("prohibited_output_columns", [])]
    return cfg


def deny_reason(key: str, policy: dict[str, Any]) -> str | None:
    for rx, reason in policy["_deny_res"]:
        if rx.search(key):
            return reason
    return None


def classify_keys(keys: list[str], policy: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    """Returns (allowed, excluded[{key, reason}]). Unknown keys are denied by default."""
    allowed, excluded = [], []
    for k in keys:
        reason = deny_reason(k, policy)
        if reason:
            excluded.append({"key": k, "reason": f"denylist:{reason}"})
            continue
        if k in policy["_allow"]:
            allowed.append(k)
            continue
        if policy.get("default_allow_unknown", False):
            allowed.append(k)
        else:
            excluded.append({"key": k, "reason": "unreviewed:not_in_allowlist"})
    return allowed, excluded


def guard_output_columns(feature_cols: list[str], policy: dict[str, Any]) -> None:
    bad = [c for c in feature_cols
           if any(rx.search(c) for rx in policy["_prohibited_res"])]
    if bad:
        raise PolicyViolation(f"prohibited columns would reach ML output: {bad}")


# ─── loading ─────────────────────────────────────────────────────────────────


def load_usable_keys(run_dir: Path) -> tuple[set[tuple[str, str]], list[dict[str, Any]], dict[str, Any]]:
    """Usable (device,key) pairs: coverage completeness complete/empty_verified only."""
    ledger = CoverageLedger("anomaly-build", run_dir / "coverage" / "telemetry_coverage.jsonl")
    usable: set[tuple[str, str]] = set()
    excluded: list[dict[str, Any]] = []
    stats = {"complete": 0, "empty": 0, "partial": 0, "failed": 0, "rows_usable": 0}
    for r in ledger.records:
        pair = (r.device_id, r.key)
        if r.completeness == "complete":
            usable.add(pair)
            stats["rows_usable"] += int(getattr(r, "rows", 0) or 0)
            stats["complete"] += 1
        elif r.completeness == "empty_verified":
            stats["empty"] += 1
        elif r.completeness in ("partial", "failed", "unknown", "retrying"):
            stats["partial" if r.completeness == "partial" else "failed"] += 1
            excluded.append({"device_id": r.device_id, "key": r.key,
                             "reason": f"coverage:{r.completeness}", "detail": r.error})
    return usable, excluded, stats


def normalize_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_ms"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts_ms"])
    df["ts_ms"] = df["ts_ms"].astype("int64")
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def iter_device_telemetry(run_dir: Path):
    """Yield (device_id, frame) one device at a time — peak memory is one device.

    Reads the per-key layout (telemetry/<device_id>/<key>.parquet, post-OOM fix)
    plus the legacy telemetry/telemetry_<device_id>.parquet layout. Skips temp files.
    """
    tele = run_dir / "telemetry"
    if not tele.is_dir():
        return
    for d_dir in sorted(p for p in tele.iterdir() if p.is_dir()):
        frames = []
        for pq in sorted(d_dir.glob("*.parquet")):
            if pq.name.startswith("."):
                continue
            try:
                frames.append(pd.read_parquet(pq))
            except Exception as exc:
                print(f"WARNING: unreadable {pq}: {exc}", file=sys.stderr)
        if frames:
            yield d_dir.name, pd.concat(frames, ignore_index=True)
    legacy = [pq for pq in sorted(tele.glob("telemetry_*.parquet")) if not pq.name.startswith(".")]
    if legacy:
        frames = []
        for pq in legacy:
            try:
                frames.append(pd.read_parquet(pq))
            except Exception as exc:
                print(f"WARNING: unreadable {pq.name}: {exc}", file=sys.stderr)
        if frames:
            yield "__legacy__", pd.concat(frames, ignore_index=True)


def iter_device_keys(run_dir: Path):
    """Yield (device_id, key) using column-only reads — cheap enough for --plan-only."""
    tele = run_dir / "telemetry"
    if not tele.is_dir():
        return
    for d_dir in sorted(p for p in tele.iterdir() if p.is_dir()):
        for pq in sorted(d_dir.glob("*.parquet")):
            if pq.name.startswith("."):
                continue
            try:
                keys = pd.read_parquet(pq, columns=["key"])["key"].unique().tolist()
            except Exception:
                continue
            for k in keys:
                yield d_dir.name, str(k)
    for pq in sorted(tele.glob("telemetry_*.parquet")):
        if pq.name.startswith("."):
            continue
        try:
            keys = pd.read_parquet(pq, columns=["key"])["key"].unique().tolist()
        except Exception:
            continue
        for k in keys:
            yield "", str(k)


def load_telemetry(run_dir: Path) -> pd.DataFrame:
    """Legacy bulk loader (kept for API compat). Prefer the streaming path in main:
    peak memory is the whole fleet — do not use on full 365-day runs."""
    frames = []
    for _, dev_df in iter_device_telemetry(run_dir):
        frames.append(dev_df)
    if not frames:
        return pd.DataFrame()
    return normalize_telemetry(pd.concat(frames, ignore_index=True))


def load_groups(run_dir: Path) -> dict[str, str]:
    """device_id -> customer_id for grouped splits (IDs for partitioning only)."""
    mapping: dict[str, str] = {}
    dev_csv = run_dir / "registry" / "devices.csv"
    if dev_csv.is_file():
        for row in pd.read_csv(dev_csv, dtype=str, keep_default_na=False).to_dict("records"):
            mapping[row.get("device_id", "")] = row.get("customer_id", "")
    return mapping


# ─── features ────────────────────────────────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    allowed: list[str],
    anchors_hours: int = 6,
    windows_hours: tuple[int, ...] = (1, 6, 24, 168),
    fill_cap_hours: float = 6.0,
    min_history_hours: float = 12.0,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Point-in-time rows: anchor T uses only records with ts <= T."""
    dictionary: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    if df.empty or not allowed:
        return pd.DataFrame(), dictionary
    df = df[df["key"].isin(allowed)].copy()
    if df.empty:
        return pd.DataFrame(), dictionary

    for did, g in df.groupby("device_id"):
        g = g.sort_values("ts_ms")
        t0 = int(g["ts_ms"].min())
        t1 = int(g["ts_ms"].max())
        step = anchors_hours * 3600_000
        first_anchor = t0 + int(min_history_hours * 3600_000)
        anchor = first_anchor - (first_anchor % step)
        by_key = {k: kg.sort_values("ts_ms") for k, kg in g.groupby("key")}
        while anchor <= t1:
            row: dict[str, Any] = {"device_id": did, "anchor_ts": pd.Timestamp(anchor, unit="ms", tz="UTC")}
            max_src = 0
            n_used = 0
            for key in allowed:
                kg = by_key.get(key)
                if kg is None:
                    continue
                hist = kg[kg["ts_ms"] <= anchor]
                if hist.empty:
                    continue
                vals = hist["value_num"].dropna()
                last_ts = int(hist["ts_ms"].max())
                max_src = max(max_src, last_ts)
                n_used += len(hist)
                age_h = (anchor - last_ts) / 3600_000
                base = key.replace(".", "_")
                if len(vals):
                    carried = float(vals.iloc[-1]) if age_h <= fill_cap_hours else float("nan")
                    row[f"{base}__last"] = carried
                    row[f"{base}__last_age_h"] = age_h
                    row[f"{base}__stale"] = int(age_h > fill_cap_hours)
                    for w in windows_hours:
                        wvals = vals[hist["ts_ms"] >= anchor - w * 3600_000]
                        row[f"{base}__count_{w}h"] = int(len(wvals))
                        if len(wvals):
                            row[f"{base}__mean_{w}h"] = float(wvals.mean())
                            row[f"{base}__std_{w}h"] = float(wvals.std()) if len(wvals) > 1 else 0.0
                            row[f"{base}__min_{w}h"] = float(wvals.min())
                            row[f"{base}__max_{w}h"] = float(wvals.max())
                else:
                    row[f"{base}__count_1h"] = 0
            if max_src:
                assert max_src <= anchor, f"leakage: max_source {max_src} > anchor {anchor}"
                row["max_source_ts"] = pd.Timestamp(max_src, unit="ms", tz="UTC")
                row["n_rows_used"] = n_used
                row["feature_ready"] = True
                rows.append(row)
            anchor += step

    frame = pd.DataFrame(rows)
    if not frame.empty:
        for col in frame.columns:
            if col in META_COLUMNS or col in ("device_id", "anchor_ts"):
                continue
            dictionary.append({"column": col, "dtype": str(frame[col].dtype),
                               "description": "backward-window sensor feature (point-in-time safe)"})
    return frame, dictionary


# ─── splits + scaler ─────────────────────────────────────────────────────────


def assign_chrono_splits(frame: pd.DataFrame, groups: dict[str, str],
                         fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
                         purge_hours: float = 192.0) -> pd.DataFrame:
    """Chronological per-group splits with purge band. Groups never split.

    Cut positions are chosen to best match the target row fractions while keeping
    every split non-empty (a single giant group can no longer eat validation).
    Purge applies ONLY at the two split boundaries — never between same-split groups.
    """
    frame = frame.copy()
    frame["group_key"] = ["customer:" + (groups.get(d) or f"device:{d}")
                          for d in frame["device_id"]]
    medians = frame.groupby("group_key")["anchor_ts"].median().sort_values()
    order = list(medians.index)
    counts = [int((frame["group_key"] == g).sum()) for g in order]
    total = sum(counts)
    targets = [fractions[0] * total, fractions[1] * total, fractions[2] * total]
    n = len(order)
    best = None
    for first in range(1, n - 1):
        for second in range(first + 1, n):
            sizes = [sum(counts[:first]), sum(counts[first:second]), sum(counts[second:])]
            if min(sizes) == 0:
                continue
            dev = sum(abs(sizes[k] - targets[k]) for k in range(3))
            if best is None or dev < best[0]:
                best = (dev, first, second)
    if best is None:  # fewer than 3 usable groups: chronological two-way split
        first = max(1, n - 1)
        best = (0.0, first, n)
    _, first, second = best
    assignment = {}
    for pos, g in enumerate(order):
        assignment[g] = "train" if pos < first else ("validation" if pos < second else "test")
    frame["split"] = [assignment[g] for g in frame["group_key"]]
    # Purge: rows within purge_hours of a split boundary go, tagged explicitly.
    med_list = medians.tolist()
    purge = pd.Timedelta(hours=purge_hours)
    drop_idx = set()
    for cut in (first, second):
        if cut <= 0 or cut >= n:
            continue
        mid = med_list[cut - 1] + (med_list[cut] - med_list[cut - 1]) / 2
        lo, hi = mid - purge / 2, mid + purge / 2
        drop_idx.update(frame[(frame["anchor_ts"] >= lo) & (frame["anchor_ts"] <= hi)].index.tolist())
    if "censor_reason" not in frame.columns:
        frame["censor_reason"] = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    frame.loc[sorted(drop_idx), "split"] = "purged"
    return frame.reset_index(drop=True)


def fit_scaler(frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    train = frame[frame["split"] == "train"]
    scaler: dict[str, dict[str, float]] = {}
    for c in feature_cols:
        vals = pd.to_numeric(train[c], errors="coerce").dropna()
        mu = float(vals.mean()) if len(vals) else 0.0
        sd = float(vals.std()) if len(vals) > 1 else 1.0
        scaler[c] = {"mean": mu, "std": sd if sd else 1.0}
    return scaler


def apply_scaler(frame: pd.DataFrame, feature_cols: list[str], scaler: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = frame.copy()
    for c in feature_cols:
        mu, sd = scaler[c]["mean"], scaler[c]["std"]
        vals = pd.to_numeric(frame[c], errors="coerce")
        frame[c] = (vals - mu) / sd
    return frame


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build leakage-free anomaly dataset from raw telemetry.")
    p.add_argument("--run-dir", required=True, help="Pull run dir (runs/<run_id>/) with telemetry/*.parquet.")
    p.add_argument("--feature-policy", default="feature_policy.json")
    p.add_argument("--out-dir", default="", help="Output dir (default: training_data/anomaly_<run_id>/).")
    p.add_argument("--anchors-hours", type=int, default=6)
    p.add_argument("--windows-hours", default="1,6,24,168")
    p.add_argument("--fill-cap-hours", type=float, default=6.0)
    p.add_argument("--min-history-hours", type=float, default=12.0)
    p.add_argument("--purge-hours", type=float, default=192.0)
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    run_dir = Path(args.run_dir)
    for forbidden in FORBIDDEN_INPUT_FILES:
        if forbidden in str(run_dir):
            print(f"ERROR: refusing rule-derived input {run_dir}", file=sys.stderr)
            return 2
    policy = load_policy(Path(args.feature_policy))
    out_dir = Path(args.out_dir) if args.out_dir else Path("training_data") / f"anomaly_{run_dir.name}"
    windows = tuple(int(w) for w in args.windows_hours.split(",") if w.strip())

    usable, excluded_windows, cov_stats = load_usable_keys(run_dir)
    # Cheap key discovery (column-only reads): never loads full telemetry here.
    present_keys = sorted({k for _, k in iter_device_keys(run_dir)})
    allowed, excluded_keys = classify_keys(present_keys, policy)
    allowed_set = set(allowed)

    if args.plan_only:
        print(f"run            : {run_dir}")
        print(f"coverage       : {cov_stats}")
        print(f"present keys   : {len(present_keys)} -> allowed {len(allowed)}, excluded {len(excluded_keys)}")
        print(f"usable rows    : ~{cov_stats.get('rows_usable', 0)} (ledger estimate, complete windows)")
        print(f"outputs        : {out_dir}/")
        return 0

    if not allowed:
        print("ERROR: allowlist empty/blocked every key — no features permitted. "
              "Lead must review keys into feature_policy.json allowlist.", file=sys.stderr)
        atomic_write_bytes(out_dir / "exclusion_report.json", json.dumps(
            {"excluded_keys": excluded_keys, "excluded_windows": excluded_windows[:200],
             "coverage": cov_stats}, indent=2).encode("utf-8"))
        return 4

    # Streaming feature build: one device in memory at a time.
    frames: list[pd.DataFrame] = []
    dictionary: list[dict[str, str]] = []
    for _, dev_df in iter_device_telemetry(run_dir):
        dev_df = normalize_telemetry(dev_df)
        keep = [(d, k) in usable and k in allowed_set
                for d, k in zip(dev_df["device_id"].astype(str), dev_df["key"].astype(str))]
        dev_df = dev_df[pd.Series(keep, index=dev_df.index)]
        del keep
        if dev_df.empty:
            continue
        dev_frame, dev_dict = build_features(dev_df, allowed, args.anchors_hours, windows,
                                             args.fill_cap_hours, args.min_history_hours)
        del dev_df
        if dev_frame.empty:
            continue
        frames.append(dev_frame)
        dictionary.extend(dev_dict)
    if not frames:
        print("ERROR: no usable telemetry (all windows partial/failed?)", file=sys.stderr)
        return 4
    frame = pd.concat(frames, ignore_index=True)
    del frames
    if frame.empty:
        print("ERROR: feature frame empty.", file=sys.stderr)
        return 4
    feature_cols = [c for c in frame.columns if c not in META_COLUMNS and c not in ("device_id", "anchor_ts")]
    try:
        guard_output_columns(feature_cols, policy)
    except PolicyViolation as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    groups = load_groups(run_dir)
    frame = assign_chrono_splits(frame, groups, purge_hours=args.purge_hours)
    scaler = fit_scaler(frame, feature_cols)
    frame = apply_scaler(frame, feature_cols, scaler)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_dir / "anomaly_features.parquet", _to_parquet_bytes(frame))
    atomic_write_bytes(out_dir / "scaler.json", json.dumps(
        {"fit_on": "train", "columns": scaler}, indent=2).encode("utf-8"))
    dict_rows = [{"column": c, "dtype": str(frame[c].dtype),
                  "description": next((d["description"] for d in dictionary if d["column"] == c), "")}
                 for c in feature_cols]
    import csv as _csv, io as _io
    buf = _io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=["column", "dtype", "description"])
    w.writeheader()
    w.writerows(dict_rows)
    atomic_write_bytes(out_dir / "feature_dictionary.csv", buf.getvalue().encode("utf-8"))
    split_rep = {"counts": frame["split"].value_counts().to_dict(),
                 "time_ranges": {s: {"min": str(frame[frame['split'] == s]["anchor_ts"].min()),
                                     "max": str(frame[frame['split'] == s]["anchor_ts"].max())}
                                 for s in ("train", "validation", "test") if (frame["split"] == s).any()},
                 "devices": int(frame["device_id"].nunique())}
    atomic_write_bytes(out_dir / "split_report.json", json.dumps(split_rep, indent=2).encode("utf-8"))
    atomic_write_bytes(out_dir / "exclusion_report.json", json.dumps(
        {"excluded_keys": excluded_keys, "excluded_windows": excluded_windows,
         "coverage": cov_stats}, indent=2).encode("utf-8"))
    card = ("# Anomaly dataset data card\n\n"
            f"- Source run: {run_dir} (raw telemetry only; complete windows only)\n"
            f"- Rows: {len(frame)} across {frame['device_id'].nunique()} devices\n"
            f"- Features: {len(feature_cols)} sensor-derived, point-in-time safe\n"
            f"- Policy: default-deny; allowlisted keys: {allowed}\n"
            "- No rule-derived scores, severities, alarms-as-features, IDs-as-features, "
            "GPS, secrets, payloads, or label proxies (fail-closed guards).\n"
            f"- Splits: chronological grouped + {args.purge_hours}h purge; scaler fit on train only.\n"
            "- Labels: NONE. Censored/negative outage labels are out of scope for this artifact.\n")
    atomic_write_bytes(out_dir / "data_card.md", card.encode("utf-8"))
    atomic_write_bytes(out_dir / "build_report.json", json.dumps(
        {"rows": len(frame), "devices": int(frame["device_id"].nunique()),
         "features": len(feature_cols), "allowed_keys": allowed,
         "splits": split_rep["counts"], "built_at": datetime.now(UTC).isoformat()}, indent=2).encode("utf-8"))
    print(f"OK: {len(frame)} rows, {len(feature_cols)} features -> {out_dir}/")
    return 0


def _to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    import io as _io
    buf = _io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
