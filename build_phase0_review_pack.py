#!/usr/bin/env python
"""Phase-0 human-review pack: heuristic CRITICAL vs snapshot-derived offline windows.

Scaffold only — produces the JOIN TABLE for the lead + Seple ops verdict.
It never decides ground truth itself.

Inputs (all paths overridable; defaults are the post-harvest fresh versions):
  --grid            fresh daily grid, e.g. ts_daily_snapshots_20260921_1201.csv
                    (device-day rows; score lives in ts_fault_score /
                    ts_severity / ts_top_reasons — the preferred pack source)
  --timeseries      ml_training_timeseries.jsonl (score lives in
                    _meta.severity / _meta.score / _meta.date — used only for
                    a grid-vs-JSONL agreement cross-check, not pack rows)
  --jsonl           legacy snapshot-level JSONL, e.g. ml_training_v11.jsonl
                    (used only when --grid is absent)
  --snapshots-dir   current_state_snapshots/ (current_state_YYYYMMDD.csv files)
  --recoveries      offline_recoveries.csv (optional; derived by
                    pull_current_state_snapshots.py)

Output (gitignored dir by default):
  training_data/phase0_review_pack.csv — one row per heuristic-CRITICAL
  device (or device-day), with snapshot coverage attached and a
  ``human_verdict`` column left blank for ops to fill
  (true-fault / false-alarm / inconclusive).

Read-only w.r.t. the tenant and harvest_cache: no API calls, no cache access.
Exit 0 even when snapshots are absent (reports ``no-snapshot-coverage``).

Usage:
    python build_phase0_review_pack.py --help
    python build_phase0_review_pack.py --plan-only
    python build_phase0_review_pack.py
    python build_phase0_review_pack.py --jsonl ml_training_v11.jsonl --min-score 70
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

FAULT_CLASS_RE = re.compile(r"fault_class\s*:\s*([A-Za-z]+)", re.I)
FAULT_SCORE_RE = re.compile(r"fault_score\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.I)
REASONS_RE = re.compile(r"reasons\s*:\s*(.+)$", re.I | re.S)

DEFAULT_JSONL = Path("jsonl") / "ml_training_v11.jsonl"
DEFAULT_SNAPSHOTS_DIR = Path("current_state_snapshots")
DEFAULT_OUT = Path("training_data") / "phase0_review_pack.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    p.add_argument("--grid", default="",
                   help="Fresh daily grid CSV (device-day rows with ts_severity). "
                        "When given, it replaces --jsonl as the pack source.")
    p.add_argument("--timeseries", default="",
                   help="jsonl/ml_training_timeseries.jsonl for a grid-vs-JSONL severity cross-check.")
    p.add_argument("--severity", default="CRITICAL",
                   help="Grid/JSONL severity value to pack (default: CRITICAL).")
    p.add_argument("--sample-n", type=int, default=60,
                   help="Stratified sample size (round-robin across bank x month, "
                        "highest score first within each stratum). 0 disables the sample file.")
    p.add_argument("--snapshots-dir", default=str(DEFAULT_SNAPSHOTS_DIR))
    p.add_argument("--recoveries", default="",
                   help="Path to offline_recoveries.csv (default: <snapshots-dir>/offline_recoveries.csv if present).")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--min-score", type=float, default=70.0,
                   help="Heuristic score threshold when fault_class is missing (default: 70).")
    p.add_argument("--plan-only", action="store_true",
                   help="Print what would be read/written, touch nothing.")
    return p.parse_args(argv)


def mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return "missing"


def load_heuristic_critical(jsonl_path: Path, min_score: float) -> tuple[list[dict], dict]:
    """Return (critical_rows, stats). Raises FileNotFoundError if missing."""
    critical: list[dict] = []
    stats = {"n_lines": 0, "n_critical": 0, "classes": {}}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["n_lines"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            output = str(rec.get("output", ""))
            meta = rec.get("_meta", {}) if isinstance(rec.get("_meta"), dict) else {}
            m_class = FAULT_CLASS_RE.search(output)
            m_score = FAULT_SCORE_RE.search(output)
            m_reasons = REASONS_RE.search(output)
            cls = m_class.group(1).upper() if m_class else ""
            score = float(m_score.group(1)) if m_score else float(meta.get("score") or 0.0)
            stats["classes"][cls or "UNKNOWN"] = stats["classes"].get(cls or "UNKNOWN", 0) + 1
            is_critical = (cls == "CRITICAL") or (not cls and score >= min_score)
            if not is_critical:
                continue
            stats["n_critical"] += 1
            critical.append({
                "device_id": str(meta.get("device_id", "")),
                "bank": str(meta.get("bank", "")),
                "is_real_device": str(meta.get("is_real_device", "")),
                "heuristic_class": cls or f"SCORE>={min_score:g}",
                "heuristic_score": score,
                "heuristic_reasons": (m_reasons.group(1).strip()[:300] if m_reasons else ""),
                "input_snippet": str(rec.get("input", ""))[:300],
            })
    return critical, stats


def load_grid_critical(grid_path: Path, severity: str) -> tuple[list[dict], dict]:
    """Stream the (large) daily grid; keep device-day rows at `severity`.

    Score location (confirmed 2026-09-21): trailing columns ts_fault_score /
    ts_severity / ts_top_reasons; identity in device_id / date / bank_name /
    branch_name / device_name. The grid is written with a UTF-8 BOM, so it
    must be opened with utf-8-sig (plain utf-8 yields a '?device_id' key and
    a blank join column). Never loads the whole file."""
    wanted = severity.upper()
    critical: list[dict] = []
    stats = {"n_rows": 0, "n_critical": 0, "severities": {}}
    with grid_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            stats["n_rows"] += 1
            sev = str(row.get("ts_severity", "")).strip().upper()
            stats["severities"][sev or "UNKNOWN"] = stats["severities"].get(sev or "UNKNOWN", 0) + 1
            if sev != wanted:
                continue
            stats["n_critical"] += 1
            try:
                score = float(row.get("ts_fault_score") or 0.0)
            except ValueError:
                score = 0.0
            critical.append({
                "device_id": row.get("device_id", ""),
                "date": row.get("date", ""),
                "bank": row.get("bank_name", ""),
                "branch": row.get("branch_name", ""),
                "device_name": row.get("device_name", ""),
                "is_real_device": row.get("is_real_device", ""),
                "heuristic_class": sev,
                "heuristic_score": score,
                "heuristic_reasons": str(row.get("ts_top_reasons", ""))[:300],
                "input_snippet": "",
            })
    return critical, stats


def count_timeseries_severity(ts_path: Path) -> dict:
    """Stream the (large) timeseries JSONL; score lives in _meta.severity /
    _meta.score. Returns a severity -> count map for the cross-check."""
    counts: dict = {"_lines": 0, "_json_errors": 0}
    with ts_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counts["_lines"] += 1
            try:
                meta = json.loads(line).get("_meta", {}) or {}
            except json.JSONDecodeError:
                counts["_json_errors"] += 1
                continue
            sev = str(meta.get("severity", "")).strip().upper() or "UNKNOWN"
            counts[sev] = counts.get(sev, 0) + 1
    return counts


def load_snapshot_offline(snapshots_dir: Path) -> tuple[dict, dict]:
    """Map device_id -> offline snapshot info. Empty when no CSVs exist yet."""
    info: dict = {}
    stats = {"n_files": 0, "n_rows": 0, "n_offline_rows": 0}
    if not snapshots_dir.is_dir():
        return info, stats
    for csv_path in sorted(snapshots_dir.glob("current_state_*.csv")):
        stats["n_files"] += 1
        try:
            with csv_path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    stats["n_rows"] += 1
                    dev = row.get("device_id", "")
                    if not dev:
                        continue
                    offline = str(row.get("is_offline", "")).lower() in {"1", "true", "yes"}
                    entry = info.setdefault(dev, {"n_snapshots": 0, "n_offline": 0,
                                                  "last_snapshot": "", "last_offline": ""})
                    entry["n_snapshots"] += 1
                    snap_ts = row.get("snapshot_utc", "")
                    if snap_ts > entry["last_snapshot"]:
                        entry["last_snapshot"] = snap_ts
                    if offline:
                        stats["n_offline_rows"] += 1
                        entry["n_offline"] += 1
                        if snap_ts > entry["last_offline"]:
                            entry["last_offline"] = snap_ts
        except OSError:
            continue
    return info, stats


def load_recoveries(recoveries_path: Path) -> dict:
    """Map device_id -> list of recovery rows (may be empty)."""
    out: dict = {}
    if not recoveries_path.is_file():
        return out
    with recoveries_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row.get("device_id", ""), []).append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jsonl_path = Path(args.jsonl)
    snapshots_dir = Path(args.snapshots_dir)
    rec_path = Path(args.recoveries) if args.recoveries else snapshots_dir / "offline_recoveries.csv"
    out_path = Path(args.out)

    grid_path = Path(args.grid) if args.grid else None
    ts_path = Path(args.timeseries) if args.timeseries else None

    if args.plan_only:
        print(f"would read pack source      : "
              f"{grid_path or jsonl_path} ({mtime_iso(grid_path or jsonl_path)})")
        if ts_path is not None:
            print(f"would cross-check timeseries: {ts_path} ({mtime_iso(ts_path)})")
        print(f"would read snapshots dir   : {snapshots_dir}/ "
              f"({len(list(snapshots_dir.glob('current_state_*.csv')))} files)" if snapshots_dir.is_dir()
              else f"would read snapshots dir   : {snapshots_dir}/ (missing)")
        print(f"would read recoveries      : {rec_path} ({mtime_iso(rec_path)})")
        print(f"would write review pack    : {out_path}")
        print("no tenant API calls, no harvest_cache access.")
        return 0

    critical: list[dict] = []
    if grid_path is not None:
        if not grid_path.is_file():
            print(f"ERROR: grid not found: {grid_path}", file=sys.stderr)
            return 2
        critical, gstats = load_grid_critical(grid_path, args.severity)
        print(f"grid input      : {grid_path} (mtime {mtime_iso(grid_path)})")
        print(f"  rows={gstats['n_rows']} {args.severity}={gstats['n_critical']} "
              f"severities={gstats['severities']}")
        if ts_path is not None:
            if not ts_path.is_file():
                print(f"ERROR: timeseries not found: {ts_path}", file=sys.stderr)
                return 2
            counts = count_timeseries_severity(ts_path)
            print(f"timeseries check: {ts_path} (mtime {mtime_iso(ts_path)})")
            print(f"  lines={counts.pop('_lines')} json_errors={counts.pop('_json_errors')} "
                  f"severities={counts}")
            grid_crit = gstats['n_critical']
            ts_crit = counts.get(args.severity.upper(), 0)
            print(f"  agreement: grid {args.severity}={grid_crit} vs "
                  f"timeseries {args.severity}={ts_crit} "
                  f"({'MATCH' if grid_crit == ts_crit else 'MISMATCH — investigate before review'}).")
    else:
        if not jsonl_path.is_file():
            print(f"ERROR: heuristic JSONL not found: {jsonl_path}", file=sys.stderr)
            print("Wait for the harvest to finish and pass the fresh file via --jsonl.", file=sys.stderr)
            return 2
        critical, jstats = load_heuristic_critical(jsonl_path, args.min_score)
        print(f"heuristic input : {jsonl_path} (mtime {mtime_iso(jsonl_path)})")
        print(f"  lines={jstats['n_lines']} critical={jstats['n_critical']} classes={jstats['classes']}")

    offline_info, sstats = load_snapshot_offline(snapshots_dir)
    recoveries = load_recoveries(rec_path)
    print(f"snapshots dir   : {snapshots_dir}/ files={sstats['n_files']} "
          f"rows={sstats['n_rows']} offline_rows={sstats['n_offline_rows']}")
    print(f"recoveries      : {rec_path} ({mtime_iso(rec_path)}, "
          f"{sum(len(v) for v in recoveries.values())} rows)")
    if not sstats["n_files"]:
        print("NOTE: no current_state_*.csv yet — every row will be "
              "'no-snapshot-coverage'. Re-run after the nightly snapshot lands.")
    print("NOTE: stale-file guard — do NOT use the Sep-21 11:01 kernel "
          "re-export as Phase-0 input; take the freshly-written post-harvest files.")

    fieldnames = ["device_id", "date", "bank", "branch", "device_name",
                  "is_real_device",
                  "heuristic_class", "heuristic_score",
                  "heuristic_reasons", "n_snapshots", "n_offline_snapshots",
                  "last_snapshot_utc", "last_offline_utc", "n_recovery_rows",
                  "coverage", "human_verdict", "input_snippet"]
    rows = []
    for c in critical:
        dev = c["device_id"]
        snap = offline_info.get(dev, {})
        recs = recoveries.get(dev, [])
        coverage = ("no-snapshot-coverage" if not sstats["n_files"]
                    else ("offline-seen" if snap.get("n_offline") else "no-offline-seen"))
        rows.append({
            "device_id": dev,
            "date": c.get("date", ""),
            "bank": c["bank"],
            "branch": c.get("branch", ""),
            "device_name": c.get("device_name", ""),
            "is_real_device": c.get("is_real_device", ""),
            "heuristic_class": c["heuristic_class"],
            "heuristic_score": c["heuristic_score"],
            "heuristic_reasons": c["heuristic_reasons"],
            "n_snapshots": snap.get("n_snapshots", 0),
            "n_offline_snapshots": snap.get("n_offline", 0),
            "last_snapshot_utc": snap.get("last_snapshot", ""),
            "last_offline_utc": snap.get("last_offline", ""),
            "n_recovery_rows": len(recs),
            "coverage": coverage,
            "human_verdict": "",
            "input_snippet": c["input_snippet"],
        })

    # Full pack sorted for navigability (bank, month, score desc) so no
    # single-bank pile dominates the top of the file.
    rows.sort(key=lambda r: (r["bank"], r.get("date", "")[:7],
                             -float(r["heuristic_score"] or 0.0),
                             r["device_id"], r.get("date", "")))

    test_rows = [r for r in rows
                 if str(r.get("is_real_device", "")).strip().lower() in {"", "0", "false", "no"}]
    print(f"test-device share: {len(test_rows)}/{len(rows)} CRITICAL rows are "
          f"non-real devices (filter is_real_device=FALSE to dismiss in one shot).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} review rows -> {out_path}")

    if args.sample_n and args.sample_n > 0:
        strata: dict = {}
        for r in rows:
            strata.setdefault((r["bank"], r.get("date", "")[:7]), []).append(r)
        for members in strata.values():
            members.sort(key=lambda r: (-float(r["heuristic_score"] or 0.0),
                                        r["device_id"], r.get("date", "")))
        sample: list[dict] = []
        buckets = sorted(strata.values(), key=lambda b: (-len(b), b[0]["bank"]))
        i = 0
        while len(sample) < args.sample_n and any(len(b) > i for b in buckets):
            for b in buckets:
                if len(sample) >= args.sample_n:
                    break
                if len(b) > i:
                    sample.append(b[i])
            i += 1
        sample_path = out_path.parent / "phase0_review_sample.csv"
        with sample_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(sample)
        by_bank: dict = {}
        for r in sample:
            by_bank[r["bank"]] = by_bank.get(r["bank"], 0) + 1
        print(f"wrote {len(sample)} stratified sample rows -> {sample_path}")
        print(f"  strata={len(strata)} bank-mix={by_bank}")
    print("Next: lead + Seple ops fill human_verdict "
          "(true-fault / false-alarm / inconclusive) per row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
