#!/usr/bin/env python
"""Nightly risk chain: fresh telemetry -> features -> production score -> report.

Why a chain: score_daily_risk.py needs a CURRENT features parquet. Nothing
produced one — the initial run scored the frozen 2026-10-05 pull (anchors
from March/August). This chain closes the loop nightly:

  1. pull_all_data.py     10-day mini-pull, full fleet (telemetry only)
  2. build_anomaly_dataset features from the fresh run (v2 feature set)
  3. score_daily_risk.py  production model -> audit_reports CSV
  4. NO-DATA gap list: eligible devices with zero fresh telemetry are appended
     as tier NO_DATA (presumed offline) — the sensor-blind fleet must not
     silently vanish from the nightly report.
  5. Prune: chain outputs older than 3 days are deleted (disk hygiene).

Exit codes: 0 = report written (partial pulls tolerated), 1 = step failure.
Label semantics note: the model predicts a device-level ALARM EVENT within
24h (any non-camera alarm), not a verified power/DVR outage. See
training_data/cv_oct_full/ops_card.json before changing thresholds.

Usage:
    python nightly_risk_chain.py                # full fleet
    python nightly_risk_chain.py --max-devices 6   # smoke test
"""
import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PRUNE_DAYS = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("nightly_risk_chain")


def run(cmd: list[str], accept: tuple[int, ...] = (0,)) -> int:
    LOG.info(">> %s", " ".join(str(c) for c in cmd))
    rc = subprocess.call([str(c) for c in cmd])
    if rc not in accept:
        LOG.error("step failed rc=%s (accepted: %s)", rc, accept)
        return rc
    return 0


def prune(pattern_dir: Path, prefix: str) -> None:
    cutoff = datetime.now().timestamp() - PRUNE_DAYS * 86400
    if not pattern_dir.is_dir():
        return
    for p in sorted(pattern_dir.glob(f"{prefix}*")):
        try:
            if p.stat().st_mtime < cutoff:
                import shutil
                shutil.rmtree(p, ignore_errors=True)
                LOG.info("pruned %s", p)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-devices", type=int, default=0, help="Smoke-test device cap (0 = full fleet)")
    ap.add_argument("--telemetry-days", type=int, default=10)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"nightly_risk_{stamp}"
    run_dir = ROOT / "data_harvest" / "runs" / run_id
    feat_dir = ROOT / "training_data" / "nightly" / run_id
    report = ROOT / "audit_reports" / f"nightly_alarm_risk_{stamp}.csv"

    pull_cmd = [PY, "-X", "utf8", ROOT / "pull_all_data.py",
                "--insecure-skip-tls-verify",
                "--telemetry-days", str(args.telemetry_days),
                "--skip-events", "--skip-alarms",
                "--run-id", run_id]
    if args.max_devices:
        pull_cmd += ["--max-devices", str(args.max_devices)]

    # 1) fresh pull — exit 5 (incomplete windows) is tolerated: the coverage
    #    ledger gates the feature build, and silent devices become NO_DATA.
    rc = run(pull_cmd, accept=(0, 5))
    if rc:
        return rc

    # 2) features
    rc = run([PY, "-X", "utf8", ROOT / "build_anomaly_dataset.py",
              "--run-dir", run_dir, "--out-dir", feat_dir])
    if rc:
        return rc
    feats = feat_dir / "anomaly_features.parquet"
    if not feats.is_file():
        LOG.error("features parquet missing after build")
        return 1

    # 3) score
    rc = run([PY, "-X", "utf8", ROOT / "score_daily_risk.py",
              "--input", feats, "--out", report])
    if rc:
        return rc

    # 4) merge device names + NO_DATA gap list
    import pandas as pd
    scored = pd.read_csv(report, dtype={"device_id": str})
    reg_path = run_dir / "registry" / "devices.csv"
    if reg_path.is_file():
        reg = pd.read_csv(reg_path, dtype=str, keep_default_na=False)
        eligible = reg[reg["eligible"].str.lower().isin(("true", "1"))]
        names = dict(zip(reg["device_id"], reg["name"]))
        scored["device_name"] = scored["device_id"].map(names)
        missing = eligible[~eligible["device_id"].isin(set(scored["device_id"]))]
        gap = pd.DataFrame({
            "device_id": missing["device_id"],
            "device_name": missing["name"],
            "risk_tier": "NO_DATA",
            "note": f"no telemetry in the last {args.telemetry_days} days - presumed offline/sensor-blind",
        })
        out = pd.concat([scored, gap], ignore_index=True)
        prob_col = "alarm_event_prob" if "alarm_event_prob" in out.columns else "risk_score"
        out = out.sort_values(prob_col, ascending=False, na_position="last")
        out.to_csv(report, index=False)
        LOG.info("report: %d scored + %d NO_DATA -> %s", len(scored), len(gap), report)
    else:
        LOG.warning("no registry in run dir; skipped NO_DATA merge")

    # 5) prune old chain artifacts
    prune(ROOT / "data_harvest" / "runs", "nightly_risk_")
    prune(ROOT / "training_data" / "nightly", "nightly_risk_")
    LOG.info("chain complete -> %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
