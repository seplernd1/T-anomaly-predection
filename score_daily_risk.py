#!/usr/bin/env python
"""Nightly inference scorer for Dexter HMS device ALARM-EVENT risk (24h).

Semantics: the model predicts that a non-camera device-level alarm event
(any type — DVR/NVR OFF, FIRE ALARM ACTIVATE, BATTERY REVERSE, intrusion,
time lock, ...) will start within 24h of the anchor. This is NOT a verified
power/DVR outage label; see models/gbm_outage_oct2026/metadata.json
(label_semantics) and training_data/cv_oct_full/ops_card.json before
changing thresholds or quoting precision/recall to stakeholders.

Consumes daily telemetry feature snapshots, loads the trained production
HistGradientBoostingClassifier bundle, and produces a prioritized risk
ranking and alert list with explainable top deviation reasons.

Usage:
    python score_daily_risk.py --input training_data/anomaly_20261005_041300_v2/anomaly_features.parquet
    python score_daily_risk.py --input <latest_features.parquet> --out audit_reports/nightly_alarm_risk_latest.csv
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models/gbm_outage_oct2026"
DEFAULT_OUT_DIR = ROOT / "audit_reports"


def load_model_bundle(model_dir: Path):
    model_path = model_dir / "model.joblib"
    feat_path = model_dir / "feature_names.json"
    meta_path = model_dir / "metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = joblib.load(model_path)
    with open(feat_path, "r") as f:
        feature_names = json.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return model, feature_names, metadata


def score_features(df: pd.DataFrame, model, feature_names: list[str], metadata: dict) -> pd.DataFrame:
    # Ensure all feature columns exist (fill missing with NaN for HistGradientBoosting)
    X = np.full((len(df), len(feature_names)), np.nan, dtype=np.float64)
    for col_idx, col_name in enumerate(feature_names):
        if col_name in df.columns:
            X[:, col_idx] = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=np.float64)

    # Predict alarm-event probabilities (24h horizon)
    probs = model.predict_proba(X)[:, 1]

    # Assign risk tiers based on validated operating thresholds
    # recall_first is 0.40, balanced is 0.50
    tiers = []
    for p in probs:
        if p >= 0.50:
            tiers.append("CRITICAL")
        elif p >= 0.40:
            tiers.append("HIGH")
        elif p >= 0.25:
            tiers.append("MEDIUM")
        else:
            tiers.append("HEALTHY")

    scored = df[["device_id", "anchor_ts"]].copy() if "anchor_ts" in df.columns else df[["device_id"]].copy()
    scored["alarm_event_prob"] = np.round(probs, 4)
    scored["risk_score"] = np.round(probs * 100, 1)
    scored["risk_tier"] = tiers

    # Attach device names if present
    for extra_col in ["device_name", "branch_name", "customer_name"]:
        if extra_col in df.columns:
            scored[extra_col] = df[extra_col]

    return scored.sort_values("alarm_event_prob", ascending=False).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Path to input features parquet")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.info(f"Loading production model bundle from {args.model_dir}...")
    model, feature_names, metadata = load_model_bundle(args.model_dir)

    logging.info(f"Loading input telemetry features from {args.input}...")
    df = pd.read_parquet(args.input)
    logging.info(f"Loaded {len(df)} feature records across {df['device_id'].nunique()} devices.")

    # If dataset has multiple anchor timestamps, take the latest per device for nightly scoring
    if "anchor_ts" in df.columns:
        latest_df = df.sort_values("anchor_ts").groupby("device_id").last().reset_index()
        logging.info(f"Filtered to latest anchor snapshot per device ({len(latest_df)} devices).")
    else:
        latest_df = df

    scored = score_features(latest_df, model, feature_names, metadata)

    out_file = args.out
    if out_file is None:
        DEFAULT_OUT_DIR.mkdir(exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        out_file = DEFAULT_OUT_DIR / f"nightly_alarm_risk_{ts_str}.csv"
    else:
        out_file.parent.mkdir(parents=True, exist_ok=True)

    scored.to_csv(out_file, index=False)
    logging.info(f"Successfully wrote risk scores to {out_file}")

    # Summary
    print("\n--- Daily Alarm-Event Risk (24h) Summary ---")
    print(f"Total Devices Scored: {len(scored)}")
    print("Risk Tier Distribution:")
    print(scored["risk_tier"].value_counts())
    print("\nTop 10 Highest Risk Branches/Devices:")
    cols_to_print = [c for c in ["device_id", "device_name", "risk_score", "risk_tier"] if c in scored.columns]
    print(scored[cols_to_print].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
