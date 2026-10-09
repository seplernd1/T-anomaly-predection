#!/usr/bin/env python
"""Train and serialize the production HistGradientBoosting ALARM-EVENT risk model.

Labels: y=1 means a non-camera device-level alarm event (any type) fires within
24h of the anchor — NOT a verified power/DVR outage (see metadata label_semantics).

Trains on all measured samples from the October dataset (37,892 samples,
5,115 verified positive outages) using the frozen, validated hyperparameters.

Outputs to models/gbm_outage_oct2026/:
  - model.joblib          Serialized HistGradientBoostingClassifier
  - feature_names.json    List of active feature columns
  - metadata.json         Training stats, timestamps, and operating thresholds
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parent
DEFAULT_FEATS = ROOT / "training_data/anomaly_20261005_041300_v2/anomaly_features.parquet"
DEFAULT_LABELS = ROOT / "training_data/supervised_outage_oct2026.parquet"
DEFAULT_OUT = ROOT / "models/gbm_outage_oct2026"

META_COLS = {
    "device_id", "anchor_ts", "split", "group_key", "censor_reason",
    "label_status", "y_outage_24h", "y", "max_source_ts", "n_rows_used",
    "n_rows_ignored_future", "n_feature_values"
}

OPERATING_THRESHOLDS = {
    "max_recall": {"threshold": 0.20, "description": "83% recall, 25% precision (~46 alerts/day)"},
    "recall_first": {"threshold": 0.40, "description": "72% recall, 29% precision (~35 alerts/day) [RECOMMENDED]"},
    "balanced": {"threshold": 0.50, "description": "67% recall, 30% precision (~31 alerts/day)"},
    "high_precision": {"threshold": 0.65, "description": "58% recall, 31% precision (~26 alerts/day)"},
}


def make_production_clf():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=7,
    )


def train_and_save(feats_path: Path, labels_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Loading features from {feats_path}...")
    fr = pd.read_parquet(feats_path)
    logging.info(f"Loading labels from {labels_path}...")
    lb = pd.read_parquet(labels_path)

    lb["anchor_ts"] = pd.to_datetime(lb["anchor_ts"], utc=True)
    fr["anchor_ts"] = pd.to_datetime(fr["anchor_ts"], utc=True)

    m = fr.merge(
        lb[["device_id", "anchor_ts", "y_outage_24h", "label_status"]],
        on=["device_id", "anchor_ts"],
        how="inner",
    )
    m = m[m["label_status"].isin(["measured", "measured_negative"])].reset_index(drop=True)
    m["y"] = (m["y_outage_24h"] == 1).astype(int)

    feat_cols = [c for c in m.columns if c not in META_COLS]
    nu = m[feat_cols].nunique(dropna=True)
    active_feats = [c for c in feat_cols if nu.get(c, 0) > 1]

    n_samples = len(m)
    n_pos = int(m["y"].sum())
    n_neg = n_samples - n_pos
    logging.info(
        f"Training set: {n_samples} samples ({n_pos} positive outages, {n_neg} negatives, "
        f"{m['device_id'].nunique()} devices) across {len(active_feats)} features."
    )

    clf = make_production_clf()
    X = m[active_feats].to_numpy(dtype=np.float64)
    y = m["y"].to_numpy(dtype=int)

    logging.info("Fitting production HistGradientBoostingClassifier...")
    clf.fit(X, y)
    logging.info("Model fitted successfully.")

    # Save artifacts
    model_file = out_dir / "model.joblib"
    joblib.dump(clf, model_file)
    logging.info(f"Saved model weights to {model_file}")

    feat_file = out_dir / "feature_names.json"
    with open(feat_file, "w") as f:
        json.dump(active_feats, f, indent=2)
    logging.info(f"Saved feature list ({len(active_feats)} features) to {feat_file}")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "HistGradientBoostingClassifier",
        "label_semantics": (
            "y = 1 when a non-camera device-level ALARM EVENT (any type: DVR/NVR OFF, "
            "FIRE ALARM ACTIVATE, BATTERY REVERSE, INTRUSION, TIME LOCK, ...) starts "
            "within 24h of the anchor. This is NOT a verified power/DVR outage label; "
            "threshold metrics describe alarm-event alerting, not outage prediction."
        ),
        "training_samples": n_samples,
        "positive_samples": n_pos,
        "negative_samples": n_neg,
        "positive_prevalence": round(n_pos / n_samples, 4),
        "unique_devices": int(m["device_id"].nunique()),
        "num_features": len(active_feats),
        "features_path": str(feats_path),
        "labels_path": str(labels_path),
        "hyperparameters": {
            "max_iter": 300,
            "learning_rate": 0.06,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
            "random_state": 7,
        },
        "operating_thresholds": OPERATING_THRESHOLDS,
    }
    meta_file = out_dir / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata to {meta_file}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default=str(DEFAULT_FEATS), type=Path)
    p.add_argument("--labels", default=str(DEFAULT_LABELS), type=Path)
    p.add_argument("--out", default=str(DEFAULT_OUT), type=Path)
    args = p.parse_args()
    train_and_save(args.features, args.labels, args.out)


if __name__ == "__main__":
    main()
