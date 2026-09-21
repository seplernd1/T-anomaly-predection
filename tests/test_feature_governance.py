"""Governance tests: exclusion, leakage, splits, ackTs, coverage censoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from build_anomaly_dataset import (
    apply_scaler,
    assign_chrono_splits,
    build_features,
    classify_keys,
    fit_scaler,
    guard_output_columns,
    load_policy,
    load_usable_keys,
)
from build_training_dataset import (
    R_HORIZON_INCOMPLETE,
    R_INSIDE_OUTAGE,
    R_NO_VERIFIED_SOURCE,
    TrainingDataError,
    assert_no_prohibited_features,
    build_evidence_frame,
    build_outage_windows,
    label_samples,
    load_config,
)
from pull_outage_labels import build_alarm_labels

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "training_config.json")
POLICY = load_policy(ROOT / "feature_policy.json")


# ---------------------------------------------------------------------------
# 12: partial/failed windows cannot enter anomaly datasets
# ---------------------------------------------------------------------------
def test_partial_windows_excluded_from_usable(tmp_path):
    run = tmp_path / "runs" / "r1"
    cov = run / "coverage"
    cov.mkdir(parents=True)
    recs = [
        {"run_id": "r1", "device_id": "d1", "key": "k1", "endpoint": "e",
         "requested_start_ms": 0, "requested_end_ms": 2, "chunk_start_ms": 0, "chunk_end_ms": 2,
         "completeness": "complete", "rows": 5},
        {"run_id": "r1", "device_id": "d1", "key": "k2", "endpoint": "e",
         "requested_start_ms": 0, "requested_end_ms": 2, "chunk_start_ms": 0, "chunk_end_ms": 2,
         "completeness": "partial", "rows": 5, "error": "capped"},
        {"run_id": "r1", "device_id": "d2", "key": "k1", "endpoint": "e",
         "requested_start_ms": 0, "requested_end_ms": 2, "chunk_start_ms": 0, "chunk_end_ms": 2,
         "completeness": "failed", "rows": 0, "error": "boom"},
    ]
    with (cov / "telemetry_coverage.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    usable, excluded, stats = load_usable_keys(run)
    assert usable == {("d1", "k1")}
    assert {(e["device_id"], e["key"]) for e in excluded} == {("d1", "k2"), ("d2", "k1")}


# ---------------------------------------------------------------------------
# 13/14: sensitive + rule-derived fields blocked, fail-closed
# ---------------------------------------------------------------------------
def test_denylist_blocks_sensitive_and_rule_derived():
    allowed, excluded = classify_keys(
        ["temperature", "imei", "fault_score", "ts_severity", "heartbeat", "latitude"],
        POLICY,
    )
    assert allowed == []
    reasons = {e["key"]: e["reason"] for e in excluded}
    assert reasons["imei"].startswith("denylist:")
    assert reasons["fault_score"].startswith("denylist:")
    assert reasons["heartbeat"].startswith("denylist:")
    assert reasons["temperature"] == "unreviewed:not_in_allowlist"


def test_output_guard_raises():
    with pytest.raises(Exception):
        guard_output_columns(["temp__mean_1h", "fault_score__mean_1h"], POLICY)
    guard_output_columns(["temp__mean_1h"], POLICY)  # no raise


def test_supervised_builder_output_guard():
    frame = pd.DataFrame({"device_id": ["d"], "anchor_ts": [pd.Timestamp("2026-01-01", tz="UTC")],
                          "temp__mean_1h": [1.0], "ts_severity__x": [0.0]})
    with pytest.raises(TrainingDataError):
        assert_no_prohibited_features(frame, CFG)
    ok = frame.drop(columns=["ts_severity__x"])
    assert_no_prohibited_features(ok, CFG)


# ---------------------------------------------------------------------------
# 15: future telemetry cannot enter a feature row
# ---------------------------------------------------------------------------
def test_future_points_ignored_and_max_source_bounded():
    rows = [{"device_id": "d", "key": "temp", "ts_ms": h * 3600_000, "value_num": float(h)}
            for h in range(0, 21)]
    rows.append({"device_id": "d", "key": "temp", "ts_ms": 100 * 3600_000, "value_num": 999.0})
    df = pd.DataFrame(rows)
    frame, _ = build_features(df, ["temp"], anchors_hours=6, windows_hours=(1, 6))
    assert not frame.empty
    assert (frame["max_source_ts"] <= frame["anchor_ts"]).all()
    early = frame[frame["anchor_ts"] <= pd.Timestamp(20 * 3600_000, unit="ms", tz="UTC")]
    assert (early["temp__last"] != 999.0).all()


# ---------------------------------------------------------------------------
# 16/18: schema/scaler from training data only
# ---------------------------------------------------------------------------
def test_scaler_fit_on_train_only():
    frame = pd.DataFrame({
        "device_id": ["d"] * 6, "split": ["train"] * 3 + ["test"] * 3,
        "anchor_ts": pd.date_range("2026-01-01", periods=6, tz="UTC"),
        "f": [0.0, 1.0, 2.0, 100.0, 101.0, 102.0],
    })
    scaler = fit_scaler(frame, ["f"])
    assert scaler["f"]["mean"] == pytest.approx(1.0)
    out = apply_scaler(frame, ["f"], scaler)
    assert out.loc[out["split"] == "train", "f"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out.loc[out["split"] == "test", "f"].iloc[0] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# 17: chronological per-device splits
# ---------------------------------------------------------------------------
def test_chrono_splits_group_pure_and_ordered():
    base = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for i, (dev, day) in enumerate([("d1", 1)] * 70 + [("d2", 40)] * 15 + [("d3", 80)] * 15):
        rows.append({"device_id": dev, "anchor_ts": base + pd.Timedelta(days=day),
                     "f": float(i)})
    frame = pd.DataFrame(rows)
    out = assign_chrono_splits(frame, {}, purge_hours=1.0)
    assert out.groupby("group_key")["split"].nunique().max() == 1
    med = out.groupby("split")["anchor_ts"].median()
    assert med["train"] < med["validation"] < med["test"]
    assert set(out[out["group_key"].str.endswith("d1")]["split"]) == {"train"}


# ---------------------------------------------------------------------------
# 19: ackTs never closes an outage
# ---------------------------------------------------------------------------
def test_ack_alone_leaves_outage_open():
    device = {"id": "d1", "name": "n"}
    ack_only = [{"startTs": 1000, "ackTs": 2000, "type": "offline"}]
    labels = build_alarm_labels(device, ack_only)
    assert len(labels) == 1
    assert labels[0].offline_end in ("", None)
    closed = [{"startTs": 1000, "endTs": 3000, "ackTs": 2000, "type": "offline"}]
    labels2 = build_alarm_labels(device, closed)
    assert labels2[0].offline_end not in ("", None)


# ---------------------------------------------------------------------------
# 20/21: coverage censoring + stale history
# ---------------------------------------------------------------------------
def _samples(anchors):
    return pd.DataFrame([{"device_id": "d1", "anchor_ts": pd.Timestamp(a, tz="UTC")} for a in anchors])


def test_no_verified_source_censors_everything():
    out = label_samples(_samples(["2026-01-15 12:00"]), pd.DataFrame(), pd.DataFrame(),
                        pd.Timestamp("2026-02-01", tz="UTC"), CFG)
    assert (out["label_status"] == "censored").all()
    assert (out["censor_reason"] == R_NO_VERIFIED_SOURCE).all()
    assert out["y_outage_24h"].isna().all()


def test_stale_history_blocks_future_claims():
    ev = pd.DataFrame([
        {"device_id": "d1", "event_type": "LC_EVENT", "time": pd.Timestamp("2026-01-10 01:00", tz="UTC"),
         "event_id": "e1", "payload": {"method": "onDisconnect"}},
        {"device_id": "d1", "event_type": "LC_EVENT", "time": pd.Timestamp("2026-01-10 03:00", tz="UTC"),
         "event_id": "e2", "payload": {"method": "onConnect"}},
    ])
    evidence = build_evidence_frame(ev, CFG)
    windows = build_outage_windows(evidence, CFG)
    out = label_samples(_samples(["2026-01-31 12:00"]), windows, evidence,
                        pd.Timestamp("2026-02-01", tz="UTC"), CFG)
    assert out.iloc[0]["label_status"] == "censored"
    assert out.iloc[0]["censor_reason"] == R_HORIZON_INCOMPLETE


# ---------------------------------------------------------------------------
# 22: alarm/lifecycle evidence actually consumed by the label builder
# ---------------------------------------------------------------------------
def test_lifecycle_pair_produces_positive_label():
    ev = pd.DataFrame([
        {"device_id": "d1", "event_type": "LC_EVENT", "time": pd.Timestamp("2026-01-10 01:00", tz="UTC"),
         "event_id": "e1", "payload": {"method": "onDisconnect"}},
        {"device_id": "d1", "event_type": "LC_EVENT", "time": pd.Timestamp("2026-01-10 03:00", tz="UTC"),
         "event_id": "e2", "payload": {"method": "onConnect"}},
    ])
    evidence = build_evidence_frame(ev, CFG)
    assert set(evidence["evidence_type"]) == {"lc_disconnect", "lc_connect"}
    assert (evidence["evidence_strength"] == "strong").all()
    # Raw values are never stored; only the hashed reference column exists.
    assert "redacted_evidence_reference" in evidence.columns
    assert "payload" not in evidence.columns
    windows = build_outage_windows(evidence, CFG)
    assert len(windows) == 1 and bool(windows.iloc[0]["window_closed"])
    out = label_samples(_samples(["2026-01-09 12:00", "2026-01-10 02:00"]), windows, evidence,
                        pd.Timestamp("2026-02-01", tz="UTC"), CFG)
    pre = out[out["anchor_ts"] == pd.Timestamp("2026-01-09 12:00", tz="UTC")].iloc[0]
    assert pre["y_outage_24h"] == 1 and pre["label_status"] == "measured"
    inside = out[out["anchor_ts"] == pd.Timestamp("2026-01-10 02:00", tz="UTC")].iloc[0]
    assert inside["label_status"] == "censored" and inside["censor_reason"] == R_INSIDE_OUTAGE
