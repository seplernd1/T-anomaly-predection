"""Tests for the alarm-evidence bridge and label-source coverage gating.

Covers the final wiring required by the pipeline contract:
* alarm CSV -> device_event rows (Part C: alarms actually consumed by the
  label builder, ackTs never closes an outage),
* camera-channel alarms are weak per-channel evidence, never device outages,
* per-device label-source coverage gating (Part E): missing/failed/stale
  source coverage censors instead of manufacturing negatives.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alarms_bridge import (
    BRIDGE_MARKER,
    alarms_frame_to_event_rows,
    label_source_coverage_from_per_device,
)
from build_training_dataset import (
    R_NO_SOURCE_COVERAGE,
    R_STALE_SOURCE,
    STRONG,
    WEAK,
    build_evidence_frame,
    build_outage_windows,
    label_samples,
    load_config,
    verified_windows,
)

CFG = load_config(None)


def _alarm_row(**over) -> dict:
    row = {
        "device_id": "dev-1",
        "device_name": "BRANCH-1",
        "alarm_type": "DVR/NVR OFF",
        "severity": "CRITICAL",
        "status": "CLEARED_UNACK",
        "start": "2026-03-01T10:00:00+00:00",
        "end": "2026-03-01T12:00:00+00:00",
        "ack": "2026-03-01T11:00:00+00:00",
        "clear": "2026-03-01T12:00:00+00:00",
        "payload": "{}",
    }
    row.update(over)
    return row


def _alarms_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bridge: structure and ack exclusion
# ---------------------------------------------------------------------------


def test_bridge_emits_two_rows_per_cleared_alarm():
    rows = alarms_frame_to_event_rows(_alarms_frame([_alarm_row()]))
    assert len(rows) == 2
    kinds = [r["payload"]["state"] for r in rows.to_dict("records")]
    assert kinds == ["ALARM ACTIVATED", "ALARM CLEARED"]
    assert rows.iloc[0]["event_type"] == "ALARM"
    # offline row at startTs, cleared row at clearTs
    assert rows.iloc[0]["time"] == "2026-03-01T10:00:00+00:00"
    assert rows.iloc[1]["time"] == "2026-03-01T12:00:00+00:00"


def test_bridge_ack_never_appears_in_payload():
    rows = alarms_frame_to_event_rows(_alarms_frame([_alarm_row()]))
    for rec in rows.to_dict("records"):
        assert "ack" not in rec["payload"]
        assert "ackTs" not in rec["payload"]
        # the clear row is driven by clearTs only
    cleared = rows.iloc[1]["payload"]
    assert cleared["clearTs"] == "2026-03-01T12:00:00+00:00"


def test_bridge_uncleared_alarm_yields_only_offline_row():
    rows = alarms_frame_to_event_rows(_alarms_frame([_alarm_row(clear="")]))
    assert len(rows) == 1
    assert rows.iloc[0]["payload"]["state"] == "ALARM ACTIVATED"
    assert rows.iloc[0]["payload"]["clearTs"] is None


def test_bridge_ms_epoch_timestamps_converted_to_iso():
    rows = alarms_frame_to_event_rows(
        _alarms_frame([_alarm_row(start=1772352000000, clear=1772359200000)])
    )
    assert rows.iloc[0]["time"].startswith("2026-")
    assert "T" in rows.iloc[0]["time"]


# ---------------------------------------------------------------------------
# Bridge -> classifier: camera-channel vs whole-device semantics
# ---------------------------------------------------------------------------


def _classify(alarm_rows: pd.DataFrame):
    evidence = build_evidence_frame(alarm_rows, CFG)
    return evidence


def test_camera_channel_alarm_is_weak_and_never_closes_outages():
    frame = _alarms_frame(
        [
            _alarm_row(alarm_type="CAMERA TAMPER CH 9", status="CLEARED_UNACK"),
            _alarm_row(alarm_type="CAMERA DISCONNECT CH 2", status="CLEARED_UNACK"),
        ]
    )
    evidence = _classify(alarms_frame_to_event_rows(frame))
    assert not evidence.empty
    assert (evidence["evidence_type"] == "alarm_channel").all()
    assert (evidence["evidence_strength"] == WEAK).all()
    # weak evidence can never build a verified outage window
    windows = build_outage_windows(evidence, CFG)
    assert windows.empty


def test_device_outage_alarm_pair_produces_verified_window():
    frame = _alarms_frame([_alarm_row()])
    evidence = _classify(alarms_frame_to_event_rows(frame))
    offline = evidence[evidence["evidence_type"] == "alarm_offline"]
    cleared = evidence[evidence["evidence_type"] == "alarm_cleared"]
    assert len(offline) == 1 and len(cleared) == 1
    assert (evidence["evidence_strength"] == STRONG).all()
    windows = verified_windows(build_outage_windows(evidence, CFG), CFG)
    assert len(windows) == 1
    w = windows.iloc[0]
    assert w["window_closed"] is True or bool(w["window_closed"])
    assert w["outage_end"] == pd.Timestamp("2026-03-01T12:00:00+00:00")


def test_device_off_alarm_matches_dedicated_alarm_regex():
    # "DVR/NVR OFF" contains neither "offline" nor "down" — the bridge's
    # controlled ACTIVATED state must classify the startTs row as offline
    # (the dedicated alarm semantics path), not as cleared.
    evidence = _classify(alarms_frame_to_event_rows(_alarms_frame([_alarm_row()])))
    offline = evidence[evidence["evidence_type"] == "alarm_offline"]
    cleared = evidence[evidence["evidence_type"] == "alarm_cleared"]
    assert len(offline) == 1 and len(cleared) == 1
    # the offline row sits at startTs (the activation), not at clearTs
    assert offline.iloc[0]["evidence_ts"] == pd.Timestamp("2026-03-01T10:00:00+00:00")


# ---------------------------------------------------------------------------
# Label-source coverage ledger
# ---------------------------------------------------------------------------


def test_coverage_ledger_marks_failed_queries():
    per_device = {
        "dev-ok": {"rows": 3, "oldest": "2026-01-01T00:00:00+00:00", "newest": "2026-03-01T00:00:00+00:00", "error": ""},
        "dev-bad": {"rows": 0, "oldest": "", "newest": "", "error": "HTTP 500 after retries"},
    }
    cov = label_source_coverage_from_per_device(per_device, 0, 1)
    by_dev = cov.set_index("device_id")["status"].to_dict()
    assert by_dev["dev-ok"] == "ok"
    assert by_dev["dev-bad"] == "failed"


# ---------------------------------------------------------------------------
# Per-device coverage gating in label_samples
# ---------------------------------------------------------------------------


def _samples_for(devices: list[str], anchors: list[str]) -> pd.DataFrame:
    rows = []
    for dev, a in zip(devices, anchors):
        rows.append(
            {
                "device_id": dev,
                "anchor_ts": pd.Timestamp(a),
                "split": "test",
                "censor_reason": None,
            }
        )
    return pd.DataFrame(rows)


def _verified_window_for(dev: str, start: str, end: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "device_id": dev,
                "outage_start": pd.Timestamp(start),
                "outage_end": pd.Timestamp(end),
                "duration_minutes": 120.0,
                "start_evidence_type": "alarm_offline",
                "end_evidence_type": "alarm_cleared",
                "window_closed": True,
                "start_event_id": "s",
                "end_event_id": "e",
            }
        ]
    )


def _device_window_frame(dev: str, start: str, end: str, closed: bool = True) -> pd.DataFrame:
    return _verified_window_for(dev, start, end).assign(window_closed=closed)


def _full_evidence(dev: str) -> pd.DataFrame:
    return build_evidence_frame(
        alarms_frame_to_event_rows(_alarms_frame([_alarm_row(device_id=dev)])), CFG
    )


def _coverage_frame(records: list[dict]) -> dict[str, pd.DataFrame]:
    return {"alarms": pd.DataFrame(records)}


def test_missing_coverage_censors_instead_of_negative():
    # device with verified history but NO entry in the coverage ledger
    # (the ledger exists — it covers another device — so the gate is active)
    samples = _samples_for(["dev-1"], ["2026-03-05T00:00:00Z"])
    windows = _device_window_frame("dev-1", "2026-03-01T10:00:00Z", "2026-03-01T12:00:00Z")
    evidence = _full_evidence("dev-1")
    cov = _coverage_frame([{"device_id": "dev-other", "status": "ok", "data_newest": "2026-03-09T00:00:00Z"}])
    out = label_samples(
        samples, windows, evidence, pd.Timestamp("2026-03-10T00:00:00Z"), CFG,
        source_coverage=cov,
    )
    row = out.iloc[0]
    assert row["label_status"] == "censored"
    assert row["censor_reason"] == R_NO_SOURCE_COVERAGE
    assert pd.isna(row["y_outage_24h"])


def test_failed_query_coverage_censors_instead_of_negative():
    samples = _samples_for(["dev-1"], ["2026-03-05T00:00:00Z"])
    windows = _device_window_frame("dev-1", "2026-03-01T10:00:00Z", "2026-03-01T12:00:00Z")
    evidence = _full_evidence("dev-1")
    cov = _coverage_frame([{"device_id": "dev-1", "status": "failed", "data_newest": None}])
    out = label_samples(
        samples, windows, evidence, pd.Timestamp("2026-03-10T00:00:00Z"), CFG,
        source_coverage=cov,
    )
    assert out.iloc[0]["censor_reason"] == R_NO_SOURCE_COVERAGE


def test_stale_source_data_censors_recent_anchors():
    # alarm data stream stopped in March; anchors sit in September
    samples = _samples_for(["dev-1"], ["2026-09-10T00:00:00Z"])
    windows = _device_window_frame("dev-1", "2026-03-01T10:00:00Z", "2026-03-01T12:00:00Z")
    evidence = _full_evidence("dev-1")
    cov = _coverage_frame(
        [{"device_id": "dev-1", "status": "ok", "data_newest": "2026-03-30T00:00:00Z"}]
    )
    out = label_samples(
        samples, windows, evidence, pd.Timestamp("2026-09-20T00:00:00Z"), CFG,
        source_coverage=cov,
    )
    assert out.iloc[0]["label_status"] == "censored"
    assert out.iloc[0]["censor_reason"] == R_STALE_SOURCE


def test_good_coverage_allows_negative_label():
    samples = _samples_for(["dev-1"], ["2026-03-05T00:00:00Z"])
    windows = _device_window_frame("dev-1", "2026-03-01T10:00:00Z", "2026-03-01T12:00:00Z")
    evidence = _full_evidence("dev-1")
    cov = _coverage_frame(
        [{"device_id": "dev-1", "status": "ok", "data_newest": "2026-03-09T00:00:00Z"}]
    )
    out = label_samples(
        samples, windows, evidence, pd.Timestamp("2026-03-10T00:00:00Z"), CFG,
        source_coverage=cov,
    )
    row = out.iloc[0]
    assert row["label_status"] == "measured"
    assert row["y_outage_24h"] == 0


def test_positive_label_survives_coverage_gate():
    samples = _samples_for(["dev-1"], ["2026-02-28T20:00:00Z"])
    windows = _device_window_frame("dev-1", "2026-03-01T10:00:00Z", "2026-03-01T12:00:00Z")
    evidence = _full_evidence("dev-1")
    cov = _coverage_frame(
        [{"device_id": "dev-1", "status": "ok", "data_newest": "2026-03-09T00:00:00Z"}]
    )
    out = label_samples(
        samples, windows, evidence, pd.Timestamp("2026-03-10T00:00:00Z"), CFG,
        source_coverage=cov,
    )
    row = out.iloc[0]
    assert row["y_outage_24h"] == 1
    assert row["label_status"] == "measured"


def test_mixed_devices_are_gated_independently():
    samples = _samples_for(
        ["dev-covered", "dev-uncovered"],
        ["2026-03-05T00:00:00Z", "2026-03-05T00:00:00Z"],
    )
    windows = pd.concat(
        [
            _device_window_frame("dev-covered", "2026-03-01T10:00:00Z", "2026-03-01T12:00:00Z"),
            _device_window_frame("dev-uncovered", "2026-03-01T10:00:00Z", "2026-03-01T12:00:00Z"),
        ],
        ignore_index=True,
    )
    evidence = pd.concat([_full_evidence("dev-covered"), _full_evidence("dev-uncovered")], ignore_index=True)
    cov = _coverage_frame([{"device_id": "dev-covered", "status": "ok", "data_newest": "2026-03-09T00:00:00Z"}])
    out = label_samples(
        samples, windows, evidence, pd.Timestamp("2026-03-10T00:00:00Z"), CFG,
        source_coverage=cov,
    ).set_index("device_id")
    assert out.loc["dev-covered", "y_outage_24h"] == 0
    assert out.loc["dev-uncovered", "label_status"] == "censored"
    assert out.loc["dev-uncovered", "censor_reason"] == R_NO_SOURCE_COVERAGE


def test_gate_disabled_by_config_flag_behaves_like_before():
    cfg = load_config(None)
    cfg["labels"]["require_source_coverage"] = False
    # isolate the flag: disable the legacy span requirement too
    cfg["labels"]["min_device_evidence_span_days"] = 0
    samples = _samples_for(["dev-1"], ["2026-03-05T00:00:00Z"])
    windows = _device_window_frame("dev-1", "2026-03-01T10:00:00Z", "2026-03-01T12:00:00Z")
    evidence = _full_evidence("dev-1")
    out = label_samples(
        samples, windows, evidence, pd.Timestamp("2026-03-10T00:00:00Z"), cfg,
        source_coverage=None,
    )
    assert out.iloc[0]["label_status"] == "measured"
    assert out.iloc[0]["y_outage_24h"] == 0
