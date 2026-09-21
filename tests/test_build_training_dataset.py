"""Tests for the leakage-free training-data pipeline.

The central guarantee is tested directly: nothing observed after the anchor
timestamp may influence a feature row. Every other test covers one clause of
the data contract (quarantine, exclusions, splits, evidence, labels).
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_training_dataset import (
    META_COLUMNS,
    R_HORIZON_INCOMPLETE,
    R_INSIDE_OUTAGE,
    R_NO_VERIFIED_SOURCE,
    R_PURGE,
    R_STALE,
    R_UNVERIFIED_NEGATIVE,
    SPLIT_PURGED,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    STRONG,
    WEAK,
    FeatureLeakageError,
    ReadOnlyDb,
    ReferenceViolation,
    TrainingDataError,
    apply_eligibility,
    assert_read_only_sql,
    assign_splits,
    build_device_samples,
    build_evidence_frame,
    build_outage_windows,
    build_quality_metrics,
    compute_features_at,
    day_windows,
    excluded_reason,
    extract_telemetry,
    feature_column_names,
    finalize_feature_frame,
    find_exclusion_violations,
    label_samples,
    load_config,
    quarantine_telemetry,
    resolve_purge_hours,
    select_feature_keys,
    split_report,
    to_utc,
    verified_windows,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "training_config.json")
ANCHOR = pd.Timestamp("2026-03-10 12:00", tz="UTC")


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def staged_row(device: str, key: str, ts, num: float | None = None, text: str | None = None,
               customer: str = "CUST-1") -> dict:
    return {
        "time": to_utc(ts),
        "device_id": device,
        "key": key,
        "customer_id": customer,
        "value_kind": "numeric" if num is not None else "text",
        "value_num": num,
        "value_text": text,
    }


def staged(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["time"] = frame["time"].map(to_utc)
    return frame


def raw_row(device: str, key: str, ts, num=None, text=None) -> dict:
    return {
        "time": ts,
        "device_id": device,
        "key": key,
        "tenant_id": "T1",
        "customer_id": "CUST-1",
        "value_num": num,
        "value_text": text,
    }


def assert_features_equal(left: dict, right: dict, label: str = "") -> None:
    assert set(left) == set(right), f"{label} key mismatch: {set(left) ^ set(right)}"
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, float) or isinstance(b, float):
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b), f"{label} differs at {key}: {a} vs {b}"
        else:
            assert a == b, f"{label} differs at {key}: {a} vs {b}"


def evidence_row(device: str, ts, evidence_type: str, strength: str = STRONG, event_id: str = "e1") -> dict:
    return {
        "device_id": device,
        "evidence_ts": to_utc(ts),
        "evidence_type": evidence_type,
        "evidence_strength": strength,
        "source_event_id": event_id,
        "source_event_type": "LC_EVENT",
        "observed_at": to_utc(ts),
        "payload_fields_present": "method",
        "redacted_evidence_reference": f"ev:deadbeef:{evidence_type}",
    }


def sample_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["anchor_ts"] = frame["anchor_ts"].map(to_utc)
    return frame


# ---------------------------------------------------------------------------
# read-only guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT time FROM public.device_telemetry WHERE time >= :t0",
        "WITH d AS (SELECT device_id FROM public.device_event) SELECT * FROM d",
        "  select count(*) from public.device_telemetry;  ",
    ],
)
def test_read_only_guard_allows_reads(sql):
    assert assert_read_only_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO public.device_telemetry (key) VALUES ('x')",
        "UPDATE public.device_telemetry SET value_num = 0",
        "DELETE FROM public.device_telemetry",
        "DROP TABLE public.device_telemetry",
        "TRUNCATE public.device_telemetry",
        "SET default_transaction_read_only = off",
        "CREATE INDEX i ON public.device_telemetry (time)",
        "SELECT 1; DROP TABLE public.device_telemetry",
    ],
)
def test_read_only_guard_rejects_writes(sql):
    with pytest.raises(ReferenceViolation):
        assert_read_only_sql(sql)


def test_db_layer_cannot_execute_a_write_even_with_injected_executor():
    executed: list[str] = []
    db = ReadOnlyDb(executor=lambda sql, params: executed.append(sql) or [], cfg=CFG)
    with pytest.raises(ReferenceViolation):
        db.query("DELETE FROM public.device_telemetry")
    assert executed == []


# ---------------------------------------------------------------------------
# bounded partitions
# ---------------------------------------------------------------------------
def test_day_windows_are_bounded_and_cover_the_range():
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = start + timedelta(days=3, hours=5)
    windows = list(day_windows(start, end, 1))
    assert windows[0][0] == start
    assert windows[-1][1] == end
    for t0, t1 in windows:
        assert t1 - t0 <= timedelta(days=1)
        assert t1 > t0


# ---------------------------------------------------------------------------
# quarantine
# ---------------------------------------------------------------------------
def test_quarantine_drops_epoch_like_future_and_duplicate_rows():
    now = pd.Timestamp.now(tz="UTC")
    raw = pd.DataFrame(
        [
            raw_row("d1", "battery_voltage", pd.Timestamp("2026-03-01", tz="UTC"), num=12.0),
            raw_row("d1", "battery_voltage", pd.Timestamp("1970-01-01 00:00:01", tz="UTC"), num=99.0),
            raw_row("d1", "battery_voltage", pd.Timestamp("1970-01-01", tz="UTC"), num=99.0),
            raw_row("d1", "battery_voltage", now + timedelta(days=30), num=99.0),
            raw_row("d1", "battery_voltage", pd.Timestamp("2026-03-01", tz="UTC"), num=13.0),
            raw_row("d1", "battery_voltage", None, num=1.0),
            raw_row("d1", "battery_voltage", "not-a-timestamp", num=1.0),
            raw_row("d1", "battery_voltage", pd.Timestamp("2026-03-02", tz="UTC")),
        ]
    )
    clean, quarantined, stats = quarantine_telemetry(raw, CFG)

    assert len(clean) == 1
    assert float(clean["value_num"].iloc[0]) == 13.0  # duplicate kept the last write
    assert stats["epoch_like_timestamp"] == 2
    assert stats["future_timestamp"] == 1
    assert stats["null_timestamp"] == 1
    assert stats["unparseable_timestamp"] == 1
    assert stats["no_value"] == 1
    assert stats["duplicate_device_key_time"] == 1
    reasons = set(quarantined["quarantine_reason"])
    assert {"epoch_like_timestamp", "future_timestamp", "duplicate_device_key_time"} <= reasons
    assert not clean["time"].isna().any()


def test_quarantine_keeps_text_only_rows_as_text():
    raw = pd.DataFrame([raw_row("d1", "network_type", pd.Timestamp("2026-03-01", tz="UTC"), text="4G")])
    clean, _quarantined, _stats = quarantine_telemetry(raw, CFG)
    assert len(clean) == 1
    assert clean["value_kind"].iloc[0] == "text"


# ---------------------------------------------------------------------------
# leakage: the core contract
# ---------------------------------------------------------------------------
def test_features_ignore_every_row_after_the_anchor():
    history = staged(
        [
            staged_row("d1", "battery_voltage", ANCHOR - timedelta(hours=3), num=12.0),
            staged_row("d1", "battery_voltage", ANCHOR - timedelta(hours=1), num=12.5),
            staged_row("d1", "network_type", ANCHOR - timedelta(hours=2), text="4G"),
        ]
    )
    future = staged(
        [
            staged_row("d1", "battery_voltage", ANCHOR + timedelta(minutes=1), num=9999.0),
            staged_row("d1", "battery_voltage", ANCHOR + timedelta(hours=5), num=-1e9),
            staged_row("d1", "network_type", ANCHOR + timedelta(minutes=5), text="OFFLINE fault"),
        ]
    )

    baseline, meta_base = compute_features_at(ANCHOR, history, CFG, ["battery_voltage"], ["network_type"])
    poisoned, meta_poisoned = compute_features_at(
        ANCHOR, pd.concat([history, future], ignore_index=True), CFG, ["battery_voltage"], ["network_type"]
    )

    assert_features_equal(baseline, poisoned, "feature row")
    assert meta_base["max_source_ts"] == meta_poisoned["max_source_ts"]
    assert meta_poisoned["n_rows_ignored_future"] == len(future)
    assert meta_poisoned["max_source_ts"] <= ANCHOR
    assert baseline["battery_voltage__last"] == pytest.approx(12.5)
    assert baseline["network_type__text_bad_flag"] == 0.0


def test_strict_mode_refuses_to_compute_with_future_rows():
    frame = staged([staged_row("d1", "battery_voltage", ANCHOR + timedelta(seconds=1), num=1.0)])
    with pytest.raises(FeatureLeakageError):
        compute_features_at(ANCHOR, frame, CFG, ["battery_voltage"], [], strict=True)


def test_no_leakage_checker_flags_a_violating_sample():
    good = sample_frame([{"device_id": "d1", "anchor_ts": ANCHOR, "max_source_ts": ANCHOR - timedelta(hours=1)}])
    build_device_samples("d1", staged([staged_row("d1", "battery_voltage", ANCHOR - timedelta(hours=1), num=1.0)]),
                         [ANCHOR], CFG, ["battery_voltage"], [])
    bad = sample_frame([{"device_id": "d1", "anchor_ts": ANCHOR, "max_source_ts": ANCHOR + timedelta(hours=1)}])
    from build_training_dataset import verify_no_leakage

    verify_no_leakage(good)
    with pytest.raises(FeatureLeakageError):
        verify_no_leakage(bad)


def test_samples_record_source_timestamp_and_never_exceed_the_anchor():
    telemetry = staged(
        [staged_row("d1", "battery_voltage", ANCHOR - timedelta(hours=h), num=12.0 + h) for h in range(1, 40)]
    )
    anchors = [ANCHOR - timedelta(hours=6), ANCHOR]
    samples = build_device_samples("d1", telemetry, anchors, CFG, ["battery_voltage"], [], customer_id="C1")
    assert len(samples) == 2
    assert (samples["max_source_ts"] <= samples["anchor_ts"]).all()
    # reading 1h..39h before ANCHOR; the first anchor at -6h can only see 34 of them
    assert samples["n_rows_used"].tolist() == [34, 39]


# ---------------------------------------------------------------------------
# imputation / freshness
# ---------------------------------------------------------------------------
def test_imputation_is_capped_and_age_is_always_exposed():
    fresh = staged([staged_row("d1", "battery_voltage", ANCHOR - timedelta(hours=1), num=12.0)])
    stale = staged([staged_row("d1", "battery_voltage", ANCHOR - timedelta(hours=10), num=12.0)])
    very_stale = staged([staged_row("d1", "battery_voltage", ANCHOR - timedelta(hours=30), num=12.0)])

    f, _ = compute_features_at(ANCHOR, fresh, CFG, ["battery_voltage"], [])
    s, _ = compute_features_at(ANCHOR, stale, CFG, ["battery_voltage"], [])
    v, _ = compute_features_at(ANCHOR, very_stale, CFG, ["battery_voltage"], [])

    assert f["battery_voltage__last"] == pytest.approx(12.0)
    assert f["battery_voltage__ffill_capped"] == 1.0
    assert f["battery_voltage__stale"] == 0.0

    # Beyond the fill cap the value is NOT carried forward, but its age remains.
    assert np.isnan(s["battery_voltage__last"])
    assert s["battery_voltage__ffill_capped"] == 0.0
    assert s["battery_voltage__last_age_h"] == pytest.approx(10.0)
    assert s["g__last_age_h"] == pytest.approx(10.0)
    assert s["battery_voltage__stale"] == 0.0

    assert v["battery_voltage__stale"] == 1.0
    assert v["g__stale_key_frac"] == pytest.approx(1.0)


def test_missingness_is_reported_per_window():
    dense = staged([staged_row("d1", "battery_voltage", ANCHOR - timedelta(minutes=5 * i), num=12.0) for i in range(12)])
    sparse = staged([staged_row("d1", "battery_voltage", ANCHOR - timedelta(hours=23), num=12.0)])
    dense_f, _ = compute_features_at(ANCHOR, dense, CFG, ["battery_voltage"], [])
    sparse_f, _ = compute_features_at(ANCHOR, sparse, CFG, ["battery_voltage"], [])
    assert dense_f["battery_voltage__missing_frac_1h"] < sparse_f["battery_voltage__missing_frac_1h"]
    assert sparse_f["battery_voltage__missing_frac_24h"] == pytest.approx(1.0 - (1 / 288), abs=1e-3)


def test_numeric_and_text_keys_are_handled_separately():
    frame = staged(
        [
            staged_row("d1", "signal_strength", ANCHOR - timedelta(hours=1), num=-70.0),
            staged_row("d1", "network_type", ANCHOR - timedelta(hours=1), text="4G"),
            staged_row("d1", "network_type", ANCHOR - timedelta(hours=2), text="OFFLINE"),
        ]
    )
    numeric_only, _ = compute_features_at(ANCHOR, frame, CFG, ["signal_strength"], [])
    text_only, _ = compute_features_at(ANCHOR, frame, CFG, [], ["network_type"])
    assert any(c.startswith("signal_strength__") for c in numeric_only)
    assert not any("__text_" in c for c in numeric_only)
    assert any(c.startswith("network_type__text_") for c in text_only)
    assert not any(c.endswith("__mean_24h") for c in text_only)
    assert text_only["network_type__text_bad_flag"] == 0.0  # most recent state wins
    assert text_only["network_type__text_changes_24h"] == 1.0


# ---------------------------------------------------------------------------
# exclusions
# ---------------------------------------------------------------------------
def test_exclusion_patterns_cover_identifiers_gps_targets_and_secrets():
    for name, expected in [
        ("device_id", "identifier"),
        ("latitude", "gps_location"),
        ("imei_id", "network_identifier"),
        ("tb_password", "secret"),
        ("ts_fault_score", "rule_derived_score"),
        ("severity", "alert_or_severity"),
        ("cameraOffline", "target_like_or_label_proxy"),
        ("lastConnectTime", "target_like_or_label_proxy"),
        ("gatewayStatus", "derived_status_flag"),
        ("basSystemIntegration", "raw_payload"),
    ]:
        assert excluded_reason(name, CFG) is not None, name


def test_selection_and_generated_columns_never_violate_exclusions():
    keys_path = ROOT / "live_tenant_keys.json"
    if not keys_path.exists():
        pytest.skip("live_tenant_keys.json fixture unavailable")
    tenant_keys = json.loads(keys_path.read_text(encoding="utf-8"))
    text_like = re.compile(r"version|operator|network_type|sw_|hostname|title", re.I)
    stats = pd.DataFrame(
        [
            {
                "key": key,
                "n_rows": count,
                "n_numeric": 0 if text_like.search(key) else count,
                "n_text": count if text_like.search(key) else 0,
            }
            for key, count in tenant_keys.items()
        ]
    )
    numeric_keys, text_keys, excluded = select_feature_keys(stats, CFG)
    assert numeric_keys
    assert text_keys, [k for k in tenant_keys if text_like.search(k)]
    assert "imei_id" not in numeric_keys
    assert "latitude" not in numeric_keys
    assert not any("imei" in k.lower() or "latitude" in k.lower() for k in numeric_keys + text_keys)
    columns = feature_column_names(numeric_keys, text_keys, CFG)
    assert find_exclusion_violations(columns, CFG) == []
    assert excluded  # the fixture contains identifier/raw/spec keys


def test_finalize_refuses_to_emit_an_excluded_feature_column():
    frame = sample_frame(
        [
            {
                "device_id": "d1",
                "anchor_ts": ANCHOR,
                "split": SPLIT_TRAIN,
                "label_status": "censored",
                "censor_reason": R_NO_VERIFIED_SOURCE,
                "n_rows_used": 5,
                "n_feature_values": 5,
                "g__last_age_h": 0.1,
                "severity": 3.0,
            }
        ]
    )
    with pytest.raises(TrainingDataError):
        finalize_feature_frame(frame, CFG)


def test_device_id_is_metadata_never_a_feature():
    telemetry = staged([staged_row("d1", "battery_voltage", ANCHOR - timedelta(hours=1), num=12.0)])
    samples = build_device_samples("d1", telemetry, [ANCHOR], CFG, ["battery_voltage"], [])
    finalized = finalize_feature_frame(samples, CFG)
    features = [c for c in finalized.columns if c not in META_COLUMNS]
    assert "device_id" in META_COLUMNS and "device_id" not in features
    assert not any(c.startswith("device_id") for c in features)


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def synthetic_samples() -> pd.DataFrame:
    rows: list[dict] = []
    for idx, customer in enumerate(["A", "B", "C", "D"]):
        base = pd.Timestamp("2026-01-01", tz="UTC") + timedelta(days=idx * 40)
        for device in range(3):
            for step in range(0, 40 * 24, 6):
                rows.append(
                    {
                        "device_id": f"{customer}-dev{device}",
                        "anchor_ts": base + timedelta(hours=step),
                        "customer_id": customer,
                        "n_rows_used": 20,
                        "n_feature_values": 10,
                        "g__last_age_h": 0.2,
                    }
                )
    return sample_frame(rows)


def test_group_splits_are_chronological_group_pure_and_purged():
    samples = assign_splits(synthetic_samples(), CFG)
    purge_hours = resolve_purge_hours(CFG)

    valid = samples[samples["split"].isin([SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST])]
    assert set(valid["split"]) == {SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST}

    # group purity: a customer/device appears in exactly one split
    per_group = valid.groupby("group_key")["split"].nunique()
    assert (per_group == 1).all()
    assert set(valid["customer_id"]) == {"A", "B", "C", "D"}

    # purged rows are explicit and excluded from every split
    purged = samples[samples["split"] == SPLIT_PURGED]
    assert (purged["censor_reason"] == R_PURGE).all()

    report = split_report(samples, purge_hours)
    for pair in report["pairs"]:
        assert pair.get("status") != "empty_split"
        assert pair["chronological"] is True
        assert pair["group_overlap_count"] == 0
        assert pair["purge_satisfied"] is True, pair

    # ordering is monotone in time across splits -> not a random-row split
    ordered = valid.sort_values("anchor_ts")["split"].tolist()
    rank = {SPLIT_TRAIN: 0, SPLIT_VALIDATION: 1, SPLIT_TEST: 2}
    assert [rank[s] for s in ordered] == sorted(rank[s] for s in ordered)


def test_time_mode_splits_are_also_chronological_with_purge_gaps():
    cfg = load_config(ROOT / "training_config.json")
    cfg["splits"]["mode"] = "time"
    samples = assign_splits(synthetic_samples(), cfg)
    report = split_report(samples, resolve_purge_hours(cfg))
    for pair in report["pairs"]:
        assert pair["status"] != "empty_split"
        assert pair["chronological"] is True
        assert pair["purge_satisfied"] is True


# ---------------------------------------------------------------------------
# labels: verified only, censored otherwise
# ---------------------------------------------------------------------------
def base_samples(anchors: list[pd.Timestamp], device: str = "d1") -> pd.DataFrame:
    return sample_frame(
        [
            {
                "device_id": device,
                "anchor_ts": anchor,
                "customer_id": "C1",
                "split": SPLIT_TRAIN,
                "censor_reason": None,
                "n_rows_used": 50,
                "n_feature_values": 20,
                "g__last_age_h": 0.2,
            }
            for anchor in anchors
        ]
    )


def test_no_verified_evidence_censors_everything_and_creates_no_labels():
    samples = base_samples([ANCHOR, ANCHOR + timedelta(hours=6)])
    empty_windows = pd.DataFrame(columns=["device_id", "outage_start", "outage_end", "window_closed"])
    labelled = label_samples(samples, empty_windows, pd.DataFrame(), ANCHOR + timedelta(days=30), CFG)
    assert (labelled["label_status"] == "censored").all()
    assert (labelled["censor_reason"] == R_NO_VERIFIED_SOURCE).all()
    assert labelled["y_outage_24h"].isna().all()
    eligible = apply_eligibility(labelled, CFG)
    assert (eligible["eligibility"] == "excluded").all()


def test_positive_label_requires_a_verified_outage_start_in_the_horizon():
    outage_start = ANCHOR + timedelta(hours=2)
    evidence = pd.DataFrame(
        [
            evidence_row("d1", ANCHOR - timedelta(days=10), "lc_connect", event_id="c0"),
            evidence_row("d1", outage_start, "lc_disconnect", event_id="d1"),
            evidence_row("d1", outage_start + timedelta(hours=3), "lc_connect", event_id="c2"),
            evidence_row("d1", ANCHOR + timedelta(days=5), "lc_connect", event_id="c3"),
        ]
    )
    windows = build_outage_windows(evidence, CFG)
    assert len(windows) == 1 and bool(windows["window_closed"].iloc[0])

    samples = base_samples([ANCHOR, ANCHOR + timedelta(hours=26)])
    labelled = label_samples(samples, windows, evidence, ANCHOR + timedelta(days=30), CFG)
    by_anchor = labelled.set_index("anchor_ts")["y_outage_24h"]
    assert by_anchor.loc[ANCHOR] == 1
    # the window starts after the second anchor's horizon
    assert by_anchor.loc[ANCHOR + timedelta(hours=26)] == 0


def test_weak_evidence_alone_can_never_create_a_positive_label():
    evidence = pd.DataFrame(
        [
            evidence_row("d1", ANCHOR + timedelta(hours=1), "current_attr_snapshot", strength=WEAK),
            evidence_row("d1", ANCHOR + timedelta(hours=2), "current_attr_snapshot", strength=WEAK),
        ]
    )
    windows = build_outage_windows(evidence, CFG)
    assert windows.empty
    labelled = label_samples(base_samples([ANCHOR]), windows, evidence, ANCHOR + timedelta(days=30), CFG)
    assert labelled["y_outage_24h"].isna().all()
    assert labelled["censor_reason"].iloc[0] == R_NO_VERIFIED_SOURCE


def test_open_outage_windows_are_not_used_as_verified_labels():
    evidence = pd.DataFrame([evidence_row("d1", ANCHOR + timedelta(hours=1), "lc_disconnect")])
    windows = build_outage_windows(evidence, CFG)
    assert len(windows) == 1 and not bool(windows["window_closed"].iloc[0])
    assert verified_windows(windows, CFG).empty
    labelled = label_samples(base_samples([ANCHOR]), windows, evidence, ANCHOR + timedelta(days=30), CFG)
    assert labelled["y_outage_24h"].isna().all()


def test_anchor_inside_an_outage_is_censored():
    start = ANCHOR - timedelta(hours=3)
    evidence = pd.DataFrame(
        [
            evidence_row("d1", ANCHOR - timedelta(days=20), "lc_connect", event_id="c0"),
            evidence_row("d1", start, "lc_disconnect", event_id="o1"),
            evidence_row("d1", ANCHOR + timedelta(hours=1), "lc_connect", event_id="c1"),
            evidence_row("d1", ANCHOR + timedelta(days=8), "lc_connect", event_id="c2"),
        ]
    )
    windows = build_outage_windows(evidence, CFG)
    labelled = label_samples(base_samples([ANCHOR]), windows, evidence, ANCHOR + timedelta(days=30), CFG)
    assert labelled["y_outage_24h"].isna().all()
    assert labelled["censor_reason"].iloc[0] == R_INSIDE_OUTAGE


def test_anchor_without_a_full_future_window_is_censored():
    evidence = pd.DataFrame(
        [
            evidence_row("d1", ANCHOR - timedelta(days=10), "lc_connect", event_id="c0"),
            evidence_row("d1", ANCHOR - timedelta(days=9), "lc_disconnect", event_id="o0"),
            evidence_row("d1", ANCHOR - timedelta(days=9) + timedelta(hours=1), "lc_connect", event_id="c1"),
        ]
    )
    windows = build_outage_windows(evidence, CFG)
    coverage_end = ANCHOR + timedelta(hours=2)  # horizon ends at +24h
    labelled = label_samples(base_samples([ANCHOR]), windows, evidence, coverage_end, CFG)
    assert labelled["y_outage_24h"].isna().all()
    assert labelled["censor_reason"].iloc[0] == R_HORIZON_INCOMPLETE


def test_negative_label_requires_device_level_evidence_coverage():
    evidence = pd.DataFrame(
        [
            evidence_row("d1", ANCHOR - timedelta(hours=5), "lc_connect", event_id="c0"),
            evidence_row("d1", ANCHOR - timedelta(hours=4), "lc_disconnect", event_id="o0"),
            evidence_row("d1", ANCHOR - timedelta(hours=3), "lc_connect", event_id="c1"),
        ]
    )
    windows = build_outage_windows(evidence, CFG)
    labelled = label_samples(base_samples([ANCHOR]), windows, evidence, ANCHOR + timedelta(days=30), CFG)
    assert labelled["y_outage_24h"].isna().all()
    assert labelled["censor_reason"].iloc[0] == R_UNVERIFIED_NEGATIVE


def long_span_evidence() -> pd.DataFrame:
    """A device with verified evidence spanning >7 days and one distant outage."""
    return pd.DataFrame(
        [
            evidence_row("d1", ANCHOR - timedelta(days=10), "lc_connect", event_id="c0"),
            evidence_row("d1", ANCHOR - timedelta(days=8), "lc_disconnect", event_id="o0"),
            evidence_row("d1", ANCHOR - timedelta(days=8) + timedelta(hours=3), "lc_connect", event_id="c1"),
            evidence_row("d1", ANCHOR + timedelta(days=5), "lc_connect", event_id="c2"),
        ]
    )


def test_purged_rows_are_never_relabelled():
    samples = base_samples([ANCHOR])
    samples["split"] = SPLIT_PURGED
    samples["censor_reason"] = R_PURGE
    evidence = long_span_evidence()
    windows = build_outage_windows(evidence, CFG)
    labelled = label_samples(samples, windows, evidence, ANCHOR + timedelta(days=30), CFG)
    assert labelled["censor_reason"].iloc[0] == R_PURGE
    assert labelled["y_outage_24h"].isna().all()


def test_stale_samples_are_marked_excluded():
    evidence = long_span_evidence()
    windows = build_outage_windows(evidence, CFG)
    samples = base_samples([ANCHOR])
    samples.loc[0, "g__last_age_h"] = 999.0
    labelled = label_samples(samples, windows, evidence, ANCHOR + timedelta(days=30), CFG)
    assert labelled["y_outage_24h"].iloc[0] == 0  # label itself is measurable
    eligible = apply_eligibility(labelled, CFG)
    assert eligible["censor_reason"].iloc[0] == R_STALE
    assert eligible["eligibility"].iloc[0] == "excluded"
    assert eligible["anomaly_eligible"].iloc[0] == True  # noqa: E712 - numpy bool


# ---------------------------------------------------------------------------
# evidence audit
# ---------------------------------------------------------------------------
def test_evidence_audit_has_required_columns_and_redacts_payload_values():
    events = pd.DataFrame(
        [
            {
                "id": 1,
                "event_id": "evt-1",
                "device_id": "d1",
                "event_type": "LC_EVENT",
                "time": pd.Timestamp("2026-03-10", tz="UTC"),
                "payload": {
                    "method": "onDisconnect",
                    "data": {"currentAttr": {"tb_password": "sup3rsecret", "active": False}},
                },
            }
        ]
    )
    audit = build_evidence_frame(events, CFG)
    required = {
        "device_id",
        "evidence_ts",
        "evidence_type",
        "evidence_strength",
        "source_event_id",
        "observed_at",
        "redacted_evidence_reference",
    }
    assert required <= set(audit.columns)
    assert audit["evidence_type"].iloc[0] == "lc_disconnect"
    assert audit["evidence_strength"].iloc[0] == STRONG
    assert audit["source_event_id"].iloc[0] == "evt-1"
    blob = audit.astype(str).to_json()
    assert "sup3rsecret" not in blob
    assert "tb_password" not in blob


def test_currentattr_snapshot_is_weak_evidence_only():
    events = pd.DataFrame(
        [
            {
                "id": 2,
                "event_id": "evt-2",
                "device_id": "d1",
                "event_type": "SOMETHING_ELSE",
                "time": pd.Timestamp("2026-03-10", tz="UTC"),
                "payload": {"data": {"currentAttr": {"active": False, "lastDisconnectTime": 1}},
                            "note": "token=abc123"},
            }
        ]
    )
    audit = build_evidence_frame(events, CFG)
    assert audit["evidence_type"].iloc[0] == "current_attr_snapshot"
    assert audit["evidence_strength"].iloc[0] == WEAK
    assert "abc123" not in audit.astype(str).to_json()


def test_unclassifiable_events_produce_no_evidence_row():
    events = pd.DataFrame(
        [
            {
                "id": 3,
                "event_id": "evt-3",
                "device_id": "d1",
                "event_type": "STATS",
                "time": pd.Timestamp("2026-03-10", tz="UTC"),
                "payload": {"whatever": 1},
            }
        ]
    )
    assert build_evidence_frame(events, CFG).empty


# ---------------------------------------------------------------------------
# extraction orchestration (bounded, batched)
# ---------------------------------------------------------------------------
def test_extraction_uses_daily_partitions_and_key_batches(tmp_path):
    captured: list[tuple[str, dict]] = []
    midnight = pd.Timestamp("2026-03-01", tz="UTC")

    def executor(sql: str, params: dict):
        captured.append((sql, params))
        if "GROUP BY" in sql:
            return [
                {"key": f"k{i}", "n_rows": 100, "n_numeric": 100, "n_text": 0, "n_devices": 1}
                for i in range(20)
            ]
        if sql.startswith("SELECT time"):
            return [
                {
                    "time": params["t0"],
                    "device_id": "d1",
                    "key": key,
                    "customer_id": "C1",
                    "value_num": 1.0,
                    "value_text": None,
                }
                for key in params["keys"]
            ]
        return []

    db = ReadOnlyDb(executor=executor, cfg=CFG)
    staging = tmp_path / "staging"
    result = extract_telemetry(db, CFG, midnight, midnight + timedelta(days=2), staging)

    assert result.partitions == 2
    telemetry_calls = [p for s, p in captured if s.startswith("SELECT time")]
    assert telemetry_calls
    for params in telemetry_calls:
        assert params["t1"] - params["t0"] <= timedelta(days=1)
        assert 0 < len(params["keys"]) <= CFG["extraction"]["key_batch_size"]

    files = sorted((staging / "telemetry").rglob("*.parquet"))
    assert len(files) == 2 * 3  # 20 keys / 8 per batch -> 3 batches per day
    assert result.rows_clean == 40
    assert result.devices_seen == 1
    assert result.keys_seen == 20
    assert not result.errors
    assert (staging / "devices.json").exists()


def test_extraction_records_partition_failures_without_aborting(tmp_path):
    def executor(sql: str, params: dict):
        if "GROUP BY" in sql:
            return [{"key": "k0", "n_rows": 5, "n_numeric": 5, "n_text": 0, "n_devices": 1}]
        raise RuntimeError("statement timeout")

    db = ReadOnlyDb(executor=executor, cfg=CFG)
    midnight = pd.Timestamp("2026-03-01", tz="UTC")
    result = extract_telemetry(db, CFG, midnight, midnight + timedelta(days=1), tmp_path / "staging")
    assert result.rows_clean == 0
    assert result.errors and "statement timeout" in result.errors[0]


# ---------------------------------------------------------------------------
# quality report
# ---------------------------------------------------------------------------
def test_quality_metrics_and_report_render_the_censored_verdict(tmp_path):
    samples = base_samples([ANCHOR, ANCHOR + timedelta(hours=6)])
    labelled = label_samples(samples, pd.DataFrame(), pd.DataFrame(), ANCHOR + timedelta(days=30), CFG)
    labelled = apply_eligibility(labelled, CFG)
    metrics = build_quality_metrics(
        labelled, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        split_report(labelled, resolve_purge_hours(CFG)), CFG, ANCHOR + timedelta(days=30),
    )
    assert metrics["supervised_training_ready"] is False
    assert metrics["samples"]["label_censored"] == 2
    assert metrics["samples"]["positive_labels"] == 0

    from build_training_dataset import write_quality_report

    md_path = tmp_path / "report.md"
    json_path = tmp_path / "metrics.json"
    write_quality_report(metrics, md_path, json_path)
    body = md_path.read_text(encoding="utf-8")
    assert "Supervised 24h-label training ready: **NO**" in body
    assert R_NO_VERIFIED_SOURCE in body
    assert json.loads(json_path.read_text(encoding="utf-8"))["supervised_training_ready"] is False
