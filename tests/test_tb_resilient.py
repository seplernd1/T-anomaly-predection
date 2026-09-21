"""Tests for tb_resilient: retry, paging, cap-splitting, checkpoints, policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from tb_resilient import (
    AuthError,
    CheckpointStore,
    CoverageLedger,
    CoverageRecord,
    DevicePolicy,
    PermanentError,
    RetryExhaustedError,
    RetryPolicy,
    atomic_write_bytes,
    fetch_timeseries_paged,
    http_get,
    redact_evidence,
    resolve_verify_tls,
    resolve_window,
    sanitize_csv_value,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakeResp:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text else (json.dumps(payload)[:300] if payload is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


FAST = RetryPolicy(max_attempts=4, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0)


def point(ts, v=1):
    return {"ts": ts, "value": v}


# ---------------------------------------------------------------------------
# 8/9/10: retry behavior
# ---------------------------------------------------------------------------
def test_429_retry_after_honored():
    sleeps = []
    sess = FakeSession([FakeResp(429, headers={"Retry-After": "7"}), FakeResp(200, {"ok": True})])
    resp, retries = http_get(sess, "http://x", policy=FAST, sleep=sleeps.append)
    assert resp.status_code == 200 and retries == 1
    assert sleeps == [7.0]


def test_5xx_then_success():
    sess = FakeSession([FakeResp(500), FakeResp(503), FakeResp(200, {"ok": True})])
    resp, retries = http_get(sess, "http://x", policy=FAST, sleep=lambda s: None)
    assert resp.status_code == 200 and retries == 2


def test_timeout_and_connection_retry_then_exhaust():
    sess = FakeSession([requests.ConnectTimeout("c"), requests.ReadTimeout("t"),
                        FakeResp(500), FakeResp(500)])
    with pytest.raises(RetryExhaustedError):
        http_get(sess, "http://x", policy=FAST, sleep=lambda s: None)
    assert sess.calls == 4


def test_auth_never_retried():
    for status in (401, 403):
        sess = FakeSession([FakeResp(status)])
        with pytest.raises(AuthError):
            http_get(sess, "http://x", policy=FAST, sleep=lambda s: None)
        assert sess.calls == 1


def test_permanent_4xx_never_retried():
    for status in (400, 404, 405, 422):
        sess = FakeSession([FakeResp(status)])
        with pytest.raises(PermanentError):
            http_get(sess, "http://x", policy=FAST, sleep=lambda s: None)
        assert sess.calls == 1


# ---------------------------------------------------------------------------
# 2/3/4/5: paging, cap, same-timestamp, chunk boundaries
# ---------------------------------------------------------------------------
def make_server(points, server_cap=5000):
    """page_fn(cursor, end, limit): server truncates every response at server_cap."""
    def page_fn(cursor, end, limit):
        batch = [p for p in points if cursor <= p["ts"] <= end][: min(limit, server_cap)]
        return batch, 0
    return page_fn


def test_multipage_and_exact_limit_continues():
    pts = [point(1000 + i) for i in range(250)]
    out = fetch_timeseries_paged(make_server(pts, server_cap=10_000), 0, 10**12, limit=100)
    assert len(out.points) == 250 and not out.capped


def test_exact_limit_response_triggers_continuation():
    pts = [point(1000 + i) for i in range(300)]
    out = fetch_timeseries_paged(make_server(pts, server_cap=10_000), 0, 10**12, limit=100)
    assert len(out.points) == 300  # 3 full pages of exactly `limit`, then a short page


def test_server_cap_forces_window_split_until_complete():
    pts = [point(1000 + i) for i in range(12_000)]
    page_fn = make_server(pts, server_cap=5000)
    whole = resolve_window(page_fn, 0, 10**12, limit=100_000, min_chunk_ms=1)
    assert whole.completeness == "complete"
    assert len(whole.points) == 12_000
    assert whole.cap_hits >= 1


def test_same_timestamp_records_not_skipped():
    pts = [point(5000, v) for v in range(30)] + [point(6000, v) for v in range(30)]
    out = fetch_timeseries_paged(make_server(pts, server_cap=10_000), 0, 10**12, limit=10)
    assert len(out.points) == 60  # overlap cursor + dedupe keeps all same-ts rows


def test_adjacent_chunks_no_duplicates_or_gaps():
    pts = [point(1000 + i * 7) for i in range(2000)]
    page_fn = make_server(pts, server_cap=5000)
    full = resolve_window(page_fn, 0, 10**12, limit=100_000, min_chunk_ms=1)
    left = resolve_window(page_fn, 0, 8000, limit=100_000, min_chunk_ms=1)
    right = resolve_window(page_fn, 8001, 10**12, limit=100_000, min_chunk_ms=1)
    merged_ts = sorted(p["ts"] for p in list(left.points) + list(right.points))
    full_ts = sorted(p["ts"] for p in full.points)
    assert merged_ts == full_ts == sorted(p["ts"] for p in pts)


def test_stalled_cursor_marks_partial_not_hang():
    def stuck(cursor, end, limit):
        return [point(cursor, 1) for _ in range(limit)], 0  # full pages, never advances

    out = fetch_timeseries_paged(stuck, 1000, 2000, limit=5, max_pages=50)
    assert out.stalled and out.capped


# ---------------------------------------------------------------------------
# 6/7: empty_verified vs failed
# ---------------------------------------------------------------------------
def test_valid_zero_rows_is_empty_verified():
    outcome = resolve_window(make_server([], server_cap=5000), 0, 10**12, limit=1000)
    assert outcome.completeness == "empty_verified" and outcome.points == []


def test_failed_request_is_not_no_data():
    def boom(cursor, end, limit):
        raise RetryExhaustedError("down", 500, "", attempts=3, retries=2)

    outcome = resolve_window(boom, 0, 10**12, limit=1000)
    assert outcome.completeness == "failed"


# ---------------------------------------------------------------------------
# 11: checkpoints / resume + atomic writes
# ---------------------------------------------------------------------------
def test_checkpoint_resume_skips_done(tmp_path):
    store = CheckpointStore(tmp_path / "ckpt.json")
    store.mark_done("d1", "k", 0, 100)
    assert store.is_done("d1", "k", 0, 100)
    assert not store.is_done("d1", "k", 0, 200)
    again = CheckpointStore(tmp_path / "ckpt.json")
    assert again.is_done("d1", "k", 0, 100)


def test_atomic_write_and_ledger(tmp_path):
    p = tmp_path / "sub" / "f.bin"
    atomic_write_bytes(p, b"hello")
    assert p.read_bytes() == b"hello"
    ledger = CoverageLedger("r1", tmp_path / "cov.jsonl")
    ledger.add(CoverageRecord(run_id="r1", device_id="d", key="k", endpoint="e",
                              requested_start_ms=0, requested_end_ms=1,
                              chunk_start_ms=0, chunk_end_ms=1, rows=3,
                              completeness="complete"))
    assert ledger.summary()["by_status"] == {"complete": 1}
    assert CoverageLedger("r1", tmp_path / "cov.jsonl").summary()["records"] == 1
    assert ledger.incomplete() == []


# ---------------------------------------------------------------------------
# policy, redaction, csv safety, TLS
# ---------------------------------------------------------------------------
def test_device_policy_needs_two_signals():
    pol = DevicePolicy()
    assert pol.decide({"device_id": "a", "name": "TEST-rig", "type": "HESTIA"}, 10).eligible
    assert pol.decide({"device_id": "a", "name": "TEST-rig", "type": "HESTIA"},
                      10).reason.startswith("name_pattern_only")
    assert not pol.decide({"device_id": "b", "name": "probe", "type": "simulator"}, 50).eligible
    assert not pol.decide({"device_id": "c", "name": "demo-x", "type": "x"}, 0).eligible
    assert pol.decide({"device_id": "d", "name": "Branch1", "type": "x"}, 5).eligible
    allow = DevicePolicy(explicit_allow=["z"])
    assert allow.decide({"device_id": "z", "name": "test", "type": "simulator"}, 0).eligible
    deny = DevicePolicy(explicit_deny=["y"])
    assert not deny.decide({"device_id": "y", "name": "ok", "type": "x"}, 9).eligible


def test_redaction_and_csv_safety():
    out = redact_evidence({"type": "offline", "imei": "123", "ts": 5}, {"type", "ts"})
    assert out["type"] == "offline" and out["ts"] == 5
    assert out["imei"].startswith("hash:")
    assert sanitize_csv_value("=cmd|'/c") == "'=cmd|'/c"
    assert sanitize_csv_value({"a": 1}) == '{"a": 1}'


def test_tls_defaults_to_verify(capsys):
    assert resolve_verify_tls(False, False) is True
    assert resolve_verify_tls(True, False) is False
    assert "WARNING" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 1: registry pagination until exhausted
# ---------------------------------------------------------------------------
def test_registry_pagination_walks_has_next():
    from pull_all_data import pull_devices

    pages = [
        {"data": [{"id": {"id": f"d{i}"}, "name": f"n{i}"} for i in range(3)], "hasNext": True},
        {"data": [{"id": {"id": "d3"}, "name": "n3"}], "hasNext": False},
    ]

    class C:
        def get_json(self, path, timeout=30):
            return pages.pop(0)

    rows = pull_devices(C(), page_size=3, delay=0)
    assert [r["device_id"] for r in rows] == ["d0", "d1", "d2", "d3"]
