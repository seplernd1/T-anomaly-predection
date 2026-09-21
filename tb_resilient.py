#!/usr/bin/env python
"""Resilient ThingsBoard HTTP + extraction primitives.

Pure, dependency-light (stdlib + requests) building blocks used by the pull
pipeline:

* error taxonomy: AuthError / PermanentError / RetryExhaustedError
* http_get(): bounded retries with exponential backoff + jitter, Retry-After
  support for 429, retries for 429/5xx/timeouts/connection failures, NO retry
  for 401/403 (auth) or other 4xx (permanent)
* fetch_timeseries_paged(): cursor paging with API-limit (cap) detection,
  same-timestamp overlap + deterministic dedupe, stall guard
* resolve_window(): recursive time-window splitting when a chunk stays capped
* CoverageLedger: per device/key/window provenance records
* CheckpointStore: device+key+window resume state with atomic writes
* atomic_write_bytes / sanitize_csv_value / redact_evidence helpers
* DevicePolicy: auditable real-vs-test/demo/sim device classification with
  per-device reasons (never name-matching alone)

A zero-row response is only `empty_verified` when the full window resolved
without failures. Failed/unknown requests are never "no data".
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests


# ─── error taxonomy ──────────────────────────────────────────────────────────


class TbHttpError(Exception):
    """Base for classified HTTP failures."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body[:500]


class AuthError(TbHttpError):
    """401/403 — never retried. Caller must re-authenticate or fix credentials."""


class PermanentError(TbHttpError):
    """4xx (other than 429) — never retried."""


class RetryExhaustedError(TbHttpError):
    """Transient failure persisted past the retry budget."""

    def __init__(self, message: str, status: int | None = None, body: str = "",
                 attempts: int = 0, retries: int = 0):
        super().__init__(message, status, body)
        self.attempts = attempts
        self.retries = retries


# ─── retry policy ────────────────────────────────────────────────────────────


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    jitter: float = 0.25
    retry_statuses: frozenset = frozenset({429, 500, 502, 503, 504})
    respect_retry_after: bool = True
    retry_after_cap_s: float = 120.0


def _parse_retry_after(value: str | None, cap_s: float) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, min(float(value), cap_s))
    except (TypeError, ValueError):
        return None


def _backoff(policy: RetryPolicy, attempt: int) -> float:
    delay = min(policy.base_delay_s * (2.0 ** max(0, attempt - 1)), policy.max_delay_s)
    if policy.jitter:
        delay *= 1.0 + random.uniform(-policy.jitter, policy.jitter)
    return max(0.0, delay)


def http_get(
    session: Any,
    url: str,
    headers: dict[str, str] | None = None,
    timeout_s: int = 30,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, int]:
    """GET with classified retries. Returns (response, retries_used).

    Raises AuthError (401/403), PermanentError (other 4xx),
    RetryExhaustedError (transient failures past budget).
    """
    policy = policy or RetryPolicy()
    retries = 0
    last_status: int | None = None
    last_body = ""
    for attempt in range(1, policy.max_attempts + 1):
        try:
            resp = session.get(url, headers=headers or {}, timeout=timeout_s)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_status, last_body = None, f"{type(exc).__name__}: {exc}"[:300]
            if attempt >= policy.max_attempts:
                break
            retries += 1
            sleep(_backoff(policy, attempt))
            continue
        status = getattr(resp, "status_code", None)
        last_status = status
        try:
            last_body = (getattr(resp, "text", "") or "")[:300]
        except Exception:
            last_body = ""
        if status == 200:
            return resp, retries
        if status in (401, 403):
            raise AuthError(f"GET {url} -> HTTP {status} (auth, not retried)", status, last_body)
        if status == 429 or (status is not None and status in policy.retry_statuses):
            if attempt >= policy.max_attempts:
                break
            retries += 1
            wait: float | None = None
            if status == 429 and policy.respect_retry_after:
                try:
                    wait = _parse_retry_after(resp.headers.get("Retry-After"), policy.retry_after_cap_s)
                except Exception:
                    wait = None
            sleep(wait if wait is not None else _backoff(policy, attempt))
            continue
        if status is not None and 400 <= status < 500:
            raise PermanentError(f"GET {url} -> HTTP {status} (permanent, not retried)", status, last_body)
        # Any other unexpected status: treat as transient.
        if attempt >= policy.max_attempts:
            break
        retries += 1
        sleep(_backoff(policy, attempt))
    raise RetryExhaustedError(
        f"GET {url} failed after {policy.max_attempts} attempts (last status={last_status})",
        last_status, last_body, attempts=policy.max_attempts, retries=retries,
    )


# ─── paged timeseries fetch ──────────────────────────────────────────────────


@dataclass
class PageOutcome:
    points: list[dict[str, Any]] = field(default_factory=list)
    pages: int = 0
    retries: int = 0
    capped: bool = False  # last page hit the API limit -> window may be truncated
    cap_hits: int = 0
    oldest_ts: int | None = None
    newest_ts: int | None = None
    stalled: bool = False


def _point_key(ts: Any, value: Any) -> tuple[str, str]:
    try:
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        canonical = str(value)
    return (str(ts), canonical)


# Observed ThingsBoard server-side truncation: responses are cut at ~5000
# points even when a larger `limit` is requested. A batch is therefore
# *potentially truncated* when it reaches min(requested limit, server cap).
SERVER_SIDE_CAP = 5000


def fetch_timeseries_paged(
    page_fn: Callable[[int, int, int], tuple[list[dict[str, Any]], int]],
    start_ms: int,
    end_ms: int,
    limit: int,
    max_pages: int = 10_000,
) -> PageOutcome:
    """Page [start_ms, end_ms] via page_fn(cursor, end, limit) -> (batch, retries).

    Cap handling: a batch reaching min(limit, SERVER_SIDE_CAP) is treated as
    *potentially truncated*. The next cursor overlaps at last_ts (never
    last_ts+1) and points are deduped on (ts, value), so same-timestamp
    records are never skipped and adjacent chunks never duplicate. `capped`
    reflects the TERMINAL page: False once a short page proves exhaustion.
    A cursor that cannot advance triggers one escalated-limit recovery probe;
    if the server still returns a full probe page the window is marked
    stalled/partial instead of looping forever.
    """
    out = PageOutcome()
    seen: set[tuple[str, str]] = set()
    threshold = min(limit, SERVER_SIDE_CAP)
    cursor = start_ms
    out.capped = False
    while cursor <= end_ms and out.pages < max_pages:
        batch, retries = page_fn(cursor, end_ms, limit)
        out.pages += 1
        out.retries += retries
        fresh = [p for p in batch if _point_key(p.get("ts"), p.get("value")) not in seen]
        for p in fresh:
            seen.add(_point_key(p.get("ts"), p.get("value")))
        out.points.extend(fresh)
        if not batch:
            out.capped = False
            break
        numeric_ts = [int(p["ts"]) for p in batch
                      if isinstance(p.get("ts"), (int, float)) or str(p.get("ts", "")).isdigit()]
        last_ts = max(numeric_ts) if numeric_ts else None
        if len(batch) < threshold:
            out.capped = False
            break
        # Full batch: may be truncated.
        out.cap_hits += 1
        out.capped = True
        if last_ts is None or last_ts < cursor:
            out.stalled = True
            break
        if last_ts == cursor and not fresh:
            # Same-timestamp overflow: escalate the limit once to pull the
            # whole same-ts burst through a single response.
            probe_limit = min(max(limit * 10, SERVER_SIDE_CAP), 100_000)
            pbatch, pretries = page_fn(cursor, end_ms, probe_limit)
            out.pages += 1
            out.retries += pretries
            fresh2 = [p for p in pbatch if _point_key(p.get("ts"), p.get("value")) not in seen]
            for p in fresh2:
                seen.add(_point_key(p.get("ts"), p.get("value")))
            out.points.extend(fresh2)
            if len(pbatch) < probe_limit:
                # Server exhausted: everything reachable was retrieved.
                out.capped = False
                break
            out.stalled = True
            break
        cursor = last_ts  # overlap, dedupe keeps it exact
    ts_vals = [int(p["ts"]) for p in out.points
               if isinstance(p.get("ts"), (int, float)) or str(p.get("ts", "")).isdigit()]
    if ts_vals:
        out.oldest_ts, out.newest_ts = min(ts_vals), max(ts_vals)
    return out


@dataclass
class WindowOutcome:
    points: list[dict[str, Any]] = field(default_factory=list)
    completeness: str = "unknown"  # complete | empty_verified | partial | failed
    pages: int = 0
    retries: int = 0
    cap_hits: int = 0
    oldest_ts: int | None = None
    newest_ts: int | None = None
    error: str = ""


def resolve_window(
    page_fn: Callable[[int, int, int], tuple[list[dict[str, Any]], int]],
    start_ms: int,
    end_ms: int,
    limit: int,
    min_chunk_ms: int = 3600_000,
    depth: int = 0,
    max_depth: int = 12,
) -> WindowOutcome:
    """Fetch [start_ms, end_ms]; split the window while pages stay capped.

    Returns complete/empty_verified only when the ENTIRE window resolved.
    A chunk that stays capped at min_chunk_ms (or max depth) is partial —
    never silently complete.
    """
    if start_ms > end_ms:
        return WindowOutcome(completeness="empty_verified")
    if depth > max_depth:
        return WindowOutcome(completeness="partial", error="max split depth reached")
    try:
        page = fetch_timeseries_paged(page_fn, start_ms, end_ms, limit)
    except (AuthError, PermanentError, RetryExhaustedError) as exc:
        return WindowOutcome(completeness="failed", error=f"{type(exc).__name__}: {exc}"[:300])
    if not page.capped or page.stalled:
        status = "complete" if page.points else "empty_verified"
        if page.stalled and page.capped:
            status = "partial"
        return WindowOutcome(
            points=page.points, completeness=status, pages=page.pages,
            retries=page.retries, cap_hits=page.cap_hits,
            oldest_ts=page.oldest_ts, newest_ts=page.newest_ts,
            error="cursor stalled on capped page" if (page.stalled and page.capped) else "",
        )
    if (end_ms - start_ms) <= min_chunk_ms:
        return WindowOutcome(
            points=page.points, completeness="partial", pages=page.pages,
            retries=page.retries, cap_hits=page.cap_hits,
            oldest_ts=page.oldest_ts, newest_ts=page.newest_ts,
            error=f"capped at min chunk ({min_chunk_ms} ms)",
        )
    mid = start_ms + (end_ms - start_ms) // 2
    left = resolve_window(page_fn, start_ms, mid, limit, min_chunk_ms, depth + 1, max_depth)
    right = resolve_window(page_fn, mid + 1, end_ms, limit, min_chunk_ms, depth + 1, max_depth)
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for p in list(left.points) + list(right.points):
        k = _point_key(p.get("ts"), p.get("value"))
        if k not in seen:
            seen.add(k)
            merged.append(p)
    merged.sort(key=lambda p: (str(p.get("ts", "")), str(p.get("value", ""))[:50]))
    completeness = "complete" if (
        left.completeness in {"complete", "empty_verified"}
        and right.completeness in {"complete", "empty_verified"}
    ) else "partial"
    if completeness == "complete" and not merged:
        completeness = "empty_verified"
    ts_vals = [int(p["ts"]) for p in merged
               if isinstance(p.get("ts"), (int, float)) or str(p.get("ts", "")).isdigit()]
    return WindowOutcome(
        points=merged, completeness=completeness,
        pages=left.pages + right.pages, retries=left.retries + right.retries,
        cap_hits=left.cap_hits + right.cap_hits + page.cap_hits,
        oldest_ts=min(ts_vals) if ts_vals else None,
        newest_ts=max(ts_vals) if ts_vals else None,
        error="; ".join(e for e in (left.error, right.error) if e)[:300],
    )


# ─── coverage ledger ─────────────────────────────────────────────────────────


@dataclass
class CoverageRecord:
    run_id: str
    device_id: str
    key: str
    endpoint: str
    requested_start_ms: int
    requested_end_ms: int
    chunk_start_ms: int
    chunk_end_ms: int
    pages: int = 0
    status_code: int | None = None
    retries: int = 0
    rows: int = 0
    oldest_ts: int | None = None
    newest_ts: int | None = None
    retrieved_at: str = ""
    completeness: str = "unknown"  # complete|empty_verified|retrying|partial|failed|unknown
    error: str = ""


class CoverageLedger:
    """Append-only per device/key/window provenance. JSONL on disk."""

    def __init__(self, run_id: str, path: Path):
        self.run_id = run_id
        self.path = Path(path)
        self.records: list[CoverageRecord] = []
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.records.append(CoverageRecord(**json.loads(line)))
                        except Exception:
                            continue

    def add(self, record: CoverageRecord) -> None:
        if not record.retrieved_at:
            record.retrieved_at = datetime.now(timezone.utc).isoformat()
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        retries = 0
        rows = 0
        for r in self.records:
            by_status[r.completeness] = by_status.get(r.completeness, 0) + 1
            retries += r.retries
            rows += r.rows
        devices = len({r.device_id for r in self.records})
        keys = len({(r.device_id, r.key) for r in self.records})
        return {"records": len(self.records), "devices": devices, "device_keys": keys,
                "rows": rows, "retries": retries, "by_status": by_status}

    def incomplete(self) -> list[CoverageRecord]:
        return [r for r in self.records if r.completeness in {"partial", "failed", "unknown", "retrying"}]


# ─── checkpoints ─────────────────────────────────────────────────────────────


class CheckpointStore:
    """device+key+window resume state. Atomic JSON persistence."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.done: set[str] = set()
        if self.path.is_file():
            try:
                self.done = set(json.loads(self.path.read_text(encoding="utf-8")).get("done", []))
            except Exception:
                self.done = set()

    @staticmethod
    def key(device_id: str, key: str, chunk_start_ms: int, chunk_end_ms: int) -> str:
        return f"{device_id}|{key}|{chunk_start_ms}|{chunk_end_ms}"

    def is_done(self, device_id: str, key: str, chunk_start_ms: int, chunk_end_ms: int) -> bool:
        return self.key(device_id, key, chunk_start_ms, chunk_end_ms) in self.done

    def mark_done(self, device_id: str, key: str, chunk_start_ms: int, chunk_end_ms: int) -> None:
        self.done.add(self.key(device_id, key, chunk_start_ms, chunk_end_ms))
        self.save()

    def save(self) -> None:
        atomic_write_bytes(self.path, json.dumps({"done": sorted(self.done)}).encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ─── output safety ───────────────────────────────────────────────────────────


def sanitize_csv_value(value: Any) -> str:
    """Neutralize spreadsheet-formula injection; stringify nested values."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=True, default=str)
    else:
        text = str(value)
    if text[:1] in {"=", "+", "-", "@"}:
        return "'" + text
    return text


def write_csv_rows_safe(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """CSV writer with formula sanitization + atomic write."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: sanitize_csv_value(row.get(k)) for k in fieldnames})
    atomic_write_bytes(Path(path), buf.getvalue().encode("utf-8"))


_HASH_RE = re.compile(r"^[0-9a-f]{8,}$")


def redact_evidence(payload: dict[str, Any], keep_keys: Iterable[str]) -> dict[str, Any]:
    """Keep allowlisted keys verbatim; replace everything else with a hash ref.

    Restricted raw evidence stays joinable/debuggable without raw secrets,
    GPS, IMEIs or payloads ever reaching ML artifacts.
    """
    keep = set(keep_keys)
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if k in keep:
            out[k] = v
            continue
        digest = hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()[:12]
        out[k] = f"hash:{digest}"
    return out


# ─── TLS policy ────────────────────────────────────────────────────────────────


def resolve_verify_tls(insecure_flag: bool = False, legacy_verify: bool = False,
                       warn: Callable[[str], None] | None = None) -> bool:
    """TLS verification is ON by default; insecure mode is explicit opt-in.

    insecure_flag (--insecure-skip-tls-verify / TB_INSECURE_TLS=1) disables
    verification with a loud warning. Otherwise legacy_verify (--verify-tls /
    TB_VERIFY_TLS) is honored; default is verify=True.
    """
    import sys as _sys

    if insecure_flag or os.getenv("TB_INSECURE_TLS", "").strip().lower() in {"1", "true", "yes"}:
        msg = ("WARNING: TLS certificate verification DISABLED. Use only on trusted "
               "networks; fix the tenant certificate instead.")
        (warn or (lambda m: print(m, file=_sys.stderr)))(msg)
        return False
    return True


# ─── device eligibility policy ───────────────────────────────────────────────


@dataclass
class DeviceDecision:
    eligible: bool
    reason: str


DEFAULT_TEST_NAME_PATTERNS = (
    r"test", r"demo", r"simul", r"synth", r"\blab\b", r"\bqa\b",
    r"staging", r"staging-", r"dummy", r"sample",
)

DEFAULT_TEST_TYPES = (
    "test", "demo", "simulator", "synthetic",
)


class DevicePolicy:
    """Auditable real-vs-test/demo/simulator classification.

    Never name-matching alone: name patterns, device type/profile, customer
    association and observed activity (telemetry key count) all contribute,
    and every exclusion carries an explicit reason.
    """

    def __init__(
        self,
        test_name_patterns: Iterable[str] | None = None,
        test_types: Iterable[str] | None = None,
        require_customer: bool = False,
        min_keys: int = 0,
        explicit_allow: Iterable[str] | None = None,
        explicit_deny: Iterable[str] | None = None,
    ):
        self.test_name_res = [re.compile(p, re.I) for p in (test_name_patterns or DEFAULT_TEST_NAME_PATTERNS)]
        self.test_types = {t.lower() for t in (test_types or DEFAULT_TEST_TYPES)}
        self.require_customer = require_customer
        self.min_keys = min_keys
        self.explicit_allow = set(explicit_allow or ())
        self.explicit_deny = set(explicit_deny or ())

    @classmethod
    def from_env(cls) -> "DevicePolicy":
        pats = os.getenv("REAL_DEVICE_NAME_PATTERNS", "")
        return cls(
            test_name_patterns=[p.strip() for p in pats.split(",") if p.strip()] or None,
            require_customer=os.getenv("REAL_DEVICE_REQUIRE_CUSTOMER", "").lower() in {"1", "true", "yes"},
            min_keys=int(os.getenv("REAL_DEVICE_MIN_KEYS", "0") or 0),
            explicit_allow=[x.strip() for x in os.getenv("REAL_DEVICE_ALLOW", "").split(",") if x.strip()],
            explicit_deny=[x.strip() for x in os.getenv("REAL_DEVICE_DENY", "").split(",") if x.strip()],
        )

    def decide(self, device: dict[str, Any], key_count: int | None = None) -> DeviceDecision:
        did = str(device.get("device_id", ""))
        name = str(device.get("name", ""))
        dtype = str(device.get("type", "")).lower()
        if did in self.explicit_allow:
            return DeviceDecision(True, "explicit_allow")
        if did in self.explicit_deny:
            return DeviceDecision(False, "explicit_deny")
        if dtype in self.test_types:
            return DeviceDecision(False, f"test_type:{dtype}")
        for rx in self.test_name_res:
            if rx.search(name):
                # Name hit alone is a *candidate*; confirm with a second signal.
                second = dtype or "unknown-type"
                if "test" in second or "demo" in second or "sim" in second:
                    return DeviceDecision(False, f"name_pattern+type:{rx.pattern}+{dtype}")
                if key_count is not None and key_count == 0:
                    return DeviceDecision(False, f"name_pattern+no_keys:{rx.pattern}")
                return DeviceDecision(True, f"name_pattern_only:{rx.pattern}|kept_pending_review")
        if self.require_customer and not str(device.get("customer_id", "")):
            return DeviceDecision(False, "no_customer")
        if self.min_keys and (key_count or 0) < self.min_keys:
            return DeviceDecision(False, f"inactive:min_keys={self.min_keys}")
        return DeviceDecision(True, "real_device")
