#!/usr/bin/env python
"""Leakage-free training-data pipeline for device anomaly / outage-risk modelling.

Two tracks share one feature matrix:

  (A) Unsupervised anomaly detection - features only, no labels required.
      Usable as soon as telemetry is extracted (`anomaly_eligible == True`).

  (B) 24-hour outage risk - `y_outage_24h`. Emitted ONLY for samples whose
      future 24h window is covered by an INDEPENDENT verified evidence source
      (ThingsBoard lifecycle connect/disconnect events, or offline/no-data
      alarms). Everything else is `label_status == "censored"` with a reason.
      If no verified evidence exists at all, EVERY sample is censored and the
      run stops with a clear report instead of manufacturing labels.

Hard rules enforced by this module
----------------------------------
* Read-only database access: every statement must be SELECT/WITH and the
  session is opened with `default_transaction_read_only = on`.
* Bounded extraction: daily partitions, batched by key, with a row cap. The
  compressed telemetry hypertable is never aggregated whole.
* Point-in-time features: a feature row at anchor T is computed only from
  records with `time <= T`. `strict=True` raises `FeatureLeakageError` if a
  future row is supplied, and every emitted sample records `max_source_ts`
  which is asserted to be <= `anchor_ts`.
* No indefinite forward-fill: carry-forward is capped by `fill_cap_hours`;
  beyond the cap the value is NaN and an explicit feature-age field remains.
* Identifiers, GPS, secrets, raw payloads, rule-derived scores, severity and
  alarm flags, and target-like attributes are excluded from model features
  (see `exclusions.patterns` in training_config.json).
* Chronological splits grouped by customer/branch/device with a purge gap.
  Rows are never split at random.
* No credentials are ever printed, logged, or written to outputs.

Usage
-----
    python build_training_dataset.py --check-db
    python build_training_dataset.py --plan-only
    python build_training_dataset.py --start 2026-01-01 --end 2026-02-01
    python build_training_dataset.py --skip-extract          # reuse staging
    python build_training_dataset.py --report-only

Environment
-----------
    TRAIN_DB_URL   postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DB?sslmode=...
    (or the standard PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD variables)

Nothing else is required. The URL is never echoed; only a redacted descriptor
(host/db/user masks) is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("build_training_dataset")

UTC = timezone.utc
DEFAULT_CONFIG_PATH = Path("training_config.json")

STRONG = "strong"
WEAK = "weak"
NO_EVIDENCE = "none"

# Non-feature columns. Everything else in anomaly_samples.parquet is a feature.
META_COLUMNS = (
    "device_id",
    "anchor_ts",
    "customer_id",
    "branch_id",
    "group_key",
    "split",
    "eligibility",
    "anomaly_eligible",
    "feature_ready",
    "label_status",
    "censor_reason",
    "y_outage_24h",
    "label_horizon_end",
    "max_source_ts",
    "n_rows_used",
    "n_rows_ignored_future",
    "n_feature_values",
)

# Censor / eligibility reasons.
R_NO_HISTORY = "no_telemetry_history"
R_STALE = "stale_data"
R_NO_VERIFIED_SOURCE = "no_verified_outage_source"
R_UNVERIFIED_NEGATIVE = "unverified_negative_coverage"
R_OPEN_WINDOW = "open_outage_window_unverified"
R_HORIZON_INCOMPLETE = "future_label_coverage_incomplete"
R_INSIDE_OUTAGE = "inside_verified_outage"
R_PURGE = "split_purge_gap"
R_NOT_READY = "features_unusable"

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_PURGED = "purged"
VALID_SPLITS = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)

_READ_PREFIXES = ("select", "with", "show", "explain", "values", "table")
_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|create|alter|truncate|grant|revoke|vacuum|"
    r"reindex|copy|call|do|refresh|cluster|comment|set|reset|lock|listen|"
    r"notify|prepare|execute|discard)\b",
    re.I,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "db": {
        "url_env": "TRAIN_DB_URL",
        "statement_timeout_ms": 900000,
        "lock_timeout_ms": 30000,
        "idle_in_transaction_timeout_ms": 60000,
    },
    "tables": {
        "telemetry": "public.device_telemetry",
        "events": "public.device_event",
        "hierarchy_node": "public.hierarchy_node",
        "branch_identity": "public.branch_identity",
        "branch_ancestor_path": "public.branch_ancestor_path",
        "customer": "public.customer",
    },
    "columns": {
        "time": "time",
        "device_id": "device_id",
        "key": "key",
        "tenant_id": "tenant_id",
        "customer_id": "customer_id",
        "value_num": "value_num",
        "value_text": "value_text",
        "event_pk": "id",
        "event_id": "event_id",
        "event_type": "event_type",
        "event_payload": "payload",
    },
    "extraction": {
        "window_days": 1,
        "key_batch_size": 8,
        "device_batch_size": 50,
        "row_cap_per_query": 750000,
        "on_row_cap": "error",
        "key_discovery": "per_day_aggregate",
        "static_keys": [],
        "max_partitions": None,
        "quarantine": {
            "min_valid_ts": "2000-01-01T00:00:00+00:00",
            "future_slack_hours": 6,
        },
    },
    "anchors": {"every_hours": 6, "min_history_hours": 12, "align_utc": True},
    "features": {
        "numeric_windows_hours": [1, 6, 24, 168],
        "text_windows_hours": [24],
        "expected_interval_s": 300,
        "min_count_for_stats": 6,
        "fill_cap_hours": 6,
        "stale_after_hours": 24,
        "max_feature_keys": 25,
        "max_text_keys": 8,
        "watchlist_keys": [],
        "force_include_keys": [],
        "force_exclude_keys": [],
    },
    "exclusions": {"patterns": []},
    "labels": {
        "horizon_hours": 24,
        "min_outage_minutes": 5,
        "require_closed_windows": True,
        "event_types_of_interest": ["LC_EVENT", "ERROR", "ALARM", "ALARM_UPDATE"],
        "lifecycle_offline_methods": ["ondisconnect", "disconnect", "inactive", "offline"],
        "lifecycle_online_methods": ["onconnect", "connect", "onactivity", "activity"],
        "offline_regex": "offline|no[_ -]?data|inactiv|disconnect|heartbeat[_ -]?stop|down",
        "online_regex": "online|connected|activity|restored|reconnect|up",
        "evidence_strength": {
            "lc_disconnect": STRONG,
            "lc_connect": STRONG,
            "alarm_offline": STRONG,
            "alarm_cleared": STRONG,
            "error_event": WEAK,
            "current_attr_snapshot": WEAK,
            "unknown": NO_EVIDENCE,
        },
        "negative_requires_device_evidence": True,
        "min_device_evidence_span_days": 7,
        "payload_key_allowlist": ["method", "status", "msg", "type", "currentAttr"],
    },
    "splits": {
        "mode": "group",
        "group_by": "customer",
        "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "purge_hours": "auto",
    },
    "output": {
        "dir": "training_data",
        "anomaly_samples": "anomaly_samples.parquet",
        "evidence_audit": "outage_evidence_audit.parquet",
        "report_md": "data_quality_report.md",
        "metrics_json": "data_quality_metrics.json",
        "staging_dir": "staging",
    },
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class TrainingDataError(RuntimeError):
    """Base error for this module."""


class ReferenceViolation(TrainingDataError):
    """A statement that is not read-only was attempted."""


class FeatureLeakageError(TrainingDataError):
    """A record after the anchor timestamp reached a feature computation."""


class NoVerifiedLabelsError(TrainingDataError):
    """Verified outage evidence is absent; supervised labels cannot be built."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    p = Path(path)
    if not p.exists():
        LOG.warning("config %s not found; using built-in defaults", p)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with p.open("r", encoding="utf-8") as fh:
        user_cfg = json.load(fh)
    return deep_merge(DEFAULT_CONFIG, user_cfg)


# ---------------------------------------------------------------------------
# Redaction helpers (never print credentials)
# ---------------------------------------------------------------------------
def redact_url(url: str | None) -> str:
    """Return a safe descriptor for a connection URL. Never returns secrets."""
    if not url:
        return "<unset>"
    safe = re.sub(r"://[^@/]*@", "://***:***@", url)
    safe = re.sub(r"(password|passwd|pwd)=[^&;\s]+", r"\1=***", safe, flags=re.I)
    return safe


def connection_target(cfg: dict[str, Any]) -> tuple[str | None, str]:
    """Resolve the DB URL from the environment. Returns (url, redacted_target)."""
    env_name = cfg["db"]["url_env"]
    url = os.environ.get(env_name, "").strip()
    if url:
        return url, redact_url(url)

    host = os.environ.get("PGHOST", "").strip()
    user = os.environ.get("PGUSER", "").strip()
    db = os.environ.get("PGDATABASE", "").strip()
    if host or user or db:
        port = os.environ.get("PGPORT", "5432").strip() or "5432"
        return "pgparts", f"PGHOST={host or '<unset>'} PGPORT={port} PGDATABASE={db or '<unset>'} PGUSER={user or '<unset>'} (password from env)"
    return None, f"<{env_name} unset and no PG* variables>"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def to_utc(ts: Any) -> pd.Timestamp | None:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    try:
        out = pd.Timestamp(ts)
    except Exception:
        return None
    if out is pd.NaT or pd.isna(out):
        return None
    return out.tz_localize(UTC) if out.tzinfo is None else out.tz_convert(UTC)


def ensure_utc_series(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce", utc=True)
    return out


def parse_iso(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return to_utc(pd.Timestamp(value))


def day_windows(start: pd.Timestamp, end: pd.Timestamp, window_days: int = 1) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield bounded [t0, t1) partitions. Used for every telemetry/event read."""
    step = timedelta(days=max(1, int(window_days)))
    cursor = start
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


def hour_grid(start: pd.Timestamp, end: pd.Timestamp, every_hours: int, align_utc: bool = True) -> pd.DatetimeIndex:
    every = max(1, int(every_hours))
    if align_utc:
        floor_hours = (start.hour // every) * every
        cursor = start.normalize() + timedelta(hours=floor_hours)
        while cursor < start:
            cursor += timedelta(hours=every)
    else:
        cursor = start
    stamps: list[pd.Timestamp] = []
    while cursor < end:
        stamps.append(cursor)
        cursor += timedelta(hours=every)
    return pd.DatetimeIndex(stamps)


# ---------------------------------------------------------------------------
# Read-only SQL guard
# ---------------------------------------------------------------------------
def assert_read_only_sql(sql: str) -> str:
    """Reject anything that is not a single read-only statement."""
    text = (sql or "").strip()
    while text.endswith(";"):
        text = text[:-1].strip()
    if not text:
        raise ReferenceViolation("empty statement")
    lowered = text.lower()
    if not lowered.startswith(_READ_PREFIXES):
        raise ReferenceViolation(f"refusing non-read-only statement: {text[:60]!r}")
    match = _WRITE_KEYWORDS.search(text)
    if match:
        raise ReferenceViolation(f"refusing statement containing {match.group(1)!r}: {text[:60]!r}")
    return text


@dataclass
class ReadOnlyDb:
    """Minimal read-only Postgres access layer.

    `executor` can be injected for tests; otherwise SQLAlchemy/psycopg2 is used.
    """

    url: str | None = None
    cfg: dict[str, Any] | None = None
    executor: Callable[[str, dict[str, Any]], list[dict[str, Any]]] | None = None
    _engine: Any = None

    def connect(self) -> None:
        if self.executor is not None or self._engine is not None:
            return
        if not self.url or self.url == "pgparts":
            raise TrainingDataError("no database URL available; set the env var named in db.url_env")
        try:
            from sqlalchemy import create_engine, event
        except Exception as exc:  # pragma: no cover - dependency guard
            raise TrainingDataError(f"SQLAlchemy is required for live extraction: {exc}") from exc

        db_cfg = (self.cfg or DEFAULT_CONFIG)["db"]
        options = " ".join(
            [
                "-c default_transaction_read_only=on",
                f"-c statement_timeout={int(db_cfg.get('statement_timeout_ms', 900000))}",
                f"-c lock_timeout={int(db_cfg.get('lock_timeout_ms', 30000))}",
                f"-c idle_in_transaction_session_timeout={int(db_cfg.get('idle_in_transaction_timeout_ms', 60000))}",
            ]
        )
        engine = create_engine(
            self.url,
            future=True,
            pool_pre_ping=True,
            connect_args={"options": options},
        )

        @event.listens_for(engine, "connect")
        def _force_read_only(dbapi_conn, _record):  # pragma: no cover - live only
            try:
                cur = dbapi_conn.cursor()
                cur.execute("SET default_transaction_read_only = on")
                cur.close()
            except Exception:
                pass

        self._engine = engine

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        safe_sql = assert_read_only_sql(sql)
        params = params or {}
        self.connect()
        if self.executor is not None:
            rows = self.executor(safe_sql, params)
            return pd.DataFrame(rows)
        from sqlalchemy import text

        with self._engine.connect() as conn:
            result = conn.execute(text(safe_sql), params)
            rows = [dict(r._mapping) for r in result]
        return pd.DataFrame(rows)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------
def quarantine_telemetry(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Split raw telemetry into clean rows and quarantined rows with reasons."""
    cols = cfg["columns"]
    tcol, dcol, kcol = cols["time"], cols["device_id"], cols["key"]
    vnum, vtxt = cols["value_num"], cols["value_text"]
    q = cfg["extraction"]["quarantine"]
    min_valid = to_utc(q["min_valid_ts"]) or pd.Timestamp("2000-01-01", tz=UTC)
    future_limit = pd.Timestamp.now(tz=UTC) + timedelta(hours=float(q["future_slack_hours"]))

    stats = {
        "rows_in": int(len(df)),
        "unparseable_timestamp": 0,
        "null_timestamp": 0,
        "epoch_like_timestamp": 0,
        "future_timestamp": 0,
        "no_value": 0,
        "duplicate_device_key_time": 0,
        "rows_clean": 0,
    }
    if df.empty:
        return df.copy(), df.copy(), stats

    out = df.copy()
    out["_ts"] = ensure_utc_series(out[tcol])
    out["_reason"] = pd.Series([""] * len(out), index=out.index)

    raw_blank = out[tcol].isna()
    parsed_blank = out["_ts"].isna()
    out.loc[parsed_blank & raw_blank, "_reason"] = "null_timestamp"
    out.loc[parsed_blank & ~raw_blank, "_reason"] = "unparseable_timestamp"
    stats["null_timestamp"] = int((parsed_blank & raw_blank).sum())
    stats["unparseable_timestamp"] = int((parsed_blank & ~raw_blank).sum())

    epoch_like = out["_ts"].notna() & (out["_ts"] < min_valid)
    out.loc[epoch_like & (out["_reason"] == ""), "_reason"] = "epoch_like_timestamp"
    stats["epoch_like_timestamp"] = int(epoch_like.sum())

    future = out["_ts"].notna() & (out["_ts"] > future_limit)
    out.loc[future & (out["_reason"] == ""), "_reason"] = "future_timestamp"
    stats["future_timestamp"] = int(future.sum())

    num_nan = pd.to_numeric(out.get(vnum), errors="coerce").isna() if vnum in out else pd.Series(True, index=out.index)
    txt_nan = out[vtxt].isna() if vtxt in out else pd.Series(True, index=out.index)
    no_value = num_nan & txt_nan
    out.loc[no_value & (out["_reason"] == ""), "_reason"] = "no_value"
    stats["no_value"] = int(no_value.sum())

    valid = out["_reason"] == ""
    clean = out[valid].copy()
    dup_mask = clean.duplicated(subset=[dcol, kcol, "_ts"], keep="last")
    if dup_mask.any():
        dup_rows = clean[dup_mask].copy()
        dup_rows["_reason"] = "duplicate_device_key_time"
        stats["duplicate_device_key_time"] = int(dup_mask.sum())
    else:
        dup_rows = clean.iloc[0:0].copy()
    clean = clean[~dup_mask].copy()

    clean["value_kind"] = np.where(
        pd.to_numeric(clean.get(vnum), errors="coerce").notna(), "numeric", "text"
    )
    clean["value_num"] = pd.to_numeric(clean.get(vnum), errors="coerce")
    clean["value_text"] = clean.get(vtxt)
    clean = clean.rename(columns={tcol: "time", dcol: "device_id", kcol: "key"})
    if cols.get("customer_id") in clean.columns:
        clean = clean.rename(columns={cols["customer_id"]: "customer_id"})

    quarantined = pd.concat([out[~valid], dup_rows], ignore_index=True)
    if not quarantined.empty:
        quarantined = quarantined.rename(columns={tcol: "time", dcol: "device_id", kcol: "key"})
        quarantined["quarantine_reason"] = quarantined["_reason"]

    stats["rows_clean"] = int(len(clean))
    keep = ["time", "device_id", "key", "customer_id", "value_kind", "value_num", "value_text"]
    clean = clean[[c for c in keep if c in clean.columns]]
    qkeep = ["time", "device_id", "key", "quarantine_reason"]
    if not quarantined.empty:
        quarantined = quarantined[[c for c in qkeep if c in quarantined.columns]]
    return clean.reset_index(drop=True), quarantined.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Feature exclusions
# ---------------------------------------------------------------------------
def excluded_reason(name: str, cfg: dict[str, Any]) -> str | None:
    for pattern, reason in cfg["exclusions"]["patterns"]:
        if re.search(pattern, str(name), re.I):
            return reason
    return None


def find_exclusion_violations(feature_columns: Sequence[str], cfg: dict[str, Any]) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for col in feature_columns:
        reason = excluded_reason(col, cfg)
        if reason:
            hits.append((col, reason))
    return hits


def safe_name(key: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(key)).strip("_").lower()
    return cleaned or "unnamed"


def select_feature_keys(
    key_stats: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Pick numeric/text feature keys from per-key stats, minus excluded keys."""
    f = cfg["features"]
    force_in = set(f.get("force_include_keys") or [])
    force_out = set(f.get("force_exclude_keys") or [])
    watchlist = set(f.get("watchlist_keys") or [])

    numeric: list[tuple[str, float]] = []
    text: list[tuple[str, float]] = []
    excluded: dict[str, list[str]] = {"identifier": [], "excluded_other": []}

    if key_stats is None or key_stats.empty:
        return [], [], excluded

    for row in key_stats.itertuples(index=False):
        key = str(getattr(row, "key"))
        n_num = float(getattr(row, "n_numeric", 0) or 0)
        n_txt = float(getattr(row, "n_text", 0) or 0)
        if key in force_out:
            excluded["excluded_other"].append(key)
            continue
        if key not in force_in:
            reason = excluded_reason(key, cfg)
            if reason:
                bucket = "identifier" if reason in {"identifier", "network_identifier", "gps_location", "secret"} else "excluded_other"
                excluded[bucket].append(key)
                continue
        if n_num >= n_txt and n_num > 0:
            numeric.append((key, n_num))
        elif n_txt > 0:
            text.append((key, n_txt))

    def rank(items: list[tuple[str, float]]) -> list[str]:
        ordered = sorted(items, key=lambda kv: (-kv[1], kv[0]))
        picked = [k for k, _ in ordered if k in watchlist]
        picked += [k for k, _ in ordered if k not in watchlist]
        return picked

    numeric_keys = rank(numeric)[: int(f["max_feature_keys"])]
    text_keys = rank(text)[: int(f["max_text_keys"])]
    return numeric_keys, text_keys, excluded


def feature_column_names(numeric_keys: Sequence[str], text_keys: Sequence[str], cfg: dict[str, Any]) -> list[str]:
    f = cfg["features"]
    windows = [int(w) for w in f["numeric_windows_hours"]]
    text_windows = [int(w) for w in f["text_windows_hours"]]
    cols: list[str] = []
    for key in numeric_keys:
        col = safe_name(key)
        for w in windows:
            cols += [
                f"{col}__n_{w}h",
                f"{col}__mean_{w}h",
                f"{col}__min_{w}h",
                f"{col}__max_{w}h",
                f"{col}__std_{w}h",
                f"{col}__slope_{w}h",
                f"{col}__missing_frac_{w}h",
            ]
        cols += [f"{col}__last", f"{col}__last_age_h", f"{col}__stale", f"{col}__ffill_capped"]
    for key in text_keys:
        col = safe_name(key)
        for w in text_windows:
            cols += [
                f"{col}__text_n_{w}h",
                f"{col}__text_distinct_{w}h",
                f"{col}__text_changes_{w}h",
            ]
        cols += [
            f"{col}__text_last_age_h",
            f"{col}__text_last_len",
            f"{col}__text_bad_flag",
            f"{col}__text_last_numeric",
        ]
    cols += [
        "g__n_rows_total",
        "g__n_rows_24h",
        "g__n_keys_24h",
        "g__n_numeric_keys",
        "g__n_text_keys",
        "g__last_age_h",
        "g__stale_key_frac",
        "g__missing_frac_mean_24h",
    ]
    return cols


# ---------------------------------------------------------------------------
# Feature computation (point-in-time)
# ---------------------------------------------------------------------------
def _stats_block(times: pd.Series, values: pd.Series, anchor: pd.Timestamp, window_s: float,
                 min_n: int, expected_interval_s: float) -> dict[str, float]:
    if window_s > 0:
        start = anchor - timedelta(seconds=window_s)
        mask = times > start
        t = times[mask]
        v = values[mask]
    else:
        t, v = times, values
    n = int(len(v))
    out = {
        "n": float(n),
        "mean": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "std": float("nan"),
        "slope": float("nan"),
        "missing_frac": float("nan"),
    }
    expected = max(1.0, window_s / max(1.0, expected_interval_s))
    out["missing_frac"] = float(max(0.0, min(1.0, 1.0 - (n / expected))))
    if n:
        arr = v.astype(float).to_numpy()
        out["mean"] = float(np.nanmean(arr))
        out["min"] = float(np.nanmin(arr))
        out["max"] = float(np.nanmax(arr))
    if n >= max(2, min_n) and n > 1:
        arr = v.astype(float).to_numpy()
        out["std"] = float(np.nanstd(arr, ddof=1)) if n > 1 else float("nan")
        ns = t.astype("int64").to_numpy(dtype="float64")
        x = (ns - ns[0]) / 3.6e12  # hours
        if len(x) > 1 and not np.allclose(x, x[0]):
            try:
                out["slope"] = float(np.polyfit(x, arr, 1)[0])
            except Exception:
                out["slope"] = float("nan")
    return out


def compute_features_at(
    anchor_ts: Any,
    telemetry: pd.DataFrame,
    cfg: dict[str, Any],
    numeric_keys: Sequence[str],
    text_keys: Sequence[str],
    strict: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute one point-in-time feature row at `anchor_ts`.

    Only rows with `time <= anchor_ts` may enter. With `strict=True` a future row
    raises `FeatureLeakageError`; otherwise future rows are dropped and counted
    in `meta["n_rows_ignored_future"]`.
    """
    anchor = to_utc(anchor_ts)
    if anchor is None:
        raise TrainingDataError("anchor_ts is required")

    f = cfg["features"]
    horizon_s = float(f["stale_after_hours"]) * 3600.0
    fill_cap_s = float(f["fill_cap_hours"]) * 3600.0
    min_n = int(f["min_count_for_stats"])
    expected_interval_s = float(f["expected_interval_s"])
    numeric_windows = [int(w) for w in f["numeric_windows_hours"]]
    text_windows = [int(w) for w in f["text_windows_hours"]]

    if telemetry is None or telemetry.empty:
        return {}, {
            "max_source_ts": pd.NaT,
            "n_rows_used": 0,
            "n_rows_ignored_future": 0,
            "n_keys_seen": 0,
            "last_age_s": float("nan"),
        }

    frame = telemetry.copy()
    if "time" not in frame.columns:
        raise TrainingDataError("telemetry frame must contain a 'time' column")
    frame["time"] = ensure_utc_series(frame["time"])
    frame = frame[frame["time"].notna()]

    future_mask = frame["time"] > anchor
    n_future = int(future_mask.sum())
    if strict and n_future:
        worst = frame.loc[future_mask, "time"].max()
        raise FeatureLeakageError(
            f"{n_future} record(s) after anchor {anchor.isoformat()} reached the feature computation "
            f"(latest {worst.isoformat()})"
        )
    used = frame[~future_mask].sort_values("time")

    feats: dict[str, Any] = {}
    stale_keys = 0
    last_ages: list[float] = []
    missing_fracs_24h: list[float] = []
    n_numeric_keys = 0
    n_text_keys = 0

    for key in numeric_keys:
        col = safe_name(key)
        sub = used[(used["key"] == key) & (used["value_kind"] == "numeric")].dropna(subset=["value_num"])
        sub = sub.sort_values("time")
        if sub.empty:
            for w in numeric_windows:
                feats[f"{col}__n_{w}h"] = 0.0
                feats[f"{col}__mean_{w}h"] = float("nan")
                feats[f"{col}__min_{w}h"] = float("nan")
                feats[f"{col}__max_{w}h"] = float("nan")
                feats[f"{col}__std_{w}h"] = float("nan")
                feats[f"{col}__slope_{w}h"] = float("nan")
                feats[f"{col}__missing_frac_{w}h"] = 1.0
                if w == 24:
                    missing_fracs_24h.append(1.0)
            feats[f"{col}__last"] = float("nan")
            feats[f"{col}__last_age_h"] = float("nan")
            feats[f"{col}__stale"] = 1.0
            feats[f"{col}__ffill_capped"] = 0.0
            continue

        n_numeric_keys += 1
        times = sub["time"]
        values = sub["value_num"]
        for w in numeric_windows:
            block = _stats_block(times, values, anchor, w * 3600.0, min_n, expected_interval_s)
            feats[f"{col}__n_{w}h"] = block["n"]
            feats[f"{col}__mean_{w}h"] = block["mean"]
            feats[f"{col}__min_{w}h"] = block["min"]
            feats[f"{col}__max_{w}h"] = block["max"]
            feats[f"{col}__std_{w}h"] = block["std"]
            feats[f"{col}__slope_{w}h"] = block["slope"]
            feats[f"{col}__missing_frac_{w}h"] = block["missing_frac"]
            if w == 24:
                missing_fracs_24h.append(block["missing_frac"])

        last_age_s = float(max(0.0, (anchor - times.iloc[-1]).total_seconds()))
        last_ages.append(last_age_s)
        last_value = float(values.iloc[-1])
        # Capped imputation: the carried value is only exposed while fresh.
        if last_age_s <= fill_cap_s:
            feats[f"{col}__last"] = last_value
            feats[f"{col}__ffill_capped"] = 1.0
        else:
            feats[f"{col}__last"] = float("nan")
            feats[f"{col}__ffill_capped"] = 0.0
        feats[f"{col}__last_age_h"] = last_age_s / 3600.0
        stale = last_age_s > horizon_s
        feats[f"{col}__stale"] = 1.0 if stale else 0.0
        if stale:
            stale_keys += 1

    offline_re = re.compile(cfg["labels"]["offline_regex"], re.I)
    for key in text_keys:
        col = safe_name(key)
        sub = used[(used["key"] == key) & (used["value_kind"] == "text")].dropna(subset=["value_text"])
        sub = sub.sort_values("time")
        for w in text_windows:
            start = anchor - timedelta(hours=w)
            win = sub[sub["time"] > start]
            feats[f"{col}__text_n_{w}h"] = float(len(win))
            feats[f"{col}__text_distinct_{w}h"] = float(win["value_text"].astype(str).nunique()) if len(win) else 0.0
            changes = 0
            prev = None
            for val in win["value_text"].astype(str):
                if prev is not None and val != prev:
                    changes += 1
                prev = val
            feats[f"{col}__text_changes_{w}h"] = float(changes)
        if sub.empty:
            feats[f"{col}__text_last_age_h"] = float("nan")
            feats[f"{col}__text_last_len"] = float("nan")
            feats[f"{col}__text_bad_flag"] = float("nan")
            feats[f"{col}__text_last_numeric"] = float("nan")
            continue
        n_text_keys += 1
        last_ts = sub["time"].iloc[-1]
        last_text = str(sub["value_text"].iloc[-1])
        feats[f"{col}__text_last_age_h"] = float(max(0.0, (anchor - last_ts).total_seconds())) / 3600.0
        feats[f"{col}__text_last_len"] = float(len(last_text))
        feats[f"{col}__text_bad_flag"] = 1.0 if offline_re.search(last_text) else 0.0
        feats[f"{col}__text_last_numeric"] = 1.0 if re.fullmatch(r"-?\d+(\.\d+)?", last_text.strip()) else 0.0

    last_24h = used[used["time"] > anchor - timedelta(hours=24)]
    feats["g__n_rows_total"] = float(len(used))
    feats["g__n_rows_24h"] = float(len(last_24h))
    feats["g__n_keys_24h"] = float(last_24h["key"].nunique()) if len(last_24h) else 0.0
    feats["g__n_numeric_keys"] = float(n_numeric_keys)
    feats["g__n_text_keys"] = float(n_text_keys)
    feats["g__last_age_h"] = (min(last_ages) / 3600.0) if last_ages else float("nan")
    n_tracked = max(1, n_numeric_keys)
    feats["g__stale_key_frac"] = float(stale_keys) / float(n_tracked)
    feats["g__missing_frac_mean_24h"] = (
        float(np.nanmean(missing_fracs_24h)) if missing_fracs_24h else float("nan")
    )

    meta = {
        "max_source_ts": used["time"].max() if len(used) else pd.NaT,
        "n_rows_used": int(len(used)),
        "n_rows_ignored_future": n_future,
        "n_keys_seen": int(used["key"].nunique()) if len(used) else 0,
        "last_age_s": min(last_ages) if last_ages else float("nan"),
    }
    return feats, meta


def verify_no_leakage(samples: pd.DataFrame) -> None:
    """Assert every sample only used records at or before its anchor."""
    if samples.empty:
        return
    anchors = ensure_utc_series(samples["anchor_ts"])
    sources = ensure_utc_series(samples["max_source_ts"])
    bad = sources.notna() & (sources > anchors)
    if bool(bad.any()):
        worst = samples.loc[bad].head(5)
        raise FeatureLeakageError(
            "feature rows sourced data from the future:\n"
            + worst[["device_id", "anchor_ts", "max_source_ts"]].to_string(index=False)
        )


def build_device_samples(
    device_id: str,
    telemetry: pd.DataFrame,
    anchors: Sequence[pd.Timestamp],
    cfg: dict[str, Any],
    numeric_keys: Sequence[str],
    text_keys: Sequence[str],
    customer_id: str | None = None,
    branch_id: str | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    times = ensure_utc_series(telemetry["time"]) if not telemetry.empty else pd.Series(dtype="datetime64[ns, UTC]")
    for anchor in anchors:
        # Explicit point-in-time slice at the call site, so `strict=True` verifies
        # the slice itself rather than the whole device history.
        allowed = telemetry[times <= anchor]
        n_future = int(len(telemetry) - len(allowed))
        feats, meta = compute_features_at(anchor, allowed, cfg, numeric_keys, text_keys, strict=strict)
        meta["n_rows_ignored_future"] = n_future
        row: dict[str, Any] = {
            "device_id": device_id,
            "anchor_ts": anchor,
            "customer_id": customer_id,
            "branch_id": branch_id,
            "max_source_ts": meta["max_source_ts"],
            "n_rows_used": meta["n_rows_used"],
            "n_rows_ignored_future": meta["n_rows_ignored_future"],
            "n_feature_values": int(sum(1 for v in feats.values() if not (isinstance(v, float) and np.isnan(v)))),
        }
        row.update(feats)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        verify_no_leakage(frame)
    return frame


# ---------------------------------------------------------------------------
# Splits (chronological, grouped, purged)
# ---------------------------------------------------------------------------
def resolve_purge_hours(cfg: dict[str, Any]) -> float:
    raw = cfg["splits"].get("purge_hours", "auto")
    if isinstance(raw, (int, float)):
        return float(raw)
    longest_window = max([int(w) for w in cfg["features"]["numeric_windows_hours"]] or [0])
    horizon = int(cfg["labels"]["horizon_hours"])
    fill_cap = float(cfg["features"]["fill_cap_hours"])
    return float(max(longest_window, horizon) + fill_cap)


def group_key_for(row: pd.Series | dict[str, Any], group_by: str, fallback_device: str) -> str:
    getter = row.get if isinstance(row, dict) else row.get
    if group_by == "customer":
        value = getter("customer_id")
    elif group_by == "branch":
        value = getter("branch_id")
    else:
        value = None
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        value = fallback_device
    return f"{group_by}:{value}"


def assign_splits(samples: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Time-ordered splits with group purity, then a purge band at boundaries.

    `group` mode (default): each customer/branch/device belongs to exactly one
    split, and the purge band is cut at the midpoint between adjacent splits'
    group medians. `time` mode: sample-level chronological cuts.
    """
    if samples.empty:
        return samples.assign(split=pd.Series(dtype="object"))

    out = samples.copy()
    out["anchor_ts"] = ensure_utc_series(out["anchor_ts"])
    split_cfg = cfg["splits"]
    fractions = split_cfg["fractions"]
    purge_h = resolve_purge_hours(cfg)
    out["_median_anchor"] = out["anchor_ts"]

    if split_cfg["mode"] == "time":
        ordered = out.sort_values("anchor_ts")
        n = len(ordered)
        n_train = int(round(n * float(fractions["train"])))
        n_val = int(round(n * float(fractions["validation"])))
        labels = np.array([SPLIT_TEST] * n, dtype=object)
        labels[:n_train] = SPLIT_TRAIN
        labels[n_train : n_train + n_val] = SPLIT_VALIDATION
        ordered = ordered.assign(split=labels)
        result = ordered.sort_index()
        return apply_purge_band(result, purge_h, mode="time")

    group_by = split_cfg["group_by"]
    out["group_key"] = [
        group_key_for(row, group_by, str(row["device_id"])) for _, row in out.iterrows()
    ]
    stats = (
        out.groupby("group_key")
        .agg(n=("anchor_ts", "size"), median_ts=("anchor_ts", "median"))
        .reset_index()
        .sort_values("median_ts")
    )
    total = int(stats["n"].sum())
    targets = {
        SPLIT_TRAIN: float(fractions["train"]) * total,
        SPLIT_VALIDATION: float(fractions["validation"]) * total,
        SPLIT_TEST: float(fractions["test"]) * total,
    }
    order = [SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST]
    counts = [int(n) for n in stats["n"].tolist()]
    cuts = choose_group_cuts(counts, targets, order)
    assignment: dict[str, str] = {}
    for position, group_key in enumerate(stats["group_key"].tolist()):
        if position < cuts[0]:
            assignment[str(group_key)] = SPLIT_TRAIN
        elif position < cuts[1]:
            assignment[str(group_key)] = SPLIT_VALIDATION
        else:
            assignment[str(group_key)] = SPLIT_TEST

    out["split"] = out["group_key"].map(assignment).fillna(SPLIT_TRAIN)
    out = apply_purge_band(out, purge_h, mode="group")
    return out.drop(columns=["_median_anchor"])


def choose_group_cuts(
    counts: Sequence[int], targets: dict[str, float], order: Sequence[str]
) -> tuple[int, int]:
    """Pick the two group-boundary positions whose split sizes best match the targets.

    Groups are already sorted by median anchor time, so the cuts keep the splits
    chronological while never splitting a group. Empty splits are avoided when at
    least three groups exist.
    """
    n = len(counts)
    best: tuple[float, int, int] | None = None
    for first_cut in range(1, n):
        for second_cut in range(first_cut + 1, n + 1):
            sizes = {
                order[0]: sum(counts[:first_cut]),
                order[1]: sum(counts[first_cut:second_cut]),
                order[2]: sum(counts[second_cut:]),
            }
            if min(sizes.values()) == 0:
                continue
            deviation = sum(abs(sizes[key] - targets[key]) for key in order)
            if best is None or deviation < best[0]:
                best = (deviation, first_cut, second_cut)
    if best is not None:
        return best[1], best[2]

    # Fewer than three usable groups: fall back to a two-way chronological split.
    fallback = None
    for first_cut in range(1, n):
        train_size = sum(counts[:first_cut])
        deviation = abs(train_size - targets[order[0]])
        if fallback is None or deviation < fallback[0]:
            fallback = (deviation, first_cut)
    if fallback is None:
        return n, n
    return fallback[1], n


def apply_purge_band(samples: pd.DataFrame, purge_hours: float, mode: str = "group") -> pd.DataFrame:
    """Drop rows inside the purge band between adjacent splits.

    For a boundary `b` between a left and a right split, every left row after
    `b - purge/2` and every right row before `b + purge/2` is purged. After this,
    `min(right) - max(left) >= purge_hours`, so no train row can share its label
    horizon (or feature window) with a validation/test row.
    """
    out = samples.copy()
    if out.empty:
        return out
    out["anchor_ts"] = ensure_utc_series(out["anchor_ts"])
    if "censor_reason" not in out.columns:
        out["censor_reason"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    half = timedelta(hours=purge_hours / 2.0)
    pairs = ((SPLIT_TRAIN, SPLIT_VALIDATION), (SPLIT_VALIDATION, SPLIT_TEST))

    boundaries: list[tuple[pd.Timestamp, str, str]] = []
    if mode == "group":
        medians = out[out["split"].isin(VALID_SPLITS)].groupby(["split", "group_key"])["anchor_ts"].median().reset_index()
        for left, right in pairs:
            left_rows = medians[medians["split"] == left]
            right_rows = medians[medians["split"] == right]
            if left_rows.empty or right_rows.empty:
                continue
            left_max = left_rows["anchor_ts"].max()
            right_min = right_rows["anchor_ts"].min()
            boundaries.append((left_max + (right_min - left_max) / 2, left, right))
    else:
        for left, right in pairs:
            left_rows = out.loc[out["split"] == left, "anchor_ts"]
            right_rows = out.loc[out["split"] == right, "anchor_ts"]
            if left_rows.empty or right_rows.empty:
                continue
            boundaries.append((left_rows.max() + (right_rows.min() - left_rows.max()) / 2, left, right))

    for boundary, left, right in boundaries:
        purge = ((out["split"] == left) & (out["anchor_ts"] > boundary - half)) | (
            (out["split"] == right) & (out["anchor_ts"] < boundary + half)
        )
        out.loc[purge, "censor_reason"] = R_PURGE
        out.loc[purge, "split"] = SPLIT_PURGED
    return out


def split_report(samples: pd.DataFrame, purge_hours: float) -> dict[str, Any]:
    """Diagnostics: chronology, purge gaps and group purity per adjacent pair."""
    report: dict[str, Any] = {
        "purge_hours": purge_hours,
        "counts": {s: int((samples["split"] == s).sum()) for s in list(VALID_SPLITS) + [SPLIT_PURGED]},
        "pairs": [],
    }
    for left, right in ((SPLIT_TRAIN, SPLIT_VALIDATION), (SPLIT_VALIDATION, SPLIT_TEST)):
        left_df = samples[samples["split"] == left]
        right_df = samples[samples["split"] == right]
        if left_df.empty or right_df.empty:
            report["pairs"].append({"left": left, "right": right, "status": "empty_split"})
            continue
        left_max = ensure_utc_series(left_df["anchor_ts"]).max()
        right_min = ensure_utc_series(right_df["anchor_ts"]).min()
        gap_h = float((right_min - left_max).total_seconds()) / 3600.0
        left_groups = set(left_df["group_key"]) if "group_key" in left_df else set(left_df["device_id"])
        right_groups = set(right_df["group_key"]) if "group_key" in right_df else set(right_df["device_id"])
        overlap = sorted(left_groups & right_groups)
        report["pairs"].append(
            {
                "status": "ok",
                "left": left,
                "right": right,
                "chronological": bool(right_min >= left_max),
                "gap_hours": gap_h,
                "purge_satisfied": bool(gap_h >= purge_hours),
                "group_overlap_count": len(overlap),
                "group_overlap_sample": overlap[:5],
            }
        )
    return report


# ---------------------------------------------------------------------------
# Outage evidence (independent sources only)
# ---------------------------------------------------------------------------
def _parse_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8", errors="replace")
        except Exception:
            return {}
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            return {"_raw_text": payload[:500]}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {"_value": payload}


def _payload_method(payload: dict[str, Any]) -> str:
    for container in (payload, payload.get("data") if isinstance(payload.get("data"), dict) else {}):
        for field in ("method", "eventType", "event", "type", "status", "state"):
            value = container.get(field) if isinstance(container, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def classify_event_evidence(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Map one device_event row to an evidence record. Raw values are discarded."""
    labels_cfg = cfg["labels"]
    event_type = str(row.get("event_type") or "").strip().upper()
    payload = _parse_payload(row.get("payload"))
    method = _payload_method(payload).lower()
    offline_re = re.compile(labels_cfg["offline_regex"], re.I)
    online_re = re.compile(labels_cfg["online_regex"], re.I)

    evidence_type = "unknown"
    if event_type in {"LC_EVENT", "LIFECYCLE"}:
        if method in [m.lower() for m in labels_cfg["lifecycle_offline_methods"]]:
            evidence_type = "lc_disconnect"
        elif method in [m.lower() for m in labels_cfg["lifecycle_online_methods"]]:
            evidence_type = "lc_connect"
    elif event_type.startswith("ALARM"):
        blob = " ".join(str(payload.get(k, "")) for k in ("type", "status", "msg", "message", "alarmType"))
        if offline_re.search(blob) and not online_re.search(blob):
            evidence_type = "alarm_offline"
        elif online_re.search(blob):
            evidence_type = "alarm_cleared"
    elif event_type == "ERROR":
        evidence_type = "error_event"

    # The `currentAttr` snapshot is a point-in-time state copy, NOT history.
    if evidence_type == "unknown" and isinstance(payload.get("data"), dict) and payload["data"].get("currentAttr"):
        evidence_type = "current_attr_snapshot"

    strength = (labels_cfg["evidence_strength"] or {}).get(evidence_type, NO_EVIDENCE)
    if strength == NO_EVIDENCE and evidence_type == "unknown":
        return None

    allow = set(labels_cfg.get("payload_key_allowlist") or [])
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    seen_keys = sorted(k for k in set(list(payload.keys()) + list(data.keys())) if k in allow)
    detail_bits = [f"{k}={str((data or payload).get(k))[:40]}" for k in seen_keys if k in {"method", "status", "state"}]
    digest = hashlib.sha256(
        f"{row.get('device_id')}|{event_type}|{row.get('event_id') or row.get('id')}".encode("utf-8")
    ).hexdigest()[:16]

    return {
        "device_id": str(row.get("device_id") or ""),
        "evidence_ts": to_utc(row.get("time")),
        "observed_at": to_utc(row.get("observed_at") or row.get("ingested_at") or row.get("time")),
        "evidence_type": evidence_type,
        "evidence_strength": strength,
        "source_event_id": str(row.get("event_id") or row.get("id") or ""),
        "source_event_type": event_type,
        "payload_fields_present": ",".join(seen_keys),
        "redacted_evidence_reference": f"ev:{digest}:{event_type}:{';'.join(detail_bits)[:80]}",
    }


def build_evidence_frame(events: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "device_id",
        "evidence_ts",
        "evidence_type",
        "evidence_strength",
        "source_event_id",
        "source_event_type",
        "observed_at",
        "payload_fields_present",
        "redacted_evidence_reference",
    ]
    rows: list[dict[str, Any]] = []
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)
    for record in events.to_dict("records"):
        classified = classify_event_evidence(record, cfg)
        if classified is not None:
            rows.append(classified)
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["device_id", "evidence_ts"]).reset_index(drop=True)
    return frame


def build_outage_windows(evidence: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Pair strong offline/online evidence into outage windows per device."""
    columns = [
        "device_id",
        "outage_start",
        "outage_end",
        "duration_minutes",
        "start_evidence_type",
        "end_evidence_type",
        "window_closed",
        "start_event_id",
        "end_event_id",
    ]
    if evidence is None or evidence.empty:
        return pd.DataFrame(columns=columns)
    strong = evidence[evidence["evidence_strength"] == STRONG].copy()
    if strong.empty:
        return pd.DataFrame(columns=columns)
    strong = strong.sort_values(["device_id", "evidence_ts"])
    min_minutes = float(cfg["labels"]["min_outage_minutes"])

    windows: list[dict[str, Any]] = []
    for device_id, group in strong.groupby("device_id"):
        open_start: pd.Series | None = None
        for _, row in group.iterrows():
            etype = row["evidence_type"]
            if etype in {"lc_disconnect", "alarm_offline"}:
                if open_start is None:
                    open_start = row
            elif etype in {"lc_connect", "alarm_cleared"} and open_start is not None:
                duration = float((row["evidence_ts"] - open_start["evidence_ts"]).total_seconds()) / 60.0
                if duration >= min_minutes:
                    windows.append(
                        {
                            "device_id": device_id,
                            "outage_start": open_start["evidence_ts"],
                            "outage_end": row["evidence_ts"],
                            "duration_minutes": duration,
                            "start_evidence_type": open_start["evidence_type"],
                            "end_evidence_type": row["evidence_type"],
                            "window_closed": True,
                            "start_event_id": open_start["source_event_id"],
                            "end_event_id": row["source_event_id"],
                        }
                    )
                open_start = None
        if open_start is not None:
            windows.append(
                {
                    "device_id": device_id,
                    "outage_start": open_start["evidence_ts"],
                    "outage_end": pd.NaT,
                    "duration_minutes": float("nan"),
                    "start_evidence_type": open_start["evidence_type"],
                    "end_evidence_type": "",
                    "window_closed": False,
                    "start_event_id": open_start["source_event_id"],
                    "end_event_id": "",
                }
            )
    return pd.DataFrame(windows, columns=columns)


def verified_windows(windows: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if windows is None or windows.empty:
        return windows
    if cfg["labels"].get("require_closed_windows", True):
        return windows[windows["window_closed"]].reset_index(drop=True)
    return windows.reset_index(drop=True)


def device_evidence_span_days(evidence: pd.DataFrame) -> pd.Series:
    if evidence is None or evidence.empty:
        return pd.Series(dtype="float64")
    strong = evidence[evidence["evidence_strength"] == STRONG]
    if strong.empty:
        return pd.Series(dtype="float64")
    span = strong.groupby("device_id")["evidence_ts"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400.0)
    counts = strong.groupby("device_id").size()
    return pd.concat([span.rename("span_days"), counts.rename("n_strong")], axis=1)


def label_samples(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    evidence: pd.DataFrame,
    coverage_end: pd.Timestamp,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Attach `y_outage_24h` / `label_status` / `censor_reason`.

    A positive label requires verified (strong) evidence. Weak evidence and
    current-attribute snapshots can never produce a positive.
    """
    labels_cfg = cfg["labels"]
    horizon_h = float(labels_cfg["horizon_hours"])
    out = samples.copy()
    out["anchor_ts"] = ensure_utc_series(out["anchor_ts"])
    out["y_outage_24h"] = pd.NA
    out["label_status"] = "censored"
    out["label_horizon_end"] = out["anchor_ts"] + timedelta(hours=horizon_h)
    if "censor_reason" not in out.columns:
        out["censor_reason"] = None

    verified = verified_windows(windows, cfg)
    all_windows = windows if windows is not None else pd.DataFrame()

    if verified is None or verified.empty:
        out.loc[out["censor_reason"].isna(), "censor_reason"] = R_NO_VERIFIED_SOURCE
        return out

    spans = device_evidence_span_days(evidence)
    min_span = float(labels_cfg.get("min_device_evidence_span_days", 0) or 0)
    require_evidence = bool(labels_cfg.get("negative_requires_device_evidence", True))

    verified_by_device = {dev: grp for dev, grp in verified.groupby("device_id")}
    open_by_device = (
        {dev: grp for dev, grp in all_windows[~all_windows["window_closed"]].groupby("device_id")}
        if not all_windows.empty and "window_closed" in all_windows
        else {}
    )

    positive_flags: list[bool] = []
    statuses: list[str] = []
    reasons: list[Any] = []
    values: list[Any] = []

    for row in out.itertuples(index=False):
        reason = getattr(row, "censor_reason", None)
        anchor = row.anchor_ts
        horizon_end = anchor + timedelta(hours=horizon_h)
        device = row.device_id

        if getattr(row, "split", None) == SPLIT_PURGED:
            positive_flags.append(False)
            statuses.append("censored")
            reasons.append(reason or R_PURGE)
            values.append(pd.NA)
            continue

        if horizon_end > coverage_end:
            positive_flags.append(False)
            statuses.append("censored")
            reasons.append(R_HORIZON_INCOMPLETE)
            values.append(pd.NA)
            continue

        open_grp = open_by_device.get(device)
        if open_grp is not None and bool(
            ((open_grp["outage_start"] <= anchor) & (open_grp["outage_end"].isna())).any()
        ):
            positive_flags.append(False)
            statuses.append("censored")
            reasons.append(R_INSIDE_OUTAGE)
            values.append(pd.NA)
            continue

        grp = verified_by_device.get(device)
        starts_in_horizon = 0
        if grp is not None:
            in_horizon = grp[(grp["outage_start"] > anchor) & (grp["outage_start"] <= horizon_end)]
            starts_in_horizon = int(len(in_horizon))
            inside = grp[(grp["outage_start"] <= anchor) & (grp["outage_end"] >= anchor)]
            if len(inside):
                positive_flags.append(False)
                statuses.append("censored")
                reasons.append(R_INSIDE_OUTAGE)
                values.append(pd.NA)
                continue

        if starts_in_horizon:
            positive_flags.append(True)
            statuses.append("measured")
            reasons.append(reason)
            values.append(1)
            continue

        span_ok = True
        if require_evidence:
            if device not in spans.index:
                span_ok = False
            else:
                span_ok = float(spans.loc[device, "span_days"]) >= min_span
        if not span_ok:
            positive_flags.append(False)
            statuses.append("censored")
            reasons.append(R_UNVERIFIED_NEGATIVE)
            values.append(pd.NA)
            continue

        if open_by_device.get(device) is not None:
            positive_flags.append(False)
            statuses.append("censored")
            reasons.append(R_OPEN_WINDOW)
            values.append(pd.NA)
            continue

        positive_flags.append(False)
        statuses.append("measured")
        reasons.append(reason)
        values.append(0)

    out["y_outage_24h"] = pd.array(values, dtype="Int64")
    out["label_status"] = statuses
    out["censor_reason"] = reasons
    out["label_horizon_end"] = out["anchor_ts"] + timedelta(hours=horizon_h)
    return out


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def apply_eligibility(samples: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = samples.copy()
    stale_h = float(cfg["features"]["stale_after_hours"])
    reasons = out.get("censor_reason", pd.Series([None] * len(out), index=out.index)).astype(object)

    ready = pd.Series(True, index=out.index)
    if "n_rows_used" in out:
        ready &= pd.to_numeric(out["n_rows_used"], errors="coerce").fillna(0) > 0
    ready &= out["n_feature_values"].fillna(0) > 0
    no_history = ~ready
    reasons = reasons.where(~no_history, R_NO_HISTORY)
    out["feature_ready"] = ready

    age = pd.to_numeric(
        out.get("g__last_age_h", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    stale = age.notna() & (age > stale_h)
    reasons = reasons.where(~stale, reasons.where(reasons.notna(), R_STALE))

    out["censor_reason"] = reasons
    measured = out["label_status"] == "measured"
    in_split = out["split"].isin(VALID_SPLITS)
    # A supervised-eligible row needs usable features, a measured label, a real
    # split and fresh data. Stale rows stay available to the anomaly track only.
    out["eligibility"] = np.where(ready & measured & in_split & ~stale, "eligible", "excluded")
    out["anomaly_eligible"] = ready & in_split
    return out


def finalize_feature_frame(samples: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = samples.copy()
    numeric_cols = [
        c for c in out.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(out[c])
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    ordered = [c for c in META_COLUMNS if c in out.columns] + [c for c in out.columns if c not in META_COLUMNS]
    out = out[ordered]
    violations = find_exclusion_violations([c for c in out.columns if c not in META_COLUMNS], cfg)
    if violations:
        raise TrainingDataError(
            "excluded attributes leaked into model features: "
            + ", ".join(f"{c} ({r})" for c, r in violations[:10])
        )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SQL builders
# ---------------------------------------------------------------------------
def _fq(cfg: dict[str, Any], table_key: str) -> str:
    return str(cfg["tables"][table_key])


def telemetry_sql(cfg: dict[str, Any]) -> str:
    c = cfg["columns"]
    return (
        f"SELECT {c['time']} AS time, {c['device_id']} AS device_id, {c['key']} AS key, "
        f"{c['customer_id']} AS customer_id, {c['value_num']} AS value_num, {c['value_text']} AS value_text "
        f"FROM {_fq(cfg, 'telemetry')} "
        f"WHERE {c['time']} >= :t0 AND {c['time']} < :t1 AND {c['key']} = ANY(:keys)"
    )


def telemetry_key_stats_sql(cfg: dict[str, Any]) -> str:
    c = cfg["columns"]
    return (
        f"SELECT {c['key']} AS key, "
        f"COUNT(*) AS n_rows, "
        f"COUNT({c['value_num']}) AS n_numeric, "
        f"COUNT({c['value_text']}) AS n_text, "
        f"COUNT(DISTINCT {c['device_id']}) AS n_devices "
        f"FROM {_fq(cfg, 'telemetry')} "
        f"WHERE {c['time']} >= :t0 AND {c['time']} < :t1 "
        f"GROUP BY {c['key']}"
    )


def events_sql(cfg: dict[str, Any]) -> str:
    c = cfg["columns"]
    return (
        f"SELECT {c['event_pk']} AS id, {c['event_id']} AS event_id, {c['device_id']} AS device_id, "
        f"{c['event_type']} AS event_type, {c['time']} AS time, {c['event_payload']} AS payload "
        f"FROM {_fq(cfg, 'events')} "
        f"WHERE {c['time']} >= :t0 AND {c['time']} < :t1 AND {c['event_type']} = ANY(:types)"
    )


def _chunks(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


@dataclass
class ExtractionResult:
    staging_dir: Path
    partitions: int
    rows_clean: int
    rows_quarantined: int
    keys_seen: int
    devices_seen: int
    covered_start: pd.Timestamp | None
    covered_end: pd.Timestamp | None
    quarantine_stats: dict[str, int]
    key_stats: pd.DataFrame
    errors: list[str]


def extract_telemetry(
    db: ReadOnlyDb,
    cfg: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    staging_dir: Path,
    max_partitions: int | None = None,
) -> ExtractionResult:
    """Extract telemetry in bounded daily partitions, batched by key."""
    ext = cfg["extraction"]
    out_dir = staging_dir / "telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    key_stats_rows: list[dict[str, Any]] = []
    quarantine_rows: list[pd.DataFrame] = []
    quarantine_stats: dict[str, int] = {}
    devices: set[str] = set()
    rows_clean = 0
    rows_quarantined = 0
    keys_seen: set[str] = set()
    errors: list[str] = []
    covered_start: pd.Timestamp | None = None
    covered_end: pd.Timestamp | None = None
    partition_count = 0

    static_keys = [k for k in (ext.get("static_keys") or [])]
    for t0, t1 in day_windows(start, end, int(ext["window_days"])):
        if max_partitions is not None and partition_count >= int(max_partitions):
            LOG.info("stopping extraction after %s partitions (--max-partitions)", partition_count)
            break
        partition_count += 1
        day_label = t0.strftime("%Y-%m-%d")

        keys = list(static_keys)
        if ext.get("key_discovery", "per_day_aggregate") == "per_day_aggregate" or not keys:
            stats_df = db.query(telemetry_key_stats_sql(cfg), {"t0": t0, "t1": t1})
            if not stats_df.empty:
                stats_df["partition"] = day_label
                key_stats_rows.extend(stats_df.to_dict("records"))
                discovered = [str(k) for k in stats_df["key"].tolist() if str(k).strip()]
                if ext.get("key_discovery") == "per_day_aggregate":
                    keys = discovered
                else:
                    keys = keys or discovered
            elif not keys:
                LOG.warning("partition %s: no keys discovered", day_label)

        keys_seen.update(keys)
        written = 0
        for batch_idx, batch in enumerate(_chunks(sorted(keys), int(ext["key_batch_size"]))):
            try:
                raw = db.query(telemetry_sql(cfg), {"t0": t0, "t1": t1, "keys": batch})
            except Exception as exc:
                errors.append(f"partition {day_label} batch {batch_idx}: {exc}")
                LOG.error("partition %s batch %s failed: %s", day_label, batch_idx, exc)
                continue
            if raw.empty:
                continue
            cap = int(ext["row_cap_per_query"])
            if cap and len(raw) >= cap:
                message = f"partition {day_label} batch {batch_idx} hit the row cap ({cap})"
                if ext.get("on_row_cap", "error") == "error":
                    errors.append(message)
                    LOG.error("%s - reduce extraction.window_days or narrow the key batch", message)
                    continue
                LOG.warning(message)
            clean, quarantined, stats = quarantine_telemetry(raw, cfg)
            rows_clean += stats["rows_clean"]
            rows_quarantined += int(len(quarantined))
            for reason, count in stats.items():
                if reason in {"rows_in", "rows_clean"}:
                    continue
                quarantine_stats[reason] = quarantine_stats.get(reason, 0) + int(count)
            if not clean.empty:
                devices.update(str(d) for d in clean["device_id"].dropna().unique())
                local_max = ensure_utc_series(clean["time"]).max()
                local_min = ensure_utc_series(clean["time"]).min()
                covered_end = local_max if covered_end is None else max(covered_end, local_max)
                covered_start = local_min if covered_start is None else min(covered_start, local_min)
                target = out_dir / f"day={day_label}" / f"part-{batch_idx:03d}.parquet"
                target.parent.mkdir(parents=True, exist_ok=True)
                clean.to_parquet(target, index=False)
                written += 1
            if not quarantined.empty:
                quarantine_rows.append(quarantined)
        LOG.info("partition %s: %s key-batch file(s) written", day_label, written)

    key_stats = pd.DataFrame(key_stats_rows)
    if not key_stats.empty:
        key_stats = (
            key_stats.groupby("key", as_index=False)[["n_rows", "n_numeric", "n_text"]].sum()
        )
        key_stats.to_csv(staging_dir / "key_stats.csv", index=False)
    if quarantine_rows:
        pd.concat(quarantine_rows, ignore_index=True).to_csv(
            staging_dir / "quarantine.csv", index=False
        )
    (staging_dir / "devices.json").write_text(
        json.dumps(sorted(devices), indent=2), encoding="utf-8"
    )

    return ExtractionResult(
        staging_dir=staging_dir,
        partitions=partition_count,
        rows_clean=rows_clean,
        rows_quarantined=rows_quarantined,
        keys_seen=len(keys_seen),
        devices_seen=len(devices),
        covered_start=covered_start,
        covered_end=covered_end,
        quarantine_stats=quarantine_stats,
        key_stats=key_stats,
        errors=errors,
    )


def extract_events(
    db: ReadOnlyDb,
    cfg: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    staging_dir: Path,
    max_partitions: int | None = None,
) -> pd.DataFrame:
    """Extract event rows in daily partitions (evidence collector only)."""
    ext = cfg["extraction"]
    out_dir = staging_dir / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    types = [str(t) for t in cfg["labels"].get("event_types_of_interest", [])]
    frames: list[pd.DataFrame] = []
    for idx, (t0, t1) in enumerate(day_windows(start, end, int(ext["window_days"]))):
        if max_partitions is not None and idx >= int(max_partitions):
            break
        try:
            raw = db.query(events_sql(cfg), {"t0": t0, "t1": t1, "types": types})
        except Exception as exc:
            LOG.error("event partition %s failed: %s", t0.date(), exc)
            continue
        if raw.empty:
            continue
        raw = raw.copy()
        raw["time"] = ensure_utc_series(raw["time"])
        raw.to_parquet(out_dir / f"day={t0.strftime('%Y-%m-%d')}.parquet", index=False)
        frames.append(raw)
    if not frames:
        return pd.DataFrame(columns=["id", "event_id", "device_id", "event_type", "time", "payload"])
    return pd.concat(frames, ignore_index=True)


def load_staging(
    staging_dir: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    device_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    root = staging_dir / "telemetry"
    if not root.exists():
        return pd.DataFrame(columns=["time", "device_id", "key", "value_kind", "value_num", "value_text"])
    filters: list[tuple[str, str, Any]] = []
    if start is not None:
        filters.append(("time", ">=", start))
    if end is not None:
        filters.append(("time", "<", end))
    if device_ids:
        filters.append(("device_id", "in", list(device_ids)))
    try:
        return pd.read_parquet(root, filters=filters or None)
    except Exception as exc:
        LOG.error("failed to read staging parquet: %s", exc)
        raise


def load_events_staging(staging_dir: Path) -> pd.DataFrame:
    root = staging_dir / "events"
    if not root.exists():
        return pd.DataFrame(columns=["id", "event_id", "device_id", "event_type", "time", "payload"])
    return pd.read_parquet(root)


def load_device_metadata(db: ReadOnlyDb, cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Best-effort device -> {customer_id, branch_id} map from the hierarchy tables."""
    mapping: dict[str, dict[str, str]] = {}
    table = cfg["tables"].get("branch_identity")
    if not table:
        return mapping
    try:
        cols = db.query(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = :schema AND table_name = :table",
            {"schema": str(table).split(".")[0], "table": str(table).split(".")[-1]},
        )
        names = [str(c) for c in cols.get("column_name", pd.Series(dtype=str)).tolist()]
        dev_col = next((c for c in names if re.search(r"device.*id$|^device_id$", c, re.I)), None)
        branch_col = next((c for c in names if re.search(r"branch.*id$|^branch_id$", c, re.I)), None)
        cust_col = next((c for c in names if re.search(r"customer.*id$|^customer_id$", c, re.I)), None)
        if not dev_col or not branch_col:
            LOG.warning("branch_identity lacks device/branch id columns; branch grouping disabled")
            return mapping
        select_cols = ", ".join(dict.fromkeys([dev_col, branch_col] + ([cust_col] if cust_col else [])))
        rows = db.query(f"SELECT {select_cols} FROM {table}")
        for row in rows.to_dict("records"):
            device = str(row.get(dev_col) or "")
            if not device:
                continue
            mapping[device] = {
                "branch_id": str(row.get(branch_col) or ""),
                "customer_id": str(row.get(cust_col) or "") if cust_col else "",
            }
    except Exception as exc:
        LOG.warning("device metadata lookup skipped: %s", exc)
    return mapping


# ---------------------------------------------------------------------------
# Data-quality report
# ---------------------------------------------------------------------------
def build_quality_metrics(
    samples: pd.DataFrame,
    extraction: ExtractionResult | None,
    evidence: pd.DataFrame,
    windows: pd.DataFrame,
    verified: pd.DataFrame,
    splits: dict[str, Any],
    cfg: dict[str, Any],
    coverage_end: pd.Timestamp | None,
    excluded_keys: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "config_path": str(DEFAULT_CONFIG_PATH),
        "coverage_end": coverage_end.isoformat() if coverage_end is not None else None,
        "splits": splits,
        "excluded_keys": {k: len(v) for k, v in (excluded_keys or {}).items()},
    }

    if extraction is not None:
        metrics["extraction"] = {
            "partitions": extraction.partitions,
            "rows_clean": extraction.rows_clean,
            "rows_quarantined": extraction.rows_quarantined,
            "keys_seen": extraction.keys_seen,
            "devices_seen": extraction.devices_seen,
            "cover_start": extraction.covered_start.isoformat() if extraction.covered_start is not None else None,
            "cover_end": extraction.covered_end.isoformat() if extraction.covered_end is not None else None,
            "quarantine": extraction.quarantine_stats,
            "errors": extraction.errors[:20],
            "error_count": len(extraction.errors),
        }
        ks = extraction.key_stats
        if ks is not None and not ks.empty:
            metrics["coverage_by_key"] = (
                ks.sort_values("n_rows", ascending=False)
                .head(50)
                .to_dict("records")
            )

    if samples is not None and not samples.empty:
        metrics["samples"] = {
            "total": int(len(samples)),
            "eligible": int((samples["eligibility"] == "eligible").sum()),
            "excluded": int((samples["eligibility"] == "excluded").sum()),
            "anomaly_eligible": int(samples["anomaly_eligible"].sum()),
            "label_measured": int((samples["label_status"] == "measured").sum()),
            "label_censored": int((samples["label_status"] == "censored").sum()),
            "positive_labels": int(pd.to_numeric(samples["y_outage_24h"], errors="coerce").fillna(0).sum()),
            "censor_reasons": samples["censor_reason"].value_counts(dropna=False).to_dict(),
            "devices": int(samples["device_id"].nunique()),
        }
        if "customer_id" in samples:
            metrics["samples"]["customers"] = int(samples["customer_id"].nunique(dropna=True))
        nan_col = pd.Series(np.nan, index=samples.index)
        freshness = pd.to_numeric(samples.get("g__last_age_h", nan_col), errors="coerce").dropna()
        if len(freshness):
            metrics["freshness_hours"] = {
                "p50": float(freshness.quantile(0.5)),
                "p90": float(freshness.quantile(0.9)),
                "p99": float(freshness.quantile(0.99)),
                "max": float(freshness.max()),
            }
        missing = pd.to_numeric(samples.get("g__missing_frac_mean_24h", nan_col), errors="coerce").dropna()
        if len(missing):
            metrics["missingness_24h"] = {
                "p50": float(missing.quantile(0.5)),
                "p90": float(missing.quantile(0.9)),
                "p99": float(missing.quantile(0.99)),
            }
        balance = (
            samples.groupby(["split", "device_id"])
            .size()
            .reset_index(name="n")
            .groupby("split")["n"]
            .agg(["count", "min", "max", "mean"])
            .reset_index()
            .to_dict("records")
        )
        metrics["split_balance_by_device"] = balance
        if "customer_id" in samples:
            metrics["split_balance_by_customer"] = (
                samples.groupby(["split", "customer_id"]).size().reset_index(name="n").to_dict("records")
            )

    if evidence is not None and not evidence.empty:
        metrics["evidence"] = {
            "rows": int(len(evidence)),
            "by_type": evidence["evidence_type"].value_counts().to_dict(),
            "by_strength": evidence["evidence_strength"].value_counts().to_dict(),
            "devices_with_strong_evidence": int(
                evidence.loc[evidence["evidence_strength"] == STRONG, "device_id"].nunique()
            ),
            "weak_only_devices": int(
                len(
                    set(evidence.loc[evidence["evidence_strength"] == WEAK, "device_id"])
                    - set(evidence.loc[evidence["evidence_strength"] == STRONG, "device_id"])
                )
            ),
            "weak_evidence_rows": int((evidence["evidence_strength"] == WEAK).sum()),
        }
        latency = None
        try:
            latency = (
                ensure_utc_series(evidence["observed_at"]) - ensure_utc_series(evidence["evidence_ts"])
            ).dt.total_seconds().dropna()
        except Exception:
            latency = None
        if latency is not None and len(latency):
            metrics["evidence"]["observed_minus_evidence_seconds"] = {
                "p50": float(latency.quantile(0.5)),
                "p90": float(latency.quantile(0.9)),
                "max": float(latency.max()),
            }

    metrics["outage_windows"] = {
        "total": int(len(windows)) if windows is not None else 0,
        "closed": int(windows["window_closed"].sum()) if windows is not None and not windows.empty else 0,
        "verified_used_for_labels": int(len(verified)) if verified is not None else 0,
    }
    metrics["supervised_training_ready"] = bool(verified is not None and not verified.empty)
    return metrics


def write_quality_report(metrics: dict[str, Any], md_path: Path, json_path: Path) -> None:
    json_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    lines: list[str] = []
    a = lines.append
    ready = metrics.get("supervised_training_ready", False)
    a("# Training-data quality report")
    a("")
    a(f"Generated: {metrics.get('generated_at')}")
    a(f"Supervised 24h-label training ready: **{'yes' if ready else 'NO'}**")
    a("")
    if not ready:
        a("> **No verified outage evidence is available.** Every sample is `censored`")
        a("> (`no_verified_outage_source`) and `y_outage_24h` is NULL. Do not train the")
        a("> 24h outage model yet. The unsupervised anomaly track (`anomaly_eligible`) is unaffected.")
        a("")
    ext = metrics.get("extraction")
    if ext:
        a("## Coverage")
        a("")
        a(f"- Partitions: {ext['partitions']}  |  clean rows: {ext['rows_clean']:,}  |  quarantined: {ext['rows_quarantined']:,}")
        a(f"- Keys seen: {ext['keys_seen']}  |  devices seen: {ext['devices_seen']}")
        a(f"- Data coverage: {ext['cover_start']} -> {ext['cover_end']}")
        a(f"- Extraction errors: {ext['error_count']}")
        a("")
        a("### Quarantined rows by reason")
        a("")
        a("| reason | rows |")
        a("|---|---|")
        for reason, count in sorted((ext.get("quarantine") or {}).items(), key=lambda kv: -kv[1]):
            a(f"| {reason} | {count:,} |")
        a("")
    cov = metrics.get("coverage_by_key")
    if cov:
        a("### Top telemetry keys by volume")
        a("")
        a("| key | rows | numeric | text | devices |")
        a("|---|---|---|---|---|")
        for row in cov[:20]:
            a(f"| {row.get('key')} | {row.get('n_rows'):,} | {row.get('n_numeric'):,} | {row.get('n_text'):,} | {row.get('n_devices')} |")
        a("")
    samp = metrics.get("samples")
    if samp:
        a("## Samples")
        a("")
        a(f"- Total: {samp['total']:,}  |  eligible: {samp['eligible']:,}  |  excluded: {samp['excluded']:,}")
        a(f"- Anomaly-track eligible: {samp['anomaly_eligible']:,}")
        a(f"- Labels measured: {samp['label_measured']:,}  |  censored: {samp['label_censored']:,}  |  positives: {samp['positive_labels']:,}")
        a(f"- Devices: {samp['devices']}")
        a("")
        a("### Censor reasons")
        a("")
        a("| reason | samples |")
        a("|---|---|")
        for reason, count in sorted((samp.get("censor_reasons") or {}).items(), key=lambda kv: -kv[1]):
            a(f"| {reason} | {count:,} |")
        a("")
    fresh = metrics.get("freshness_hours")
    if fresh:
        a("### Freshness / missingness")
        a("")
        a(f"- time since last reading (h): p50={fresh['p50']:.1f} p90={fresh['p90']:.1f} p99={fresh['p99']:.1f} max={fresh['max']:.1f}")
        miss = metrics.get("missingness_24h") or {}
        if miss:
            a(f"- 24h missing fraction: p50={miss['p50']:.3f} p90={miss['p90']:.3f} p99={miss['p99']:.3f}")
        a("")
    ev = metrics.get("evidence")
    if ev:
        a("## Outage evidence")
        a("")
        a(f"- Rows: {ev['rows']:,}  |  devices with strong evidence: {ev['devices_with_strong_evidence']}")
        a(f"- Weak-only devices: {ev['weak_only_devices']}  |  weak rows: {ev['weak_evidence_rows']:,}")
        a(f"- By type: {ev.get('by_type')}")
        a(f"- By strength: {ev.get('by_strength')}")
        if ev.get("observed_minus_evidence_seconds"):
            lat = ev["observed_minus_evidence_seconds"]
            a(f"- Observation latency (s): p50={lat['p50']:.0f} p90={lat['p90']:.0f} max={lat['max']:.0f}")
        a("")
    win = metrics.get("outage_windows")
    if win:
        a(f"## Outage windows: {win['total']} total, {win['closed']} closed, {win['verified_used_for_labels']} verified")
        a("")
    sp = metrics.get("splits") or {}
    if sp:
        a("## Splits")
        a("")
        a(f"- purge gap: {sp.get('purge_hours')} h")
        a(f"- counts: {sp.get('counts')}")
        a("")
        a("| left | right | chronological | gap (h) | purge satisfied | group overlap |")
        a("|---|---|---|---|---|---|")
        for pair in sp.get("pairs", []):
            if pair.get("status") == "empty_split":
                a(f"| {pair['left']} | {pair['right']} | - | - | - | empty split |")
                continue
            a(
                f"| {pair['left']} | {pair['right']} | {pair['chronological']} | {pair['gap_hours']:.1f} | "
                f"{pair['purge_satisfied']} | {pair['group_overlap_count']} |"
            )
        a("")
    bal = metrics.get("split_balance_by_device")
    if bal:
        a("### Split balance (devices)")
        a("")
        a("| split | devices | min samples | max samples | mean |")
        a("|---|---|---|---|---|")
        for row in bal:
            a(f"| {row['split']} | {row['count']} | {row['min']} | {row['max']} | {row['mean']:.1f} |")
        a("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def check_db_structure(db: ReadOnlyDb, cfg: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"tables": {}, "ok": True}
    required = {
        cfg["tables"]["telemetry"]: [cfg["columns"][k] for k in ("time", "device_id", "key", "value_num", "value_text")],
        cfg["tables"]["events"]: [cfg["columns"][k] for k in ("device_id", "event_type", "time", "event_payload")],
    }
    for table, expected in required.items():
        schema, _, name = str(table).rpartition(".")
        try:
            cols = db.query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = :schema AND table_name = :table",
                {"schema": schema or "public", "table": name},
            )
            names = [str(c) for c in cols.get("column_name", pd.Series(dtype=str)).tolist()]
            missing = [c for c in expected if c not in names]
            report["tables"][table] = {"columns": names, "missing": missing}
            if missing:
                report["ok"] = False
        except Exception as exc:
            report["tables"][table] = {"error": str(exc)}
            report["ok"] = False
    return report


def run_pipeline(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.anchor_every_hours:
        cfg["anchors"]["every_hours"] = int(args.anchor_every_hours)
    if args.group_by:
        cfg["splits"]["group_by"] = str(args.group_by)
    out_dir = Path(args.out_dir or cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / cfg["output"]["staging_dir"]

    url, target = connection_target(cfg)
    LOG.info("database target: %s", target)

    if args.check_db:
        if not url:
            print(f"No database configuration found. {target}")
            print("Set the env var named in db.url_env (or PGHOST/PGUSER/PGDATABASE) and retry.")
            return 2
        db = ReadOnlyDb(url=url, cfg=cfg)
        try:
            print(f"Connecting read-only to {target}")
            probe = db.query("SELECT 1 AS ok")
            report = check_db_structure(db, cfg)
            print(json.dumps(report, indent=2))
            print(f"probe ok={'-' if probe.empty else bool(probe.iloc[0]['ok'])}")
        except TrainingDataError as exc:
            print(f"Database check failed: {exc}")
            return 2
        finally:
            db.close()
        return 0 if report.get("ok") else 4

    if args.plan_only:
        start = parse_iso(args.start) or (pd.Timestamp.now(tz=UTC) - timedelta(days=30))
        end = parse_iso(args.end) or pd.Timestamp.now(tz=UTC)
        partitions = list(day_windows(start, end, int(cfg["extraction"]["window_days"])))
        print("Extraction plan (no database calls made)")
        print(f"  window      : {start.isoformat()} -> {end.isoformat()}")
        print(f"  partitions  : {len(partitions)} x {cfg['extraction']['window_days']} day(s)")
        print(f"  key batches : {cfg['extraction']['key_batch_size']} keys per query")
        print(f"  row cap     : {cfg['extraction']['row_cap_per_query']:,} per query ({cfg['extraction']['on_row_cap']} on breach)")
        print(f"  anchors     : every {cfg['anchors']['every_hours']}h, min history {cfg['anchors']['min_history_hours']}h")
        print(f"  purge gap   : {resolve_purge_hours(cfg)}h (auto = max(longest window, label horizon) + fill cap)")
        print(f"  outputs     : {out_dir}")
        return 0

    if not url and not args.skip_extract:
        print(f"No database configuration found. {target}")
        print("Set the env var named in db.url_env (or PGHOST/PGUSER/PGDATABASE) and retry.")
        print("Nothing was extracted and no labels were created.")
        return 2
    if not url:
        LOG.warning("no database configured; recomputing features from staged parquet only")

    start = parse_iso(args.start) or (pd.Timestamp.now(tz=UTC) - timedelta(days=30))
    end = parse_iso(args.end) or pd.Timestamp.now(tz=UTC)
    db = ReadOnlyDb(url=url, cfg=cfg)
    extraction: ExtractionResult | None = None
    evidence = pd.DataFrame()
    windows = pd.DataFrame()
    verified = pd.DataFrame()
    coverage_end: pd.Timestamp | None = None
    samples = pd.DataFrame()
    excluded_keys: dict[str, list[str]] = {}

    try:
        if not args.skip_extract:
            extraction = extract_telemetry(
                db, cfg, start, end, staging_dir, max_partitions=args.max_partitions
            )
            events = extract_events(db, cfg, start, end, staging_dir, max_partitions=args.max_partitions)
        else:
            events = load_events_staging(staging_dir)
            key_stats_path = staging_dir / "key_stats.csv"
            key_stats = pd.read_csv(key_stats_path) if key_stats_path.exists() else pd.DataFrame()
            extraction = ExtractionResult(
                staging_dir=staging_dir,
                partitions=0,
                rows_clean=0,
                rows_quarantined=0,
                keys_seen=int(len(key_stats)) if not key_stats.empty else 0,
                devices_seen=0,
                covered_start=None,
                covered_end=None,
                quarantine_stats={},
                key_stats=key_stats,
                errors=[],
            )

        evidence = build_evidence_frame(events, cfg)
        windows = build_outage_windows(evidence, cfg)
        verified = verified_windows(windows, cfg)

        telemetry = load_staging(staging_dir, start=start, end=end)
        if telemetry.empty:
            print("No staged telemetry found. Run extraction first (remove --skip-extract).")
            return 4
        telemetry["time"] = ensure_utc_series(telemetry["time"])
        coverage_end = telemetry["time"].max()

        numeric_keys, text_keys, excluded_keys = select_feature_keys(extraction.key_stats, cfg)
        if args.max_feature_keys:
            numeric_keys = numeric_keys[: int(args.max_feature_keys)]
        LOG.info("feature keys: %s numeric, %s text", len(numeric_keys), len(text_keys))
        column_names = feature_column_names(numeric_keys, text_keys, cfg)
        violations = find_exclusion_violations(column_names, cfg)
        if violations:
            raise TrainingDataError(f"excluded columns would be generated: {violations[:5]}")

        metadata = load_device_metadata(db, cfg)
        devices = sorted(telemetry["device_id"].dropna().unique().tolist())
        if args.max_devices:
            devices = devices[: int(args.max_devices)]
        anchors_all = hour_grid(start, end, int(cfg["anchors"]["every_hours"]), bool(cfg["anchors"]["align_utc"]))

        frames: list[pd.DataFrame] = []
        for device in devices:
            dev_tel = telemetry[telemetry["device_id"] == device]
            if dev_tel.empty:
                continue
            first_ts = dev_tel["time"].min()
            anchors = anchors_all[
                (anchors_all >= first_ts + timedelta(hours=float(cfg["anchors"]["min_history_hours"])))
                & (anchors_all <= coverage_end)
            ]
            if len(anchors) == 0:
                continue
            meta = metadata.get(str(device), {})
            customer_id = meta.get("customer_id") or (
                str(dev_tel["customer_id"].dropna().iloc[0]) if "customer_id" in dev_tel and dev_tel["customer_id"].notna().any() else None
            )
            frames.append(
                build_device_samples(
                    device_id=str(device),
                    telemetry=dev_tel,
                    anchors=list(anchors),
                    cfg=cfg,
                    numeric_keys=numeric_keys,
                    text_keys=text_keys,
                    customer_id=customer_id,
                    branch_id=meta.get("branch_id"),
                    strict=bool(args.strict_leakage_check),
                )
            )
        if not frames:
            print("No anchors could be built (no telemetry history).")
            return 4
        samples = pd.concat(frames, ignore_index=True)
        verify_no_leakage(samples)

        samples = assign_splits(samples, cfg)
        purge_h = resolve_purge_hours(cfg)
        samples = label_samples(samples, windows, evidence, coverage_end, cfg)
        samples = apply_eligibility(samples, cfg)
        samples = finalize_feature_frame(samples, cfg)
        verify_no_leakage(samples)

        samples_path = out_dir / cfg["output"]["anomaly_samples"]
        samples.to_parquet(samples_path, index=False)
        evidence_path = out_dir / cfg["output"]["evidence_audit"]
        evidence.to_parquet(evidence_path, index=False)

        metrics = build_quality_metrics(
            samples, extraction, evidence, windows, verified,
            split_report(samples, purge_h), cfg, coverage_end, excluded_keys,
        )
        write_quality_report(
            metrics,
            out_dir / cfg["output"]["report_md"],
            out_dir / cfg["output"]["metrics_json"],
        )
        LOG.info("wrote %s (%s rows)", samples_path, len(samples))
        print(json.dumps(metrics.get("samples", {}), indent=2, default=str))

        if not metrics.get("supervised_training_ready"):
            print("")
            print("=" * 78)
            print("STOP: no verified outage evidence found in device_event.")
            print("-" * 78)
            print("  * No lifecycle disconnect/reconnect pairs and no offline/no-data alarms")
            print("    were present for this window, so no sample can be labelled.")
            print("  * Every row is label_status='censored', censor_reason='" + R_NO_VERIFIED_SOURCE + "'.")
            print("  * y_outage_24h is NULL for all rows. No supervised model was trained.")
            print("  * The unsupervised anomaly track is unaffected: use")
            print("    anomaly_eligible == True rows and the feature columns only.")
            print("  * Independent sources to look for: device_event lifecycle rows,")
            print("    offline/no-data alarms, maintenance or ticket records.")
            print("=" * 78)
            return 3
        return 0
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leakage-free anomaly/outage training data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--start", default=None, help="ISO timestamp (inclusive)")
    parser.add_argument("--end", default=None, help="ISO timestamp (exclusive)")
    parser.add_argument("--check-db", action="store_true", help="Verify connection + expected columns, then exit")
    parser.add_argument("--plan-only", action="store_true", help="Print the extraction plan without touching the DB")
    parser.add_argument("--skip-extract", action="store_true", help="Reuse staged parquet instead of querying")
    parser.add_argument("--max-partitions", type=int, default=None, help="Limit daily partitions (smoke tests)")
    parser.add_argument("--max-devices", type=int, default=None)
    parser.add_argument("--max-feature-keys", type=int, default=None)
    parser.add_argument("--anchor-every-hours", type=int, default=None)
    parser.add_argument("--group-by", choices=["customer", "branch", "device"], default=None)
    parser.add_argument(
        "--strict-leakage-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise if any record after the anchor reaches a feature row (default: on)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return run_pipeline(args)
    except FeatureLeakageError as exc:
        LOG.error("leakage check failed: %s", exc)
        return 5
    except TrainingDataError as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
