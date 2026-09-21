#!/usr/bin/env python
"""Alarm evidence bridge: ThingsBoard alarm CSV -> device_event-shaped rows.

Why this exists
---------------
The label builder (`build_training_dataset.py`) consumes device_event rows
with columns {id, event_id, device_id, event_type, time, payload}. The pull
pipeline writes alarms to `alarms.csv` with columns
{device_id, device_name, alarm_type, severity, status, start, end, ack,
clear, payload}. Until now nothing connected the two — collected alarms were
dead data for labeling. This module is the bridge.

Semantics that keep the labels honest
-------------------------------------
* ``alarm_type`` is forwarded as payload ``type`` so the builder's ALARM
  branch regexes can classify offline/online evidence.
* Each cleared alarm yields TWO evidence rows: an offline row at ``start``
  and a cleared row at ``clearTs`` — never at ``ackTs``. Acknowledgement is
  a human action, not a recovery timestamp, so ``ack`` is deliberately
  excluded from the payload entirely.
* Camera-channel alarms (``CAMERA TAMPER CH n``, ``CAMERA DISCONNECT CH n``)
  describe a single video channel, NOT a device outage. They are forwarded
  with ``scope=not_device_outage``; the builder classifies them as
  ``alarm_channel`` which the default config marks weak — a camera dying is
  not the branch going offline.
* Times are emitted as ISO-8601 UTC strings; ms-epoch inputs are converted.
* Alarms that never cleared yield only the offline row -> the builder keeps
  an open (unverified) window, which censors downstream instead of inventing
  a recovery.

Usage::

    python alarms_bridge.py alarms.csv --out training_data/staging/events/alarms_as_events.parquet
    # library use:
    from alarms_bridge import alarms_csv_to_event_rows
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ALARM_EVENT_TYPE = "ALARM"

#: Camera-channel alarms describe one video channel, not device availability.
CAMERA_CHANNEL_RE = re.compile(r"^CAMERA\s+(TAMPER|DISCONNECT)", re.I)

BRIDGE_MARKER = "alarm_csv_bridge_v1"


def _to_iso(value: Any) -> str:
    """ISO-8601 UTC from ISO strings or ms-since-epoch values; '' if absent.

    Numeric values >= 1e11 are treated as ms since epoch (a 1970-nanosecond
    interpretation of such magnitudes is never a valid ingest timestamp —
    those are quarantined upstream anyway).
    """
    if value is None or value == "":
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        num = float(value)
        if num >= 1e11:  # ms since epoch (>= 1973-03 in ms); ns would be ~1e18
            return pd.Timestamp(num, unit="ms", tz="UTC").isoformat()
        if num >= 1e17:  # ns since epoch
            return pd.Timestamp(num, unit="ns", tz="UTC").isoformat()
        return ""  # too small to be a real epoch timestamp
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return ""
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    except Exception:
        return ""


def alarms_frame_to_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert an alarms DataFrame (pull_all_data schema) to event rows."""
    required = {"device_id", "alarm_type", "status", "start", "end", "ack", "clear", "payload"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"alarms frame missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for r in df.to_dict("records"):
        alarm_type = str(r.get("alarm_type", ""))
        status = str(r.get("status", ""))
        start_iso = _to_iso(r.get("start"))
        clear_iso = _to_iso(r.get("clear"))
        scope = "not_device_outage" if CAMERA_CHANNEL_RE.match(alarm_type) else "device_outage"

        def make_row(kind: str, time_iso: str, state_text: str, suffix: str) -> dict[str, Any]:
            eid = f"alarm-{r.get('device_id')}-{r.get('start')}-{suffix}"
            payload = {
                "type": alarm_type,
                "status": status,
                "state": state_text,  # drives the builder's offline/online regexes
                "startTs": start_iso,
                "endTs": _to_iso(r.get("end")),
                "clearTs": clear_iso or None,
                "scope": scope,
                "bridge": BRIDGE_MARKER,
            }
            # NOTE: no "ack" field — acknowledgement time must never be
            # interpretable as an outage recovery timestamp downstream.
            return {
                "id": eid,
                "event_id": eid,
                "device_id": r.get("device_id"),
                "event_type": ALARM_EVENT_TYPE,
                "time": time_iso,
                "payload": payload,
            }

        if not start_iso:
            continue  # cannot place the alarm in time; skip rather than guess
        rows.append(make_row("offline", start_iso, "ALARM ACTIVATED", "start"))
        if clear_iso:
            rows.append(make_row("cleared", clear_iso, "ALARM CLEARED", "clear"))
        # No clearTs -> no cleared row: the window stays open (unresolved).
    return pd.DataFrame(
        rows,
        columns=["id", "event_id", "device_id", "event_type", "time", "payload"],
    )


def alarms_csv_to_event_rows(path: str | Path) -> pd.DataFrame:
    """Read an alarms CSV and return device_event-shaped rows."""
    return alarms_frame_to_event_rows(pd.read_csv(path))


def label_source_coverage_from_per_device(
    per_device: dict[str, dict[str, Any]],
    query_start_ms: int,
    query_end_ms: int,
    source: str = "alarms",
) -> pd.DataFrame:
    """Build the label-source coverage ledger from pull step stats.

    One row per device actually queried: query coverage (start/end/status)
    plus observed data recency, so the builder can separately enforce
    "the interval was queried" and "the source stream is not stale".
    """
    from datetime import datetime, timezone

    def iso(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()

    rows = []
    for device_id, stat in per_device.items():
        error = str(stat.get("error", "") or "")
        rows.append(
            {
                "device_id": device_id,
                "source": source,
                "query_start": iso(query_start_ms),
                "query_end": iso(query_end_ms),
                "status": "failed" if error else "ok",
                "rows": int(stat.get("rows", 0) or 0),
                "data_oldest": stat.get("oldest", "") or None,
                "data_newest": stat.get("newest", "") or None,
                "error": error[:200],
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bridge alarm CSVs into label-builder event rows.")
    p.add_argument("alarms_csv", type=Path)
    p.add_argument("--out", type=Path, default=Path("training_data/staging/events/alarms_as_events.parquet"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = alarms_csv_to_event_rows(args.alarms_csv)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out, index=False)
    print(f"bridged {len(rows)} evidence rows from {args.alarms_csv} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
