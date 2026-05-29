#!/usr/bin/env python
"""Pull daily ThingsBoard current-state snapshots and derive recoveries.

The nightly snapshot captures all devices, including devices that are online.
Keeping those rows lets us close a previously-open outage when a later snapshot
shows active/online state or a newer lastConnectTime.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from pull_outage_labels import (
    DEFAULT_CURRENT_ATTR_KEYS,
    ThingsBoardClient,
    bool_env,
    classify_value,
    compact_raw,
    duration_seconds,
    parse_time_to_ms,
    utc_iso,
)


DEFAULT_OUTPUT_DIR = Path("current_state_snapshots")
DEFAULT_RECOVERY_OUTPUT = DEFAULT_OUTPUT_DIR / "offline_recoveries.csv"


@dataclass
class CurrentStateSnapshot:
    snapshot_utc: str
    snapshot_ms: int
    device_id: str
    device_name: str
    active: str
    status: str
    is_offline: bool
    lastDisconnectTime: str
    lastDisconnectTime_iso: str
    lastConnectTime: str
    lastConnectTime_iso: str
    lastActivityTime: str
    lastActivityTime_iso: str
    inactivityAlarmTime: str
    inactivityAlarmTime_iso: str
    inactive_since: str
    inactive_since_iso: str
    inactive_reason: str
    raw: str


@dataclass
class SnapshotRecovery:
    device_id: str
    device_name: str
    offline_start: str
    offline_end: str
    reconnect_time: str
    outage_reason: str
    source: str
    source_key: str
    source_event_id: str
    duration_sec: int | None
    confidence: str
    raw: str


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def device_id_from(device: dict[str, Any]) -> str:
    raw_id = device.get("id")
    if isinstance(raw_id, dict):
        return str(raw_id.get("id", ""))
    return str(raw_id or "")


def time_pair(attrs: dict[str, Any], key: str) -> tuple[str, str]:
    raw = attrs.get(key)
    parsed = parse_time_to_ms(raw)
    return stringify(raw), utc_iso(parsed)


def build_snapshot(
    device: dict[str, Any],
    attrs: dict[str, Any],
    snapshot_ms: int,
) -> CurrentStateSnapshot:
    active_state = classify_value(attrs.get("active"), "active")
    status_state = classify_value(attrs.get("status"), "status")
    is_offline = active_state == "offline" or status_state == "offline"
    last_disconnect, last_disconnect_iso = time_pair(attrs, "lastDisconnectTime")
    last_connect, last_connect_iso = time_pair(attrs, "lastConnectTime")
    last_activity, last_activity_iso = time_pair(attrs, "lastActivityTime")
    inactivity_alarm, inactivity_alarm_iso = time_pair(attrs, "inactivityAlarmTime")
    inactive_since, inactive_since_iso = time_pair(attrs, "inactive_since")

    return CurrentStateSnapshot(
        snapshot_utc=utc_iso(snapshot_ms),
        snapshot_ms=snapshot_ms,
        device_id=device_id_from(device),
        device_name=str(device.get("name", "")),
        active=stringify(attrs.get("active")),
        status=stringify(attrs.get("status")),
        is_offline=is_offline,
        lastDisconnectTime=last_disconnect,
        lastDisconnectTime_iso=last_disconnect_iso,
        lastConnectTime=last_connect,
        lastConnectTime_iso=last_connect_iso,
        lastActivityTime=last_activity,
        lastActivityTime_iso=last_activity_iso,
        inactivityAlarmTime=inactivity_alarm,
        inactivityAlarmTime_iso=inactivity_alarm_iso,
        inactive_since=inactive_since,
        inactive_since_iso=inactive_since_iso,
        inactive_reason=stringify(attrs.get("inactive_reason")),
        raw=compact_raw(attrs),
    )


def row_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def row_int(row: dict[str, str], key: str) -> int | None:
    return parse_time_to_ms(row.get(key))


def row_offline_start_ms(row: dict[str, str]) -> int | None:
    return (
        row_int(row, "inactive_since")
        or row_int(row, "lastDisconnectTime")
        or row_int(row, "inactivityAlarmTime")
        or row_int(row, "lastActivityTime")
    )


def row_connect_ms(row: dict[str, str]) -> int | None:
    return row_int(row, "lastConnectTime")


def read_snapshot_rows(snapshot_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(snapshot_dir.glob("current_state_*.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_snapshot_file"] = path.name
                rows.append(row)
    return rows


def derive_recoveries(snapshot_dir: Path) -> list[SnapshotRecovery]:
    rows_by_device: dict[str, list[dict[str, str]]] = {}
    for row in read_snapshot_rows(snapshot_dir):
        device_id = row.get("device_id", "")
        if device_id:
            rows_by_device.setdefault(device_id, []).append(row)

    recoveries: list[SnapshotRecovery] = []
    seen: set[tuple[str, int, int]] = set()

    for device_id, rows in rows_by_device.items():
        rows.sort(key=lambda row: row_int(row, "snapshot_ms") or 0)
        open_start_ms: int | None = None
        open_row: dict[str, str] | None = None

        for row in rows:
            is_offline = row_bool(row.get("is_offline"))
            if is_offline:
                candidate_start = row_offline_start_ms(row)
                if candidate_start and open_start_ms != candidate_start:
                    open_start_ms = candidate_start
                    open_row = row
                continue

            if open_start_ms is None or open_row is None:
                continue

            connect_ms = row_connect_ms(row)
            snapshot_ms = row_int(row, "snapshot_ms")
            end_ms = connect_ms if connect_ms and connect_ms > open_start_ms else snapshot_ms
            if not end_ms or end_ms < open_start_ms:
                continue

            key = (device_id, open_start_ms, end_ms)
            if key in seen:
                open_start_ms = None
                open_row = None
                continue

            seen.add(key)
            confidence = "medium" if connect_ms and connect_ms > open_start_ms else "low"
            recoveries.append(
                SnapshotRecovery(
                    device_id=device_id,
                    device_name=row.get("device_name") or open_row.get("device_name", ""),
                    offline_start=utc_iso(open_start_ms),
                    offline_end=utc_iso(end_ms),
                    reconnect_time=utc_iso(end_ms),
                    outage_reason="derived from nightly current-state snapshots",
                    source="current_state_snapshot",
                    source_key="active/lastDisconnectTime/lastConnectTime",
                    source_event_id="",
                    duration_sec=duration_seconds(open_start_ms, end_ms),
                    confidence=confidence,
                    raw=compact_raw({"offline_snapshot": open_row, "online_snapshot": row}),
                )
            )
            open_start_ms = None
            open_row = None

    return sorted(recoveries, key=lambda item: (item.device_id, item.offline_start))


def write_csv(path: Path, rows: list[Any], field_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull ThingsBoard current-state snapshots.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output", default="", help="CSV output path. Defaults to output-dir/current_state_YYYYMMDD.csv.")
    parser.add_argument("--jsonl-output", default="", help="Optional JSONL output path.")
    parser.add_argument("--recovery-output", default=str(DEFAULT_RECOVERY_OUTPUT))
    parser.add_argument("--no-derive-recoveries", action="store_true")
    parser.add_argument("--device-offset", type=int, default=0, help="Skip this many devices after sorting by name.")
    parser.add_argument("--max-devices", type=int, default=None, help="Limit devices for smoke tests.")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--request-delay", type=float, default=float(os.getenv("REQUEST_DELAY", "0.05")))
    parser.add_argument(
        "--verify-tls",
        action="store_true",
        default=bool_env("TB_VERIFY_TLS", False),
        help="Verify TLS certificates. Default follows TB_VERIFY_TLS, otherwise false.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    if load_dotenv is not None:
        load_dotenv(override=True)

    args = parse_args(argv)
    host = os.getenv("TB_HOST", "").strip()
    email = os.getenv("TB_EMAIL", "").strip()
    password = os.getenv("TB_PASSWORD", "").strip()
    missing = [name for name, value in (("TB_HOST", host), ("TB_EMAIL", email), ("TB_PASSWORD", password)) if not value]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    today = datetime.now().strftime("%Y%m%d")
    output_path = Path(args.output) if args.output else output_dir / f"current_state_{today}.csv"
    jsonl_path = Path(args.jsonl_output) if args.jsonl_output else None
    recovery_path = Path(args.recovery_output)

    snapshot_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    client = ThingsBoardClient(host, email, password, verify_tls=args.verify_tls)
    fetch_limit = None
    if args.max_devices is not None:
        fetch_limit = max(args.device_offset, 0) + args.max_devices
    devices = client.paginate_devices(args.page_size, args.request_delay, fetch_limit)
    if args.device_offset:
        devices = devices[args.device_offset :]
    if args.max_devices is not None:
        devices = devices[: args.max_devices]

    snapshots: list[CurrentStateSnapshot] = []
    errors: list[str] = []
    print(f"Authenticated. Devices: {len(devices)}. Snapshot: {utc_iso(snapshot_ms)}")

    for idx, device in enumerate(devices, start=1):
        device_id = device_id_from(device)
        name = str(device.get("name", ""))
        if not device_id:
            continue
        if idx == 1 or idx % 10 == 0 or idx == len(devices):
            print(f"[{idx}/{len(devices)}] {name}")
        try:
            attrs = client.get_current_attrs(device_id, DEFAULT_CURRENT_ATTR_KEYS)
            snapshots.append(build_snapshot(device, attrs, snapshot_ms))
        except Exception as exc:
            errors.append(f"{device_id} {name}: {exc}")
        time.sleep(args.request_delay)

    write_csv(output_path, snapshots, list(CurrentStateSnapshot.__dataclass_fields__.keys()))
    if jsonl_path:
        write_jsonl(jsonl_path, snapshots)
    print(f"Wrote {len(snapshots)} current-state snapshots -> {output_path}")

    if not args.no_derive_recoveries:
        recoveries = derive_recoveries(output_dir)
        write_csv(recovery_path, recoveries, list(SnapshotRecovery.__dataclass_fields__.keys()))
        print(f"Wrote {len(recoveries)} derived recoveries -> {recovery_path}")

    if errors:
        print(f"Errors: {len(errors)}. First errors:", file=sys.stderr)
        for err in errors[:10]:
            print(f"  {err}", file=sys.stderr)

    return 0 if snapshots else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
