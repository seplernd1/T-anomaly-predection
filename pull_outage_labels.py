#!/usr/bin/env python
"""Pull real-ish outage labels from ThingsBoard.

The existing harvest produces heuristic fault labels. This script builds a
separate outage-label dataset with columns suitable for supervised training:

    device_id, offline_start, offline_end, reconnect_time, outage_reason

It tries, in order:
1. ThingsBoard device events (connect/disconnect/activity/inactivity).
2. ThingsBoard alarms whose type/details look like offline/no-data alarms.
3. Timeseries state transitions from discovered connectivity keys.
4. Current device-state attributes for still-open outages.

Credentials are read from .env/environment:
    TB_HOST, TB_EMAIL, TB_PASSWORD
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Iterable
from urllib.parse import quote

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


DEFAULT_EVENT_TYPES = (
    "LC_EVENT",
    "ERROR",
    "STATS",
    "DEBUG",
)

DEFAULT_CURRENT_ATTR_KEYS = (
    "active",
    "status",
    "lastConnectTime",
    "lastDisconnectTime",
    "lastActivityTime",
    "inactivityAlarmTime",
    "inactive_since",
    "inactive_reason",
)

DEFAULT_KEY_REGEX = (
    r"(active|connect|disconnect|offline|inactiv|lastActivity|"
    r"lastConnect|lastDisconnect|heartbeat.*offline|device_status|"
    r"gatewayStatus|gw_status|status$)"
)

OFFLINE_RE = re.compile(
    r"\b(disconnect(?:ed)?|offline|inactive|inactivity|no[_ -]?data|"
    r"heartbeat[_ -]?stop|down)\b",
    re.I,
)
ONLINE_RE = re.compile(r"\b(connect(?:ed)?|online|active|activity|up)\b", re.I)

OFFLINE_VALUES = {
    "false",
    "0",
    "off",
    "offline",
    "inactive",
    "disconnected",
    "disconnect",
    "down",
    "no_data",
    "no data",
}
ONLINE_VALUES = {
    "true",
    "1",
    "on",
    "online",
    "active",
    "connected",
    "connect",
    "up",
    "healthy",
}


@dataclass
class OutageLabel:
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


def utc_iso(ms: int | float | None) -> str:
    if ms is None:
        return ""
    try:
        ms_int = int(float(ms))
    except Exception:
        return ""
    if ms_int <= 0:
        return ""
    return datetime.fromtimestamp(ms_int / 1000, tz=timezone.utc).isoformat()


def parse_time_to_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        return v if v > 10_000_000_000 else v * 1000

    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "nan", "n/a", "-"}:
        return None
    if s.isdigit():
        v = int(s)
        return v if v > 10_000_000_000 else v * 1000

    normalized = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y, %I:%M:%S %p",
        "%d/%m/%Y, %I:%M %p",
    ):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            continue
    return None


def duration_seconds(start_ms: int | None, end_ms: int | None) -> int | None:
    if not start_ms or not end_ms or end_ms < start_ms:
        return None
    return int((end_ms - start_ms) / 1000)


def compact_raw(value: Any, limit: int = 1200) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        raw = str(value)
    return raw if len(raw) <= limit else raw[: limit - 3] + "..."


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def classify_value(value: Any, key: str = "") -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "online" if value else "offline"
    s = str(value).strip().lower()
    if not s:
        return "unknown"
    if s in OFFLINE_VALUES:
        return "offline"
    if s in ONLINE_VALUES:
        return "online"
    if OFFLINE_RE.search(s):
        return "offline"
    if ONLINE_RE.search(s) and not re.search(r"disconnect", s, re.I):
        return "online"
    if key.lower() == "active" and s in {"false", "0"}:
        return "offline"
    return "unknown"


def classify_text(value: Any) -> str:
    text = compact_raw(value, 2000)
    if OFFLINE_RE.search(text):
        return "offline"
    if ONLINE_RE.search(text) and not re.search(r"disconnect", text, re.I):
        return "online"
    return "unknown"


class ThingsBoardClient:
    def __init__(self, host: str, email: str, password: str, verify_tls: bool):
        self.host = host.rstrip("/")
        self.session = requests.Session()
        self.session.verify = verify_tls
        if not verify_tls:
            try:
                requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
            except Exception:
                pass
        resp = self.session.post(
            f"{self.host}/api/auth/login",
            json={"username": email, "password": password},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ThingsBoard login failed HTTP {resp.status_code}: {resp.text[:300]}")
        token = resp.json()["token"]
        self.headers = {
            "X-Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def get_json(self, path: str, timeout: int = 30) -> Any:
        url = path if path.startswith("http") else f"{self.host}{path}"
        resp = self.session.get(url, headers=self.headers, timeout=timeout)
        if resp.status_code == 401:
            raise RuntimeError("JWT expired or unauthorized")
        if resp.status_code in {404, 405}:
            return None
        if resp.status_code != 200:
            raise RuntimeError(f"GET {url} failed HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def paginate_devices(self, page_size: int, delay: float, max_devices: int | None = None) -> list[dict[str, Any]]:
        devices: list[dict[str, Any]] = []
        page = 0
        while True:
            data = self.get_json(
                "/api/tenant/devices"
                f"?pageSize={page_size}&page={page}"
                "&sortProperty=name&sortOrder=ASC"
            )
            if not isinstance(data, dict):
                break
            devices.extend(data.get("data", []))
            if max_devices and len(devices) >= max_devices:
                return devices[:max_devices]
            if not data.get("hasNext"):
                break
            page += 1
            time.sleep(delay)
        return devices

    def get_current_attrs(self, device_id: str, keys: Iterable[str]) -> dict[str, Any]:
        key_str = ",".join(quote(k, safe="") for k in keys)
        merged: dict[str, Any] = {}
        for scope in ("SERVER_SCOPE", "CLIENT_SCOPE", "SHARED_SCOPE"):
            data = self.get_json(
                f"/api/plugins/telemetry/DEVICE/{device_id}/values/attributes/{scope}?keys={key_str}",
                timeout=20,
            )
            if isinstance(data, list):
                for item in data:
                    key = item.get("key")
                    if key and key not in merged:
                        merged[key] = item.get("value")
        return merged

    def get_timeseries_keys(self, device_id: str) -> list[str]:
        data = self.get_json(f"/api/plugins/telemetry/DEVICE/{device_id}/keys/timeseries", timeout=20)
        return [x for x in (data or []) if isinstance(x, str)]

    def get_timeseries(
        self,
        device_id: str,
        key: str,
        start_ms: int,
        end_ms: int,
        limit: int,
        delay: float,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        cursor = start_ms
        safe_key = quote(key, safe="")
        while cursor <= end_ms:
            data = self.get_json(
                f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries"
                f"?keys={safe_key}&startTs={cursor}&endTs={end_ms}"
                f"&limit={limit}&orderBy=ASC&agg=NONE&useStrictDataTypes=false",
                timeout=45,
            )
            batch = []
            if isinstance(data, dict):
                batch = data.get(key) or []
            if not batch:
                break
            entries.extend(batch)
            last_ts = max(int(x.get("ts", cursor)) for x in batch if x.get("ts") is not None)
            if len(batch) < limit or last_ts < cursor:
                break
            cursor = last_ts + 1
            time.sleep(delay)
        return entries

    def get_events(
        self,
        device_id: str,
        tenant_id: str,
        event_type: str,
        start_ms: int,
        end_ms: int,
        page_size: int,
        delay: float,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page = 0
        while True:
            tenant_query = f"&tenantId={quote(tenant_id, safe='')}" if tenant_id else ""
            path = (
                f"/api/events/DEVICE/{device_id}/{quote(event_type, safe='')}"
                f"?startTime={start_ms}&endTime={end_ms}"
                f"&pageSize={page_size}&page={page}{tenant_query}"
                "&sortProperty=createdTime&sortOrder=ASC"
            )
            url = f"{self.host}{path}"
            resp = self.session.get(url, headers=self.headers, timeout=30)
            if resp.status_code == 401:
                raise RuntimeError("JWT expired or unauthorized")
            if resp.status_code in {404, 405}:
                break
            if resp.status_code == 400 and "not supported" in resp.text.lower():
                break
            if resp.status_code != 200:
                raise RuntimeError(f"GET {url} failed HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            if not isinstance(data, dict):
                break
            out.extend(data.get("data", []))
            if not data.get("hasNext"):
                break
            page += 1
            time.sleep(delay)
        return out

    def get_alarms(
        self,
        device_id: str,
        start_ms: int,
        end_ms: int,
        page_size: int,
        delay: float,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page = 0
        while True:
            data = self.get_json(
                f"/api/alarm/DEVICE/{device_id}"
                f"?startTime={start_ms}&endTime={end_ms}"
                f"&searchStatus=ANY&pageSize={page_size}&page={page}"
                "&sortProperty=createdTime&sortOrder=ASC",
                timeout=30,
            )
            if data is None:
                break
            if not isinstance(data, dict):
                break
            out.extend(data.get("data", []))
            if not data.get("hasNext"):
                break
            page += 1
            time.sleep(delay)
        return out


def label_from_pair(
    device: dict[str, Any],
    start_ms: int,
    end_ms: int | None,
    reason: str,
    source: str,
    source_key: str = "",
    source_event_id: str = "",
    confidence: str = "medium",
    raw: Any = None,
) -> OutageLabel:
    device_id = device.get("id", {}).get("id", "") if isinstance(device.get("id"), dict) else str(device.get("id", ""))
    end_iso = utc_iso(end_ms)
    return OutageLabel(
        device_id=device_id,
        device_name=str(device.get("name", "")),
        offline_start=utc_iso(start_ms),
        offline_end=end_iso,
        reconnect_time=end_iso,
        outage_reason=reason,
        source=source,
        source_key=source_key,
        source_event_id=source_event_id,
        duration_sec=duration_seconds(start_ms, end_ms),
        confidence=confidence,
        raw=compact_raw(raw or {}),
    )


def event_time_ms(event: dict[str, Any]) -> int | None:
    for key in ("createdTime", "ts", "time", "eventTime"):
        parsed = parse_time_to_ms(event.get(key))
        if parsed:
            return parsed
    return None


def build_event_labels(device: dict[str, Any], events: list[dict[str, Any]]) -> list[OutageLabel]:
    labels: list[OutageLabel] = []
    current_start: tuple[int, dict[str, Any]] | None = None

    for event in sorted(events, key=lambda e: event_time_ms(e) or 0):
        ts = event_time_ms(event)
        if not ts:
            continue
        state = classify_text(event)
        event_type = str(event.get("type") or event.get("eventType") or "")
        event_id = ""
        if isinstance(event.get("id"), dict):
            event_id = str(event["id"].get("id", ""))
        elif event.get("id") is not None:
            event_id = str(event.get("id"))

        if state == "offline":
            if current_start is None:
                current_start = (ts, event)
        elif state == "online" and current_start is not None:
            start_ms, start_event = current_start
            labels.append(
                label_from_pair(
                    device,
                    start_ms,
                    ts,
                    reason=f"{event_type or 'event'}: reconnect after offline event",
                    source="event",
                    source_key=event_type,
                    source_event_id=event_id,
                    confidence="high",
                    raw={"start_event": start_event, "end_event": event},
                )
            )
            current_start = None

    if current_start is not None:
        start_ms, start_event = current_start
        labels.append(
            label_from_pair(
                device,
                start_ms,
                None,
                reason="open outage from last offline event",
                source="event",
                source_key=str(start_event.get("type") or start_event.get("eventType") or ""),
                source_event_id="",
                confidence="high",
                raw=start_event,
            )
        )
    return labels


def build_alarm_labels(device: dict[str, Any], alarms: list[dict[str, Any]]) -> list[OutageLabel]:
    labels: list[OutageLabel] = []
    for alarm in alarms:
        if classify_text(alarm) != "offline":
            continue
        start_ms = (
            parse_time_to_ms(alarm.get("startTs"))
            or parse_time_to_ms(alarm.get("createdTime"))
            or parse_time_to_ms(alarm.get("createdTimeMs"))
        )
        if not start_ms:
            continue
        end_ms = (
            parse_time_to_ms(alarm.get("endTs"))
            or parse_time_to_ms(alarm.get("clearTs"))
            or parse_time_to_ms(alarm.get("ackTs"))
        )
        alarm_id = ""
        if isinstance(alarm.get("id"), dict):
            alarm_id = str(alarm["id"].get("id", ""))
        elif alarm.get("id") is not None:
            alarm_id = str(alarm.get("id"))
        labels.append(
            label_from_pair(
                device,
                start_ms,
                end_ms,
                reason=str(alarm.get("type") or "offline/no-data alarm"),
                source="alarm",
                source_key=str(alarm.get("type") or ""),
                source_event_id=alarm_id,
                confidence="high" if end_ms else "medium",
                raw=alarm,
            )
        )
    return labels


def build_timeseries_labels(
    device: dict[str, Any],
    series_by_key: dict[str, list[dict[str, Any]]],
) -> list[OutageLabel]:
    labels: list[OutageLabel] = []

    # Direct last-disconnect/last-connect timestamp telemetry.
    disconnects: list[tuple[int, str, dict[str, Any]]] = []
    connects: list[tuple[int, str, dict[str, Any]]] = []
    for key, entries in series_by_key.items():
        key_l = key.lower()
        for entry in entries:
            value_ms = parse_time_to_ms(entry.get("value"))
            event_ms = value_ms or parse_time_to_ms(entry.get("ts"))
            if not event_ms:
                continue
            if "lastdisconnect" in key_l or re.search(r"disconnect.*last|offline.*last", key_l):
                disconnects.append((event_ms, key, entry))
            elif "lastconnect" in key_l or re.search(r"connect.*last", key_l):
                connects.append((event_ms, key, entry))

    connects.sort(key=lambda x: x[0])
    for start_ms, key, raw in sorted(disconnects, key=lambda x: x[0]):
        end_ms = next((ts for ts, _, _ in connects if ts > start_ms), None)
        labels.append(
            label_from_pair(
                device,
                start_ms,
                end_ms,
                reason=f"{key} timestamp transition",
                source="timeseries_timestamp",
                source_key=key,
                confidence="medium" if end_ms else "low",
                raw=raw,
            )
        )

    # State transition telemetry such as active=false -> active=true.
    for key, entries in series_by_key.items():
        current_start: tuple[int, dict[str, Any]] | None = None
        for entry in sorted(entries, key=lambda e: int(e.get("ts", 0))):
            ts = parse_time_to_ms(entry.get("ts"))
            if not ts:
                continue
            state = classify_value(entry.get("value"), key)
            if state == "offline":
                if current_start is None:
                    current_start = (ts, entry)
            elif state == "online" and current_start is not None:
                start_ms, start_raw = current_start
                labels.append(
                    label_from_pair(
                        device,
                        start_ms,
                        ts,
                        reason=f"{key} offline->online transition",
                        source="timeseries_state",
                        source_key=key,
                        confidence="medium",
                        raw={"start": start_raw, "end": entry},
                    )
                )
                current_start = None
        if current_start is not None:
            start_ms, start_raw = current_start
            labels.append(
                label_from_pair(
                    device,
                    start_ms,
                    None,
                    reason=f"{key} currently/offline without later online state",
                    source="timeseries_state",
                    source_key=key,
                    confidence="low",
                    raw=start_raw,
                )
            )

    return labels


def build_current_attr_label(device: dict[str, Any], attrs: dict[str, Any]) -> list[OutageLabel]:
    labels: list[OutageLabel] = []
    active_state = classify_value(attrs.get("active"), "active")
    status_state = classify_value(attrs.get("status"), "status")
    if active_state != "offline" and status_state != "offline":
        return labels

    start_ms = (
        parse_time_to_ms(attrs.get("inactive_since"))
        or parse_time_to_ms(attrs.get("lastDisconnectTime"))
        or parse_time_to_ms(attrs.get("inactivityAlarmTime"))
        or parse_time_to_ms(attrs.get("lastActivityTime"))
    )
    if not start_ms:
        return labels
    reasons = []
    if active_state == "offline":
        reasons.append(f"active={attrs.get('active')}")
    if status_state == "offline":
        reasons.append(f"status={attrs.get('status')}")
    if attrs.get("inactive_reason"):
        reasons.append(f"inactive_reason={attrs.get('inactive_reason')}")
    if attrs.get("inactivityAlarmTime"):
        reasons.append("has_inactivityAlarmTime")
    if not reasons:
        reasons.append("current offline/inactive attributes")
    reason = "; ".join(str(x) for x in reasons)
    labels.append(
        label_from_pair(
            device,
            start_ms,
            None,
            reason=reason,
            source="current_attributes",
            source_key="active/status/lastDisconnectTime",
            confidence="low",
            raw=attrs,
        )
    )
    return labels


def dedupe_labels(labels: list[OutageLabel], tolerance_sec: int = 60) -> list[OutageLabel]:
    ranked = {"high": 0, "medium": 1, "low": 2}
    labels = sorted(
        labels,
        key=lambda x: (
            x.device_id,
            x.offline_start,
            ranked.get(x.confidence, 9),
            x.source,
        ),
    )
    out: list[OutageLabel] = []
    seen: dict[tuple[str, int], OutageLabel] = {}
    for label in labels:
        start_ms = parse_time_to_ms(label.offline_start)
        if not start_ms:
            continue
        bucket = int(start_ms / 1000 / tolerance_sec)
        key = (label.device_id, bucket)
        existing = seen.get(key)
        if existing is None:
            seen[key] = label
            out.append(label)
            continue
        if ranked.get(label.confidence, 9) < ranked.get(existing.confidence, 9):
            out.remove(existing)
            out.append(label)
            seen[key] = label
    return sorted(out, key=lambda x: (x.device_id, x.offline_start, x.source))


def write_outputs(labels: list[OutageLabel], csv_path: str, jsonl_path: str | None) -> None:
    fields = list(asdict(labels[0]).keys()) if labels else list(OutageLabel.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            writer.writerow(asdict(label))

    if jsonl_path:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull ThingsBoard outage labels.")
    parser.add_argument("--days", type=int, default=365, help="Lookback window in days.")
    parser.add_argument("--start-ms", type=int, default=None, help="Explicit start epoch ms.")
    parser.add_argument("--end-ms", type=int, default=None, help="Explicit end epoch ms.")
    parser.add_argument("--output", default="", help="CSV output path.")
    parser.add_argument("--jsonl-output", default="", help="Optional JSONL output path.")
    parser.add_argument("--device-offset", type=int, default=0, help="Skip this many devices after sorting by name.")
    parser.add_argument("--max-devices", type=int, default=None, help="Limit devices for smoke tests.")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=1000, help="Timeseries page limit.")
    parser.add_argument("--request-delay", type=float, default=float(os.getenv("REQUEST_DELAY", "0.05")))
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--skip-alarms", action="store_true")
    parser.add_argument("--skip-timeseries", action="store_true")
    parser.add_argument("--skip-current", action="store_true")
    parser.add_argument("--event-types", default=",".join(DEFAULT_EVENT_TYPES))
    parser.add_argument("--key-regex", default=DEFAULT_KEY_REGEX)
    parser.add_argument(
        "--extra-keys",
        default="",
        help="Comma-separated timeseries keys to always fetch in addition to discovered matches.",
    )
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

    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    end_ms = args.end_ms or now_ms
    start_ms = args.start_ms or int((datetime.now(tz=timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = args.output or f"outage_labels_{stamp}.csv"
    jsonl_output = args.jsonl_output or ""

    client = ThingsBoardClient(host, email, password, verify_tls=args.verify_tls)
    fetch_limit = None
    if args.max_devices is not None:
        fetch_limit = max(args.device_offset, 0) + args.max_devices
    devices = client.paginate_devices(args.page_size, args.request_delay, fetch_limit)
    if args.device_offset:
        devices = devices[args.device_offset :]
    if args.max_devices is not None:
        devices = devices[: args.max_devices]
    print(f"Authenticated. Devices: {len(devices)}. Window: {utc_iso(start_ms)} -> {utc_iso(end_ms)}")

    event_types = [x.strip() for x in args.event_types.split(",") if x.strip()]
    key_re = re.compile(args.key_regex, re.I)
    extra_keys = {x.strip() for x in args.extra_keys.split(",") if x.strip()}

    all_labels: list[OutageLabel] = []
    errors: list[str] = []

    for idx, device in enumerate(devices, start=1):
        device_id = device.get("id", {}).get("id", "") if isinstance(device.get("id"), dict) else str(device.get("id", ""))
        tenant_id = (
            device.get("tenantId", {}).get("id", "")
            if isinstance(device.get("tenantId"), dict)
            else str(device.get("tenantId", "") or "")
        )
        name = str(device.get("name", ""))
        if not device_id:
            continue
        if idx == 1 or idx % 10 == 0 or idx == len(devices):
            print(f"[{idx}/{len(devices)}] {name}")

        if not args.skip_events:
            try:
                events: list[dict[str, Any]] = []
                for event_type in event_types:
                    events.extend(
                        client.get_events(
                            device_id,
                            tenant_id,
                            event_type,
                            start_ms,
                            end_ms,
                            args.page_size,
                            args.request_delay,
                        )
                    )
                all_labels.extend(build_event_labels(device, events))
            except Exception as exc:
                errors.append(f"{device_id} {name} events: {exc}")

        if not args.skip_alarms:
            try:
                alarms = client.get_alarms(device_id, start_ms, end_ms, args.page_size, args.request_delay)
                all_labels.extend(build_alarm_labels(device, alarms))
            except Exception as exc:
                errors.append(f"{device_id} {name} alarms: {exc}")

        if not args.skip_timeseries:
            try:
                discovered = client.get_timeseries_keys(device_id)
                selected_keys = sorted({k for k in discovered if key_re.search(k)} | extra_keys)
                series_by_key: dict[str, list[dict[str, Any]]] = {}
                for key in selected_keys:
                    entries = client.get_timeseries(
                        device_id,
                        key,
                        start_ms,
                        end_ms,
                        args.limit,
                        args.request_delay,
                    )
                    if entries:
                        series_by_key[key] = entries
                all_labels.extend(build_timeseries_labels(device, series_by_key))
            except Exception as exc:
                errors.append(f"{device_id} {name} timeseries: {exc}")

        if not args.skip_current:
            try:
                attrs = client.get_current_attrs(device_id, DEFAULT_CURRENT_ATTR_KEYS)
                all_labels.extend(build_current_attr_label(device, attrs))
            except Exception as exc:
                errors.append(f"{device_id} {name} current_attrs: {exc}")

        time.sleep(args.request_delay)

    labels = dedupe_labels(all_labels)
    write_outputs(labels, output, jsonl_output or None)
    print(f"Wrote {len(labels)} outage labels -> {output}")
    if jsonl_output:
        print(f"Wrote JSONL -> {jsonl_output}")
    if errors:
        print(f"Errors: {len(errors)}. First errors:", file=sys.stderr)
        for err in errors[:10]:
            print(f"  {err}", file=sys.stderr)
    return 0 if labels else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
