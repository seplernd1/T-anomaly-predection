#!/usr/bin/env python
"""Pull ALL data types from the ThingsBoard tenant in one run.

Single entry point for the "harvest everything" layer:
  1. Devices          -> devices.csv
  2. Customers        -> customers.csv
  3. Assets           -> assets.csv
  4. Relations        -> relations.csv (full asset/relation graph)
  5. Device attributes-> device_attributes.csv (SERVER/CLIENT/SHARED scopes)
  6. Telemetry keys   -> device_telemetry_keys.csv (key discovery per device)
  7. Telemetry series -> telemetry_<key>.csv per device (date-limited fetch)
  8. Device events    -> events_<type>.csv (LC_EVENT, ERROR, STATS, DEBUG)
  9. Alarms           -> alarms.csv
 10. Run manifest     -> manifest.json (row counts, per-step errors, timing)

Reuses the proven client/auth/scoping logic from pull_outage_labels.py
and the snapshot builder from pull_current_state_snapshots.py.

Usage:
    python pull_all_data.py                        # full run
    python pull_all_data.py --telemetry-days 90    # limit telemetry lookback
    python pull_all_data.py --max-devices 5 --plan-only   # smoke test

Outputs are written to  data_harvest/all_data_<ts>/  by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from pull_outage_labels import (
    DEFAULT_CURRENT_ATTR_KEYS,
    DEFAULT_EVENT_TYPES,
    ThingsBoardClient,
    bool_env,
    parse_time_to_ms,
    utc_iso,
)


DEFAULT_OUTPUT_ROOT = Path("data_harvest")
FIVE_DAYS_MS = 5 * 24 * 3600 * 1000


@dataclass
class StepResult:
    name: str
    rows: int = 0
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flat(value: Any) -> str:
    """Stringify nested values for CSV output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, default=str)
    return str(value)


def entity_id(entity: dict[str, Any]) -> str:
    raw = entity.get("id")
    if isinstance(raw, dict):
        return str(raw.get("id", ""))
    return str(raw or "")


# ─── Step 1-3: registry entities ─────────────────────────────────────────────


def pull_customers(client: ThingsBoardClient, page_size: int, delay: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 0
    while True:
        data = client.get_json(
            f"/api/customers?pageSize={page_size}&page={page}&sortProperty=title&sortOrder=ASC"
        )
        if not isinstance(data, dict):
            break
        for c in data.get("data", []):
            rows.append(
                {
                    "customer_id": entity_id(c),
                    "title": c.get("title", ""),
                    "country": c.get("country", ""),
                    "state": c.get("state", ""),
                    "city": c.get("city", ""),
                    "email": c.get("email", ""),
                    "created": utc_iso(c.get("createdTime")),
                }
            )
        if not data.get("hasNext"):
            break
        page += 1
        time.sleep(delay)
    return rows


def pull_assets(client: ThingsBoardClient, page_size: int, delay: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 0
    while True:
        data = client.get_json(
            f"/api/tenant/assets?pageSize={page_size}&page={page}&sortProperty=name&sortOrder=ASC"
        )
        if not isinstance(data, dict):
            break
        for a in data.get("data", []):
            aid = entity_id(a)
            rows.append(
                {
                    "asset_id": aid,
                    "name": a.get("name", ""),
                    "type": a.get("type", ""),
                    "label": a.get("label", ""),
                    "customer_id": (a.get("customerId") or {}).get("id", ""),
                    "created": utc_iso(a.get("createdTime")),
                }
            )
        if not data.get("hasNext"):
            break
        page += 1
        time.sleep(delay)
    return rows


def pull_devices(client: ThingsBoardClient, page_size: int, delay: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 0
    while True:
        data = client.get_json(
            f"/api/tenant/devices?pageSize={page_size}&page={page}&sortProperty=name&sortOrder=ASC"
        )
        if not isinstance(data, dict):
            break
        for d in data.get("data", []):
            rows.append(
                {
                    "device_id": entity_id(d),
                    "name": d.get("name", ""),
                    "type": d.get("type", ""),
                    "label": d.get("label", ""),
                    "customer_id": (d.get("customerId") or {}).get("id", ""),
                    "device_profile_id": (d.get("deviceProfileId") or {}).get("id", ""),
                    "tenant_id": (d.get("tenantId") or {}).get("id", ""),
                    "created": utc_iso(d.get("createdTime")),
                }
            )
        if not data.get("hasNext"):
            break
        page += 1
        time.sleep(delay)
    return rows


# ─── Step 4: relations graph ─────────────────────────────────────────────────


def pull_relations(
    client: ThingsBoardClient,
    devices: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    delay: float,
    include_all_assets: bool = False,
) -> list[dict[str, Any]]:
    """Walk parent relations for devices (the bank->branch->device graph).

    The device walk already recurses up through parent assets to the customer,
    so the full ancestry is captured per device. Set include_all_assets=True
    to additionally walk standalone asset trees not reachable from any device.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def walk(etype: str, eid: str, depth: int = 0) -> None:
        if depth > 8:
            return
        data = client.get_json(f"/api/relations?toId={quote(eid, safe='')}&toType={etype}")
        if not isinstance(data, list):
            return
        for rel in data:
            frm = rel.get("from") or {}
            ptype = str(frm.get("entityType", ""))
            pid = str(frm.get("id", ""))
            if not ptype or not pid:
                continue
            key = (ptype, pid, eid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "from_type": ptype,
                    "from_id": pid,
                    "to_type": etype,
                    "to_id": eid,
                    "relation_type": rel.get("type", ""),
                }
            )
            walk(ptype, pid, depth + 1)

    for d in devices:
        walk("DEVICE", d["device_id"], 0)
        time.sleep(delay)

    if include_all_assets:
        for a in assets:
            walk("ASSET", a["asset_id"], 0)
            time.sleep(delay)
    return rows


# ─── Step 5: attributes (all scopes) ─────────────────────────────────────────


def pull_device_attributes(
    client: ThingsBoardClient,
    devices: list[dict[str, Any]],
    delay: float,
    state_attrs: bool,
) -> list[dict[str, Any]]:
    """All attributes for each device. state_attrs=True also pulls the
    connectivity keys used by the nightly snapshot (active, lastConnect*)."""
    rows: list[dict[str, Any]] = []
    keys_param = ""
    if state_attrs:
        keys_param = "?keys=" + ",".join(quote(k, safe="") for k in DEFAULT_CURRENT_ATTR_KEYS)

    for d in devices:
        did = d["device_id"]
        for scope in ("SERVER_SCOPE", "CLIENT_SCOPE", "SHARED_SCOPE"):
            try:
                suffix = f"/values/attributes/{scope}"
                data = client.get_json(f"/api/plugins/telemetry/DEVICE/{did}{suffix}{keys_param}")
                if isinstance(data, list):
                    for item in data:
                        rows.append(
                            {
                                "device_id": did,
                                "device_name": d["name"],
                                "scope": scope,
                                "key": item.get("key", ""),
                                "value": flat(item.get("value")),
                                "last_update_ts": utc_iso(item.get("lastUpdateTs")),
                            }
                        )
            except Exception as exc:
                rows.append(
                    {
                        "device_id": did,
                        "device_name": d["name"],
                        "scope": scope,
                        "key": "_error",
                        "value": str(exc)[:300],
                        "last_update_ts": "",
                    }
                )
        time.sleep(delay)
    return rows


# ─── Step 6: telemetry key discovery ─────────────────────────────────────────


def pull_telemetry_keys(
    client: ThingsBoardClient,
    devices: list[dict[str, Any]],
    delay: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    keys_by_device: dict[str, list[str]] = {}
    for d in devices:
        did = d["device_id"]
        try:
            keys = client.get_timeseries_keys(did)
        except Exception as exc:
            keys = []
            rows.append(
                {
                    "device_id": did,
                    "device_name": d["name"],
                    "key": "_error",
                    "error": str(exc)[:300],
                }
            )
        keys_by_device[did] = keys
        for k in keys:
            rows.append({"device_id": did, "device_name": d["name"], "key": k, "error": ""})
        time.sleep(delay)
    return rows, keys_by_device


# ─── Step 7: telemetry series ────────────────────────────────────────────────


def load_alias_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def apply_aliases(keys_by_device: dict[str, list[str]], alias_map: dict[str, str]) -> None:
    """Rename keys in-place: alias -> canonical, keeping one entry per canonical name.

    NOTE: used only for reporting (device_telemetry_keys.csv). Telemetry fetches
    must use the raw tenant key names; aliasing for series output happens at
    row-write time in pull_telemetry_series.
    """
    if not alias_map:
        return
    for did, keys in keys_by_device.items():
        renamed: list[str] = []
        for k in keys:
            nk = alias_map.get(k, k)
            if nk not in renamed:
                renamed.append(nk)
        keys_by_device[did] = renamed


def pull_telemetry_series(
    client: ThingsBoardClient,
    devices: list[dict[str, Any]],
    keys_by_device: dict[str, list[str]],
    out_dir: Path,
    start_ms: int,
    end_ms: int,
    delay: float,
    limit: int,
    key_filter: str | None,
    max_keys_per_device: int,
    alias_map: dict[str, str],
    result: StepResult,
) -> None:
    """Fetch per-device timeseries, chunked into <=5-day windows (TB limit).
    One CSV per device: telemetry_<device_id>.csv with columns
    device_id, raw_key, key, ts, ts_iso, value. Fetches use the RAW tenant key
    name; `key` in the output is the canonical spec name via alias_map."""
    key_re = re.compile(key_filter) if key_filter else None
    for d in devices:
        did = d["device_id"]
        keys = keys_by_device.get(did, [])
        if key_re:
            keys = [k for k in keys if key_re.search(k)]
        if max_keys_per_device:
            keys = keys[:max_keys_per_device]
        if not keys:
            continue

        entries_out: list[dict[str, Any]] = []
        for key in keys:
            try:
                entries = client.get_timeseries(did, key, start_ms, end_ms, limit, delay)
            except Exception as exc:
                result.errors.append(f"telemetry {did} {key}: {str(exc)[:200]}")
                continue
            canonical = alias_map.get(key) or alias_map.get(key.lower()) or key
            for e in entries:
                ts = parse_time_to_ms(e.get("ts"))
                entries_out.append(
                    {
                        "device_id": did,
                        "raw_key": key,
                        "key": canonical,
                        "ts": e.get("ts", ""),
                        "ts_iso": utc_iso(ts) if ts else "",
                        "value": flat(e.get("value")),
                    }
                )
            time.sleep(delay)

        if entries_out:
            path = out_dir / f"telemetry_{did}.csv"
            write_csv_rows(
                path,
                entries_out,
                ["device_id", "raw_key", "key", "ts", "ts_iso", "value"],
            )
            result.files.append(path.name)
            result.rows += len(entries_out)


# ─── Step 8-9: events + alarms ───────────────────────────────────────────────


def pull_events(
    client: ThingsBoardClient,
    devices: list[dict[str, Any]],
    event_types: list[str],
    start_ms: int,
    end_ms: int,
    page_size: int,
    delay: float,
    result: StepResult,
    out_dir: Path | None = None,
) -> None:
    by_type: dict[str, list[dict[str, Any]]] = {et: [] for et in event_types}
    for d in devices:
        did = d["device_id"]
        tenant_id = d.get("tenant_id", "")
        for et in event_types:
            try:
                events = client.get_events(did, tenant_id, et, start_ms, end_ms, page_size, delay)
            except Exception as exc:
                result.errors.append(f"events {did} {et}: {str(exc)[:200]}")
                continue
            for e in events:
                ts = parse_time_to_ms(e.get("createdTime"))
                by_type[et].append(
                    {
                        "device_id": did,
                        "device_name": d.get("name", ""),
                        "event_type": et,
                        "created": utc_iso(ts) if ts else "",
                        "event_id": entity_id(e) or flat(e.get("id")),
                        "payload": flat(e),
                    }
                )
            time.sleep(delay)

    for et, rows in by_type.items():
        if not rows:
            continue
        path = out_dir / f"events_{et.lower()}.csv"
        write_csv_rows(
            path,
            rows,
            ["device_id", "device_name", "event_type", "created", "event_id", "payload"],
        )
        result.files.append(path.name)
        result.rows += len(rows)


def pull_alarms(
    client: ThingsBoardClient,
    devices: list[dict[str, Any]],
    start_ms: int,
    end_ms: int,
    page_size: int,
    delay: float,
    result: StepResult,
    out_dir: Path | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    for d in devices:
        did = d["device_id"]
        try:
            alarms = client.get_alarms(did, start_ms, end_ms, page_size, delay)
        except Exception as exc:
            result.errors.append(f"alarms {did}: {str(exc)[:200]}")
            continue
        for a in alarms:
            start = parse_time_to_ms(a.get("startTs"))
            rows.append(
                {
                    "device_id": did,
                    "device_name": d.get("name", ""),
                    "alarm_type": a.get("type", ""),
                    "severity": a.get("severity", ""),
                    "status": a.get("status", ""),
                    "start": utc_iso(start) if start else "",
                    "end": utc_iso(parse_time_to_ms(a.get("endTs"))),
                    "ack": utc_iso(parse_time_to_ms(a.get("ackTs"))),
                    "clear": utc_iso(parse_time_to_ms(a.get("clearTs"))),
                    "payload": flat(a),
                }
            )
        time.sleep(delay)
    if rows:
        path = out_dir / "alarms.csv"
        write_csv_rows(
            path,
            rows,
            [
                "device_id",
                "device_name",
                "alarm_type",
                "severity",
                "status",
                "start",
                "end",
                "ack",
                "clear",
                "payload",
            ],
        )
        result.files.append(path.name)
        result.rows += len(rows)


# ─── orchestrator ────────────────────────────────────────────────────────────


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull ALL ThingsBoard data types in one run.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Root output folder (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--telemetry-days", type=int, default=365, help="Telemetry lookback in days.")
    parser.add_argument("--events-days", type=int, default=365, help="Events/alarms lookback in days.")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=1000, help="Timeseries page limit.")
    parser.add_argument("--request-delay", type=float, default=float(os.getenv("REQUEST_DELAY", "0.05")))
    parser.add_argument("--device-offset", type=int, default=0)
    parser.add_argument("--max-devices", type=int, default=None, help="Limit devices for smoke tests.")
    parser.add_argument("--key-filter", default="", help="Regex to limit telemetry keys (e.g. 'active|heartbeat').")
    parser.add_argument(
        "--alias-map",
        default="",
        help="JSON file mapping tenant key -> canonical spec name; renamed on output when both forms appear.",
    )
    parser.add_argument(
        "--max-keys-per-device",
        type=int,
        default=0,
        help="Cap keys fetched per device (0 = all). Useful for smoke tests.",
    )
    parser.add_argument("--skip-telemetry", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--skip-alarms", action="store_true")
    parser.add_argument("--skip-relations", action="store_true")
    parser.add_argument(
        "--all-asset-relations",
        action="store_true",
        help="Also walk standalone asset trees (slower; device ancestry is always included).",
    )
    parser.add_argument("--no-state-attrs", action="store_true", help="Skip connectivity attr keys filter.")
    parser.add_argument("--plan-only", action="store_true", help="List steps and exit without pulling.")
    parser.add_argument(
        "--verify-tls",
        action="store_true",
        default=bool_env("TB_VERIFY_TLS", False),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    if load_dotenv is not None:
        load_dotenv(override=True)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args(argv)

    steps = [
        "devices/customers/assets registry",
        "relations graph",
        "device attributes",
        "telemetry keys",
        "telemetry series",
        "events",
        "alarms",
        "manifest",
    ]
    if args.plan_only:
        print("Plan (nothing fetched):")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")
        print(f"telemetry_days={args.telemetry_days} events_days={args.events_days} "
              f"max_devices={args.max_devices} key_filter={args.key_filter or '(none)'}")
        return 0

    host = os.getenv("TB_HOST", "").strip()
    email = os.getenv("TB_EMAIL", "").strip()
    password = os.getenv("TB_PASSWORD", "").strip()
    missing = [n for n, v in (("TB_HOST", host), ("TB_EMAIL", email), ("TB_PASSWORD", password)) if not v]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2

    run_ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.output_root) / f"all_data_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    telemetry_start_ms = int(end_ms - args.telemetry_days * 24 * 3600 * 1000)
    events_start_ms = int(end_ms - args.events_days * 24 * 3600 * 1000)

    client = ThingsBoardClient(host, email, password, verify_tls=args.verify_tls)
    alias_map = load_alias_map(args.alias_map)
    if alias_map:
        print(f"Alias map loaded: {len(alias_map)} renames from {args.alias_map}")
    print(f"Authenticated. Output: {out_dir}")

    manifest: dict[str, Any] = {
        "run_ts": run_ts,
        "host": host,
        "telemetry_days": args.telemetry_days,
        "events_days": args.events_days,
        # `window` is the events/alarms window; telemetry uses its own (below).
        "window": {"start": utc_iso(events_start_ms), "end": utc_iso(end_ms)},
        "telemetry_window": {"start": utc_iso(telemetry_start_ms), "end": utc_iso(end_ms)},
        "steps": {},
    }

    def run_step(name: str, fn, *fn_args, **fn_kwargs) -> StepResult:
        result = StepResult(name=name)
        t0 = time.time()
        print(f"[{name}] starting ...")
        try:
            fn(result, *fn_args, **fn_kwargs)
        except Exception as exc:
            result.errors.append(f"step failed: {str(exc)[:300]}")
        result.seconds = round(time.time() - t0, 1)
        manifest["steps"][name] = asdict(result)
        print(f"[{name}] rows={result.rows} files={len(result.files)} "
              f"errors={len(result.errors)} in {result.seconds}s")
        return result

    # 1. Registry: devices + customers + assets
    reg: dict[str, list[dict[str, Any]]] = {}

    def step_registry(result: StepResult) -> None:
        devices = pull_devices(client, args.page_size, args.request_delay)
        customers = pull_customers(client, args.page_size, args.request_delay)
        assets = pull_assets(client, args.page_size, args.request_delay)
        if args.device_offset:
            devices = devices[args.device_offset :]
        if args.max_devices is not None:
            devices = devices[: args.max_devices]
        reg["devices"] = devices
        reg["customers"] = customers
        reg["assets"] = assets
        write_csv_rows(out_dir / "devices.csv", devices,
                       ["device_id", "name", "type", "label", "customer_id", "device_profile_id", "tenant_id", "created"])
        write_csv_rows(out_dir / "customers.csv", customers,
                       ["customer_id", "title", "country", "state", "city", "email", "created"])
        write_csv_rows(out_dir / "assets.csv", assets,
                       ["asset_id", "name", "type", "label", "customer_id", "created"])
        result.rows = len(devices) + len(customers) + len(assets)
        result.files.extend(["devices.csv", "customers.csv", "assets.csv"])

    run_step("registry", step_registry)

    devices = reg.get("devices", [])

    # 2. Relations
    if not args.skip_relations:

        def step_relations(result: StepResult) -> None:
            rels = pull_relations(
                client,
                devices,
                reg.get("assets", []),
                args.request_delay,
                include_all_assets=args.all_asset_relations,
            )
            write_csv_rows(out_dir / "relations.csv", rels,
                           ["from_type", "from_id", "to_type", "to_id", "relation_type"])
            result.rows = len(rels)
            result.files.append("relations.csv")

        run_step("relations", step_relations)

    # 3. Attributes
    def step_attributes(result: StepResult) -> None:
        attr_rows = pull_device_attributes(
            client, devices, args.request_delay, state_attrs=not args.no_state_attrs
        )
        write_csv_rows(out_dir / "device_attributes.csv", attr_rows,
                       ["device_id", "device_name", "scope", "key", "value", "last_update_ts"])
        result.rows = len(attr_rows)
        result.files.append("device_attributes.csv")

    run_step("attributes", step_attributes)

    # 4. Telemetry keys (raw tenant names; aliases applied only for reporting)
    key_rows: list[dict[str, Any]] = []
    keys_by_device: dict[str, list[str]] = {}

    def step_telemetry_keys(result: StepResult) -> None:
        nonlocal key_rows, keys_by_device
        key_rows, keys_by_device = pull_telemetry_keys(client, devices, args.request_delay)
        write_csv_rows(out_dir / "device_telemetry_keys.csv", key_rows,
                       ["device_id", "device_name", "key", "error"])
        result.rows = len(key_rows)
        result.files.append("device_telemetry_keys.csv")

    run_step("telemetry_keys", step_telemetry_keys)

    # 5. Telemetry series
    if not args.skip_telemetry:

        def step_telemetry(result: StepResult) -> None:
            tele_dir = out_dir / "telemetry"
            result.files.append("telemetry/")
            pull_telemetry_series(
                client,
                devices,
                keys_by_device,
                tele_dir,
                telemetry_start_ms,
                end_ms,
                args.request_delay,
                args.limit,
                args.key_filter or None,
                args.max_keys_per_device,
                alias_map,
                result,
            )

        run_step("telemetry", step_telemetry)

    # 6. Events
    if not args.skip_events:
        event_types = list(DEFAULT_EVENT_TYPES)

        def step_events(result: StepResult) -> None:
            pull_events(client, devices, event_types, events_start_ms, end_ms,
                        args.page_size, args.request_delay, result, out_dir)

        run_step("events", step_events)

    # 7. Alarms
    if not args.skip_alarms:

        def step_alarms(result: StepResult) -> None:
            pull_alarms(client, devices, events_start_ms, end_ms, args.page_size, args.request_delay, result, out_dir)

        run_step("alarms", step_alarms)

    # 8. Manifest
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest -> {manifest_path}")

    total_rows = sum(s.get("rows", 0) for s in manifest["steps"].values())
    total_errors = sum(len(s.get("errors", [])) for s in manifest["steps"].values())
    print(f"DONE: {total_rows} rows, {total_errors} errors across {len(manifest['steps'])} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
