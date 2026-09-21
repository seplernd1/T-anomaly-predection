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

from tb_resilient import (
    AuthError,
    CoverageLedger,
    CoverageRecord,
    CheckpointStore,
    DevicePolicy,
    PermanentError,
    RetryExhaustedError,
    RetryPolicy,
    atomic_write_bytes,
    http_get,
    resolve_verify_tls,
    resolve_window,
    sanitize_csv_value,
    write_csv_rows_safe,
)


DEFAULT_OUTPUT_ROOT = Path("data_harvest")
FIVE_DAYS_MS = 5 * 24 * 3600 * 1000
RUNS_DIR_NAME = "runs"

EXIT_INCOMPLETE = 5


def resolve_verify_tls(args: argparse.Namespace) -> bool:
    """TLS verification is ON by default; see tb_resilient.resolve_verify_tls."""
    from tb_resilient import resolve_verify_tls as _resolve

    return _resolve(bool(getattr(args, "insecure_skip_tls_verify", False)),
                    bool(getattr(args, "verify_tls", False)))


@dataclass
class StepResult:
    name: str
    rows: int = 0
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    # Sanitized (formula-injection neutralized) + atomic.
    write_csv_rows_safe(path, rows, fieldnames)


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
    per_device: dict[str, dict[str, Any]] | None = None,
) -> None:
    by_type: dict[str, list[dict[str, Any]]] = {et: [] for et in event_types}
    for d in devices:
        did = d["device_id"]
        dev_stat = {"rows": 0, "oldest": "", "newest": "", "error": ""}
        tenant_id = d.get("tenant_id", "")
        for et in event_types:
            try:
                events = client.get_events(did, tenant_id, et, start_ms, end_ms, page_size, delay)
            except Exception as exc:
                result.errors.append(f"events {did} {et}: {str(exc)[:200]}")
                dev_stat["error"] = str(exc)[:200]
                continue
            for e in events:
                ts = parse_time_to_ms(e.get("createdTime"))
                iso = utc_iso(ts) if ts else ""
                if iso and (not dev_stat["oldest"] or iso < dev_stat["oldest"]):
                    dev_stat["oldest"] = iso
                if iso and (not dev_stat["newest"] or iso > dev_stat["newest"]):
                    dev_stat["newest"] = iso
                by_type[et].append(
                    {
                        "device_id": did,
                        "device_name": d.get("name", ""),
                        "event_type": et,
                        "created": iso,
                        "event_id": entity_id(e) or flat(e.get("id")),
                        "payload": flat(e),
                    }
                )
                dev_stat["rows"] += 1
            time.sleep(delay)
        if per_device is not None:
            per_device[did] = dev_stat

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
    per_device: dict[str, dict[str, Any]] | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    for d in devices:
        did = d["device_id"]
        dev_stat = {"rows": 0, "oldest": "", "newest": "", "error": ""}
        try:
            alarms = client.get_alarms(did, start_ms, end_ms, page_size, delay)
        except Exception as exc:
            result.errors.append(f"alarms {did}: {str(exc)[:200]}")
            dev_stat["error"] = str(exc)[:200]
            if per_device is not None:
                per_device[did] = dev_stat
            continue
        for a in alarms:
            start = parse_time_to_ms(a.get("startTs"))
            iso = utc_iso(start) if start else ""
            if iso and (not dev_stat["oldest"] or iso < dev_stat["oldest"]):
                dev_stat["oldest"] = iso
            if iso and (not dev_stat["newest"] or iso > dev_stat["newest"]):
                dev_stat["newest"] = iso
            rows.append(
                {
                    "device_id": did,
                    "device_name": d.get("name", ""),
                    "alarm_type": a.get("type", ""),
                    "severity": a.get("severity", ""),
                    "status": a.get("status", ""),
                    "start": iso,
                    "end": utc_iso(parse_time_to_ms(a.get("endTs"))),
                    "ack": utc_iso(parse_time_to_ms(a.get("ackTs"))),
                    "clear": utc_iso(parse_time_to_ms(a.get("clearTs"))),
                    "payload": flat(a),
                }
            )
            dev_stat["rows"] += 1
        time.sleep(delay)
        if per_device is not None:
            per_device[did] = dev_stat
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
        help="Legacy flag: verify TLS certificates (now the default; kept for compatibility).",
    )
    parser.add_argument(
        "--insecure-skip-tls-verify",
        action="store_true",
        default=False,
        help="Explicit opt-in to DISABLE TLS verification (or TB_INSECURE_TLS=1). "
             "Emits a warning; use only on trusted networks.",
    )
    parser.add_argument(
        "--run-id", default="",
        help="Run folder name under <output-root>/runs/ (default: UTC timestamp).",
    )
    parser.add_argument(
        "--resume-from", default="",
        help="Resume an interrupted run: reuse checkpoints/coverage from runs/<ID>/ "
             "and skip completed device+key+window chunks.",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Exit 0 even with incomplete/failed windows (default: exit 5).",
    )
    parser.add_argument(
        "--min-chunk-hours", type=float, default=6.0,
        help="Smallest time chunk when splitting capped windows (default: 6h).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="HTTP retry attempts for transient failures (default: 5).",
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

    run_ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or run_ts
    out_dir = Path(args.output_root) / RUNS_DIR_NAME / run_id
    reg_dir = out_dir / "registry"
    tele_dir = out_dir / "telemetry"
    events_dir = out_dir / "events"
    alarms_dir = out_dir / "alarms"
    coverage_dir = out_dir / "coverage"
    quarantine_dir = out_dir / "quarantine"
    reports_dir = out_dir / "reports"
    for d in (reg_dir, tele_dir, events_dir, alarms_dir, coverage_dir, quarantine_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    verify_tls = resolve_verify_tls(args)
    retry_policy = RetryPolicy(max_attempts=max(1, args.max_retries))

    resumed_from = ""
    prior_manifest: dict[str, Any] = {}
    if args.resume_from:
        prior = Path(args.output_root) / RUNS_DIR_NAME / args.resume_from
        ckpt_path = prior / "coverage" / "checkpoints.json"
        if ckpt_path.is_file():
            import shutil as _shutil
            _shutil.copyfile(ckpt_path, coverage_dir / "checkpoints.json")
            resumed_from = args.resume_from
            print(f"Resuming: loaded checkpoints from runs/{args.resume_from}/")
        else:
            print(f"WARNING: --resume-from {args.resume_from}: no checkpoints.json found; full run.",
                  file=sys.stderr)
        try:
            prior_manifest = json.loads((prior / "manifest.json").read_text(encoding="utf-8"))
        except Exception:
            prior_manifest = {}

    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    if resumed_from and isinstance(prior_manifest.get("telemetry_window"), dict):
        # Resume continues the PRIOR window exactly so checkpoint keys match;
        # a fresh window requires a fresh run (no --resume-from).
        frozen = parse_time_to_ms(prior_manifest["telemetry_window"].get("end"))
        if frozen:
            end_ms = frozen
            print(f"Resuming: frozen window end {utc_iso(end_ms)} from runs/{resumed_from}/")
    telemetry_start_ms = int(end_ms - args.telemetry_days * 24 * 3600 * 1000)
    events_start_ms = int(end_ms - args.events_days * 24 * 3600 * 1000)

    client = ThingsBoardClient(host, email, password, verify_tls=verify_tls)
    alias_map = load_alias_map(args.alias_map)
    if alias_map:
        print(f"Alias map loaded: {len(alias_map)} renames from {args.alias_map}")
    print(f"Authenticated (TLS verify={'on' if verify_tls else 'OFF-INSECURE'}). Output: {out_dir}")

    ledger = CoverageLedger(run_id, coverage_dir / "telemetry_coverage.jsonl")
    checkpoints = CheckpointStore(coverage_dir / "checkpoints.json")
    policy = DevicePolicy.from_env()
    source_coverage: list[dict[str, Any]] = []
    cap_hits_total = 0
    retrieved_at = datetime.now(tz=timezone.utc).isoformat()

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "run_ts": run_ts,
        "host": host,
        "tls_verify": verify_tls,
        "resumed_from": resumed_from,
        "allow_partial": bool(args.allow_partial),
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
        # Full pagination until exhausted (hasNext); never a silent small default.
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
        for c in customers:
            c["retrieved_at"] = retrieved_at
            c["endpoint"] = "/api/customers"
        for a in assets:
            a["retrieved_at"] = retrieved_at
            a["endpoint"] = "/api/tenant/assets"
        write_csv_rows(reg_dir / "customers.csv", customers,
                       ["customer_id", "title", "country", "state", "city", "email", "created",
                        "retrieved_at", "endpoint"])
        write_csv_rows(reg_dir / "assets.csv", assets,
                       ["asset_id", "name", "type", "label", "customer_id", "created",
                        "retrieved_at", "endpoint"])
        result.rows = len(devices) + len(customers) + len(assets)
        result.files.extend(["registry/customers.csv", "registry/assets.csv"])
        source_coverage.append({"run_id": run_id, "source": "registry",
                                "devices": len(devices), "customers": len(customers),
                                "assets": len(assets), "status": "complete", "error": ""})

    run_step("registry", step_registry)

    devices = reg.get("devices", [])

    # 2. Relations
    if not args.skip_relations:

        def step_relations(result: StepResult) -> None:
            # Full graph over all discovered devices (eligibility filters only
            # the heavy per-device pulls: attributes/telemetry/events/alarms).
            rels = pull_relations(
                client,
                devices,
                reg.get("assets", []),
                args.request_delay,
                include_all_assets=args.all_asset_relations,
            )
            write_csv_rows(reg_dir / "relations.csv", rels,
                           ["from_type", "from_id", "to_type", "to_id", "relation_type"])
            result.rows = len(rels)
            result.files.append("registry/relations.csv")

        run_step("relations", step_relations)

    # 4. Telemetry keys (raw tenant names; aliases applied only for reporting)
    key_rows: list[dict[str, Any]] = []
    keys_by_device: dict[str, list[str]] = {}

    def step_telemetry_keys(result: StepResult) -> None:
        nonlocal key_rows, keys_by_device
        key_rows, keys_by_device = pull_telemetry_keys(client, devices, args.request_delay)
        write_csv_rows(reg_dir / "device_telemetry_keys.csv", key_rows,
                       ["device_id", "device_name", "key", "error"])
        result.rows = len(key_rows)
        result.files.append("registry/device_telemetry_keys.csv")

    run_step("telemetry_keys", step_telemetry_keys)

    # 4b. Eligibility: real vs test/demo/simulator (auditable, never name-only).
    eligible: list[dict[str, Any]] = []
    key_errors: set[str] = set()

    def step_eligibility(result: StepResult) -> None:
        nonlocal eligible
        for r in key_rows:
            if r.get("key") == "_error" and r.get("error"):
                key_errors.add(r["device_id"])
        decisions: list[dict[str, Any]] = []
        for d in devices:
            kc = len(keys_by_device.get(d["device_id"], []))
            if d["device_id"] in key_errors:
                dec = DeviceDecision(False, "key_discovery_failed")
            else:
                dec = policy.decide(d, kc)
            d["eligible"] = "true" if dec.eligible else "false"
            d["eligibility_reason"] = dec.reason
            d["retrieved_at"] = retrieved_at
            d["endpoint"] = "/api/tenant/devices"
            if dec.eligible:
                eligible.append(d)
            else:
                decisions.append({
                    "device_id": d["device_id"], "device_name": d.get("name", ""),
                    "type": d.get("type", ""), "customer_id": d.get("customer_id", ""),
                    "reason": dec.reason,
                })
        write_csv_rows(reg_dir / "devices.csv", devices,
                       ["device_id", "name", "type", "label", "customer_id", "device_profile_id",
                        "tenant_id", "created", "eligible", "eligibility_reason",
                        "retrieved_at", "endpoint"])
        write_csv_rows(reports_dir / "device_exclusions.csv", decisions,
                       ["device_id", "device_name", "type", "customer_id", "reason"])
        result.rows = len(devices)
        result.files.extend(["registry/devices.csv", "reports/device_exclusions.csv"])
        print(f"[eligibility] eligible={len(eligible)} excluded={len(decisions)}")

    run_step("eligibility", step_eligibility)

    # 3b. Attributes (eligible devices only; runs after eligibility).
    def step_attributes(result: StepResult) -> None:
        attr_rows = pull_device_attributes(
            client, eligible, args.request_delay, state_attrs=not args.no_state_attrs
        )
        write_csv_rows(reg_dir / "device_attributes.csv", attr_rows,
                       ["device_id", "device_name", "scope", "key", "value", "last_update_ts"])
        result.rows = len(attr_rows)
        result.files.append("registry/device_attributes.csv")
        for d in eligible:
            source_coverage.append({"run_id": run_id, "device_id": d["device_id"], "source": "attributes",
                                    "rows": sum(1 for r in attr_rows if r["device_id"] == d["device_id"]),
                                    "oldest": "", "newest": "", "error": "", "status": "complete"})

    run_step("attributes", step_attributes)

    # 5. Telemetry series (resilient: retry, cap paging + window splitting,
    #    per device/key coverage, checkpoints, parquet output).
    gaps: list[dict[str, Any]] = []

    if not args.skip_telemetry:

        def step_telemetry(result: StepResult) -> None:
            nonlocal cap_hits_total
            import pandas as pd

            key_re = re.compile(args.key_filter) if args.key_filter else None
            min_chunk_ms = int(args.min_chunk_hours * 3600 * 1000)
            result.files.append("telemetry/")
            for d in eligible:
                did = d["device_id"]
                keys = keys_by_device.get(did, [])
                if key_re:
                    keys = [k for k in keys if key_re.search(k)]
                if args.max_keys_per_device:
                    keys = keys[: args.max_keys_per_device]
                dev_rows: list[dict[str, Any]] = []
                for raw_key in keys:
                    if checkpoints.is_done(did, raw_key, telemetry_start_ms, end_ms):
                        continue
                    safe = quote(raw_key, safe="")

                    def page_fn(cursor_ms: int, end: int, limit: int,
                                _did: str = did, _safe: str = safe,
                                _rk: str = raw_key) -> tuple[list[dict[str, Any]], int]:
                        url = (
                            f"{client.host}/api/plugins/telemetry/DEVICE/{_did}/values/timeseries"
                            f"?keys={_safe}&startTs={cursor_ms}&endTs={end}"
                            f"&limit={limit}&orderBy=ASC&agg=NONE&useStrictDataTypes=false"
                        )
                        resp, retries = http_get(client.session, url, client.headers, 45, retry_policy)
                        data = resp.json()
                        batch = data.get(_rk) or [] if isinstance(data, dict) else []
                        return batch, retries

                    outcome = resolve_window(page_fn, telemetry_start_ms, end_ms,
                                             args.limit, min_chunk_ms)
                    cap_hits_total += outcome.cap_hits
                    ledger.add(CoverageRecord(
                        run_id=run_id, device_id=did, key=raw_key, endpoint="timeseries",
                        requested_start_ms=telemetry_start_ms, requested_end_ms=end_ms,
                        chunk_start_ms=telemetry_start_ms, chunk_end_ms=end_ms,
                        pages=outcome.pages,
                        status_code=200 if outcome.completeness in ("complete", "empty_verified") else None,
                        retries=outcome.retries, rows=len(outcome.points),
                        oldest_ts=outcome.oldest_ts, newest_ts=outcome.newest_ts,
                        completeness=outcome.completeness, error=outcome.error,
                    ))
                    if outcome.completeness in ("complete", "empty_verified"):
                        checkpoints.mark_done(did, raw_key, telemetry_start_ms, end_ms)
                    else:
                        gaps.append({"device_id": did, "key": raw_key,
                                     "window_start": utc_iso(telemetry_start_ms),
                                     "window_end": utc_iso(end_ms),
                                     "completeness": outcome.completeness,
                                     "pages": outcome.pages, "retries": outcome.retries,
                                     "cap_hits": outcome.cap_hits, "error": outcome.error})
                    canonical = alias_map.get(raw_key) or alias_map.get(raw_key.lower()) or raw_key
                    for e in outcome.points:
                        ts = parse_time_to_ms(e.get("ts"))
                        dev_rows.append({
                            "device_id": did, "raw_key": raw_key, "key": canonical,
                            "ts": e.get("ts", ""), "ts_iso": utc_iso(ts) if ts else "",
                            "value": flat(e.get("value")),
                        })
                    time.sleep(args.request_delay)
                if dev_rows:
                    pd.DataFrame(dev_rows).to_parquet(tele_dir / f"telemetry_{did}.parquet", index=False)
                    result.files.append(f"telemetry/telemetry_{did}.parquet")
                    result.rows += len(dev_rows)

        run_step("telemetry", step_telemetry)

    # 6. Events (eligible devices only; tenantId required by this TB version)
    events_stats: dict[str, dict[str, Any]] = {}
    if not args.skip_events:
        event_types = list(DEFAULT_EVENT_TYPES)

        def step_events(result: StepResult) -> None:
            pull_events(client, eligible, event_types, events_start_ms, end_ms,
                        args.page_size, args.request_delay, result, events_dir, events_stats)

        run_step("events", step_events)
        for did, st in events_stats.items():
            source_coverage.append({"run_id": run_id, "device_id": did, "source": "events",
                                    "window_start": utc_iso(events_start_ms),
                                    "window_end": utc_iso(end_ms), **st,
                                    "status": "failed" if st["error"] else ("complete" if st["rows"] else "empty"),
                                    })

    # 7. Alarms (eligible devices only)
    alarms_stats: dict[str, dict[str, Any]] = {}
    if not args.skip_alarms:

        def step_alarms(result: StepResult) -> None:
            pull_alarms(client, eligible, events_start_ms, end_ms, args.page_size,
                        args.request_delay, result, alarms_dir, alarms_stats)

        run_step("alarms", step_alarms)
        for did, st in alarms_stats.items():
            source_coverage.append({"run_id": run_id, "device_id": did, "source": "alarms",
                                    "window_start": utc_iso(events_start_ms),
                                    "window_end": utc_iso(end_ms), **st,
                                    "status": "failed" if st["error"] else ("complete" if st["rows"] else "empty"),
                                    })

    # 8. Coverage, quarantine, completeness report, manifest
    if gaps:
        write_csv_rows(quarantine_dir / "telemetry_gaps.csv", gaps,
                       ["device_id", "key", "window_start", "window_end", "completeness",
                        "pages", "retries", "cap_hits", "error"])
    atomic_write_bytes(coverage_dir / "source_coverage.jsonl",
                       "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in source_coverage).encode("utf-8"))

    ledger_summary = ledger.summary()
    error_cats: dict[str, int] = {}
    for step in manifest["steps"].values():
        for e in step.get("errors", []):
            cat = e.split(":")[0].split(" ")[0][:60]
            error_cats[cat] = error_cats.get(cat, 0) + 1
    completeness = {
        "run_id": run_id,
        "discovered_devices": len(devices),
        "eligible_devices": len(eligible),
        "excluded_devices": len(devices) - len(eligible),
        "telemetry": ledger_summary,
        "cap_hits_total": cap_hits_total,
        "error_categories": error_cats,
        "unresolved_gaps": len(ledger.incomplete()),
        "sources": {s: len([r for r in source_coverage if r["source"] == s])
                    for s in {r["source"] for r in source_coverage}},
    }
    manifest["eligibility"] = {"eligible": len(eligible), "excluded": len(devices) - len(eligible)}
    manifest["coverage"] = ledger_summary
    manifest["completeness"] = completeness
    atomic_write_bytes(reports_dir / "completeness.json",
                       json.dumps(completeness, indent=2, ensure_ascii=False).encode("utf-8"))
    manifest_path = out_dir / "manifest.json"
    atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    print(f"Manifest -> {manifest_path}")

    total_rows = sum(s.get("rows", 0) for s in manifest["steps"].values())
    total_errors = sum(len(s.get("errors", [])) for s in manifest["steps"].values())
    incomplete = len(ledger.incomplete())
    print(f"DONE: {total_rows} rows, {total_errors} step errors, "
          f"{incomplete} incomplete windows, {cap_hits_total} cap hits "
          f"across {len(manifest['steps'])} steps.")
    if incomplete or total_errors:
        msg = (f"INCOMPLETE: {incomplete} unresolved windows, {total_errors} step errors. "
               f"See {quarantine_dir}/ and {reports_dir}/completeness.json.")
        print(msg, file=sys.stderr)
        if not args.allow_partial:
            return EXIT_INCOMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
