#!/usr/bin/env python
"""Verify the new key spec (key_spec_manifest.csv) against live tenant keys.

Scans every device's telemetry keys on ThingsBoard, joins with the manifest,
and writes key_spec_verification.csv with live-device counts per key.

Run:  python verify_key_spec.py [--output key_spec_verification.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

MANIFEST_PATH = Path("key_spec_manifest.csv")
DEFAULT_OUTPUT = Path("key_spec_verification.csv")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify key spec manifest against live tenant keys.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--request-delay", type=float, default=float(os.getenv("REQUEST_DELAY", "0.05")))
    parser.add_argument("--json", default="", help="Also dump raw key->device-count map to this JSON file.")
    parser.add_argument("--verify-tls", action="store_true", default=os.getenv("TB_VERIFY_TLS", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--insecure-skip-tls-verify", action="store_true", default=False,
                        help="Explicit opt-in to DISABLE TLS verification (or TB_INSECURE_TLS=1).")
    return parser.parse_args(argv)


def scan_live_keys(host: str, email: str, password: str, page_size: int, delay: float, verify_tls: bool) -> Counter:
    session = requests.Session()
    session.verify = verify_tls
    if not verify_tls:
        try:
            requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
        except Exception:
            pass
    resp = session.post(
        f"{host}/api/auth/login",
        json={"username": email, "password": password},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Login failed HTTP {resp.status_code}")
    headers = {"X-Authorization": f"Bearer {resp.json()['token']}"}

    devices: list[str] = []
    page = 0
    while True:
        data = session.get(
            f"{host}/api/tenant/devices?pageSize={page_size}&page={page}&sortProperty=name&sortOrder=ASC",
            headers=headers,
            timeout=30,
        ).json()
        devices.extend(d["id"]["id"] for d in data.get("data", []))
        if not data.get("hasNext"):
            break
        page += 1
        time.sleep(delay)

    counts: Counter = Counter()
    for i, did in enumerate(devices, start=1):
        try:
            keys = session.get(
                f"{host}/api/plugins/telemetry/DEVICE/{did}/keys/timeseries",
                headers=headers,
                timeout=20,
            ).json()
        except Exception:
            keys = []
        for k in keys or []:
            counts[k] += 1
        if i % 80 == 0 or i == len(devices):
            print(f"  scanned {i}/{len(devices)}")
        time.sleep(delay)
    return counts


def match_live(clean_key: str, live_lower: dict[str, str], counts: Counter) -> int:
    key = clean_key.lower()
    if key in live_lower:
        return counts[live_lower[key]]
    if "." in key:
        tail = key.split(".")[-1]
        if tail in live_lower:
            return counts[live_lower[tail]]
    return 0


def main(argv: list[str]) -> int:
    if load_dotenv is not None:
        load_dotenv(override=True)
    args = parse_args(argv)

    host = os.getenv("TB_HOST", "").strip()
    email = os.getenv("TB_EMAIL", "").strip()
    password = os.getenv("TB_PASSWORD", "").strip()
    missing = [n for n, v in (("TB_HOST", host), ("TB_EMAIL", email), ("TB_PASSWORD", password)) if not v]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2

    manifest = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    print(f"Manifest: {len(manifest)} keys from {args.manifest}")

    from tb_resilient import resolve_verify_tls

    verify_tls = resolve_verify_tls(bool(args.insecure_skip_tls_verify), bool(args.verify_tls))
    print(f"TLS verify={'on' if verify_tls else 'OFF-INSECURE'}")
    counts = scan_live_keys(host, email, password, args.page_size, args.request_delay, verify_tls)
    print(f"Distinct live keys on tenant: {len(counts)}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(counts.most_common()), indent=2), encoding="utf-8"
        )
        print(f"Raw key counts -> {args.json}")

    live_lower = {k.lower(): k for k in counts}
    rows = []
    n_live = 0
    for r in manifest:
        cnt = match_live(r["clean_key"], live_lower, counts)
        if cnt:
            n_live += 1
        rows.append(
            {
                "clean_key": r["clean_key"],
                "group": r["group"],
                "intent": r["intent"],
                "live_devices": cnt,
                "status": "LIVE" if cnt else "NOT_POSTED",
            }
        )

    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clean_key", "group", "intent", "live_devices", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {out}")
    print(f"SUMMARY: {n_live}/{len(rows)} doc keys live; {len(rows) - n_live} not posted yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
