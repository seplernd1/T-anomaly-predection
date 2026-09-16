#!/usr/bin/env python
"""Create the new-spec key manifest from the docx/pdf key document.

Extracts the 3-column table (key | intent | Description) from
'Untitled document (1).docx' / 'Untitled document (1).pdf', cleans JS
snippets, normalizes namespace variants, and writes:

  key_spec_manifest.csv  - one row per logical key: raw_key, clean_key, group,
                           intent, description
  key_spec_manifest.json - grouped by subsystem group

Run:  python build_key_spec_manifest.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import docx

DOCX_PATH = Path("Untitled document (1).docx")
OUT_CSV = Path("key_spec_manifest.csv")
OUT_JSON = Path("key_spec_manifest.json")

# --------------------------------------------------------------- group rules

GROUP_RULES: list[tuple[str, re.Pattern]] = [
    ("gateway", re.compile(r"^gateway", re.I)),
    ("cctv", re.compile(r"^(cctv|rock|camera|hdd|systemstatus|hddstatus|cameratamper|cameradisconnect)", re.I)),
    ("ias", re.compile(r"^ias", re.I)),
    ("fas", re.compile(r"^fas", re.I)),
    ("bas", re.compile(r"^(bas|bassystemintegration)", re.I)),
    ("timelock", re.compile(r"^timelock", re.I)),
    ("access_control", re.compile(r"^accesscontrol", re.I)),
    ("power", re.compile(r"^(ac_voltage|battery_voltage|system_current)$", re.I)),
    ("statusbox", re.compile(r"^(system_status|statusbox)", re.I)),
]

# ------------------------------------------- derived (JS-computed) key names

DERIVED_KEYS: list[tuple[str, str]] = [
    # (marker in raw cell text, canonical derived key name)
    ("totalcapacity", "rock.hdd_total_capacity"),
    ("videodetails.length", "rock.total_cameras"),
    ("hddslot", "rock.hdd_info"),
    ("errorcount", "rock.hdd_error_count"),
    ("cameradetails", "rock.camera_details"),
    ("sdrecinfo", "rock.sd_recording_info"),
    ("zoneinfo", "bas_zone_info"),
    ("videodetails", "rock.recording_info"),
]


# ------------------------------------------------- canonical-name normalizer

VARIANTS: dict[str, str] = {
    # Namespaced duplicate of the bare statusbox keys (tenant posts the bare form)
    "system_status.statusbox_mains_on": "statusbox_mains_on",
    "system_status.statusbox_network": "statusbox_network",
    "system_status.statusbox_no_of_connected_device": "statusbox_no_of_connected_device",
    "systemstatus": "systemStatus",
    # Case drift seen on the tenant: Rock.* and rock.* both posted
    "Rock.dexter_date": "rock.dexter_date",
    "Rock.manufacturer": "rock.manufacturer",
    "Rock.model": "rock.model",
    "Rock.time": "rock.Time",
}


def normalize(raw: str) -> str:
    key = raw.strip()
    return VARIANTS.get(key, key)


def classify(key: str) -> str:
    for group, pattern in GROUP_RULES:
        if pattern.search(key):
            return group
    return "other"


# ------------------------------------------------------------- doc extraction

def extract_rows() -> list[dict[str, str]]:
    doc = docx.Document(str(DOCX_PATH))
    rows: list[dict[str, str]] = []
    for tbl in doc.tables:
        for r in tbl.rows:
            cells = [c.text.strip() for c in r.cells]
            if len(cells) < 3:
                continue
            raw_key, intent, desc = cells[0], cells[1], cells[2]
            if raw_key.lower() == "key" or not raw_key:
                continue
            rows.append({"raw_key": raw_key, "intent": intent, "description": desc})
    return rows


def is_js_snippet(raw_key: str, description: str) -> bool:
    combined = f"{raw_key}\n{description}".lower()
    markers = ("const ", "foreach", "console.log", "=>", "let ", "++;", "${")
    return any(m in combined for m in markers)


def clean_compound(raw_key: str) -> list[str]:
    """Split multi-key cells on / and newlines; drop prose."""
    parts = re.split(r"[/\n]", raw_key)
    out = []
    for p in parts:
        p = p.strip()
        if not p or p.lower() == "key":
            continue
        if " " in p and not re.match(r"^[\w.]+$", p):
            continue  # prose line inside cell
        out.append(p)
    return out


def main() -> int:
    rows = extract_rows()
    manifest: list[dict[str, str]] = []
    for row in rows:
        keys = clean_compound(row["raw_key"])
        if not keys:
            continue
        desc = row["description"].replace("\n", " ").strip()
        if is_js_snippet(row["raw_key"], desc):
            # JS-derived key: match against DERIVED_KEYS for a stable name
            snippet = f'{row["raw_key"]} {row["description"]}'.lower()
            clean_key = ""
            for marker, canonical in DERIVED_KEYS:
                if marker in snippet:
                    clean_key = canonical
                    break
            manifest.append(
                {
                    "raw_key": row["raw_key"].replace("\n", " ")[:120],
                    "clean_key": clean_key,
                    "group": classify(clean_key) if clean_key else "other",
                    "intent": row["intent"].replace("\n", " ").strip(),
                    "description": f"[derived] {desc}"[:400],
                }
            )
            continue
        for k in keys:
            ck = normalize(k)
            manifest.append(
                {
                    "raw_key": k,
                    "clean_key": ck,
                    "group": classify(ck),
                    "intent": row["intent"].replace("\n", " ").strip(),
                    "description": desc[:400],
                }
            )

    # dedupe by clean_key (case-insensitive; keep first, join intents)
    merged: dict[str, dict[str, str]] = {}
    for item in manifest:
        ck = item["clean_key"] or item["raw_key"][:60]
        mkey = ck.lower()
        if mkey in merged:
            existing = merged[mkey]
            if item["intent"] and item["intent"] not in existing["intent"]:
                existing["intent"] = f'{existing["intent"]} / {item["intent"]}'
            continue
        merged[mkey] = item
    final = list(merged.values())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_key", "clean_key", "group", "intent", "description"])
        writer.writeheader()
        writer.writerows(final)

    grouped: dict[str, list[dict[str, str]]] = {}
    for item in final:
        grouped.setdefault(item["group"], []).append(item)
    OUT_JSON.write_text(json.dumps(grouped, indent=2, ensure_ascii=False), encoding="utf-8")

    # Alias map: tenant-posted key -> canonical spec name (inverted VARIANTS,
    # only entries where the canonical name differs from the raw doc name).
    aliases = {
        raw: canon for raw, canon in VARIANTS.items() if raw != canon
    }
    Path("key_spec_aliases.json").write_text(
        json.dumps(aliases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(aliases)} aliases -> key_spec_aliases.json")

    print(f"Wrote {len(final)} logical keys -> {OUT_CSV} / {OUT_JSON}")
    for group, items in sorted(grouped.items()):
        print(f"  {group:<15} {len(items):>2} keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
