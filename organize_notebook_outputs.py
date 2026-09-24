"""One-shot migration: organize notebook outputs into excel/ json/ jsonl/ csv/.

Run once:  python organize_notebook_outputs.py
Idempotent: re-running makes no further changes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

NB = Path("TB_Full_Harvest_v11.ipynb")

# per-line exact replacements (match after strip, rewrite whole line)
LINE_MAP = {
    "XLSX_PATH  = f'tb_audit_v11_{ts}.xlsx'":
        "XLSX_PATH  = f'excel/tb_audit_v11_{ts}.xlsx'",
    "JSON_PATH  = 'dashboard_data.json'":
        "JSON_PATH  = 'json/dashboard_data.json'",
    "TS_JSONL_PATH    = 'ml_training_timeseries.jsonl'":
        "TS_JSONL_PATH    = 'jsonl/ml_training_timeseries.jsonl'",
    "COMBINED_JSONL   = 'ml_training_combined.jsonl'":
        "COMBINED_JSONL   = 'jsonl/ml_training_combined.jsonl'",
    "print('   ❌ Close the open ts_summary_*.xlsx and re-run Cell 19.')":
        "print('   ❌ Close the open excel/ts_summary_*.xlsx and re-run Cell 19.')",
}
# substring replacements (any occurrence on a line)
SUB_MAP = [
    ("f'ts_summary_", "f'excel/ts_summary_"),
    ("f'ts_daily_snapshots_", "f'csv/ts_daily_snapshots_"),
    ("f'dashboard_data_", "f'json/dashboard_data_"),
    ("'ml_training_v11.jsonl'", "'jsonl/ml_training_v11.jsonl'"),  # JSONL_PATH + cell38 read loop
]
MAKEDIRS_26 = ("os.makedirs('excel', exist_ok=True); "
               "os.makedirs('json', exist_ok=True); "
               "os.makedirs('jsonl', exist_ok=True)")
MAKEDIRS_38 = ("os.makedirs('excel', exist_ok=True); os.makedirs('csv', exist_ok=True); "
               "os.makedirs('json', exist_ok=True); os.makedirs('jsonl', exist_ok=True)")


def rewrite_cell(src: list[str], anchor_line: str, insert: str) -> list[str]:
    """Insert `insert` (one code line) right after the line starting with anchor."""
    out: list[str] = []
    done = False
    for line in src:
        out.append(line)
        if not done and line.strip().startswith(anchor_line):
            eol = "\r\n" if line.endswith("\r\n") else "\n"
            out.append(insert + eol)
            done = True
    return out


nb = json.loads(NB.read_text(encoding="utf-8"))
changed_cells = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    new_src = []
    cell_changed = False
    for line in src:
        stripped = line.rstrip("\r\n")
        eol = line[len(stripped):]
        if stripped in LINE_MAP:
            line = LINE_MAP[stripped] + eol
            cell_changed = True
        for old, new in SUB_MAP:
            if old in line:
                line = line.replace(old, new)
                cell_changed = True
        new_src.append(line)
    joined = "".join(new_src)
    if "JSONL_PATH  = 'jsonl/ml_training_v11.jsonl'" in joined and MAKEDIRS_26 not in joined:
        new_src = rewrite_cell(new_src, "JSONL_PATH", MAKEDIRS_26)
        cell_changed = True
    if "TS_CSV_FULL_PATH" in joined and MAKEDIRS_38 not in joined:
        new_src = rewrite_cell(new_src, "DASHBOARD_JSON   =", MAKEDIRS_38)
        cell_changed = True
    if cell_changed:
        cell["source"] = new_src
        changed_cells += 1

if changed_cells:
    backup = NB.with_suffix(".ipynb.pre-organize.bak")
    shutil.copyfile(NB, backup)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"updated {changed_cells} cells (backup -> {backup.name})")
else:
    print("already organized — no changes")
