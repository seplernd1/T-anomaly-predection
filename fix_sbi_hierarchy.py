"""
Patch TB_Full_Harvest_v9.ipynb to fix SBI hierarchy (LHO not showing).

Root causes:
1. Customer matching: "STATE BANK OF INDIA" doesn't substring-match "SBI LHO KOLKATA"
2. resolve_hierarchy_walk: CUSTOMER entities are never classified as lho/zo/rbo
3. SBI's LHO/RBO exist as CUSTOMER entities, not ASSETs

Run:  python fix_sbi_hierarchy.py
"""
import json, copy

NB_PATH = "TB_Full_Harvest_v9.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

def get_cell_source(nb, cell_id):
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return cell
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1 — Cell 3: Add aliases to SBI + add classify_entity_level function
# ═══════════════════════════════════════════════════════════════════════════════
c3 = get_cell_source(nb, "c3")
src = "".join(c3["source"])

# 1a. Add aliases to STATE BANK OF INDIA
old_sbi = """\
    'STATE BANK OF INDIA': {
        'depth': 6,
        'levels': ['ho','lho','zo','rbo','branch'],
        'type_map': {
            'HO':'ho','LHO':'lho','Local Head Office':'lho',
            'ZO':'zo','RBO':'rbo','Branch':'branch',
        },
    },"""

new_sbi = """\
    'STATE BANK OF INDIA': {
        'depth': 6,
        'aliases': ['SBI', 'STATE BANK'],
        'levels': ['ho','lho','zo','rbo','branch'],
        'type_map': {
            'HO':'ho','LHO':'lho','Local Head Office':'lho',
            'SBI LHO':'lho','ZONE':'zo',
            'ZO':'zo','RBO':'rbo','Branch':'branch',
        },
    },"""

if old_sbi in src:
    src = src.replace(old_sbi, new_sbi)
    print("✅ FIX 1a: Added aliases to STATE BANK OF INDIA")
else:
    print("⚠️  FIX 1a: Could not find SBI block — already patched?")

# 1b. Add classify_entity_level function after get_asset_level
classifier_fn = '''
import re
_LEVEL_KW = [
    (re.compile(r'\\bLHO\\b|Local\\s+Head\\s+Office', re.I), 'lho'),
    (re.compile(r'\\bRBO\\b|Regional\\s+Banking', re.I),     'rbo'),
    (re.compile(r'\\bNBG\\b|\\bFGMO\\b', re.I),              'nbg'),
    (re.compile(r'\\bZO\\b|\\bZONE\\b|Zonal\\s+Office', re.I), 'zo'),
    (re.compile(r'\\bRO\\b|Regional\\s+Office', re.I),       'ro'),
    (re.compile(r'\\bCO\\b|Circle\\s+Office', re.I),         'co'),
    (re.compile(r'\\bHO\\b|Head\\s+Office|Corporate\\s+Office|Central\\s+Office', re.I), 'ho'),
]
def classify_entity_level(name):
    """Detect hierarchy level from any entity name (customer or asset title)."""
    for pat, level in _LEVEL_KW:
        if pat.search(name or ''):
            return level
    return None

'''

anchor = "# ── CLIENT ATTRIBUTE KEYS"
if "classify_entity_level" not in src:
    src = src.replace(anchor, classifier_fn + anchor)
    print("✅ FIX 1b: Added classify_entity_level() function")
else:
    print("⚠️  FIX 1b: classify_entity_level already exists")

c3["source"] = [src]

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2 — Cell 5: Use aliases in customer→bank matching
# ═══════════════════════════════════════════════════════════════════════════════
c5 = get_cell_source(nb, "c5")
src5 = "".join(c5["source"])

old_match = """\
    bank_cfg = None
    for bank_name, cfg in BANK_HIERARCHY.items():
        if bank_name.lower() in title.lower() or title.lower() in bank_name.lower():
            bank_cfg = cfg
            bank_cfg['bank_name'] = bank_name
            break"""

new_match = """\
    bank_cfg = None
    for bank_name, cfg in BANK_HIERARCHY.items():
        aliases = cfg.get('aliases', [])
        tlow = title.lower()
        if (bank_name.lower() in tlow
            or tlow in bank_name.lower()
            or any(a.lower() in tlow for a in aliases)):
            bank_cfg = cfg
            bank_cfg['bank_name'] = bank_name
            break"""

if old_match in src5:
    src5 = src5.replace(old_match, new_match)
    print("✅ FIX 2:  Updated customer matching to use aliases")
else:
    print("⚠️  FIX 2:  Could not find old matching logic — already patched?")

c5["source"] = [src5]

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3 — Cell 8: Classify CUSTOMER entities as hierarchy levels in walk
# ═══════════════════════════════════════════════════════════════════════════════
c8 = get_cell_source(nb, "c8")
src8 = "".join(c8["source"])

old_cust_handler = """\
        if ptype == 'CUSTOMER' and pid in cust_map:
            c = cust_map[pid]
            result.setdefault('bank_name',  c['bank_name'])
            result.setdefault('nbg_name',   c['nbg_name'])"""

new_cust_handler = """\
        if ptype == 'CUSTOMER' and pid in cust_map:
            c = cust_map[pid]
            result.setdefault('bank_name',  c['bank_name'])
            result.setdefault('nbg_name',   c['nbg_name'])
            # ── NEW: classify customer entity as hierarchy level ──────
            clvl = classify_entity_level(c['customer_title'])
            cname = c['customer_title']
            if   clvl == 'lho': result.setdefault('lho_name', cname)
            elif clvl == 'rbo': result.setdefault('rbo_name', cname)
            elif clvl == 'zo':  result.setdefault('zo_name',  cname)
            elif clvl == 'ro':  result.setdefault('ro_name',  cname)
            elif clvl == 'co':  result.setdefault('co_name',  cname)
            elif clvl == 'ho':  result.setdefault('ho_name',  cname)
            elif clvl == 'nbg': result.setdefault('nbg_name', cname)"""

if old_cust_handler in src8:
    src8 = src8.replace(old_cust_handler, new_cust_handler)
    print("✅ FIX 3:  Updated resolve_hierarchy_walk to classify customers")
else:
    print("⚠️  FIX 3:  Could not find old CUSTOMER handler — already patched?")

c8["source"] = [src8]

# ═══════════════════════════════════════════════════════════════════════════════
# Clear outputs (force re-run to see new results)
# ═══════════════════════════════════════════════════════════════════════════════
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n🎉 Patched {NB_PATH} — re-run all cells (1–12) to see LHO in SBI hierarchy.")
