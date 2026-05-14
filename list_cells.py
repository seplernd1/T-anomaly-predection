import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('TB_Full_Harvest_v10.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_idx = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        code_idx += 1
        src = ''.join(c['source'])
        lines = src.split('\n')
        preview = '\n'.join(lines[:5])
        print(f"=== Code Cell #{code_idx} (nb index {i}) ===")
        print(preview)
        print()
