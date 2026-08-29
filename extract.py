import json
with open('ScoreSense_AI.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
with open('extracted_notebook_code.py', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb.get('cells', [])):
        if c.get('cell_type') == 'code':
            out.write(f"\n# --- Cell {(i+1)} ---\n")
            out.write("".join(c.get('source', [])))
            out.write("\n")
