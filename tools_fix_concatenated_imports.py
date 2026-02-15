"""Fix notebooks with concatenated imports (syntax errors)"""
import re, json
from pathlib import Path

def fix_notebook(path):
    raw = path.read_text(encoding='utf-8')
    # Split concatenated imports on lines that are too long (>500 chars)
    fixed = re.sub(r'(import\s+\w+[\s,\w]*)\s+(import\s+)', r'\1\n\2', raw)
    fixed = re.sub(r'(\w+)\s+(\w+=)', r'\1\n\2', fixed)
    
    try:
        json.loads(fixed)
        path.write_text(fixed, encoding='utf-8')
        return True
    except json.JSONDecodeError as e:
        print(f"STILL INVALID: {path.relative_to(Path(__file__).parent)}: {e}")
        return False

root = Path(__file__).resolve().parent
for p in root.glob("**/*.ipynb"):
    if "core/" in str(p):
        continue  # Skip corrupted core notebooks
    raw = p.read_text(encoding='utf-8')
    if len(raw) > 1000 and '\n' not in raw[:1000].split('\n')[0]:
        # First line is too long - likely concatenated
        print(f"FIX: {p.relative_to(root)}")
        fix_notebook(p)
    elif len(raw.split('\n')) < 5:
        print(f"TOO SHORT: {p.relative_to(root)}")
