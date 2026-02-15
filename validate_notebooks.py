#!/usr/bin/env python3
"""Validate notebooks and check for outputs"""
import json
import os
import sys

def validate_notebook(filepath):
    """Check if notebook is valid JSON and has outputs"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        code_cells = [c for c in cells if c.get('cell_type') == 'code']
        cells_with_outputs = [c for c in code_cells if c.get('outputs')]
        
        return {
            'valid': True,
            'total_cells': len(cells),
            'code_cells': len(code_cells),
            'cells_with_outputs': len(cells_with_outputs),
            'filepath': filepath
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'filepath': filepath
        }

def main():
    base_path = r'C:\Users\Olivier Robert\.openclaw\workspace\advanced-ml-mastery-collection'
    notebooks = []
    
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith('.ipynb'):
                notebooks.append(os.path.join(root, f))
    
    print(f"Found {len(notebooks)} notebooks\n")
    
    invalid = []
    valid_without_outputs = []
    valid_with_outputs = []
    
    for nb_path in notebooks:
        result = validate_notebook(nb_path)
        if not result['valid']:
            invalid.append(result)
        elif result['cells_with_outputs'] == 0:
            valid_without_outputs.append(result)
        else:
            valid_with_outputs.append(result)
    
    print("=== INVALID NOTEBOOKS ===")
    for nb in invalid:
        print(f"INVALID: {nb['filepath']}")
        print(f"  Error: {nb.get('error', 'Unknown')}\n")
    
    print("\n=== VALID WITHOUT OUTPUTS ===")
    for nb in valid_without_outputs:
        rel_path = os.path.relpath(nb['filepath'], base_path)
        print(f"NO OUTPUTS: {rel_path}")
        print(f"  Cells: {nb['total_cells']}, Code: {nb['code_cells']}\n")
    
    print("\n=== VALID WITH OUTPUTS ===")
    for nb in valid_with_outputs:
        rel_path = os.path.relpath(nb['filepath'], base_path)
        print(f"OK: {rel_path}")
        print(f"  Cells: {nb['total_cells']}, Code: {nb['code_cells']}, Outputs: {nb['cells_with_outputs']}\n")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total notebooks: {len(notebooks)}")
    print(f"Invalid: {len(invalid)}")
    print(f"Valid without outputs: {len(valid_without_outputs)}")
    print(f"Valid with outputs: {len(valid_with_outputs)}")
    
    return len(invalid) == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
