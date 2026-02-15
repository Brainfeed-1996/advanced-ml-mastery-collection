from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "08-Anomaly-Detection" / "Local-Outlier-Factor-Fraud.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    removed = 0
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source') or ''
        src_s = ''.join(src) if isinstance(src, list) else str(src)
        if 'AdvancedRegressor' in src_s or 'GradientBoostingRegressor' in src_s:
            cell['source'] = """# NOTE: This notebook focuses on LOF (unsupervised anomaly detection).
# A previous placeholder cell contained an unrelated supervised regressor pipeline.
# It was removed to keep the notebook coherent and executable.
"""
            removed += 1

    nbformat.write(nb, nb_path)
    print({'patched_cells': removed})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
