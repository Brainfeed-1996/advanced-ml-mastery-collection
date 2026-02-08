from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "03-Clustering" / "DBSCAN-Anomaly-Detection.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    repl = """# Visual summary
import numpy as np

unique, counts = np.unique(labels, return_counts=True)
summary = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
print('label_counts:', summary)

plt.figure(figsize=(8,4))
plt.bar([str(k) for k in summary.keys()], summary.values())
plt.title('DBSCAN label counts (-1 = noise)')
plt.xlabel('label')
plt.ylabel('count')
plt.show()
"""

    changed = False
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source') or ''
        src_s = ''.join(src) if isinstance(src, list) else str(src)
        if 'ResultVisualizer' in src_s:
            cell['source'] = repl
            changed = True
            break

    if not changed:
        raise SystemExit('visualizer cell not found')

    nbformat.write(nb, nb_path)
    print('rewrote visualizer cell')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
