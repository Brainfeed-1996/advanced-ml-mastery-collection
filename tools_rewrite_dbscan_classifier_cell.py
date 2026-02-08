from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "03-Clustering" / "DBSCAN-Anomaly-Detection.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    repl = """from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# DBSCAN clustering (unsupervised)
X = preprocessor.fit_transform(X_df)

db = DBSCAN(eps=0.8, min_samples=8)
labels = db.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
outliers = (labels == -1).mean()
print({'clusters': n_clusters, 'outlier_rate': float(outliers)})

# Silhouette (only if at least 2 clusters and no all-noise)
if n_clusters >= 2:
    mask = labels != -1
    if mask.sum() > 10 and len(set(labels[mask])) >= 2:
        sil = silhouette_score(X[mask], labels[mask])
        print({'silhouette': float(sil)})

# Visualize on first 2 dimensions (post-scaling)
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, s=12, cmap='tab10', alpha=0.7)
plt.title('DBSCAN clustering (scaled feature space)')
plt.xlabel('dim0')
plt.ylabel('dim1')
plt.show()
"""

    changed = False
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source') or ''
        src_s = ''.join(src) if isinstance(src, list) else str(src)
        if 'AdvancedClassifier' in src_s and 'RandomForestClassifier' in src_s:
            cell['source'] = repl
            changed = True
            break

    if not changed:
        raise SystemExit('classifier cell not found')

    nbformat.write(nb, nb_path)
    print('rewrote classifier cell')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
