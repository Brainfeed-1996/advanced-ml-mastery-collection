from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "03-Clustering" / "Hierarchical-Clustering-Genes.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    pipeline = """from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

X = preprocessor.fit_transform(X_df)
"""

    clustering = """from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score

Z = linkage(X, method='ward')

plt.figure(figsize=(10, 4))
dendrogram(Z, truncate_mode='lastp', p=20)
plt.title('Hierarchical clustering dendrogram (truncated)')
plt.xlabel('cluster size')
plt.ylabel('distance')
plt.show()

# Choose k clusters (portfolio-friendly default)
k = 4
labels = fcluster(Z, k, criterion='maxclust')
print({'k': k, 'silhouette': float(silhouette_score(X, labels))})

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, s=12, cmap='tab10', alpha=0.7)
plt.title('Hierarchical clustering (ward) — first 2 dims')
plt.xlabel('dim0')
plt.ylabel('dim1')
plt.show()
"""

    visual = """import numpy as np
unique, counts = np.unique(labels, return_counts=True)
summary = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
print('label_counts:', summary)
"""

    def replace_if_contains(needle: str, new_src: str) -> bool:
        for cell in nb.cells:
            if cell.get('cell_type') != 'code':
                continue
            src = cell.get('source') or ''
            src_s = ''.join(src) if isinstance(src, list) else str(src)
            if needle in src_s:
                cell['source'] = new_src
                return True
        return False

    a = replace_if_contains('RobustPipelineBuilder', pipeline)
    b = replace_if_contains('AdvancedClassifier', clustering)
    c = replace_if_contains('ResultVisualizer', visual)

    nbformat.write(nb, nb_path)
    print({'patched_pipeline': a, 'patched_clustering': b, 'patched_visual': c})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
