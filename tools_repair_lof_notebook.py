from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "08-Anomaly-Detection" / "Local-Outlier-Factor-Fraud.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    generator = """from sklearn.datasets import make_blobs

# Synthetic fraud-like dataset (offline-first)
# We create a dense normal cluster + scattered anomalies.

X, y = make_blobs(n_samples=1800, centers=1, n_features=6, cluster_std=1.0, random_state=SEED)

# Inject anomalies
rng = np.random.default_rng(SEED)
anom = rng.uniform(low=-8, high=8, size=(200, X.shape[1]))
X_all = np.vstack([X, anom])

X_df = pd.DataFrame(X_all, columns=[f"feat_{i}" for i in range(X.shape[1])])
# y_true: 0 = inlier, 1 = anomaly
y_true = np.array([0]*len(X) + [1]*len(anom))

print({'rows': len(X_df), 'anomaly_rate': float(y_true.mean())})
display(X_df.head())
"""

    pipeline = """from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

X = preprocessor.fit_transform(X_df)
"""

    lof = """from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

lof = LocalOutlierFactor(n_neighbors=25, contamination=float(y_true.mean()))
# LOF returns -1 for outliers, 1 for inliers
pred = lof.fit_predict(X)
y_pred = (pred == -1).astype(int)

print('confusion_matrix:\n', confusion_matrix(y_true, y_pred))
print('\nreport:\n', classification_report(y_true, y_pred, digits=3))

# LOF doesn't expose a proper probability, but we can use negative_outlier_factor_
# Higher means more inlier; we invert and normalize for an AUC-ish view
scores = -lof.negative_outlier_factor_
try:
    auc = roc_auc_score(y_true, scores)
    print({'roc_auc': float(auc)})
except Exception as e:
    print('roc_auc unavailable:', e)

plt.figure(figsize=(8,4))
sns.histplot(scores[y_true==0], label='inlier', kde=True, color='steelblue', alpha=0.6)
sns.histplot(scores[y_true==1], label='anomaly', kde=True, color='crimson', alpha=0.6)
plt.title('LOF anomaly score distribution')
plt.legend()
plt.show()
"""

    visual = """# 2D view (first 2 scaled dims)
plt.figure(figsize=(7,6))
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='coolwarm', s=12, alpha=0.7)
plt.title('LOF predicted anomalies (first 2 dims)')
plt.xlabel('dim0')
plt.ylabel('dim1')
plt.show()
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

    a = replace_if_contains('IndustrialDataGenerator', generator)
    b = replace_if_contains('RobustPipelineBuilder', pipeline)
    c = replace_if_contains('ResultVisualizer', visual)

    # If there's already a LOF/LocalOutlierFactor cell, replace it; else replace the first "LocalOutlierFactor" occurrence.
    d = False
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source') or ''
        src_s = ''.join(src) if isinstance(src, list) else str(src)
        if 'LocalOutlierFactor' in src_s or 'LOF' in src_s:
            cell['source'] = lof
            d = True
            break

    nbformat.write(nb, nb_path)
    print({'patched_generator': a, 'patched_pipeline': b, 'patched_model': d, 'patched_visual': c})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
