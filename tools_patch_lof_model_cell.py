from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "08-Anomaly-Detection" / "Local-Outlier-Factor-Fraud.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    repl = """from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

lof = LocalOutlierFactor(n_neighbors=25, contamination=float(y_true.mean()))
# LOF returns -1 for outliers, 1 for inliers
pred = lof.fit_predict(X)
y_pred = (pred == -1).astype(int)

print('confusion_matrix:\\n', confusion_matrix(y_true, y_pred))
print('\\nreport:\\n', classification_report(y_true, y_pred, digits=3))

# LOF doesn't expose a calibrated probability, but negative_outlier_factor_ is useful as a score
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

    changed = False
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source') or ''
        src_s = ''.join(src) if isinstance(src, list) else str(src)
        if 'LocalOutlierFactor' in src_s and 'negative_outlier_factor_' in src_s:
            cell['source'] = repl
            changed = True
            break

    if not changed:
        raise SystemExit('LOF model cell not found')

    nbformat.write(nb, nb_path)
    print('patched LOF model cell')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
