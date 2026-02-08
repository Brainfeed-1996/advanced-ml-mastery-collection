from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "08-Anomaly-Detection" / "Autoencoders-Network-Security.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    generator = """from sklearn.datasets import make_blobs

# Synthetic network telemetry (offline-first)
X, _ = make_blobs(n_samples=2500, centers=2, n_features=12, cluster_std=1.4, random_state=SEED)

rng = np.random.default_rng(SEED)
# Inject "attack" anomalies with shifted distribution
anom = rng.normal(loc=6.0, scale=2.5, size=(250, X.shape[1]))

X_all = np.vstack([X, anom])
X_df = pd.DataFrame(X_all, columns=[f"feat_{i}" for i in range(X.shape[1])])

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

    # Linear autoencoder approximation via PCA reconstruction error
    model = """from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# Train on inliers only (semi-supervised anomaly detection)
X_train = X[y_true == 0]
X_test = X

pca = PCA(n_components=6, random_state=SEED)
pca.fit(X_train)

X_proj = pca.inverse_transform(pca.transform(X_test))
recon_err = ((X_test - X_proj) ** 2).mean(axis=1)

# Threshold at percentile based on expected contamination
thr = float(np.quantile(recon_err, 1.0 - float(y_true.mean())))
y_pred = (recon_err > thr).astype(int)

print({'threshold': thr, 'pred_anomaly_rate': float(y_pred.mean())})

try:
    auc = roc_auc_score(y_true, recon_err)
    print({'roc_auc': float(auc)})
except Exception as e:
    print('roc_auc unavailable:', e)

plt.figure(figsize=(8,4))
sns.histplot(recon_err[y_true==0], label='inlier', kde=True, color='steelblue', alpha=0.6)
sns.histplot(recon_err[y_true==1], label='anomaly', kde=True, color='crimson', alpha=0.6)
plt.title('Reconstruction error (PCA autoencoder proxy)')
plt.legend(); plt.show()
"""

    visual = """# 2D view using PCA components
from sklearn.decomposition import PCA

p2 = PCA(n_components=2, random_state=SEED)
XY = p2.fit_transform(X)

plt.figure(figsize=(7,6))
plt.scatter(XY[:,0], XY[:,1], c=y_pred, cmap='coolwarm', s=12, alpha=0.7)
plt.title('Predicted anomalies in 2D PCA space')
plt.xlabel('pc1'); plt.ylabel('pc2')
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
    c = replace_if_contains('AdvancedClassifier', model)
    d = replace_if_contains('ResultVisualizer', visual)

    nbformat.write(nb, nb_path)
    print({'patched_generator': a, 'patched_pipeline': b, 'patched_model': c, 'patched_visual': d})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
