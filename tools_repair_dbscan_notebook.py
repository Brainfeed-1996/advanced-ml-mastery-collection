from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "03-Clustering" / "DBSCAN-Anomaly-Detection.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    first_imports = """# Reproducibility + standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)

import warnings
import logging

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""

    generator = """from sklearn.datasets import make_blobs

class IndustrialDataGenerator:
    def __init__(self, n_samples: int = 2000, n_features: int = 10, centers: int = 4, cluster_std: float = 1.0):
        self.n_samples = n_samples
        self.n_features = n_features
        self.centers = centers
        self.cluster_std = cluster_std

    def generate(self):
        logger.info('Generating clustering dataset (offline-first).')
        X, y = make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=self.centers,
            cluster_std=self.cluster_std,
            random_state=SEED,
        )
        df = pd.DataFrame(X, columns=[f"Dim_{i}" for i in range(self.n_features)])
        return df, y


data_gen = IndustrialDataGenerator()
X_df, y = data_gen.generate()
display(X_df.head())
"""

    pipeline = """from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

numeric_features = X_df.columns.tolist()

preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

logger.info('Preprocessing pipeline built.')
"""

    patched_first = False
    patched_gen = False
    patched_pipe = False

    # Patch first code cell
    for cell in nb.cells:
        if cell.get('cell_type') == 'code':
            cell['source'] = first_imports
            patched_first = True
            break

    # Patch generator cell
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source') or ''
        src_s = ''.join(src) if isinstance(src, list) else str(src)
        if 'IndustrialDataGenerator' in src_s and 'make_blobs' in src_s:
            cell['source'] = generator
            patched_gen = True
            break

    # Patch pipeline builder cell
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source') or ''
        src_s = ''.join(src) if isinstance(src, list) else str(src)
        if 'RobustPipelineBuilder' in src_s or ('PowerTransformer' in src_s and 'ColumnTransformer' in src_s and 'preprocessor' in src_s):
            cell['source'] = pipeline
            patched_pipe = True
            break

    if not (patched_first and patched_gen and patched_pipe):
        print({'patched_first': patched_first, 'patched_gen': patched_gen, 'patched_pipe': patched_pipe})

    nbformat.write(nb, nb_path)
    print('repaired dbscan notebook')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
