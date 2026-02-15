from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


CLEAN_IMPORTS = """# Standard imports (sanitized)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
SEED = 42
np.random.seed(SEED)

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

import warnings
import logging

# Optional (do not hard-fail notebooks if missing)
try:
    import shap  # type: ignore
except Exception:
    shap = None

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    nb_path = (root / args.notebook).resolve()
    if not nb_path.exists():
        raise SystemExit(f"missing: {nb_path}")

    nb = nbformat.read(nb_path, as_version=4)

    # find first code cell
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            src = cell.get("source") or ""
            src_s = "".join(src) if isinstance(src, list) else str(src)
            if args.force or "npimport" in src_s or "import numpy as npimport" in src_s or "Advanced Visualizationtry" in src_s:
                cell["source"] = CLEAN_IMPORTS
                nbformat.write(nb, nb_path)
                print("patched first code cell")
                return 0
            break

    print("no patch needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
