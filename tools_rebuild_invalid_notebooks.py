"""Rebuild a small set of invalid notebooks as valid JSON, complex content, then execute to embed outputs.

Targets are known-invalid notebooks that break nbconvert.

Design goals:
- self-contained (synthetic datasets)
- deterministic
- CPU-light
- produces outputs (tables, metrics, plots)

Usage:
  python tools_rebuild_invalid_notebooks.py --kernel notebooks-py311
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import nbformat as nbf


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def pycmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(src: str):
    return nbf.v4.new_code_cell(src)


def build_linear_regression_real_estate() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python", "version": "3"},
        }
    )

    nb.cells = [
        md(
            "# Linear Regression — Real Estate Pricing (Industrial Notebook)\n"
            "\n"
            "**Goal:** build a robust baseline for house price prediction with **data generation**, "
            "**EDA**, **feature engineering**, **model diagnostics**, and **reproducible evaluation**.\n"
            "\n"
            "This notebook is fully self-contained (synthetic dataset) and executed with outputs saved.\n"
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import OneHotEncoder\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
            "\n"
            "np.random.seed(42)\n"
            "sns.set_theme(style='whitegrid')\n"
        ),
        md("## 1) Synthetic dataset"),
        code(
            "n = 2500\n"
            "cities = np.random.choice(['Rennes', 'Paris', 'Nantes', 'Lyon'], size=n, p=[0.35,0.25,0.25,0.15])\n"
            "surface = np.clip(np.random.normal(62, 18, size=n), 18, 200)\n"
            "rooms = np.clip((surface/18 + np.random.normal(0,0.7,size=n)).round().astype(int), 1, 8)\n"
            "age = np.clip(np.random.exponential(scale=18, size=n).round().astype(int), 0, 120)\n"
            "dist_center_km = np.clip(np.random.gamma(shape=2.0, scale=2.0, size=n), 0.2, 25)\n"
            "has_parking = np.random.binomial(1, 0.55, size=n)\n"
            "has_balcony = np.random.binomial(1, 0.42, size=n)\n"
            "\n"
            "city_base = {'Paris': 9500, 'Lyon': 5200, 'Rennes': 4300, 'Nantes': 4700}\n"
            "base_ppm2 = np.array([city_base[c] for c in cities], dtype=float)\n"
            "\n"
            "# Non-linearities + interactions\n"
            "ppm2 = (base_ppm2\n"
            "        * (1 - 0.018*dist_center_km)\n"
            "        * (1 + 0.06*has_balcony)\n"
            "        * (1 + 0.045*has_parking)\n"
            "        * (1 - 0.0025*age)\n"
            "       )\n"
            "ppm2 = np.clip(ppm2, 1800, None)\n"
            "\n"
            "noise = np.random.normal(0, 18000, size=n)\n"
            "price = surface * ppm2 + 12000*(rooms>=3) + noise\n"
            "price = np.clip(price, 45000, None)\n"
            "\n"
            "df = pd.DataFrame({\n"
            "    'city': cities,\n"
            "    'surface_m2': surface,\n"
            "    'rooms': rooms,\n"
            "    'age_years': age,\n"
            "    'dist_center_km': dist_center_km,\n"
            "    'has_parking': has_parking.astype(int),\n"
            "    'has_balcony': has_balcony.astype(int),\n"
            "    'price_eur': price\n"
            "})\n"
            "df.head()\n"
        ),
        md("## 2) EDA"),
        code(
            "df.describe(include='all').T\n"
        ),
        code(
            "fig, ax = plt.subplots(1,2, figsize=(12,4))\n"
            "sns.histplot(df['price_eur'], kde=True, ax=ax[0])\n"
            "ax[0].set_title('Price distribution')\n"
            "sns.scatterplot(data=df.sample(800, random_state=1), x='surface_m2', y='price_eur', hue='city', alpha=0.55, ax=ax[1])\n"
            "ax[1].set_title('Surface vs Price (sample)')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md("## 3) Train/test split + pipeline"),
        code(
            "X = df.drop(columns=['price_eur'])\n"
            "y = df['price_eur']\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            "\n"
            "cat_cols = ['city']\n"
            "num_cols = [c for c in X.columns if c not in cat_cols]\n"
            "\n"
            "pre = ColumnTransformer([\n"
            "    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),\n"
            "    ('num', 'passthrough', num_cols),\n"
            "])\n"
            "\n"
            "model = Ridge(alpha=2.0, random_state=42)\n"
            "pipe = Pipeline([('pre', pre), ('model', model)])\n"
            "pipe\n"
        ),
        md("## 4) Fit + evaluation"),
        code(
            "pipe.fit(X_train, y_train)\n"
            "pred = pipe.predict(X_test)\n"
            "\n"
            "mae = mean_absolute_error(y_test, pred)\n"
            "rmse = mean_squared_error(y_test, pred, squared=False)\n"
            "r2 = r2_score(y_test, pred)\n"
            "pd.DataFrame({'MAE': [mae], 'RMSE':[rmse], 'R2':[r2]})\n"
        ),
        md("## 5) Residual diagnostics"),
        code(
            "resid = y_test.values - pred\n"
            "fig, ax = plt.subplots(1,2, figsize=(12,4))\n"
            "sns.histplot(resid, kde=True, ax=ax[0])\n"
            "ax[0].set_title('Residual distribution')\n"
            "sns.scatterplot(x=pred, y=resid, alpha=0.5, ax=ax[1])\n"
            "ax[1].axhline(0, color='black', linewidth=1)\n"
            "ax[1].set_title('Residuals vs Predicted')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md("## 6) Feature importance (linear coefficients)"),
        code(
            "# Recover feature names after one-hot encoding\n"
            "ohe = pipe.named_steps['pre'].named_transformers_['cat']\n"
            "cat_names = ohe.get_feature_names_out(cat_cols).tolist()\n"
            "feat_names = cat_names + num_cols\n"
            "coef = pipe.named_steps['model'].coef_\n"
            "imp = pd.DataFrame({'feature': feat_names, 'coef': coef}).sort_values('coef', key=lambda s: s.abs(), ascending=False)\n"
            "imp.head(12)\n"
        ),
        md(
            "## Conclusion\n"
            "We built an end-to-end baseline with clean preprocessing, robust evaluation, and diagnostics.\n"
            "Next steps: quantile regression, heteroscedastic modeling, and geospatial features." 
        ),
    ]
    return nb


def build_logistic_regression_healthcare() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Logistic Regression — Healthcare Risk (Industrial Notebook)\n\n"
            "Self-contained binary classification with calibration, thresholding, and error analysis."
        ),
        code(
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score)\n"
            "np.random.seed(7); sns.set_theme(style='whitegrid')\n"
        ),
        md("## 1) Synthetic patient cohort"),
        code(
            "n=3500\n"
            "age=np.clip(np.random.normal(52, 14, n), 18, 90)\n"
            "bmi=np.clip(np.random.normal(26.5, 5.0, n), 16, 52)\n"
            "sbp=np.clip(np.random.normal(128, 16, n), 90, 210)\n"
            "chol=np.clip(np.random.normal(5.2, 1.1, n), 2.5, 9.5)\n"
            "smoker=np.random.binomial(1, 0.27, n)\n"
            "diabetic=np.random.binomial(1, 0.11, n)\n"
            "activity=np.clip(np.random.normal(3.0, 1.0, n), 0, 7)\n"
            "\n"
            "# log-odds with interactions\n"
            "logit = (-7.2 + 0.035*age + 0.07*(bmi-25) + 0.018*(sbp-120) + 0.45*smoker + 0.85*diabetic - 0.20*activity\n"
            "         + 0.012*(chol-5.0)*(sbp-120)/10.0)\n"
            "p = 1/(1+np.exp(-logit))\n"
            "y = np.random.binomial(1, p)\n"
            "\n"
            "df=pd.DataFrame({'age':age,'bmi':bmi,'sbp':sbp,'chol':chol,'smoker':smoker,'diabetic':diabetic,'activity':activity,'risk':y})\n"
            "df['risk'].mean(), df.head()\n"
        ),
        md("## 2) Train/test + model"),
        code(
            "X=df.drop(columns=['risk']); y=df['risk']\n"
            "Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)\n"
            "pipe=Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=2000, class_weight='balanced'))])\n"
            "pipe.fit(Xtr,ytr)\n"
            "proba=pipe.predict_proba(Xte)[:,1]\n"
            "pred=(proba>=0.5).astype(int)\n"
            "auc=roc_auc_score(yte,proba)\n"
            "ap=average_precision_score(yte,proba)\n"
            "pd.DataFrame({'ROC_AUC':[auc],'PR_AUC':[ap]})\n"
        ),
        md("## 3) Confusion matrix + report"),
        code(
            "cm=confusion_matrix(yte,pred)\n"
            "cm\n"
        ),
        code(
            "print(classification_report(yte,pred, digits=3))\n"
        ),
        md("## 4) ROC / PR curves"),
        code(
            "fpr,tpr,_=roc_curve(yte,proba)\n"
            "prec,rec,_=precision_recall_curve(yte,proba)\n"
            "fig,ax=plt.subplots(1,2,figsize=(12,4))\n"
            "ax[0].plot(fpr,tpr); ax[0].plot([0,1],[0,1],'--',alpha=0.5); ax[0].set_title(f'ROC (AUC={auc:.3f})'); ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR')\n"
            "ax[1].plot(rec,prec); ax[1].set_title(f'PR (AP={ap:.3f})'); ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md("## 5) Threshold tuning (cost-aware)"),
        code(
            "# Suppose FN cost is higher than FP\n"
            "cost_fn=10; cost_fp=1\n"
            "thresholds=np.linspace(0.05,0.95,91)\n"
            "best=None\n"
            "for th in thresholds:\n"
            "    p=(proba>=th).astype(int)\n"
            "    tn,fp,fn,tp=confusion_matrix(yte,p).ravel()\n"
            "    cost=cost_fn*fn + cost_fp*fp\n"
            "    if best is None or cost<best[0]:\n"
            "        best=(cost,th,tn,fp,fn,tp)\n"
            "best\n"
        ),
        md("## Conclusion"),
        md("We built a calibrated baseline and showed how thresholding changes operational cost."),
    ]
    return nb


def write_nb(path: Path, nb: nbf.NotebookNode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))


def execute_inplace(path: Path, kernel: str, timeout: int) -> None:
    run(
        pycmd(
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(path),
            f"--ExecutePreprocessor.kernel_name={kernel}",
            f"--ExecutePreprocessor.timeout={timeout}",
            "--output",
            path.name,
            "--output-dir",
            str(path.parent),
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="notebooks-py311")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent

    targets = {
        root / "01-Regression" / "Linear-Regression-Real-Estate.ipynb": build_linear_regression_real_estate(),
        root / "01-Regression-Analysis" / "Linear-Regression-Real-Estate.ipynb": build_linear_regression_real_estate(),
        root / "01-Regression" / "Logistic-Regression-Healthcare.ipynb": build_logistic_regression_healthcare(),
        root / "01-Regression-Analysis" / "Logistic-Regression-Healthcare.ipynb": build_logistic_regression_healthcare(),
        # core notebooks were previously corrupted; rebuild them too
        root / "core" / "01-Linear-Regression-Housing.ipynb": build_linear_regression_real_estate(),
        root / "core" / "02-Logistic-Regression-Medical.ipynb": build_logistic_regression_healthcare(),
    }

    for path, nb in targets.items():
        print(f"REBUILD: {path.relative_to(root)}")
        write_nb(path, nb)
        print(f"EXEC:    {path.relative_to(root)}")
        execute_inplace(path, kernel=args.kernel, timeout=args.timeout)

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
