"""Rebuild core/03-Decision-Tree-Iris.ipynb as valid JSON + execute to embed outputs."""

from __future__ import annotations

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


def nb_decision_tree_iris() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Core 03 — Decision Tree (Iris)\n\n"
            "Notebook *industrial-grade* orienté interprétabilité : EDA, baseline, pruning (ccp_alpha),\n"
            "visualisation, extraction de règles, validation." 
        ),
        code(
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from sklearn.datasets import load_iris\n"
            "from sklearn.model_selection import train_test_split, GridSearchCV\n"
            "from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text\n"
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
            "np.random.seed(42); sns.set_theme(style='whitegrid')\n"
        ),
        md("## 1) Data"),
        code(
            "iris=load_iris(as_frame=True)\n"
            "df=iris.frame.copy()\n"
            "df.rename(columns={'target':'species_id'}, inplace=True)\n"
            "df['species']=df['species_id'].map({i:n for i,n in enumerate(iris.target_names)})\n"
            "df.head()\n"
        ),
        md("## 2) EDA"),
        code(
            "print(df.groupby('species').mean(numeric_only=True))\n"
            "sns.pairplot(df, hue='species', corner=True)\n"
            "plt.show()\n"
        ),
        md("## 3) Train/test + baseline"),
        code(
            "X=df[iris.feature_names]; y=df['species_id']\n"
            "Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)\n"
            "base=DecisionTreeClassifier(random_state=42)\n"
            "base.fit(Xtr,ytr)\n"
            "pred=base.predict(Xte)\n"
            "print('acc', accuracy_score(yte,pred))\n"
            "print(classification_report(yte,pred, target_names=iris.target_names))\n"
            "confusion_matrix(yte,pred)\n"
        ),
        md("## 4) Cost-complexity pruning path"),
        code(
            "path = base.cost_complexity_pruning_path(Xtr, ytr)\n"
            "ccp_alphas = path.ccp_alphas\n"
            "pd.Series(ccp_alphas).describe()\n"
        ),
        code(
            "fig=plt.figure(figsize=(8,4))\n"
            "plt.plot(ccp_alphas, path.impurities, marker='o')\n"
            "plt.title('Impurity vs ccp_alpha')\n"
            "plt.xlabel('ccp_alpha'); plt.ylabel('total impurity')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md("## 5) Grid search (depth + leaf + ccp_alpha)"),
        code(
            "param={\n"
            " 'max_depth':[2,3,4,5,None],\n"
            " 'min_samples_leaf':[1,2,3,4],\n"
            " 'ccp_alpha':[0.0, 0.001, 0.003, 0.006, 0.01, 0.02]\n"
            "}\n"
            "gs=GridSearchCV(DecisionTreeClassifier(random_state=42), param, cv=5)\n"
            "gs.fit(Xtr,ytr)\n"
            "gs.best_params_, gs.best_score_\n"
        ),
        md("## 6) Best model evaluation + interpretability"),
        code(
            "best=gs.best_estimator_\n"
            "pred=best.predict(Xte)\n"
            "print('acc', accuracy_score(yte,pred))\n"
            "print(classification_report(yte,pred, target_names=iris.target_names))\n"
            "cm=confusion_matrix(yte,pred)\n"
            "cm\n"
        ),
        code(
            "fig=plt.figure(figsize=(12,6))\n"
            "plot_tree(best, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, max_depth=3)\n"
            "plt.show()\n"
        ),
        code(
            "rules = export_text(best, feature_names=iris.feature_names)\n"
            "print(rules[:1400])\n"
        ),
        code(
            "imp=pd.DataFrame({'feature':iris.feature_names,'importance':best.feature_importances_}).sort_values('importance', ascending=False)\n"
            "imp\n"
        ),
        md("## Conclusion\nArbre pruné, stable, et interprétable (règles + importance + validation)."),
    ]
    return nb


def execute_inplace(path: Path, timeout: int = 900) -> None:
    run(
        pycmd(
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(path),
            "--ExecutePreprocessor.kernel_name=python3",
            f"--ExecutePreprocessor.timeout={timeout}",
            "--ExecutePreprocessor.startup_timeout=180",
            "--output",
            path.name,
            "--output-dir",
            str(path.parent),
        )
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    path = root / "core" / "03-Decision-Tree-Iris.ipynb"
    print("REBUILD: core/03-Decision-Tree-Iris.ipynb")
    nbf.write(nb_decision_tree_iris(), str(path))
    print("EXEC:    core/03-Decision-Tree-Iris.ipynb")
    execute_inplace(path)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
