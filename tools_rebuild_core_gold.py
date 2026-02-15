"""Rebuild core notebooks as valid JSON + complex content + execute to embed outputs.

Targets:
- core/01-Linear-Regression-Housing.ipynb
- core/02-Logistic-Regression-Medical.ipynb
- core/03-Decision-Tree-Iris.ipynb

Design:
- self-contained synthetic or sklearn datasets
- deterministic seeds
- outputs: tables, metrics, plots

Run:
  python tools_rebuild_core_gold.py
"""

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


def nb_linear_housing() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Core 01 — Linear Regression (Housing)\n\n"
            "Industrial baseline: data generation, EDA, pipeline, diagnostics, coefficients." 
        ),
        code(
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.preprocessing import OneHotEncoder\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
            "np.random.seed(42); sns.set_theme(style='whitegrid')\n"
        ),
        md("## 1) Dataset (synthetic but realistic)"),
    ]
    nb.cells += [
        code(
            "n=3000\n"
            "city=np.random.choice(['Rennes','Paris','Nantes','Lyon'], size=n, p=[0.35,0.25,0.25,0.15])\n"
            "surface=np.clip(np.random.normal(65, 20, n), 18, 220)\n"
            "rooms=np.clip((surface/18 + np.random.normal(0,0.8,n)).round().astype(int), 1, 9)\n"
            "age=np.clip(np.random.exponential(18, n).round().astype(int), 0, 120)\n"
            "dist=np.clip(np.random.gamma(2.1, 2.0, n), 0.2, 28)\n"
            "parking=np.random.binomial(1, 0.58, n)\n"
            "balcony=np.random.binomial(1, 0.44, n)\n"
            "base={'Paris': 9800, 'Lyon': 5400, 'Rennes': 4400, 'Nantes': 4800}\n"
            "ppm2=np.array([base[c] for c in city], dtype=float)\n"
            "ppm2 = ppm2*(1-0.017*dist)*(1+0.055*balcony)*(1+0.045*parking)*(1-0.0022*age)\n"
            "ppm2=np.clip(ppm2, 1800, None)\n"
            "noise=np.random.normal(0, 22000, n)\n"
            "price = surface*ppm2 + 10000*(rooms>=3) + noise\n"
            "price=np.clip(price, 45000, None)\n"
            "df=pd.DataFrame({'city':city,'surface_m2':surface,'rooms':rooms,'age_years':age,'dist_center_km':dist,'has_parking':parking,'has_balcony':balcony,'price_eur':price})\n"
            "df.head()\n"
        ),
        md("## 2) EDA"),
        code("df.describe(include='all').T"),
        code(
            "fig, ax = plt.subplots(1,2, figsize=(12,4))\n"
            "sns.histplot(df['price_eur'], kde=True, ax=ax[0]); ax[0].set_title('Price distribution')\n"
            "sns.scatterplot(data=df.sample(900, random_state=1), x='surface_m2', y='price_eur', hue='city', alpha=0.5, ax=ax[1]); ax[1].set_title('Surface vs price')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md("## 3) Pipeline + evaluation"),
        code(
            "X=df.drop(columns=['price_eur']); y=df['price_eur']\n"
            "Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)\n"
            "cat=['city']; num=[c for c in X.columns if c not in cat]\n"
            "pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num','passthrough',num)])\n"
            "pipe=Pipeline([('pre',pre),('model',Ridge(alpha=2.0, random_state=42))])\n"
            "pipe.fit(Xtr,ytr)\n"
            "pred=pipe.predict(Xte)\n"
            "mae=mean_absolute_error(yte,pred); rmse=mean_squared_error(yte,pred,squared=False); r2=r2_score(yte,pred)\n"
            "pd.DataFrame({'MAE':[mae],'RMSE':[rmse],'R2':[r2]})\n"
        ),
        md("## 4) Residual diagnostics"),
        code(
            "res=yte.values-pred\n"
            "fig,ax=plt.subplots(1,2,figsize=(12,4))\n"
            "sns.histplot(res,kde=True,ax=ax[0]); ax[0].set_title('Residuals')\n"
            "sns.scatterplot(x=pred,y=res,alpha=0.5,ax=ax[1]); ax[1].axhline(0,color='black',lw=1); ax[1].set_title('Residuals vs predicted')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md("## 5) Coefficients (interpretability)"),
        code(
            "ohe=pipe.named_steps['pre'].named_transformers_['cat']\n"
            "feat=ohe.get_feature_names_out(cat).tolist()+num\n"
            "coef=pipe.named_steps['model'].coef_\n"
            "imp=pd.DataFrame({'feature':feat,'coef':coef}).sort_values('coef', key=lambda s: s.abs(), ascending=False)\n"
            "imp.head(15)\n"
        ),
        md("## Conclusion\nBaseline solide et interprétable, prêt pour upgrades (quantile regression, geo features, robust losses)."),
    ]
    return nb


def nb_logistic_medical() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Core 02 — Logistic Regression (Medical Risk)\n\nIndustrial classification: calibration, thresholding, PR/ROC, error analysis."),
        code(
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score)\n"
            "np.random.seed(7); sns.set_theme(style='whitegrid')\n"
        ),
        md("## 1) Cohort synthesis"),
        code(
            "n=4000\n"
            "age=np.clip(np.random.normal(52, 14, n), 18, 90)\n"
            "bmi=np.clip(np.random.normal(27, 5.2, n), 16, 55)\n"
            "sbp=np.clip(np.random.normal(130, 17, n), 90, 215)\n"
            "chol=np.clip(np.random.normal(5.2, 1.1, n), 2.5, 10.0)\n"
            "smoker=np.random.binomial(1, 0.28, n)\n"
            "diab=np.random.binomial(1, 0.12, n)\n"
            "activity=np.clip(np.random.normal(3.0, 1.1, n), 0, 7)\n"
            "logit=(-7.1+0.033*age+0.075*(bmi-25)+0.018*(sbp-120)+0.5*smoker+0.95*diab-0.22*activity+0.012*(chol-5.0)*(sbp-120)/10.0)\n"
            "p=1/(1+np.exp(-logit))\n"
            "y=np.random.binomial(1,p)\n"
            "df=pd.DataFrame({'age':age,'bmi':bmi,'sbp':sbp,'chol':chol,'smoker':smoker,'diabetic':diab,'activity':activity,'risk':y})\n"
            "df['risk'].mean(), df.head()\n"
        ),
        md("## 2) Train/test + model"),
        code(
            "X=df.drop(columns=['risk']); y=df['risk']\n"
            "Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)\n"
            "pipe=Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced'))])\n"
            "pipe.fit(Xtr,ytr)\n"
            "proba=pipe.predict_proba(Xte)[:,1]\n"
            "pred=(proba>=0.5).astype(int)\n"
            "auc=roc_auc_score(yte,proba); ap=average_precision_score(yte,proba)\n"
            "pd.DataFrame({'ROC_AUC':[auc],'PR_AUC':[ap]})\n"
        ),
        md("## 3) Confusion + report"),
        code("confusion_matrix(yte,pred)"),
        code("print(classification_report(yte,pred, digits=3))"),
        md("## 4) ROC / PR"),
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
            "cost_fn=12; cost_fp=1\n"
            "ths=np.linspace(0.05,0.95,91)\n"
            "best=None\n"
            "for th in ths:\n"
            "    p=(proba>=th).astype(int)\n"
            "    tn,fp,fn,tp=confusion_matrix(yte,p).ravel()\n"
            "    cost=cost_fn*fn + cost_fp*fp\n"
            "    if best is None or cost<best[0]:\n"
            "        best=(cost,th,tn,fp,fn,tp)\n"
            "best\n"
        ),
        md("## Conclusion\nNotebook complet: modèle, courbes, analyse d'erreur et seuil opérationnel."),
    ]
    return nb


def nb_decision_tree_iris() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Core 03 — Decision Tree (Iris)\n\nInterprétabilité: pruning, importance, rules extraction, validation."),
        code(
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from sklearn.datasets import load_iris\n"
            "from sklearn.model_selection import train_test_split, GridSearchCV\n"
            "from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text\n"
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
            "np.random.seed(42); sns.set_theme(style='whitegrid')\n"
        ),
        md("## 1) Load dataset"),
        code(
            "iris=load_iris(as_frame=True)\n"
            "df=iris.frame.copy()\n"
            "df.rename(columns={'target':'species_id'}, inplace=True)\n"
            "df['species']=df['species_id'].map({i:n for i,n in enumerate(iris.target_names)})\n"
            "df.head()\n"
        ),
        md("## 2) EDA"),
        code(
            "sns.pairplot(df, hue='species', corner=True)\nplt.show()\n"
        ),
        md("## 3) Train/test + baseline"),
        code(
            "X=df[iris.feature_names]; y=df['species_id']\n"
            "Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)\n"
            "clf=DecisionTreeClassifier(random_state=42)\n"
            "clf.fit(Xtr,ytr)\n"
            "pred=clf.predict(Xte)\n"
            "acc=accuracy_score(yte,pred)\n"
            "acc\n"
        ),
        md("## 4) Hyperparameter search (pruning)"),
        code(
            "param={'max_depth':[2,3,4,5,None],'min_samples_leaf':[1,2,3,4],'ccp_alpha':[0.0,0.001,0.005,0.01,0.02]}\n"
            "gs=GridSearchCV(DecisionTreeClassifier(random_state=42), param, cv=5)\n"
            "gs.fit(Xtr,ytr)\n"
            "gs.best_params_, gs.best_score_\n"
        ),
        md("## 5) Evaluate best tree"),
        code(
            "best=gs.best_estimator_\n"
            "pred=best.predict(Xte)\n"
            "print('acc', accuracy_score(yte,pred))\n"
            "print(classification_report(yte,pred, target_names=iris.target_names))\n"
            "confusion_matrix(yte,pred)\n"
        ),
        md("## 6) Interpretability"),
        code(
            "fig=plt.figure(figsize=(12,6))\n"
            "plot_tree(best, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, max_depth=3)\n"
            "plt.show()\n"
        ),
        code(
            "rules = export_text(best, feature_names=iris.feature_names)\n"
            "print(rules[:1200])\n"
        ),
        code(
            "imp=pd.DataFrame({'feature':iris.feature_names,'importance':best.feature_importances_}).sort_values('importance', ascending=False)\n"
            "imp\n"
        ),
        md("## Conclusion\nArbre pruné, stable, et interprétable (règles + importance + validation)."),
    ]
    return nb


def write_nb(path: Path, nb: nbf.NotebookNode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))


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
    targets = {
        root / "core" / "01-Linear-Regression-Housing.ipynb": nb_linear_housing(),
        root / "core" / "02-Logistic-Regression-Medical.ipynb": nb_logistic_medical(),
        root / "core" / "03-Decision-Tree-Iris.ipynb": nb_decision_tree_iris(),
    }

    for p, nb in targets.items():
        print(f"REBUILD: {p.relative_to(root)}")
        write_nb(p, nb)
        print(f"EXEC:    {p.relative_to(root)}")
        execute_inplace(p, timeout=900)

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
