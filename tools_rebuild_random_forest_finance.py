"""Rebuild Random-Forest-Finance.ipynb with proper imports."""
import nbformat as nbf
from pathlib import Path

def nb_random_forest_finance():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Random Forest - Finance\n\nCredit scoring with ensemble methods."),
        nbf.v4.new_code_cell(
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from sklearn.base import BaseEstimator, TransformerMixin\n"
            "from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.metrics import (mean_squared_error, r2_score, f1_score, \n"
            "                            accuracy_score, classification_report, make_scorer)\n"
            "from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, \n"
            "                              GradientBoostingRegressor, GradientBoostingClassifier)\n"
            "from sklearn.linear_model import LogisticRegression, Ridge\n"
            "from sklearn.svm import SVC, SVR\n"
            "from sklearn.datasets import make_regression, make_classification, make_blobs\n"
            "\nimport warnings\nimport time\nimport logging\n"
            "try:\n    import shap\nexcept ImportError:\n    pass  # Simulation mode if missing\n"
            "warnings.filterwarnings('ignore')\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (12, 8)\n"
            "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')\nlogger = logging.getLogger(__name__)"
        ),
        nbf.v4.new_code_cell(
            "np.random.seed(42)\nn = 5000\n"
            "income = np.clip(np.random.normal(55000, 18000, n), 15000, 200000)\n"
            "debt = np.clip(np.random.exponential(scale=15000, size=n), 0, 100000)\n"
            "age = np.random.randint(18, 75, n)\n"
            "employment_years = np.clip(np.random.exponential(scale=5, size=n), 0, 40)\n"
            "credit_score = np.clip(850 - (debt/income)*100 + (employment_years*3) + np.random.normal(0, 30, n), 300, 850)\n"
            "has_delinquency = np.random.binomial(1, 0.15, n)\n"
            "num_credit_lines = np.random.randint(1, 15, n)\n"
            "logit = -5 + 0.00005*debt - 0.00001*income + 0.02*age + 0.03*has_delinquency - 0.05*employment_years\n"
            "p = 1/(1+np.exp(-logit))\n"
            "default = np.random.binomial(1, p)\n"
            "df = pd.DataFrame({\n"
            "    'income': income, 'debt_to_income': debt/income, 'age': age,\n"
            "    'employment_years': employment_years, 'has_delinquency': has_delinquency,\n"
            "    'num_credit_lines': num_credit_lines, 'credit_score': credit_score,\n"
            "    'default': default\n"
            "})\n"
            "print(f'Dataset shape: {df.shape}')\nprint(f'Default rate: {df[\"default\"].mean():.2%}')\ndf.head()"
        ),
        nbf.v4.new_code_cell(
            "X = df.drop(columns=['default'])\ny = df['default']\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n"
            "print(f'Train: {X_train.shape}, Test: {X_test.shape}')"
        ),
        nbf.v4.new_code_cell(
            "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import RandomizedSearchCV\n"
            "param_dist = {\n"
            "    'n_estimators': [100, 200, 300],\n"
            "    'max_depth': [5, 10, 15, None],\n"
            "    'min_samples_split': [2, 5, 10],\n"
            "    'min_samples_leaf': [1, 2, 4],\n"
            "    'class_weight': ['balanced', 'subsample']\n"
            "}\n"
            "rf = RandomForestClassifier(random_state=42, n_jobs=-1)\n"
            "search = RandomizedSearchCV(rf, param_dist, n_iter=15, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)\n"
            "search.fit(X_train, y_train)\n"
            "print(f'Best AUC: {search.best_score_:.3f}')\nprint(search.best_params_)"
        ),
        nbf.v4.new_code_cell(
            "best = search.best_estimator_\n"
            "pred = best.predict(X_test)\n"
            "proba = best.predict_proba(X_test)[:,1]\n"
            "from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix\n"
            "auc = roc_auc_score(y_test, proba)\n"
            "print(f'Test AUC: {auc:.3f}')\nprint(classification_report(y_test, pred))"
        ),
        nbf.v4.new_code_cell(
            "feat_imp = pd.DataFrame({'feature': X.columns, 'importance': best.feature_importances_})\n"
            "feat_imp = feat_imp.sort_values('importance', ascending=False)\nprint(feat_imp)"
        ),
        nbf.v4.new_markdown_cell("## Conclusion\nRandom Forest provides robust credit scoring with interpretable feature importance."),
    ]
    return nb

path = Path(__file__).resolve().parent / "01-Regression-Analysis" / "Random-Forest-Finance.ipynb"
nbf.write(nb_random_forest_finance(), str(path))
print(f"Written to {path}")
