"""Rebuild problematic notebooks with proper import structure"""
import nbformat as nbf
from pathlib import Path

def create_proper_imports_cell():
    return nbf.v4.new_code_cell(
        """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, r2_score, f1_score, 
                            accuracy_score, classification_report, make_scorer)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_regression, make_classification, make_blobs

import warnings
import time
import logging

# Advanced Visualization
try:
    import shap
except ImportError:
    pass  # Simulation mode if missing

# Configuration
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)"""
    )

def rebuild_random_forest_finance():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Random Forest - Finance\n\nCredit scoring with ensemble methods."),
        create_proper_imports_cell(),
        nbf.v4.new_code_cell("""# Synthetic financial data
np.random.seed(42)
n = 5000

income = np.clip(np.random.normal(55000, 18000, n), 15000, 200000)
debt = np.clip(np.random.exponential(scale=15000, size=n), 0, 100000)
age = np.random.randint(18, 75, n)
employment_years = np.clip(np.random.exponential(scale=5, size=n), 0, 40)
credit_score = np.clip(850 - (debt/income)*100 + (employment_years*3) + np.random.normal(0, 30, n), 300, 850)
has_delinquency = np.random.binomial(1, 0.15, n)
num_credit_lines = np.random.randint(1, 15, n)

# Target: loan default (1 = default, 0 = no default)
logit = -5 + 0.00005*debt - 0.00001*income + 0.02*age + 0.03*has_delinquency - 0.05*employment_years
p = 1/(1+np.exp(-logit))
default = np.random.binomial(1, p)

df = pd.DataFrame({
    'income': income, 'debt': income,
    'debt_to_income': debt/income, 'age': age,
    'employment_years': employment_years, 'has_delinquency': has_delinquency,
    'num_credit_lines': num_credit_lines, 'credit_score': credit_score,
    'default': default
})

print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['default'].mean():.2%}")
df.head()"""),
        nbf.v4.new_code_cell("""# Prepare features and target
X = df.drop(columns=['default'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")"""),
        nbf.v4.new_code_cell("""# Train Random Forest with hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'subsample']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(rf, param_dist, n_iter=15, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

print(f"Best AUC: {search.best_score_:.3f}")
print(search.best_params_)"""),
        nbf.v4.new_code_cell("""# Evaluate best model
best = search.best_estimator_
pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
auc = roc_auc_score(y_test, proba)
print(f"Test AUC: {auc:.3f}")
print(classification_report(y_test, pred))"""),
        nbf.v4.new_code_cell("""# Feature importance
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': best.feature_importances_})
feat_imp = feat_imp.sort_values('importance', ascending=False)
print(feat_imp)"""),
        nbf.v4.new_markdown_cell("## Conclusion\n\nRandom Forest provides robust credit scoring with interpretable feature importance.")
    ]
    return nb

# List of notebooks to rebuild with proper imports
rebuilds = {
    "02-Classification/Random-Forest-Finance.ipynb": rebuild_random_forest_finance,
}

root = Path(__file__).resolve().parent
for rel, builder in rebuilds.items():
    path = root / rel
    print(f"REBUILD: {rel}")
    nb = builder()
    nbf.write(nb, str(path))
    print(f"  -> Written to {path}")
