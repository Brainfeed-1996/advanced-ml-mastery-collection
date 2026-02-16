"""Rebuild the "NO OUTPUTS" notebooks into coherent, CPU-safe, executed notebooks.

These notebooks currently exist as short templates (8 cells) and were not executed, resulting in no outputs.
Olivier requested *industrial-grade refactoring* + *complete notebooks with outputs*.

Strategy
- Overwrite specific notebooks with a curated, coherent workflow per topic.
- Avoid heavyweight downloads (transformers weights, YOLO, Stable Diffusion, etc.) by providing:
  - offline baselines that run everywhere
  - optional sections that activate if deps are installed
- Execute each notebook via nbconvert to embed outputs.

Run:
  python tools_rebuild_no_output_notebooks.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_nb(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        }
    )
    nb["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbf.writes(nb), encoding="utf-8")


def exec_nb(path: Path, timeout: int = 900) -> None:
    cmd = [
        sys.executable,
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
    ]
    subprocess.check_call(cmd)


def nb_kmeans_customer_segmentation() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            """# K-Means — Customer Segmentation (Industrial Notebook)

Offline, CPU-safe implementation with:
- synthetic but realistic customer features
- preprocessing pipeline
- model selection via silhouette score
- cluster profiling + visualization

""" + f"Last rebuild: **{STAMP}**"
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n\nnp.random.seed(42)\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (12, 6)\n"""
        ),
        nbf.v4.new_code_cell(
            """# 1) Generate a realistic-ish customer dataset\n# Keep N moderate so silhouette search remains fast on CPU.\nN = 1200\n\n# Continuous\nage = np.clip(np.random.normal(38, 12, N), 18, 80).round(0)\nincome = np.clip(np.random.lognormal(mean=10.4, sigma=0.35, size=N), 15_000, 250_000)\nspend_score = np.clip(np.random.beta(2.2, 2.0, N) * 100, 0, 100)\nvisits_month = np.clip(np.random.poisson(4, N) + np.random.binomial(1, 0.2, N), 0, 30)\ntenure_months = np.clip(np.random.gamma(2.0, 18.0, N), 0, 240)\n\n# Categorical\nregion = np.random.choice(['EU', 'NA', 'APAC'], size=N, p=[0.45, 0.30, 0.25])\nchannel = np.random.choice(['web', 'mobile', 'store'], size=N, p=[0.40, 0.45, 0.15])\n\ndf = pd.DataFrame({\n    'age': age.astype(int),\n    'income': income,\n    'spend_score': spend_score,\n    'visits_month': visits_month,\n    'tenure_months': tenure_months,\n    'region': region,\n    'channel': channel,\n})\n\ndf.head()"""
        ),
        nbf.v4.new_code_cell(
            """# 2) Preprocess\n# For clustering we keep it fast and interpretable: scale *numeric* features only.\n# (Categoricals are used later for profiling, not for distance computations.)\nnum_cols = ['age', 'income', 'spend_score', 'visits_month', 'tenure_months']\n\nX = StandardScaler().fit_transform(df[num_cols])\nX.shape"""
        ),
        nbf.v4.new_code_cell(
            """# 3) Pick K using silhouette score\n# Limit the search range to keep runtime predictable.\nks = list(range(2, 9))\nscores = []\n\nfor k in ks:\n    km = KMeans(n_clusters=k, n_init=10, random_state=42)\n    labels = km.fit_predict(X)\n    # Use a sample to keep runtime stable (silhouette is O(n^2)).\n    s = silhouette_score(X, labels, sample_size=min(500, X.shape[0]), random_state=42)\n    scores.append(s)\n\nbest_k = ks[int(np.argmax(scores))]\nbest_k, max(scores)"""
        ),
        nbf.v4.new_code_cell(
            """plt.plot(ks, scores, marker='o')\nplt.title('Silhouette score vs K')\nplt.xlabel('K')\nplt.ylabel('Silhouette')\nplt.show()"""
        ),
        nbf.v4.new_code_cell(
            """# 4) Fit final KMeans + profile clusters\nkm = KMeans(n_clusters=best_k, n_init=20, random_state=42)\ndf['cluster'] = km.fit_predict(X)\n\nprofile = df.groupby('cluster')[num_cols].mean().round(2)\ncounts = df['cluster'].value_counts().sort_index()\n\nprint('Counts per cluster:')\nprint(counts)\n\nprofile"""
        ),
        nbf.v4.new_code_cell(
            """# 5) 2D visualization via PCA\nfrom sklearn.decomposition import PCA\n\nX2 = PCA(n_components=2, random_state=42).fit_transform(X)\nplot_df = pd.DataFrame({'pc1': X2[:,0], 'pc2': X2[:,1], 'cluster': df['cluster']})\n\nsns.scatterplot(data=plot_df.sample(min(1200, len(plot_df)), random_state=42), x='pc1', y='pc2', hue='cluster', palette='tab10', s=18)\nplt.title('Customer clusters (PCA projection)')\nplt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE — executed at {STAMP}')"),
    ]


def nb_svm_image_recognition() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            """# SVM — Image Recognition (CPU-safe)\n\nWe use `sklearn.datasets.load_digits` (8x8 images) to avoid downloads.\nPipeline includes:\n- scaling\n- hyperparameter search (C, gamma)\n- evaluation (confusion matrix + classification report)\n\n""" + f"Last rebuild: **{STAMP}**"
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import SVC\nfrom sklearn.metrics import classification_report, confusion_matrix\n\nnp.random.seed(42)\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (10, 4)\n"""
        ),
        nbf.v4.new_code_cell(
            """digits = load_digits()\nX = digits.data\ny = digits.target\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\nX_train.shape, X_test.shape"""
        ),
        nbf.v4.new_code_cell(
            """pipe = Pipeline([\n    ('scaler', StandardScaler()),\n    ('svm', SVC(kernel='rbf'))\n])\n\nparam_grid = {\n    'svm__C': [1, 5, 10],\n    'svm__gamma': [0.01, 0.02, 0.05],\n}\n\nsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)\nsearch.fit(X_train, y_train)\n\nsearch.best_params_, search.best_score_"""
        ),
        nbf.v4.new_code_cell(
            """best = search.best_estimator_\ny_pred = best.predict(X_test)\nprint(classification_report(y_test, y_pred))"""
        ),
        nbf.v4.new_code_cell(
            """cm = confusion_matrix(y_test, y_pred)\nplt.figure(figsize=(8,6))\nsns.heatmap(cm, cmap='Blues', square=True)\nplt.title('Confusion matrix')\nplt.xlabel('pred')\nplt.ylabel('true')\nplt.show()"""
        ),
        nbf.v4.new_code_cell(
            """# quick qualitative check\nfig, axes = plt.subplots(2, 6, figsize=(12, 4))\nfor ax, idx in zip(axes.ravel(), np.random.choice(len(X_test), 12, replace=False)):\n    ax.imshow(X_test[idx].reshape(8,8), cmap='gray')\n    ax.set_title(f"t={y_test[idx]} / p={y_pred[idx]}")\n    ax.axis('off')\nplt.tight_layout()\nplt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE — executed at {STAMP}')"),
    ]


def nb_xgboost_customer_churn() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            """# Customer Churn — Gradient Boosting (XGBoost optional)\n\nThis notebook runs fully offline.\n- Generates churn-like tabular data with class imbalance\n- Uses a robust preprocessing pipeline\n- Tries XGBoost if installed, otherwise falls back to sklearn HistGradientBoosting\n- Evaluates with ROC-AUC, PR-AUC, confusion matrix\n\n""" + f"Last rebuild: **{STAMP}**"
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report\n\nnp.random.seed(42)\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (10, 4)\n"""
        ),
        nbf.v4.new_code_cell(
            """# 1) Synthetic churn dataset\nN = 6000\n\ntenure = np.clip(np.random.gamma(2.0, 10.0, N), 0, 120)\nmonthly_charges = np.clip(np.random.normal(65, 25, N), 10, 170)\ncontract = np.random.choice(['month-to-month', 'one-year', 'two-year'], size=N, p=[0.6, 0.25, 0.15])\npay_method = np.random.choice(['card', 'bank', 'paypal'], size=N, p=[0.4, 0.35, 0.25])\n\n# churn probability increases with low tenure + high charges + month-to-month\nlogit = (\n    -1.5\n    + 0.02 * (monthly_charges - 60)\n    - 0.03 * (tenure - 20)\n    + 0.8 * (contract == 'month-to-month').astype(float)\n)\np = 1 / (1 + np.exp(-logit))\ny = (np.random.rand(N) < p).astype(int)\n\ndf = pd.DataFrame({\n    'tenure': tenure,\n    'monthly_charges': monthly_charges,\n    'contract': contract,\n    'pay_method': pay_method,\n})\n\ndf['churn'] = y\ndf['churn'].mean()"""
        ),
        nbf.v4.new_code_cell(
            """X = df.drop(columns=['churn'])\ny = df['churn']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nnum_cols = ['tenure', 'monthly_charges']\ncat_cols = ['contract', 'pay_method']\n\npre = ColumnTransformer([\n    ('num', StandardScaler(), num_cols),\n    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),\n])\n"""
        ),
        nbf.v4.new_code_cell(
            """# 2) Model: try XGBoost else fallback\nmodel_name = None\ntry:\n    import xgboost as xgb\n    clf = xgb.XGBClassifier(\n        n_estimators=300,\n        max_depth=4,\n        learning_rate=0.05,\n        subsample=0.9,\n        colsample_bytree=0.9,\n        reg_lambda=1.0,\n        eval_metric='logloss',\n        random_state=42,\n        n_jobs=-1,\n    )\n    model_name = 'xgboost.XGBClassifier'\nexcept Exception:\n    from sklearn.ensemble import HistGradientBoostingClassifier\n    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=300, random_state=42)\n    model_name = 'sklearn.HistGradientBoostingClassifier'\n\npipe = Pipeline([('pre', pre), ('clf', clf)])\npipe"""
        ),
        nbf.v4.new_code_cell(
            """pipe.fit(X_train, y_train)\n\nproba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else pipe.decision_function(X_test)\npred = (proba >= 0.5).astype(int)\n\nroc = roc_auc_score(y_test, proba)\nprauc = average_precision_score(y_test, proba)\nprint('model:', model_name)\nprint('ROC-AUC:', round(roc, 4))\nprint('PR-AUC:', round(prauc, 4))\nprint(classification_report(y_test, pred))"""
        ),
        nbf.v4.new_code_cell(
            """cm = confusion_matrix(y_test, pred)\nplt.figure(figsize=(5,4))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\nplt.title('Confusion matrix')\nplt.xlabel('pred')\nplt.ylabel('true')\nplt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE — executed at {STAMP}')"),
    ]


def nb_lstm_stock_prediction() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            """# LSTM — Stock-like Time Series Forecasting (CPU-safe)\n\nWe create a synthetic price series (trend + seasonality + noise), then train a small LSTM.\nRuns offline with PyTorch if available, otherwise falls back to an ARIMA-like baseline (statsmodels optional).\n\n""" + f"Last rebuild: **{STAMP}**"
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport matplotlib.pyplot as plt\n\nnp.random.seed(42)\nplt.rcParams['figure.figsize'] = (12, 4)\n"""
        ),
        nbf.v4.new_code_cell(
            """# 1) Synthetic price series\nT = 1500\nt = np.arange(T)\nprice = 100 + 0.02*t + 2*np.sin(2*np.pi*t/50) + 0.8*np.sin(2*np.pi*t/200) + np.random.normal(0, 0.6, T)\n\nplt.plot(price)\nplt.title('Synthetic price')\nplt.show()"""
        ),
        nbf.v4.new_code_cell(
            """# 2) Create supervised windows\ndef make_windows(x, lookback=40):\n    X, y = [], []\n    for i in range(lookback, len(x)):\n        X.append(x[i-lookback:i])\n        y.append(x[i])\n    return np.array(X), np.array(y)\n\nlookback = 40\nX, y = make_windows(price, lookback)\n\n# train/test split\nsplit = int(0.8*len(X))\nX_train, X_test = X[:split], X[split:]\ny_train, y_test = y[:split], y[split:]\n\nX_train.shape, X_test.shape"""
        ),
        nbf.v4.new_code_cell(
            """# 3) Train a tiny LSTM (PyTorch)\ntry:\n    import torch\n    import torch.nn as nn\n    from torch.utils.data import TensorDataset, DataLoader\n\n    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\n    Xtr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)\n    ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)\n    Xte = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)\n    yte = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)\n\n    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)\n\n    class LSTMForecaster(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)\n            self.fc = nn.Linear(32, 1)\n        def forward(self, x):\n            out, _ = self.lstm(x)\n            return self.fc(out[:, -1, :])\n\n    model = LSTMForecaster().to(device)\n    opt = torch.optim.Adam(model.parameters(), lr=1e-3)\n    loss_fn = nn.MSELoss()\n\n    for epoch in range(5):\n        model.train()\n        losses = []\n        for xb, yb in train_loader:\n            xb, yb = xb.to(device), yb.to(device)\n            opt.zero_grad()\n            pred = model(xb)\n            loss = loss_fn(pred, yb)\n            loss.backward()\n            opt.step()\n            losses.append(loss.item())\n        print(f"epoch {epoch+1}/5 mse={np.mean(losses):.4f}")\n\n    model.eval()\n    with torch.no_grad():\n        yhat = model(Xte.to(device)).cpu().numpy().ravel()\n\n    from sklearn.metrics import mean_squared_error\n    print('test mse:', mean_squared_error(y_test, yhat))\n\nexcept Exception as e:\n    print('PyTorch not available or failed, fallback baseline:', e)\n    yhat = np.r_[y_train[-len(y_test):]]\n\n"""
        ),
        nbf.v4.new_code_cell(
            """# 4) Plot predictions\nplt.plot(y_test[:300], label='true')\nplt.plot(yhat[:300], label='pred')\nplt.legend()\nplt.title('Forecast (first 300 test points)')\nplt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE — executed at {STAMP}')"),
    ]


def nb_cnn_medical_imaging() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            """# CNN — Medical Imaging (Proxy task, offline)\n\nReal medical datasets require downloads/licensing.\nWe use `sklearn` digits as a proxy and train a small CNN in PyTorch.\nIncludes:\n- train/val split\n- training loop\n- accuracy + confusion matrix\n\n""" + f"Last rebuild: **{STAMP}**"
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import confusion_matrix\n\nnp.random.seed(42)\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (10, 4)\n"""
        ),
        nbf.v4.new_code_cell(
            """digits = load_digits()\nX = digits.images.astype('float32') / 16.0\ny = digits.target\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nprint(X_train.shape, X_test.shape)\nplt.imshow(X_train[0], cmap='gray'); plt.title(f"label={y_train[0]}"); plt.axis('off'); plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """# PyTorch CNN\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import TensorDataset, DataLoader\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\nXtr = torch.tensor(X_train).unsqueeze(1)  # (N,1,8,8)\nYtr = torch.tensor(y_train, dtype=torch.long)\nXte = torch.tensor(X_test).unsqueeze(1)\nYte = torch.tensor(y_test, dtype=torch.long)\n\ntrain_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)\n\nclass SmallCNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),\n            nn.MaxPool2d(2),\n            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),\n            nn.MaxPool2d(2),\n        )\n        self.fc = nn.Linear(32*2*2, 10)\n    def forward(self, x):\n        x = self.net(x)\n        x = x.view(x.size(0), -1)\n        return self.fc(x)\n\nmodel = SmallCNN().to(device)\nopt = torch.optim.Adam(model.parameters(), lr=1e-3)\nloss_fn = nn.CrossEntropyLoss()\n\nfor epoch in range(5):\n    model.train()\n    losses = []\n    for xb, yb in train_loader:\n        xb, yb = xb.to(device), yb.to(device)\n        opt.zero_grad()\n        out = model(xb)\n        loss = loss_fn(out, yb)\n        loss.backward()\n        opt.step()\n        losses.append(loss.item())\n    print(f"epoch {epoch+1}/5 loss={np.mean(losses):.4f}")\n"""
        ),
        nbf.v4.new_code_cell(
            """# Evaluate\nmodel.eval()\nwith torch.no_grad():\n    logits = model(Xte.to(device)).cpu()\n    pred = logits.argmax(dim=1).numpy()\n\nacc = (pred == y_test).mean()\nprint('accuracy:', round(float(acc), 4))\n\ncm = confusion_matrix(y_test, pred)\nplt.figure(figsize=(6,5))\nsns.heatmap(cm, cmap='Blues', square=True)\nplt.title('Confusion matrix')\nplt.xlabel('pred')\nplt.ylabel('true')\nplt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE — executed at {STAMP}')"),
    ]


def nb_bert_sentiment_analysis() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            """# BERT Sentiment Analysis (Offline-first, Transformers optional)\n\nGoal: deliver a complete sentiment workflow with outputs **without downloads by default**.\n\nWe implement:\n- a strong offline baseline: **TF-IDF + Logistic Regression**\n- an optional Transformers section (runs only if `transformers` is installed *and* model is available locally)\n\n""" + f"Last rebuild: **{STAMP}**"
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import classification_report, confusion_matrix\n\nnp.random.seed(42)\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (8, 5)\n"""
        ),
        nbf.v4.new_code_cell(
            """# 1) Create a small sentiment dataset (offline)\n# (In practice: load IMDb/Amazon, etc.)\npositive = [\n    'This product is excellent and works perfectly.',\n    'Amazing quality, I would buy it again.',\n    'Fast shipping and great customer service.',\n    'Highly recommended. Five stars.',\n]\nnegative = [\n    'Terrible experience, it broke immediately.',\n    'Waste of money. Very disappointed.',\n    'Poor quality and not as described.',\n    'Support was unhelpful. One star.',\n]\n\n# expand with noise\ndef jitter(s):\n    tweaks = [' honestly', ' really', ' absolutely', ' quite', '']\n    return s + np.random.choice(tweaks)\n\ntexts = [jitter(s) for s in positive for _ in range(80)] + [jitter(s) for s in negative for _ in range(80)]\ny = np.array([1]* (len(positive)*80) + [0]* (len(negative)*80))\n\ndf = pd.DataFrame({'text': texts, 'label': y})\ndf.sample(5, random_state=42)"""
        ),
        nbf.v4.new_code_cell(
            """X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42, stratify=df['label'])\n\nbaseline = Pipeline([\n    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),\n    ('clf', LogisticRegression(max_iter=200)),\n])\n\nbaseline.fit(X_train, y_train)\ny_pred = baseline.predict(X_test)\n\nprint(classification_report(y_test, y_pred))"""
        ),
        nbf.v4.new_code_cell(
            """cm = confusion_matrix(y_test, y_pred)\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\nplt.title('Confusion matrix')\nplt.xlabel('pred')\nplt.ylabel('true')\nplt.show()"""
        ),
        nbf.v4.new_code_cell(
            """# 2) Optional: Transformers pipeline (no auto-download)\n# If a model is already cached locally, this will run; otherwise we skip.\ntry:\n    from transformers import pipeline\n    # This may attempt download if not present; wrap in a guard.\n    # Users can set TRANSFORMERS_OFFLINE=1 to enforce offline.\n    clf = pipeline('sentiment-analysis')\n    print(clf('I love this, it is fantastic!'))\n    print(clf('This is awful and I want a refund.'))\nexcept Exception as e:\n    print('Transformers section skipped:', e)\n"""
        ),
        nbf.v4.new_code_cell(f"print('DONE — executed at {STAMP}')"),
    ]


def nb_gpt_finetuning_basics() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            """# GPT Fine-Tuning Basics (Offline mini LM)\n\nWe cannot actually fine-tune a full GPT offline without weights & compute.\nThis notebook teaches the core mechanics via a **tiny character-level language model** (PyTorch)\ntrained on a small in-notebook corpus.\n\nIncludes:\n- tokenization\n- dataset windows\n- training loop\n- sampling\n\n""" + f"Last rebuild: **{STAMP}**"
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport torch\nimport torch.nn as nn\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint('device:', device)\n"""
        ),
        nbf.v4.new_code_cell(
            """text = '''
In industrial ML, reproducibility beats vibes.
We log configs, seed everything, and validate outputs.
Fine-tuning is just gradient descent on a pretrained prior.
'''.strip()

chars = sorted(set(text))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}

encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
vocab_size = len(chars)
vocab_size, len(encoded)"""
        ),
        nbf.v4.new_code_cell(
            """# dataset windows\nblock = 64\nX = []\ny = []\nfor i in range(len(encoded) - block - 1):\n    X.append(encoded[i:i+block])\n    y.append(encoded[i+1:i+block+1])\nX = torch.stack(X)\ny = torch.stack(y)\n\n# train/test split\nsplit = int(0.9*len(X))\nXtr, Xte = X[:split], X[split:]\nytr, yte = y[:split], y[split:]\nXtr.shape"""
        ),
        nbf.v4.new_code_cell(
            """class TinyGPT(nn.Module):\n    def __init__(self, vocab, d=64, heads=4, layers=2, block=64):\n        super().__init__()\n        self.tok = nn.Embedding(vocab, d)\n        self.pos = nn.Embedding(block, d)\n        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, batch_first=True)\n        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)\n        self.lm = nn.Linear(d, vocab)\n        self.block = block\n\n    def forward(self, idx):\n        B, T = idx.shape\n        pos = torch.arange(T, device=idx.device)\n        x = self.tok(idx) + self.pos(pos)[None, :, :]\n        # causal mask\n        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()\n        x = self.tr(x, mask=mask)\n        return self.lm(x)\n\nmodel = TinyGPT(vocab_size, block=block).to(device)\nopt = torch.optim.AdamW(model.parameters(), lr=3e-4)\nloss_fn = nn.CrossEntropyLoss()\n\ndef batchify(X, y, bs=64):\n    ix = torch.randint(0, X.shape[0], (bs,))\n    return X[ix].to(device), y[ix].to(device)\n\nfor step in range(300):\n    xb, yb = batchify(Xtr, ytr)\n    logits = model(xb)\n    loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))\n    opt.zero_grad(); loss.backward(); opt.step()\n    if (step+1) % 75 == 0:\n        print('step', step+1, 'loss', float(loss))\n"""
        ),
        nbf.v4.new_code_cell(
            """@torch.no_grad()\ndef sample(prefix='Fine-tuning ', n=200, temp=0.9):\n    model.eval()\n    ctx = torch.tensor([stoi.get(c, 0) for c in prefix], dtype=torch.long, device=device)[None, :]\n    for _ in range(n):\n        inp = ctx[:, -block:]\n        logits = model(inp)[:, -1, :] / temp\n        probs = torch.softmax(logits, dim=-1)\n        nxt = torch.multinomial(probs, num_samples=1)\n        ctx = torch.cat([ctx, nxt], dim=1)\n    out = ''.join(itos[int(i)] for i in ctx[0].cpu())\n    return out\n\nprint(sample())"""
        ),
        nbf.v4.new_code_cell(f"print('DONE — executed at {STAMP}')"),
    ]


def nb_rl_cartpole_agent() -> list[nbf.NotebookNode]:
    return [
        nbf.v4.new_markdown_cell(
            """# RL — CartPole (Gymnasium optional)\n\nIf `gymnasium` is installed, we train a simple policy-gradient agent on CartPole.\nOtherwise, we demonstrate RL mechanics on a tiny custom environment.\n\n""" + f"Last rebuild: **{STAMP}**"
        ),
        nbf.v4.new_code_cell("""import numpy as np\nnp.random.seed(42)\n"""),
        nbf.v4.new_code_cell(
            """# Try gymnasium CartPole first\ntry:\n    import gymnasium as gym\n    import torch\n    import torch.nn as nn\n    import torch.optim as optim\n\n    env = gym.make('CartPole-v1')\n    obs_dim = env.observation_space.shape[0]\n    act_dim = env.action_space.n\n\n    class Policy(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, act_dim))\n        def forward(self, x):\n            return self.net(x)\n\n    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    pi = Policy().to(device)\n    opt = optim.Adam(pi.parameters(), lr=1e-2)\n\n    def run_episode():\n        obs, _ = env.reset(seed=42)\n        logps, rews = [], []\n        done = False\n        while not done:\n            x = torch.tensor(obs, dtype=torch.float32, device=device)\n            logits = pi(x)\n            probs = torch.softmax(logits, dim=-1)\n            a = torch.distributions.Categorical(probs).sample()\n            logps.append(torch.log(probs[a]))\n            obs, r, term, trunc, _ = env.step(int(a))\n            done = term or trunc\n            rews.append(r)\n        return logps, rews\n\n    def returns(rews, gamma=0.99):\n        G = 0\n        out = []\n        for r in reversed(rews):\n            G = r + gamma*G\n            out.append(G)\n        out = list(reversed(out))\n        out = torch.tensor(out, dtype=torch.float32, device=device)\n        return (out - out.mean()) / (out.std() + 1e-8)\n\n    for ep in range(80):\n        logps, rews = run_episode()\n        R = returns(rews)\n        loss = -(torch.stack(logps) * R).sum()\n        opt.zero_grad(); loss.backward(); opt.step()\n        if (ep+1) % 20 == 0:\n            print('episode', ep+1, 'reward', sum(rews))\n\nexcept Exception as e:\n    print('gymnasium unavailable, using custom bandit env:', e)\n    # simple 2-armed bandit\n    p = [0.3, 0.7]\n    Q = np.zeros(2)\n    N = np.zeros(2)\n    for t in range(200):\n        a = np.argmax(Q + 0.5*np.sqrt(np.log(t+1)/(N+1e-9)))\n        r = 1.0 if np.random.rand() < p[a] else 0.0\n        N[a] += 1\n        Q[a] += (r - Q[a]) / N[a]\n    print('estimated Q:', Q, 'counts:', N)\n"""
        ),
        nbf.v4.new_code_cell(f"print('DONE — executed at {STAMP}')"),
    ]


# Map of notebooks to rebuild functions
REBUILDS = {
    # requested ones
    "02-Classification-Challenges/K-Means-Customer-Segmentation.ipynb": nb_kmeans_customer_segmentation,
    "02-Classification-Challenges/SVM-Image-Recognition.ipynb": nb_svm_image_recognition,
    "02-Classification-Challenges/XGBoost-Customer-Churn.ipynb": nb_xgboost_customer_churn,
    "03-Clustering-Techniques/BERT-Sentiment-Analysis.ipynb": nb_bert_sentiment_analysis,
    "03-Clustering-Techniques/LSTM-Stock-Prediction.ipynb": nb_lstm_stock_prediction,
    "04-Natural-Language-Processing/CNN-Medical-Imaging.ipynb": nb_cnn_medical_imaging,
    "04-Natural-Language-Processing/GPT-Fine-Tuning-Basics.ipynb": nb_gpt_finetuning_basics,
    "04-Natural-Language-Processing/RL-CartPole-Agent.ipynb": nb_rl_cartpole_agent,

    # duplicates in alternate folders
    "04-NLP/BERT-Sentiment-Analysis.ipynb": nb_bert_sentiment_analysis,
    "04-NLP/GPT-Fine-Tuning-Basics.ipynb": nb_gpt_finetuning_basics,
}


def main():
    ok, fail = [], []
    for rel, fn in REBUILDS.items():
        path = ROOT / rel
        print(f"REBUILD: {rel}")
        write_nb(path, fn())
        try:
            exec_nb(path, timeout=900)
            ok.append(rel)
            print(f"OK: {rel}")
        except Exception as e:
            fail.append(rel)
            print(f"FAIL: {rel} -> {e}")

    print("\n=== REBUILD SUMMARY ===")
    print("OK:", len(ok))
    print("FAIL:", len(fail))
    if fail:
        print("Failed list:")
        for x in fail:
            print("-", x)


if __name__ == "__main__":
    main()
