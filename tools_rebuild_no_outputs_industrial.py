"""Industrial rebuild for all notebooks currently reported as NO OUTPUTS.

This script overwrites template notebooks (8 cells, no outputs) with complete, explained, executable notebooks.
It targets real executions (Transformers/YOLO/Diffusers) but keeps runtime predictable by:
- using tiny/small models when possible
- using small synthetic/proxy datasets when licensing/downloads are problematic

Run:
  python tools_rebuild_no_outputs_industrial.py

Then:
  python validate_notebooks.py
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


def exec_nb(path: Path, timeout: int = 1800) -> None:
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
        "--ExecutePreprocessor.startup_timeout=300",
        "--output",
        path.name,
        "--output-dir",
        str(path.parent),
    ]
    subprocess.check_call(cmd)


def md(title: str, body: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"# {title}\n\n{body}\n\n_Last rebuild: **{STAMP}**_")


def nb_kmeans_segmentation():
    return [
        md(
            "K-Means — Customer Segmentation (Industrial)",
            "Complete clustering workflow: dataset → scaling → K selection (silhouette sampling) → profiling → PCA visualization.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\nfrom sklearn.decomposition import PCA\n\nnp.random.seed(42)\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (12, 5)\n"""
        ),
        nbf.v4.new_code_cell(
            """# Dataset (offline, realistic-ish)\nN = 1200\nage = np.clip(np.random.normal(38, 12, N), 18, 80)\nincome = np.clip(np.random.lognormal(mean=10.4, sigma=0.35, size=N), 15000, 250000)\nspend = np.clip(np.random.beta(2.2, 2.0, N) * 100, 0, 100)\nvisits = np.clip(np.random.poisson(4, N), 0, 30)\ntenure = np.clip(np.random.gamma(2.0, 18.0, N), 0, 240)\nregion = np.random.choice(['EU','NA','APAC'], size=N, p=[0.45,0.30,0.25])\nchannel = np.random.choice(['web','mobile','store'], size=N, p=[0.40,0.45,0.15])\n\ndf = pd.DataFrame({\n 'age': age, 'income': income, 'spend_score': spend, 'visits_month': visits, 'tenure_months': tenure,\n 'region': region, 'channel': channel\n})\n\ndf.head()"""
        ),
        nbf.v4.new_code_cell(
            """num_cols = ['age','income','spend_score','visits_month','tenure_months']\nX = StandardScaler().fit_transform(df[num_cols])\nX.shape"""
        ),
        nbf.v4.new_code_cell(
            """# K selection (silhouette is O(n^2) → sample for stability)\nks = list(range(2, 9))\nscores = []\nfor k in ks:\n    km = KMeans(n_clusters=k, n_init=10, random_state=42)\n    labels = km.fit_predict(X)\n    s = silhouette_score(X, labels, sample_size=min(500, X.shape[0]), random_state=42)\n    scores.append(s)\n\nbest_k = ks[int(np.argmax(scores))]\nbest_k, max(scores)"""
        ),
        nbf.v4.new_code_cell(
            """plt.plot(ks, scores, marker='o'); plt.title('Silhouette vs K'); plt.xlabel('K'); plt.ylabel('silhouette'); plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """km = KMeans(n_clusters=best_k, n_init=20, random_state=42)\ndf['cluster'] = km.fit_predict(X)\n\nprint(df['cluster'].value_counts().sort_index())\n\nprofile = df.groupby('cluster')[num_cols].mean().round(2)\nprofile"""
        ),
        nbf.v4.new_code_cell(
            """# PCA projection for visualization\nX2 = PCA(n_components=2, random_state=42).fit_transform(X)\nplot_df = pd.DataFrame({'pc1': X2[:,0], 'pc2': X2[:,1], 'cluster': df['cluster']})\n\nsns.scatterplot(data=plot_df.sample(min(1200, len(plot_df)), random_state=42), x='pc1', y='pc2', hue='cluster', palette='tab10', s=18)\nplt.title('Clusters (PCA)'); plt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_svm_image_recognition():
    return [
        md(
            "SVM — Image Recognition (digits)",
            "Uses sklearn digits dataset (8x8). Includes scaling, GridSearchCV, confusion matrix, sample predictions.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import SVC\nfrom sklearn.metrics import classification_report, confusion_matrix\n\nnp.random.seed(42)\nsns.set_style('whitegrid')\n"""
        ),
        nbf.v4.new_code_cell(
            """digits = load_digits()\nX = digits.data\ny = digits.target\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\nX_train.shape"""
        ),
        nbf.v4.new_code_cell(
            """pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf'))])\nparam_grid = {'svm__C':[1,5,10], 'svm__gamma':[0.01,0.02,0.05]}\nsearch = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)\nsearch.fit(X_train, y_train)\nsearch.best_params_, search.best_score_"""
        ),
        nbf.v4.new_code_cell(
            """best = search.best_estimator_\ny_pred = best.predict(X_test)\nprint(classification_report(y_test, y_pred))"""
        ),
        nbf.v4.new_code_cell(
            """cm = confusion_matrix(y_test, y_pred)\nsns.heatmap(cm, cmap='Blues', square=True)\nplt.title('Confusion matrix'); plt.xlabel('pred'); plt.ylabel('true'); plt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_bert_sentiment():
    return [
        md(
            "BERT Sentiment Analysis (real Transformers run)",
            "Offline baseline + real Transformers inference using a tiny model to keep downloads small.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import classification_report\n\nnp.random.seed(42)\n"""
        ),
        nbf.v4.new_code_cell(
            """pos = ['I love this product', 'Excellent quality', 'Amazing service', 'Works perfectly']\nneg = ['Terrible experience', 'Waste of money', 'Very disappointed', 'Broke immediately']\n\ntexts = [s for s in pos for _ in range(80)] + [s for s in neg for _ in range(80)]\ny = np.array([1]*(len(pos)*80) + [0]*(len(neg)*80))\n\ndf = pd.DataFrame({'text': texts, 'label': y})\nX_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42, stratify=df['label'])\n\nbaseline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))), ('clf', LogisticRegression(max_iter=200))])\nbaseline.fit(X_train, y_train)\npred = baseline.predict(X_test)\nprint('TFIDF baseline')\nprint(classification_report(y_test, pred))"""
        ),
        nbf.v4.new_code_cell(
            """# Real transformers inference (standard model, downloads weights on first run)\nfrom transformers import pipeline\n\nmodel_id = 'distilbert-base-uncased-finetuned-sst-2-english'\n\n# If a previous partial download corrupted the cache, remove and retry once.\ntry:\n    clf = pipeline('sentiment-analysis', model=model_id)\nexcept Exception as e:\n    msg = str(e).lower()\n    if 'state dictionary' in msg and 'corrupted' in msg:\n        import shutil\n        from pathlib import Path\n        cache_root = Path.home() / '.cache' / 'huggingface'\n        print('Detected corrupted HF cache, removing:', cache_root)\n        shutil.rmtree(cache_root, ignore_errors=True)\n        clf = pipeline('sentiment-analysis', model=model_id)\n    else:\n        raise\n\nprint(clf('I absolutely love this, it is fantastic'))\nprint(clf('This is horrible, I want a refund'))"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_lstm_stock():
    return [
        md(
            "LSTM — Stock Prediction (real training)",
            "Synthetic time series → windowing → LSTM training (PyTorch) → forecast plot.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import TensorDataset, DataLoader\nfrom sklearn.metrics import mean_squared_error\n\nnp.random.seed(42)\ntorch.manual_seed(42)\nplt.rcParams['figure.figsize']=(12,4)\n"""
        ),
        nbf.v4.new_code_cell(
            """T=1500\nt=np.arange(T)\nseries=100+0.02*t+2*np.sin(2*np.pi*t/50)+0.8*np.sin(2*np.pi*t/200)+np.random.normal(0,0.6,T)\nplt.plot(series); plt.title('Synthetic price'); plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """def windows(x, lookback=40):\n    X,y=[],[]\n    for i in range(lookback, len(x)):\n        X.append(x[i-lookback:i]); y.append(x[i])\n    return np.array(X), np.array(y)\n\nlookback=40\nX,y=windows(series, lookback)\nsplit=int(0.8*len(X))\nXtr,Xte=X[:split],X[split:]\nytr,yte=y[:split],y[split:]\n\nXtr_t=torch.tensor(Xtr, dtype=torch.float32).unsqueeze(-1)\nytr_t=torch.tensor(ytr, dtype=torch.float32).unsqueeze(-1)\nXte_t=torch.tensor(Xte, dtype=torch.float32).unsqueeze(-1)\n\ntrain_loader=DataLoader(TensorDataset(Xtr_t,ytr_t), batch_size=64, shuffle=True)\nXtr_t.shape"""
        ),
        nbf.v4.new_code_cell(
            """device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\nclass LSTMForecaster(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.lstm=nn.LSTM(input_size=1, hidden_size=32, batch_first=True)\n        self.fc=nn.Linear(32,1)\n    def forward(self,x):\n        out,_=self.lstm(x)\n        return self.fc(out[:,-1,:])\n\nmodel=LSTMForecaster().to(device)\nopt=torch.optim.Adam(model.parameters(), lr=1e-3)\nloss_fn=nn.MSELoss()\n\nfor epoch in range(5):\n    model.train(); losses=[]\n    for xb,yb in train_loader:\n        xb,yb=xb.to(device), yb.to(device)\n        opt.zero_grad()\n        pred=model(xb)\n        loss=loss_fn(pred,yb)\n        loss.backward(); opt.step()\n        losses.append(loss.item())\n    print('epoch', epoch+1, 'mse', float(np.mean(losses)))\n\nmodel.eval()\nwith torch.no_grad():\n    yhat=model(Xte_t.to(device)).cpu().numpy().ravel()\n\nprint('test mse:', mean_squared_error(yte, yhat))"""
        ),
        nbf.v4.new_code_cell(
            """plt.plot(yte[:300], label='true'); plt.plot(yhat[:300], label='pred'); plt.legend(); plt.title('Forecast'); plt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_cnn_medical_proxy():
    return [
        md(
            "CNN — Medical Imaging (proxy run)",
            "Uses digits dataset as a stand-in to demonstrate the full CNN training pipeline offline.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import confusion_matrix\n\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import TensorDataset, DataLoader\n\nnp.random.seed(42)\ntorch.manual_seed(42)\nsns.set_style('whitegrid')\n"""
        ),
        nbf.v4.new_code_cell(
            """digits=load_digits()\nX=digits.images.astype('float32')/16.0\ny=digits.target\n\nXtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)\n\nXtr_t=torch.tensor(Xtr).unsqueeze(1)\nXte_t=torch.tensor(Xte).unsqueeze(1)\nytr_t=torch.tensor(ytr, dtype=torch.long)\nyte_t=torch.tensor(yte, dtype=torch.long)\n\ntrain_loader=DataLoader(TensorDataset(Xtr_t,ytr_t), batch_size=64, shuffle=True)\n\nplt.imshow(Xtr[0], cmap='gray'); plt.title(f'label={ytr[0]}'); plt.axis('off'); plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\nclass SmallCNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv=nn.Sequential(\n            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),\n            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),\n        )\n        self.fc=nn.Linear(32*2*2, 10)\n    def forward(self,x):\n        x=self.conv(x)\n        x=x.view(x.size(0),-1)\n        return self.fc(x)\n\nmodel=SmallCNN().to(device)\nopt=torch.optim.Adam(model.parameters(), lr=1e-3)\nloss_fn=nn.CrossEntropyLoss()\n\nfor epoch in range(5):\n    model.train(); losses=[]\n    for xb,yb in train_loader:\n        xb,yb=xb.to(device), yb.to(device)\n        opt.zero_grad(); out=model(xb)\n        loss=loss_fn(out,yb)\n        loss.backward(); opt.step()\n        losses.append(loss.item())\n    print('epoch', epoch+1, 'loss', float(np.mean(losses)))\n\nmodel.eval()\nwith torch.no_grad():\n    pred=model(Xte_t.to(device)).cpu().argmax(dim=1).numpy()\nacc=(pred==yte).mean()\nprint('accuracy:', float(acc))\n\ncm=confusion_matrix(yte, pred)\nsns.heatmap(cm, cmap='Blues', square=True); plt.title('Confusion matrix'); plt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_rl_cartpole():
    return [
        md(
            "RL — CartPole Agent (real gymnasium run)",
            "Trains a simple policy-gradient agent on CartPole-v1 using gymnasium. Downloads nothing but requires gymnasium.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
"""
        ),
        nbf.v4.new_code_cell(
            """env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

pi = Policy().to(device)
opt = optim.Adam(pi.parameters(), lr=1e-2)

def run_episode(seed=None):
    obs, _ = env.reset(seed=seed)
    logps, rews = [], []
    done = False
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device)
        logits = pi(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        logps.append(dist.log_prob(a))
        obs, r, term, trunc, _ = env.step(int(a))
        done = term or trunc
        rews.append(r)
    return logps, rews

def returns(rews, gamma=0.99):
    G = 0.0
    out = []
    for r in reversed(rews):
        G = r + gamma * G
        out.append(G)
    out = list(reversed(out))
    out = torch.tensor(out, dtype=torch.float32, device=device)
    return (out - out.mean()) / (out.std() + 1e-8)

for ep in range(80):
    logps, rews = run_episode(seed=42)
    R = returns(rews)
    loss = -(torch.stack(logps) * R).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1) % 20 == 0:
        print('episode', ep+1, 'reward', sum(rews))

env.close()
"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_gpt_finetune_basics():
    return [
        md(
            "GPT Fine-Tuning Basics (tiny causal LM)",
            "Implements a tiny causal transformer, trains on a small corpus, and samples text.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np\nimport torch\nimport torch.nn as nn\n\ndevice=torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint('device:', device)\n"""
        ),
        nbf.v4.new_code_cell(
            """text = '''\nIn industrial ML, reproducibility beats vibes.\nWe log configs, seed everything, and validate outputs.\nFine-tuning is just gradient descent on a pretrained prior.\n'''.strip()\n\nchars=sorted(set(text))\nstoi={c:i for i,c in enumerate(chars)}\nitos={i:c for c,i in stoi.items()}\nenc=torch.tensor([stoi[c] for c in text], dtype=torch.long)\n\nvocab=len(chars)\nprint('vocab', vocab, 'len', len(enc))"""
        ),
        nbf.v4.new_code_cell(
            """block=64\nX=[]; y=[]\nfor i in range(len(enc)-block-1):\n    X.append(enc[i:i+block]); y.append(enc[i+1:i+block+1])\nX=torch.stack(X); y=torch.stack(y)\nsplit=int(0.9*len(X))\nXtr,Xte=X[:split],X[split:]; ytr,yte=y[:split],y[split:]\nXtr.shape"""
        ),
        nbf.v4.new_code_cell(
            """class TinyGPT(nn.Module):\n    def __init__(self, vocab, d=64, heads=4, layers=2, block=64):\n        super().__init__()\n        self.tok=nn.Embedding(vocab,d)\n        self.pos=nn.Embedding(block,d)\n        enc_layer=nn.TransformerEncoderLayer(d_model=d, nhead=heads, batch_first=True)\n        self.tr=nn.TransformerEncoder(enc_layer, num_layers=layers)\n        self.lm=nn.Linear(d, vocab)\n        self.block=block\n    def forward(self, idx):\n        B,T=idx.shape\n        pos=torch.arange(T, device=idx.device)\n        x=self.tok(idx)+self.pos(pos)[None,:,:]\n        mask=torch.triu(torch.ones(T,T,device=idx.device), diagonal=1).bool()\n        x=self.tr(x, mask=mask)\n        return self.lm(x)\n\nmodel=TinyGPT(vocab, block=block).to(device)\nopt=torch.optim.AdamW(model.parameters(), lr=3e-4)\nloss_fn=nn.CrossEntropyLoss()\n\ndef batch(bs=64):\n    ix=torch.randint(0, Xtr.shape[0], (bs,))\n    return Xtr[ix].to(device), ytr[ix].to(device)\n\nfor step in range(250):\n    xb,yb=batch()\n    logits=model(xb)\n    loss=loss_fn(logits.view(-1, vocab), yb.view(-1))\n    opt.zero_grad(); loss.backward(); opt.step()\n    if (step+1)%50==0:\n        print('step', step+1, 'loss', float(loss))\n"""
        ),
        nbf.v4.new_code_cell(
            """@torch.no_grad()\ndef sample(prefix='Fine-tuning ', n=180, temp=0.9):\n    model.eval()\n    ctx=torch.tensor([stoi.get(c,0) for c in prefix], dtype=torch.long, device=device)[None,:]\n    for _ in range(n):\n        inp=ctx[:,-block:]\n        logits=model(inp)[:,-1,:]/temp\n        probs=torch.softmax(logits, dim=-1)\n        nxt=torch.multinomial(probs, 1)\n        ctx=torch.cat([ctx,nxt], dim=1)\n    return ''.join(itos[int(i)] for i in ctx[0].cpu())\n\nprint(sample())"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_named_entity_spacy():
    return [
        md(
            "Named Entity Recognition — spaCy",
            "Loads spaCy + English model, runs NER, visualizes entities. Downloads model if missing.",
        ),
        nbf.v4.new_code_cell(
            """import spacy\nfrom spacy.cli import download\n\nmodel='en_core_web_sm'\ntry:\n    nlp = spacy.load(model)\nexcept Exception:\n    download(model)\n    nlp = spacy.load(model)\n\ntext = 'Olivier lives in Rennes and works on OpenClaw with GitHub. Apple released a new iPhone in Paris.'\ndoc = nlp(text)\n[(ent.text, ent.label_) for ent in doc.ents]"""
        ),
        nbf.v4.new_code_cell(
            """# Render to HTML (works in Jupyter).\n# In nbconvert execution, displacy may return None depending on environment.\nfrom spacy import displacy\nhtml = displacy.render(doc, style='ent')\nif isinstance(html, str):\n    print(html[:500])\nelse:\n    print('displacy.render returned:', type(html))\n"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_yolo():
    return [
        md(
            "YOLOv8 — Object Detection (real)",
            "Uses ultralytics YOLOv8n for inference. Downloads weights on first run.",
        ),
        nbf.v4.new_code_cell(
            """from ultralytics import YOLO\nimport numpy as np\nfrom PIL import Image, ImageDraw\n\n# Create a simple synthetic image (no external files)
img = Image.new('RGB', (640, 384), color=(30, 30, 30))
d = ImageDraw.Draw(img)
d.rectangle([200, 120, 440, 300], outline=(255, 255, 255), width=6)
img_path = 'synthetic_scene.jpg'
img.save(img_path)

model = YOLO('yolov8n.pt')
results = model(img_path, verbose=False)
results[0].boxes"""
        ),
        nbf.v4.new_code_cell(
            """# Visualize detections\nfrom matplotlib import pyplot as plt\n\nim = results[0].plot()  # numpy array
plt.figure(figsize=(10,5))
plt.imshow(im)
plt.axis('off')
plt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_stable_diffusion():
    return [
        md(
            "Stable Diffusion — Prompt Engineering (real)",
            "Runs Stable Diffusion v1-5 (downloads weights on first run) and generates an image.",
        ),
        nbf.v4.new_code_cell(
            """import torch\nfrom diffusers import StableDiffusionPipeline\n\n# Standard model (heavier download).\nmodel_id = 'runwayml/stable-diffusion-v1-5'\n\n# Use float32 on CPU for compatibility.
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)\npipe = pipe.to('cpu')\n\nprompt = 'a cyberpunk city skyline at night, neon lights, cinematic'\nimage = pipe(prompt, num_inference_steps=15).images[0]\nimage.size"""
        ),
        nbf.v4.new_code_cell(
            """import matplotlib.pyplot as plt\nplt.figure(figsize=(5,5))\nplt.imshow(image)\nplt.axis('off')\nplt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_isolation_forest_cybersecurity():
    return [
        md(
            "Isolation Forest — Cybersecurity Anomaly Detection",
            "End-to-end CPU-safe anomaly detection demo: synthetic security events → preprocessing → IsolationForest → scoring + ROC-AUC.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize']=(10,4)
"""
        ),
        nbf.v4.new_code_cell(
            """# Synthetic security-like events
N=6000
# normal traffic
bytes_in = np.random.lognormal(mean=10.0, sigma=0.35, size=N)
bytes_out = np.random.lognormal(mean=9.6, sigma=0.40, size=N)
failed_logins = np.random.poisson(0.2, size=N)
ports = np.random.choice([22,80,443,3389,445,53,8080], size=N, p=[0.05,0.35,0.35,0.05,0.05,0.10,0.05])

# inject anomalies
anom = np.random.rand(N) < 0.03
bytes_in[anom] *= np.random.uniform(6, 20, size=anom.sum())
failed_logins[anom] += np.random.poisson(8, size=anom.sum())
ports[anom] = np.random.choice([23,5900,4444,6667], size=anom.sum())

df = pd.DataFrame({
    'bytes_in': bytes_in,
    'bytes_out': bytes_out,
    'failed_logins': failed_logins,
    'port': ports,
    'is_anomaly': anom.astype(int)
})

df.head()"""
        ),
        nbf.v4.new_code_cell(
            """X = df[['bytes_in','bytes_out','failed_logins','port']]
y = df['is_anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

iso = IsolationForest(n_estimators=300, contamination=float(y_train.mean()), random_state=42)
iso.fit(X_train)

# IsolationForest: higher score => more normal. We invert to get anomaly score.
score = -iso.score_samples(X_test)
print('ROC-AUC:', roc_auc_score(y_test, score))

sns.histplot(score[y_test==0], label='normal', kde=True, stat='density', color='C0', alpha=0.4)
sns.histplot(score[y_test==1], label='anomaly', kde=True, stat='density', color='C3', alpha=0.4)
plt.title('Anomaly score distribution')
plt.legend(); plt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_autoencoders_denoising():
    return [
        md(
            "Autoencoders — Denoising (real training)",
            "Train a small denoising autoencoder on sklearn digits (offline). Shows reconstructions.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits

np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['figure.figsize']=(10,4)
"""
        ),
        nbf.v4.new_code_cell(
            """digits = load_digits()
X = digits.data.astype('float32') / 16.0

# add gaussian noise
noise = np.random.normal(0, 0.35, X.shape).astype('float32')
X_noisy = np.clip(X + noise, 0.0, 1.0)

X_t = torch.tensor(X, dtype=torch.float32)
Xn_t = torch.tensor(X_noisy, dtype=torch.float32)

loader = DataLoader(TensorDataset(Xn_t, X_t), batch_size=128, shuffle=True)
X.shape"""
        ),
        nbf.v4.new_code_cell(
            """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
    def forward(self, x):
        return self.dec(self.enc(x))

model = AE().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(8):
    model.train(); losses=[]
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward(); opt.step()
        losses.append(loss.item())
    print('epoch', epoch+1, 'mse', float(np.mean(losses)))
"""
        ),
        nbf.v4.new_code_cell(
            """# visualize reconstructions
model.eval()
with torch.no_grad():
    sample = Xn_t[:12].to(device)
    recon = model(sample).cpu().numpy()

fig, axes = plt.subplots(3, 12, figsize=(14, 4))
for i in range(12):
    axes[0,i].imshow(X_noisy[i].reshape(8,8), cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(X[i].reshape(8,8), cmap='gray'); axes[1,i].axis('off')
    axes[2,i].imshow(recon[i].reshape(8,8), cmap='gray'); axes[2,i].axis('off')
axes[0,0].set_ylabel('noisy')
axes[1,0].set_ylabel('clean')
axes[2,0].set_ylabel('recon')
plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_gan_synthetic_data_gen():
    return [
        md(
            "GAN — Synthetic Data Generation (minimal real run)",
            "Train a tiny GAN on 2D Gaussian mixture (offline) and plot generated samples.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['figure.figsize']=(6,6)
"""
        ),
        nbf.v4.new_code_cell(
            """# Real data: 2D mixture
N=4000
centers = np.array([[0,0],[3,0],[0,3],[-3,0],[0,-3]], dtype='float32')
idx = np.random.randint(0, len(centers), size=N)
real = centers[idx] + np.random.normal(0, 0.3, size=(N,2)).astype('float32')
real_t = torch.tensor(real)
"""
        ),
        nbf.v4.new_code_cell(
            """device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
D = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)

optG = torch.optim.Adam(G.parameters(), lr=2e-4)
optD = torch.optim.Adam(D.parameters(), lr=2e-4)
loss_fn = nn.BCEWithLogitsLoss()

def batch(bs=256):
    ix = torch.randint(0, real_t.shape[0], (bs,))
    return real_t[ix].to(device)

for step in range(800):
    x_real = batch()
    z = torch.randn(x_real.size(0), 8, device=device)
    x_fake = G(z).detach()

    # D
    optD.zero_grad()
    lossD = loss_fn(D(x_real), torch.ones(x_real.size(0),1,device=device)) + \
            loss_fn(D(x_fake), torch.zeros(x_fake.size(0),1,device=device))
    lossD.backward(); optD.step()

    # G
    z = torch.randn(x_real.size(0), 8, device=device)
    optG.zero_grad()
    gen = G(z)
    lossG = loss_fn(D(gen), torch.ones(gen.size(0),1,device=device))
    lossG.backward(); optG.step()

    if (step+1) % 200 == 0:
        print('step', step+1, 'lossD', float(lossD), 'lossG', float(lossG))
"""
        ),
        nbf.v4.new_code_cell(
            """# Plot samples
with torch.no_grad():
    z = torch.randn(1500, 8, device=device)
    fake = G(z).cpu().numpy()

plt.scatter(real[:1000,0], real[:1000,1], s=6, alpha=0.4, label='real')
plt.scatter(fake[:,0], fake[:,1], s=6, alpha=0.4, label='fake')
plt.legend(); plt.title('Tiny GAN: real vs fake'); plt.show()"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def nb_resnet_transfer_learning():
    return [
        md(
            "ResNet Transfer Learning (real torchvision weights)",
            "Loads a pretrained ResNet18 from torchvision, extracts embeddings, and trains a small classifier on digits-as-proxy dataset.",
        ),
        nbf.v4.new_code_cell(
            """import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
torch.manual_seed(42)
"""
        ),
        nbf.v4.new_code_cell(
            """# Use digits dataset as a tiny image dataset (8x8) then upsample to 224x224 RGB
D = load_digits()
X = D.images.astype('float32') / 16.0
Y = D.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

def to_rgb224(x):
    # x: (N,8,8)
    t = torch.tensor(x).unsqueeze(1)  # (N,1,8,8)
    t = torch.nn.functional.interpolate(t, size=(224,224), mode='bilinear', align_corners=False)
    t = t.repeat(1,3,1,1)
    return t

Xtr = to_rgb224(X_train)
Xte = to_rgb224(X_test)
Ytr = torch.tensor(y_train, dtype=torch.long)
Yte = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=32, shuffle=True)
Xtr.shape"""
        ),
        nbf.v4.new_code_cell(
            """device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pretrained resnet18 (downloads weights on first run)
weights = ResNet18_Weights.DEFAULT
backbone = resnet18(weights=weights)
backbone.fc = nn.Identity()  # output embeddings
backbone = backbone.to(device)
backbone.eval()

# simple linear classifier on top (train only head)
head = nn.Linear(512, 10).to(device)
opt = torch.optim.Adam(head.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    head.train()
    losses=[]
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            emb = backbone(xb)
        logits = head(emb)
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print('epoch', epoch+1, 'loss', float(np.mean(losses)))

# eval
head.eval()
with torch.no_grad():
    emb = backbone(Xte.to(device))
    pred = head(emb).argmax(dim=1).cpu().numpy()

print('accuracy:', accuracy_score(y_test, pred))"""
        ),
        nbf.v4.new_code_cell(f"print('DONE {STAMP}')"),
    ]


def builder_for(rel: str):
    name = Path(rel).name
    if name == 'K-Means-Customer-Segmentation.ipynb':
        return nb_kmeans_segmentation
    if name == 'SVM-Image-Recognition.ipynb':
        return nb_svm_image_recognition
    if name == 'BERT-Sentiment-Analysis.ipynb':
        return nb_bert_sentiment
    if name == 'LSTM-Stock-Prediction.ipynb':
        return nb_lstm_stock
    if name == 'CNN-Medical-Imaging.ipynb':
        return nb_cnn_medical_proxy
    if name == 'GPT-Fine-Tuning-Basics.ipynb':
        return nb_gpt_finetune_basics
    if name == 'Named-Entity-Recognition-Spacy.ipynb':
        return nb_named_entity_spacy
    if name == 'Isolation-Forest-Cybersecurity.ipynb':
        return nb_isolation_forest_cybersecurity
    if name == 'Autoencoders-Denoising.ipynb':
        return nb_autoencoders_denoising
    if name == 'GAN-Synthetic-Data-Gen.ipynb':
        return nb_gan_synthetic_data_gen
    if name == 'ResNet-Transfer-Learning.ipynb':
        return nb_resnet_transfer_learning
    if name in {'YOLOv8-Object-Detection.ipynb', 'Object-Detection-YOLOv8.ipynb'}:
        return nb_yolo
    if name == 'Stable-Diffusion-Prompt-Eng.ipynb':
        return nb_stable_diffusion
    if name == 'RL-CartPole-Agent.ipynb':
        return nb_rl_cartpole
    # default: keep existing (do not overwrite)
    return None


def main():
    # List from validate_notebooks.py (NO OUTPUTS)
    targets = [
        r"02-Classification-Challenges/K-Means-Customer-Segmentation.ipynb",
        r"02-Classification-Challenges/SVM-Image-Recognition.ipynb",
        r"02-Classification-Challenges/XGBoost-Customer-Churn.ipynb",
        r"03-Clustering-Techniques/BERT-Sentiment-Analysis.ipynb",
        r"03-Clustering-Techniques/LSTM-Stock-Prediction.ipynb",
        r"04-Natural-Language-Processing/CNN-Medical-Imaging.ipynb",
        r"04-Natural-Language-Processing/GPT-Fine-Tuning-Basics.ipynb",
        r"04-Natural-Language-Processing/RL-CartPole-Agent.ipynb",
        r"04-NLP/BERT-Sentiment-Analysis.ipynb",
        r"04-NLP/GPT-Fine-Tuning-Basics.ipynb",
        r"04-NLP/Named-Entity-Recognition-Spacy.ipynb",
        r"05-Computer-Vision/Autoencoders-Denoising.ipynb",
        r"05-Computer-Vision/CNN-Medical-Imaging.ipynb",
        r"05-Computer-Vision/GAN-Synthetic-Data-Gen.ipynb",
        r"05-Computer-Vision/Isolation-Forest-Cybersecurity.ipynb",
        r"05-Computer-Vision/ResNet-Transfer-Learning.ipynb",
        r"05-Computer-Vision/YOLOv8-Object-Detection.ipynb",
        r"06-Reinforcement-Learning/Deep-Q-Network-Atari.ipynb",
        r"06-Reinforcement-Learning/Feature-Engineering-Pipeline.ipynb",
        r"06-Reinforcement-Learning/Hyperparameter-Optimization-Optuna.ipynb",
        r"06-Reinforcement-Learning/Prophet-Market-Trends.ipynb",
        r"06-Reinforcement-Learning/Q-Learning-Maze-Solver.ipynb",
        r"06-Reinforcement-Learning/RL-CartPole-Agent.ipynb",
        r"07-Time-Series/LSTM-Stock-Prediction.ipynb",
        r"07-Time-Series/Prophet-Market-Trends.ipynb",
        r"07-Time-Series-Forecasting/Decision-Trees-Interpretability.ipynb",
        r"07-Time-Series-Forecasting/ML-Model-Monitoring-Prometheus.ipynb",
        r"07-Time-Series-Forecasting/Model-Quantization-TensorRT.ipynb",
        r"09-Generative-AI/GAN-Face-Generation.ipynb",
        r"09-Generative-AI/Named-Entity-Recognition-Spacy.ipynb",
        r"09-Generative-AI/Object-Detection-YOLOv8.ipynb",
        r"09-Generative-AI/Stable-Diffusion-Prompt-Eng.ipynb",
        r"09-Generative-AI/Transfer-Learning-ResNet.ipynb",
        r"09-Generative-AI/Variational-Autoencoders-MNIST.ipynb",
        r"10-MLOps/Edge-Computing-TinyML.ipynb",
        r"10-MLOps/Model-Quantization-TensorRT.ipynb",
        r"10-MLOps/Prometheus-ML-Monitoring.ipynb",
        r"10-MLOps-Production/Federated-Learning-Privacy.ipynb",
        r"10-MLOps-Production/ML-Edge-Computing-TinyML.ipynb",
        r"10-MLOps-Production/Transformer-Attention-Mechanisms.ipynb",
    ]

    ok, fail, overwritten = 0, 0, 0
    for rel in targets:
        path = ROOT / rel
        fn = builder_for(rel)
        if fn is not None:
            write_nb(path, fn())
            overwritten += 1
        print('EXEC:', rel)
        try:
            exec_nb(path)
            ok += 1
        except Exception as e:
            fail += 1
            print('FAIL:', rel, '->', e)

    print('\n=== SUMMARY ===')
    print('executed ok:', ok)
    print('executed fail:', fail)
    print('overwritten:', overwritten)


if __name__ == '__main__':
    main()
