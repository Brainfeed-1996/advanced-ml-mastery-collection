from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "08-Anomaly-Detection" / "KNN-Recommender-Systems.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    # Offline-first synthetic user-item interactions
    generator = """import numpy as np
import pandas as pd

rng = np.random.default_rng(SEED)

n_users = 200
n_items = 120
latent_k = 8

# Latent factors
U = rng.normal(size=(n_users, latent_k))
V = rng.normal(size=(n_items, latent_k))

# Implicit preference score + noise
scores = U @ V.T + rng.normal(scale=0.5, size=(n_users, n_items))

# Convert to sparse interactions by keeping top-N items per user
rows = []
top_n = 20
for u in range(n_users):
    idx = np.argsort(scores[u])[::-1][:top_n]
    for i in idx:
        # rating in [1, 5]
        rating = 1 + 4 * (scores[u, i] - scores[u].min()) / (scores[u].ptp() + 1e-9)
        rows.append((u, i, float(rating)))

interactions = pd.DataFrame(rows, columns=['user_id','item_id','rating'])

print(interactions.head())
print({'rows': len(interactions), 'users': interactions.user_id.nunique(), 'items': interactions.item_id.nunique()})
"""

    # Build a simple user-item matrix
    pipeline = """import numpy as np
import pandas as pd

R = interactions.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0.0)
X = R.values

print('matrix_shape:', X.shape)
"""

    # KNN recommender (item-based)
    model = """from sklearn.neighbors import NearestNeighbors

# Item vectors = columns of R
item_matrix = R.T.values

nn = NearestNeighbors(n_neighbors=6, metric='cosine')
nn.fit(item_matrix)

def recommend_for_user(user_id: int, k: int = 5):
    user_vec = R.loc[user_id].values
    liked_items = user_vec.argsort()[::-1][:3]

    candidates = set()
    for item in liked_items:
        dists, neigh = nn.kneighbors(item_matrix[item].reshape(1, -1), n_neighbors=6)
        for j in neigh[0]:
            candidates.add(int(j))

    # Score candidates by similarity-weighted sum
    recs = []
    for item in candidates:
        if user_vec[item] > 0:
            continue
        dists, neigh = nn.kneighbors(item_matrix[item].reshape(1, -1), n_neighbors=6)
        sims = 1 - dists[0]
        # weighted by user ratings on neighbor items
        score = float((sims * user_vec[neigh[0]]).sum())
        recs.append((item, score))

    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:k]

uid = 0
print('top_rated_items:', R.loc[uid].values.argsort()[::-1][:5].tolist())
print('recommendations:', recommend_for_user(uid, k=5))
"""

    visual = """# Simple visualization: rating density per user
import matplotlib.pyplot as plt
import seaborn as sns

user_n = (R > 0).sum(axis=1)
plt.figure(figsize=(8,4))
sns.histplot(user_n, bins=20, kde=True)
plt.title('Interactions per user')
plt.xlabel('# items rated')
plt.ylabel('count')
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
    c = replace_if_contains('AdvancedRegressor', model)
    d = replace_if_contains('ResultVisualizer', visual)

    nbformat.write(nb, nb_path)
    print({'patched_generator': a, 'patched_pipeline': b, 'patched_model': c, 'patched_visual': d})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
