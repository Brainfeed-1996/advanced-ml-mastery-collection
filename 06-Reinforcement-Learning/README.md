# 06 - Apprentissage par Renforcement

Ce dossier est consacré à l'apprentissage par renforcement (AR), une branche du machine learning où un agent apprend à prendre des décisions en interagissant avec un environnement pour maximiser une récompense cumulée.

**Date de dernière mise à jour :** 2026-02-16 16:05:59

## Prérequis

- Python 3.10+
- Un environnement virtuel (`venv`) est fortement recommandé. (Voir `docs/INSTALLATION.md`)
- Des dépendances lourdes comme `pytorch` et `gymnasium` sont nécessaires. (Voir `docs/OPTIONAL_HEAVY_DEPS.md`)

---

## Présentation des Notebooks (.ipynb)

### 1. `RL-CartPole-Agent.ipynb` - Entraînement d'un Agent pour CartPole avec Policy Gradient

Ce notebook est une excellente introduction pratique aux concepts fondamentaux de l'apprentissage par renforcement. Il montre comment entraîner un agent simple à résoudre l'environnement classique `CartPole-v1` de la bibliothèque `gymnasium`.

- **Objectif :** Entraîner un agent à équilibrer un poteau sur un chariot le plus longtemps possible.
- **Environnement (`CartPole-v1`) :**
    - **Espace d'observation :** Un vecteur de 4 valeurs flottantes représentant la position du chariot, sa vitesse, l'angle du poteau et sa vitesse angulaire.
    - **Espace d'action :** Deux actions discrètes : pousser le chariot vers la gauche (0) ou vers la droite (1).
    - **Récompense :** +1 pour chaque pas de temps où le poteau reste en équilibre.
- **Architecture (`Policy`) :** Un réseau de neurones simple (Policy Network) est utilisé pour approximer la politique de l'agent :
    - Il prend l'observation de l'environnement comme entrée.
    - Il possède une couche cachée avec une fonction d'activation `Tanh`.
    - La couche de sortie produit des scores (logits) pour chaque action possible.
- **Techniques Abordées :**
    - **Policy Gradient (REINFORCE) :** C'est l'une des méthodes les plus fondamentales de l'AR. L'agent exécute une politique, collecte des trajectoires (séquences d'états, d'actions et de récompenses), puis met à jour sa politique pour augmenter la probabilité des actions qui ont mené à des récompenses élevées.
    - **Calcul des Retours (Returns) :** Le notebook calcule les retours escomptés (somme des récompenses futures, potentiellement actualisées) pour chaque pas de temps. Ces retours sont ensuite normalisés pour stabiliser l'entraînement.
    - **Boucle d'entraînement :** L'agent est entraîné sur plusieurs épisodes. Pour chaque épisode, il interagit avec l'environnement, collecte des données, puis utilise la perte (calculée à partir des log-probabilités des actions et des retours) pour mettre à jour les poids de son réseau de politique via l'optimiseur `Adam`.
- **Interprétation des Résultats :** La sortie du notebook affiche la récompense totale obtenue à la fin de plusieurs épisodes. On peut observer une augmentation progressive de la récompense, indiquant que l'agent apprend avec succès à équilibrer le poteau de plus en plus longtemps.

### 2. `Deep-Q-Network-Atari.ipynb` (Corrompu)

*(Note : Ce notebook n'a pas pu être analysé en raison d'erreurs de syntaxe persistantes.)*

- **Objectif Présumé :** Implémenter un algorithme de Deep Q-Network (DQN) pour jouer à un jeu Atari. Les DQN sont une technique puissante qui a permis d'atteindre des performances surhumaines sur de nombreux jeux Atari classiques en apprenant directement à partir des pixels de l'écran.

### 3. `Feature-Engineering-Pipeline.ipynb` (Corrompu)

*(Note : Ce notebook n'a pas pu être analysé en raison d'erreurs de syntaxe persistantes.)*

- **Objectif Présumé :** Démontrer un pipeline complet de feature engineering dans un contexte de machine learning, potentiellement pour préparer des données avant de les utiliser dans un modèle d'apprentissage.

### 4. `Hyperparameter-Optimization-Optuna.ipynb` (Corrompu)

*(Note : Ce notebook n'a pas pu être analysé en raison d'erreurs de syntaxe persistantes.)*

- **Objectif Présumé :** Utiliser la bibliothèque `Optuna` pour effectuer une recherche et une optimisation automatiques des hyperparamètres d'un modèle de machine learning. Optuna est un framework moderne qui facilite la recherche d'hyperparamètres performants.

### 5. `Prophet-Market-Trends.ipynb` (Corrompu)

*(Note : Ce notebook n'a pas pu être analysé en raison d'erreurs de syntaxe persistantes.)*

- **Objectif Présumé :** Utiliser `Prophet`, une bibliothèque de Facebook pour la prévision de séries temporelles, afin d'analyser et de prédire des tendances sur des données de marché.

### 6. `Q-Learning-Maze-Solver.ipynb` (Corrompu)

*(Note : Ce notebook n'a pas pu être analysé en raison d'erreurs de syntaxe persistantes.)*

- **Objectif Présumé :** Implémenter l'algorithme de Q-Learning, un algorithme fondamental de l'AR basé sur la valeur, pour entraîner un agent à trouver la sortie d'un labyrinthe. C'est un exemple classique pour illustrer les bases du Q-Learning.

---

## Comment Exécuter les Projets

### Exécution Interactive (Recommandé)

```bash
jupyter notebook 06-Reinforcement-Learning/<nom_du_notebook>.ipynb
```

### Exécution Headless

```bash
python -m jupyter nbconvert --to notebook --execute \
  06-Reinforcement-Learning/<nom_du_notebook>.ipynb --output <nom_du_notebook>.ipynb --output-dir 06-Reinforcement-Learning
```
