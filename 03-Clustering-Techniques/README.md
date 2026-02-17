# 03 - Techniques de Clustering et Apprentissage Profond

Ce dossier explore des techniques de clustering avancées et des applications d'apprentissage profond (Deep Learning) pour des tâches de classification et de prédiction non supervisées et séquentielles.

**Date de dernière mise à jour :** 2026-02-16 16:05:59

## Prérequis

- Python 3.10+
- Un environnement virtuel (`venv`) est fortement recommandé. (Voir `docs/INSTALLATION.md`)
- Des dépendances lourdes comme `pytorch` et `transformers` sont nécessaires pour certains notebooks. (Voir `docs/OPTIONAL_HEAVY_DEPS.md`)

---

## Présentation des Notebooks (.ipynb)

### 1. `BERT-Sentiment-Analysis.ipynb` - Analyse de Sentiments avec BERT

Ce notebook montre comment utiliser un modèle Transformer pré-entraîné (BERT) pour une tâche de classification de texte : l'analyse de sentiments. Il compare l'approche BERT à une méthode de base (TF-IDF + Régression Logistique).

- **Objectif :** Classifier des phrases comme ayant un sentiment positif ou négatif.
- **Données :** Un jeu de données simple et synthétique est créé pour illustrer la tâche, avec des phrases clairement positives et négatives.
- **Techniques Abordées :**
    - **Modèle de base :** Création d'un pipeline `scikit-learn` avec `TfidfVectorizer` pour convertir le texte en vecteurs et `LogisticRegression` pour la classification. C'est une approche "offline" qui ne nécessite pas de services externes.
    - **Inférence avec Transformers :** Utilisation de la bibliothèque `transformers` de Hugging Face pour charger un modèle pré-entraîné spécialisé dans l'analyse de sentiments (`sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english`). Ce modèle, bien que "tiny" (petit), est capable de comprendre le contexte sémantique des phrases.
    - **Pipeline `sentiment-analysis` :** Démonstration de la simplicité d'utilisation des pipelines de `transformers` pour effectuer des prédictions sur de nouvelles phrases.

- **Problème Actuel :** Le notebook rencontre une `ValueError` lors du chargement du modèle BERT, indiquant que le dictionnaire d'état du modèle est corrompu. Cela peut être dû à un problème de téléchargement ou de cache. La résolution typique est de vider le cache de Hugging Face (`~/.cache/huggingface`) et de réessayer.

### 2. `LSTM-Stock-Prediction.ipynb` - Prédiction de Séries Temporelles avec LSTM

Ce projet illustre l'utilisation d'un réseau de neurones récurrent (RNN) de type LSTM (Long Short-Term Memory) pour prédire l'évolution d'une série temporelle, simulant la prédiction de cours de la bourse.

- **Objectif :** Prédire les valeurs futures d'une série temporelle en se basant sur les observations passées.
- **Données :** Une série temporelle synthétique est générée avec une tendance, des saisonnalités (périodicités) et du bruit, pour simuler un comportement de marché réaliste.
- **Techniques Abordées :**
    - **Préparation des données pour les RNNs :** Transformation de la série temporelle en "fenêtres" de données. Chaque fenêtre contient une séquence de `lookback` observations (les features, `X`) et la valeur suivante à prédire (la cible, `y`).
    - **Création d'un modèle LSTM avec PyTorch :** Définition d'une classe `LSTMForecaster` qui hérite de `torch.nn.Module`, avec une couche LSTM et une couche linéaire (entièrement connectée) pour la sortie.
    - **Entraînement du modèle :** Utilisation d'un `DataLoader` de PyTorch pour gérer les batchs de données, et entraînement du modèle avec l'optimiseur Adam et la fonction de perte MSE (Mean Squared Error).
    - **Évaluation et Visualisation :** Calcul de l'erreur quadratique moyenne sur l'ensemble de test et visualisation de la prédiction du modèle par rapport aux valeurs réelles.

### 3. `PCA-Dimensionality-Reduction.ipynb` - Réduction de Dimension avec l'ACP

Ce notebook explore l'Analyse en Composantes Principales (ACP ou PCA en anglais), une technique de réduction de dimensionnalité non supervisée. L'ACP est utilisée ici pour visualiser des données multi-dimensionnelles et comme étape de prétraitement avant la classification.

- **Objectif :** Réduire le nombre de variables d'un jeu de données tout en préservant le maximum d'information, et évaluer l'impact de cette réduction sur la performance d'un modèle de classification.
- **Données :** Le jeu de données `wine` de `scikit-learn`, qui contient les résultats d'une analyse chimique de vins issus de trois cépages différents. Le jeu de données a 13 caractéristiques, ce qui le rend difficile à visualiser directement.
- **Techniques Abordées :**
    - **Analyse de la variance expliquée :** Utilisation de l'ACP pour déterminer combien de composantes principales sont nécessaires pour capturer un certain pourcentage (par ex. 95%) de la variance totale des données.
    - **Visualisation 2D :** Projection des données sur les deux premières composantes principales (PC1 et PC2) pour visualiser la séparation des trois classes de vin dans un espace à 2 dimensions.
    - **ACP dans un pipeline de classification :** Comparaison de la performance (accuracy, F1-score) d'un modèle de `LogisticRegression` entraîné sur les données brutes vs. sur les données après réduction de dimension par ACP.
    - **Optimisation jointe :** Utilisation de `GridSearchCV` pour trouver simultanément les meilleurs hyperparamètres de l'ACP (`n_components`) et d'un classifieur SVM (`C` et `gamma`), montrant comment optimiser l'ensemble du pipeline de traitement.

---

## Comment Exécuter les Projets

### Exécution Interactive (Recommandé)

```bash
jupyter notebook 03-Clustering-Techniques/<nom_du_notebook>.ipynb
```

### Exécution Headless

```bash
python -m jupyter nbconvert --to notebook --execute \
  03-Clustering-Techniques/<nom_du_notebook>.ipynb --output <nom_du_notebook>.ipynb --output-dir 03-Clustering-Techniques
```
