# 07 - Séries Temporelles

Ce dossier est dédié à l'analyse et à la prévision de séries temporelles, une compétence cruciale en data science pour comprendre et prédire des tendances, des saisonnalités et des cycles dans des données dépendant du temps.

**Date de dernière mise à jour :** 2026-02-15 22:39:18

## Prérequis

- Python 3.10+
- Un environnement virtuel (`venv`) est fortement recommandé. (Voir `docs/INSTALLATION.md`)
- Des dépendances lourdes comme `statsmodels` et `pytorch` sont nécessaires. (Voir `docs/OPTIONAL_HEAVY_DEPS.md`)

---

## Présentation des Notebooks (.ipynb)

### 1. `ARIMA-Sales-Forecasting.ipynb` - Prévision de Ventes avec ARIMA

Ce notebook offre un guide complet sur la modélisation ARIMA (AutoRegressive Integrated Moving Average), une technique statistique puissante pour l'analyse de séries temporelles.

- **Objectif :** Prédire des données de ventes ou de passagers sur la base de données historiques. Le notebook utilise le jeu de données classique `AirPassengers` ou, à défaut, génère une série temporelle réaliste avec une tendance et une saisonnalité claires.
- **Techniques Abordées :**
    - **Analyse Exploratoire (EDA) :** Visualisation de la série temporelle pour identifier la tendance (croissance à long terme), la saisonnalité (cycles réguliers) et le bruit résiduel.
    - **Décomposition :** Utilisation de `statsmodels` pour décomposer la série en ses composantes de tendance, de saisonnalité et de résidus, permettant une meilleure compréhension de la structure des données.
    - **Tests de Stationnarité :** L'un des prérequis d'ARIMA est que la série soit stationnaire (moyenne et variance constantes dans le temps). Le notebook montre comment vérifier la stationnarité et comment la différencier (calculer la différence entre des observations consécutives) pour la rendre stationnaire.
    - **Sélection de l'Ordre ARIMA (p, d, q) :** ARIMA est défini par trois paramètres :
        - **p (ordre autorégressif) :** Nombre d'observations passées à inclure dans le modèle.
        - **d (ordre de différenciation) :** Nombre de fois où les données brutes sont différenciées.
        - **q (ordre de la moyenne mobile) :** Taille de la fenêtre de moyenne mobile.
        Le notebook effectue une recherche par grille (Grid Search) pour trouver la meilleure combinaison (p, d, q) en se basant sur le critère d'information d'Akaike (AIC), qui mesure le compromis entre la qualité de l'ajustement du modèle et sa complexité.
    - **Modèles de Référence (Baselines) :** Avant de construire le modèle ARIMA, des modèles de référence simples sont établis pour évaluer les performances :
        - **Naïf :** Prédit que la prochaine valeur sera identique à la dernière valeur observée.
        - **Naïf Saisonnier :** Prédit que la prochaine valeur sera identique à la valeur observée lors de la saison précédente (par exemple, il y a 12 mois pour des données mensuelles).
    - **Backtesting (Validation Croisée sur Séries Temporelles) :** Pour évaluer de manière réaliste la performance du modèle, une technique de backtesting avec origine mobile (rolling-origin) est utilisée. Le modèle est entraîné sur une partie initiale des données, fait une prévision, puis est ré-entraîné en incluant une observation de plus, et ainsi de suite. Cela simule la manière dont le modèle serait utilisé en production.
- **Interprétation des Résultats :** Le notebook compare les erreurs (MAE et RMSE) du modèle ARIMA à celles des modèles de référence, montrant l'amélioration significative des performances. Une visualisation finale montre les prévisions du modèle ARIMA par rapport aux données de test réelles, illustrant sa capacité à capturer la tendance et la saisonnalité.

### 2. `LSTM-Stock-Prediction.ipynb` (Corrompu)

*(Note : Ce notebook n'a pas pu être analysé en raison d'erreurs de syntaxe persistantes.)*

- **Objectif Présumé :** Utiliser un réseau de neurones récurrent de type LSTM (Long Short-Term Memory) pour prédire des cours de la bourse. Les LSTM sont particulièrement adaptés à l'apprentissage de dépendances à long terme dans des séquences, ce qui les rend populaires pour la modélisation de séries temporelles financières.

### 3. `Prophet-Market-Trends.ipynb` (Corrompu)

*(Note : Ce notebook n'a pas pu être analysé en raison d'erreurs de syntaxe persistantes.)*

- **Objectif Présumé :** Mettre en œuvre `Prophet`, une bibliothèque développée par Facebook, pour analyser et prédire des tendances sur des données de marché. Prophet est conçu pour être facile à utiliser et robuste face aux données manquantes, aux changements de tendance et aux fortes saisonnalités.

### 4. `WalkForward-Backtesting-Baselines.ipynb` (Corrompu)

*(Note : Ce notebook n'a pas pu être analysé en raison d'erreurs de syntaxe persistantes.)*

- **Objectif Présumé :** Se concentrer spécifiquement sur la méthode de validation croisée `Walk-Forward` (ou backtesting avec origine mobile) pour évaluer rigoureusement la performance de modèles de référence (baselines) sur des données de séries temporelles.

---

## Comment Exécuter les Projets

### Exécution Interactive (Recommandé)

```bash
jupyter notebook 07-Time-Series/<nom_du_notebook>.ipynb
```

### Exécution Headless

```bash
python -m jupyter nbconvert --to notebook --execute \
  07-Time-Series/<nom_du_notebook>.ipynb --output <nom_du_notebook>.ipynb --output-dir 07-Time-Series
```
