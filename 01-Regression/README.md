# 01 - Régression

Ce dossier est consacré à la régression, une technique fondamentale de l'apprentissage supervisé qui vise à prédire une valeur numérique continue. Les notebooks présentés ici explorent différents types de modèles de régression, de la régression linéaire simple à des approches plus complexes comme la régression polynomiale et la régression logistique (utilisée pour la classification binaire, mais basée sur des principes de régression).

**Date de dernière mise à jour :** 2026-02-08 17:42:46

---

## Présentation des Notebooks

### 1. `Linear-Regression-Real-Estate.ipynb` - Régression Linéaire (Immobilier)

Ce notebook construit un modèle de référence robuste pour la prédiction des prix de l'immobilier. Il couvre l'ensemble du processus, de la génération de données synthétiques à l'évaluation du modèle, en passant par l'ingénierie des fonctionnalités et le diagnostic des résidus.

- **Concepts Clés :**
    - **Génération de Données Synthétiques :** Création d'un jeu de données réaliste avec des relations linéaires et non-linéaires.
    - **Analyse Exploratoire des Données (EDA) :** Visualisation des distributions et des relations entre les variables.
    - **Pipeline de Prétraitement :** Utilisation de `ColumnTransformer` et `OneHotEncoder` pour gérer les variables catégorielles.
    - **Modélisation :** Entraînement d'un modèle de régression `Ridge` pour éviter le surapprentissage.
    - **Évaluation du Modèle :** Calcul des métriques MAE, RMSE et R² pour évaluer les performances.
    - **Analyse des Résidus :** Examen de la distribution des erreurs pour diagnostiquer les problèmes potentiels du modèle.

### 2. `Logistic-Regression-Healthcare.ipynb` - Régression Logistique (Santé)

Ce notebook illustre l'utilisation de la régression logistique pour une tâche de classification binaire dans le domaine de la santé. Il aborde des concepts importants tels que la calibration du modèle, le choix du seuil de décision et l'analyse des erreurs.

- **Concepts Clés :**
    - **Génération de Données Synthétiques :** Création d'un jeu de données simulant les facteurs de risque pour une cohorte de patients.
    - **Pipeline de Modélisation :** Utilisation de `StandardScaler` pour normaliser les données avant l'entraînement.
    - **Calibration du Modèle :** Utilisation de l'argument `class_weight='balanced'` pour gérer le déséquilibre des classes.
    - **Évaluation du Modèle :** Analyse approfondie des performances à l'aide de la matrice de confusion, du rapport de classification, des courbes ROC et PR.
    - **Optimisation du Seuil :** Recherche du seuil de décision optimal en fonction des coûts associés aux faux positifs et faux négatifs.

### 3. `Polynomial-Regression-Energy.ipynb` - Régression Polynomiale (Énergie)

Ce notebook aborde un problème de régression non-linéaire en utilisant des caractéristiques polynomiales pour prédire la demande en énergie. Il met en évidence l'importance de la régularisation et de la validation croisée pour éviter le surapprentissage.

- **Concepts Clés :**
    - **Génération de Données Synthétiques :** Création d'un jeu de données réaliste avec des relations non-linéaires complexes.
    - **Analyse Exploratoire des Données (EDA) :** Visualisation de la relation non-linéaire entre la température et la demande en énergie.
    - **Ingénierie des Caractéristiques :** Utilisation de `PolynomialFeatures` pour créer des termes d'interaction et des puissances des variables d'entrée.
    - **Comparaison de Modèles :** Comparaison des performances d'un modèle de régression linéaire simple avec un modèle de régression polynomiale régularisé (`Ridge`).
    - **Optimisation d'Hyperparamètres :** Utilisation de `GridSearchCV` pour trouver le meilleur degré polynomial et le meilleur paramètre de régularisation `alpha`.
