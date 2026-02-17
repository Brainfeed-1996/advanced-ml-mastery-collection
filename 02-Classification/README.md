# 02 - Classification

Ce dossier se concentre sur les tâches de classification, une catégorie fondamentale du machine learning supervisé où l'objectif est de prédire une étiquette de classe discrète. Les exemples fournis illustrent des algorithmes de classification populaires et leurs applications, notamment les Forêts Aléatoires, les Machines à Vecteurs de Support (SVM) et XGBoost.

**Date de dernière mise à jour :** 2026-02-08 17:42:46

---

## Présentation des Notebooks

### 1. `Random-Forest-Finance.ipynb` - Classification avec Forêt Aléatoire (Finance)

Ce notebook utilise un modèle de Forêt Aléatoire pour une tâche de classification dans le domaine financier. Il couvre l'ensemble du processus, de la préparation des données à l'évaluation du modèle, en passant par l'interprétation de l'importance des fonctionnalités.

- **Concepts Clés :**
    - **Génération de Données Synthétiques :** Création d'un jeu de données réaliste pour simuler le risque de crédit.
    - **Analyse Exploratoire des Données (EDA) :** Visualisation des distributions et des relations entre les variables pour identifier les prédicteurs potentiels.
    - **Pipeline de Prétraitement :** Utilisation de `ColumnTransformer` et `OneHotEncoder` pour gérer les variables catégorielles et numériques.
    - **Optimisation d'Hyperparamètres :** Utilisation de `RandomizedSearchCV` pour trouver les meilleurs hyperparamètres pour le modèle de Forêt Aléatoire.
    - **Importance des Variables :** Analyse de l'importance des fonctionnalités pour comprendre les facteurs les plus influents dans la prédiction.

### 2. `SVM-Handwritten-Digits.ipynb` - Classification avec SVM (Chiffres Manuscrits)

Ce notebook applique les Machines à Vecteurs de Support (SVM) pour la reconnaissance de chiffres manuscrits à partir du jeu de données `digits` de scikit-learn. Il montre comment les SVM peuvent être utilisés pour des tâches de classification multi-classes et comment optimiser leurs hyperparamètres.

- **Concepts Clés :**
    - **Chargement de Données :** Utilisation du jeu de données `digits` intégré à scikit-learn.
    - **Analyse Exploratoire des Données (EDA) :** Visualisation des images de chiffres pour comprendre la nature des données.
    - **Comparaison de Modèles :** Comparaison des performances d'un modèle SVM avec noyau RBF à des modèles de base comme la régression logistique.
    - **Optimisation d'Hyperparamètres :** Utilisation de `GridSearchCV` pour trouver les meilleurs paramètres `C` et `gamma` pour le SVM.
    - **Évaluation du Modèle :** Utilisation de rapports de classification et de matrices de confusion pour évaluer la performance du modèle final.

### 3. `XGBoost-Customer-Churn.ipynb` - Classification avec XGBoost (Churn Client)

Ce notebook utilise XGBoost, un algorithme de boosting de gradient, pour prédire le churn (départ) de clients. Il met en évidence la puissance de XGBoost pour obtenir des performances de pointe sur des données tabulaires et compare ses performances à des modèles de base.

- **Concepts Clés :**
    - **Génération de Données Synthétiques :** Création d'un jeu de données réaliste simulant le comportement des clients et leur probabilité de churn.
    - **Pipeline de Prétraitement :** Gestion des variables catégorielles et numériques avec `ColumnTransformer` et `OneHotEncoder`.
    - **Comparaison de Modèles :** Évaluation des performances de XGBoost par rapport à un modèle de base (régression logistique) et un classificateur `HistGradientBoosting`.
    - **Optimisation d'Hyperparamètres :** Utilisation de `RandomizedSearchCV` pour trouver les meilleurs hyperparamètres pour le modèle `HistGradientBoosting`.
    - **Évaluation du Modèle :** Utilisation de l'AUC ROC comme métrique principale pour évaluer et comparer les modèles.
