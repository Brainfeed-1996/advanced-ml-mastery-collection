# 02 - Classification

Ce dossier se concentre sur les défis de la classification, une tâche d'apprentissage supervisé où l'objectif est de prédire une étiquette de classe discrète. Les notebooks de ce dossier explorent des techniques de classification avancées, telles que le gradient boosting et les machines à vecteurs de support (SVM), ainsi qu'une introduction au clustering avec l'algorithme K-Means pour la segmentation de la clientèle.

**Date de dernière mise à jour :** 2026-02-08 17:42:46

---

## Présentation des Notebooks

### 1. `XGBoost-Customer-Churn.ipynb` - Prédiction de l'Attrition Client avec Gradient Boosting

Ce notebook aborde un problème courant en entreprise : la prédiction de l'attrition client (churn). Il montre comment utiliser le gradient boosting pour construire un modèle de classification robuste, en tenant compte du déséquilibre des classes.

- **Concepts Clés :**
    - **Génération de Données Synthétiques :** Création d'un jeu de données tabulaires imitant les caractéristiques des clients et leur probabilité d'attrition.
    - **Pipeline de Prétraitement Robuste :** Utilisation de `StandardScaler` pour les variables numériques et `OneHotEncoder` pour les variables catégorielles.
    - **Modélisation avec Gradient Boosting :** Utilisation de `XGBoost` (si installé) ou de `HistGradientBoostingClassifier` de scikit-learn comme alternative.
    - **Évaluation du Modèle pour Classes Déséquilibrées :** Utilisation de métriques appropriées telles que l'AUC-ROC et l'AUC-PR (Precision-Recall) pour évaluer le modèle en présence de classes déséquilibrées.

### 2. `SVM-Image-Recognition.ipynb` - Reconnaissance d'Images avec SVM

Ce notebook utilise une machine à vecteurs de support (SVM) avec un noyau RBF pour une tâche de reconnaissance d'images sur le jeu de données `digits` de scikit-learn. Il illustre un workflow complet de classification d'images, y compris le prétraitement, l'optimisation des hyperparamètres et l'évaluation détaillée.

- **Concepts Clés :**
    - **Jeu de Données `digits` :** Utilisation d'un jeu de données classique de scikit-learn pour la reconnaissance de chiffres manuscrits.
    - **Pipeline de Prétraitement et Modélisation :** Combinaison de `StandardScaler` et `SVC` dans un pipeline pour simplifier le workflow.
    - **Optimisation d'Hyperparamètres :** Utilisation de `GridSearchCV` pour trouver les meilleurs paramètres `C` et `gamma` pour le noyau RBF.
    - **Évaluation Approfondie :** Analyse des performances à l'aide du rapport de classification et de la matrice de confusion pour comprendre les erreurs du modèle.

### 3. `K-Means-Customer-Segmentation.ipynb` - Segmentation de la Clientèle avec K-Means

Bien que le K-Means soit un algorithme de clustering (apprentissage non supervisé), ce notebook est inclus ici car il est souvent utilisé comme une étape préliminaire à la classification. Il montre comment segmenter une clientèle en groupes homogènes en fonction de leur comportement.

- **Concepts Clés :**
    - **Génération de Données Synthétiques :** Création d'un jeu de données client avec des caractéristiques telles que l'âge, le revenu et le score de dépense.
    - **Mise à l'Échelle des Données :** Utilisation de `StandardScaler` pour normaliser les données, une étape cruciale pour le K-Means.
    - **Sélection du Nombre de Clusters (K) :** Utilisation du score de silhouette pour déterminer le nombre optimal de clusters.
    - **Analyse des Clusters :** Profilage des segments de clientèle en analysant les caractéristiques moyennes de chaque groupe.
    - **Visualisation avec PCA :** Utilisation de l'analyse en composantes principales (PCA) pour visualiser les clusters en 2D.
