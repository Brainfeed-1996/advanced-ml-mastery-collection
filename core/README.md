# Core - Notebooks Fondamentaux

Ce dossier `core` rassemble des notebooks essentiels qui servent de fondations et de baselines pour les tâches de machine learning les plus courantes. Chaque notebook est conçu pour être une référence "industrielle", c'est-à-dire robuste, bien documenté et illustrant les meilleures pratiques, de l'analyse exploratoire à l'interprétabilité du modèle.

**Date de dernière mise à jour :** 2026-02-15 22:45:00

---

## Présentation des Notebooks

### 1. `01-Linear-Regression-Housing.ipynb` - Régression Linéaire (Immobilier)

Ce notebook établit une baseline solide pour les problèmes de régression. Il utilise un jeu de données synthétiques mais réalistes sur les prix de l'immobilier pour illustrer un workflow complet.

- **Concepts Clés :**
    - **Génération de Données Synthétiques :** Création d'un jeu de données avec des caractéristiques et des relations plausibles, simulant les prix de appartements dans plusieurs villes en fonction de leur surface, âge, etc.
    - **Analyse Exploratoire des Données (EDA) :** Visualisation des distributions et des corrélations pour comprendre les données avant la modélisation.
    - **Pipeline de Prétraitement :** Utilisation de `ColumnTransformer` pour appliquer un encodage `OneHotEncoder` aux caractéristiques catégorielles (comme la ville).
    - **Modélisation :** Entraînement d'un modèle de régression linéaire simple mais robuste (`Ridge`) pour prédire les prix.
    - **Évaluation et Diagnostics :** Calcul des métriques standards (MAE, RMSE, R²) et analyse des résidus pour valider la performance et les hypothèses du modèle.
    - **Interprétabilité :** Extraction et analyse des coefficients du modèle pour comprendre l'influence de chaque caractéristique sur la prédiction du prix.

### 2. `02-Logistic-Regression-Medical.ipynb` - Régression Logistique (Risque Médical)

Ce notebook est une référence pour les tâches de classification binaire, en particulier dans des contextes critiques comme le domaine médical où l'évaluation du modèle va bien au-delà de la simple exactitude.

- **Concepts Clés :**
    - **Synthèse de Cohorte :** Génération de données de patients synthétiques (âge, IMC, tension, etc.) avec une probabilité de "risque" définie par une fonction logistique, simulant une cohorte pour une étude médicale.
    - **Modélisation et Calibration :** Entraînement d'un modèle de régression logistique avec gestion du déséquilibre des classes (`class_weight='balanced'`).
    - **Analyse des Probabilités :** Le notebook ne se contente pas de prédire une classe (0 ou 1), mais analyse les probabilités prédites, qui sont souvent plus importantes dans un contexte de risque.
    - **Courbes ROC et PR :** Tracer les courbes ROC (Receiver Operating Characteristic) et PR (Precision-Recall) pour évaluer la performance du classifieur sur l'ensemble des seuils de décision possibles. L'aire sous la courbe (AUC) pour chacune est calculée.
    - **Seuillage Opérationnel (Thresholding) :** Le notebook montre comment choisir un seuil de décision optimal en fonction des coûts métiers (ici, le coût d'un faux négatif est plus élevé qu'un faux positif).
    - **Rapport de Classification et Matrice de Confusion :** Analyse détaillée des erreurs du modèle (faux positifs, faux négatifs) pour le seuil choisi.

### 3. `03-Decision-Tree-Iris.ipynb` - Arbre de Décision (Iris)

Utilisant le jeu de données classique "Iris", ce notebook se concentre sur l'interprétabilité, une force majeure des arbres de décision.

- **Concepts Clés :**
    - **Baseline :** Entraînement d'un premier arbre de décision sans contraintes pour évaluer sa performance initiale.
    - **Élagage par Complexité de Coût (Cost-Complexity Pruning) :** C'est une technique clé pour lutter contre le surapprentissage. Le notebook explore le "chemin d'élagage" en faisant varier le paramètre `ccp_alpha` pour trouver le meilleur compromis entre la complexité de l'arbre et sa capacité de généralisation.
    - **Recherche par Grille (GridSearch) :** Utilisation de `GridSearchCV` pour trouver la meilleure combinaison d'hyperparamètres, incluant `max_depth`, `min_samples_leaf`, et le `ccp_alpha` optimal identifié précédemment.
    - **Visualisation de l'Arbre :** Le notebook montre comment visualiser l'arbre de décision final avec `plot_tree`, ce qui le rend entièrement transparent et facile à expliquer.
    - **Extraction de Règles :** Utilisation de `export_text` pour extraire les règles de décision de l'arbre sous un format lisible par un humain (par exemple, "SI petal length <= 2.45 ALORS classe = setosa").
    - **Importance des Caractéristiques :** Analyse de l'attribut `feature_importances_` pour quantifier quelles caractéristiques (longueur des pétales, etc.) ont le plus contribué aux décisions de l'arbre.
