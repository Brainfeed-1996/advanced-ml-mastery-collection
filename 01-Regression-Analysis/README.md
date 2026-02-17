# 01 - Analyse de Régression

Ce dossier est une extension du dossier `01-Regression`, se concentrant sur des analyses plus approfondies et des diagnostics de modèles pour les problèmes de régression. Les notebooks ici reprennent les mêmes études de cas (immobilier, santé, finance) mais avec une approche plus orientée vers l'évaluation et l'interprétation des modèles.

**Date de dernière mise à jour :** 2026-02-08 17:42:46

---

## Présentation des Notebooks

### 1. `Linear-Regression-Real-Estate.ipynb` - Analyse de Régression Linéaire (Immobilier)

Ce notebook approfondit l'analyse du modèle de régression linéaire pour la prédiction des prix immobiliers. Il se concentre sur les diagnostics avancés pour valider la robustesse du modèle.

- **Concepts Clés :**
    - **Diagnostics des Résidus :** Analyse approfondie de la distribution des erreurs, de l'hétéroscédasticité et des points influents.
    - **Interprétation des Coefficients :** Discussion sur la signification des coefficients du modèle et leur impact sur la prédiction.
    - **Validation Croisée :** Utilisation de la validation croisée pour une évaluation plus fiable de la performance du modèle.

### 2. `Logistic-Regression-Healthcare.ipynb` - Analyse de Régression Logistique (Santé)

Ce notebook va au-delà de la simple classification binaire en explorant des techniques d'évaluation plus fines pour le modèle de risque médical.

- **Concepts Clés :**
    - **Calibration du Modèle :** Évaluation de la fiabilité des probabilités prédites par le modèle.
    - **Analyse de Sensibilité au Seuil :** Étude de l'impact du choix du seuil de classification sur les métriques de performance.
    - **Coût des Erreurs :** Intégration des coûts métiers dans l'évaluation du modèle pour une prise de décision plus éclairée.

### 3. `Random-Forest-Finance.ipynb` - Analyse de Forêt Aléatoire (Finance)

Ce notebook introduit un modèle non linéaire, la forêt aléatoire, pour un problème de régression en finance. L'accent est mis sur l'interprétabilité du modèle et la comparaison avec des modèles plus simples.

- **Concepts Clés :**
    - **Importance des Variables :** Utilisation de la forêt aléatoire pour estimer l'importance de chaque variable dans la prédiction.
    - **Comparaison de Modèles :** Comparaison des performances de la forêt aléatoire avec un modèle de régression linéaire pour évaluer les gains de la non-linéarité.
    - **Visualisation des Prédictions :** Création de graphiques pour visualiser les prédictions du modèle et comprendre son comportement.
