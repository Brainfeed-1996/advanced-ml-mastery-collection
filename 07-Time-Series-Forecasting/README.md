# 07 - Prévision de Séries Temporelles

Ce dossier se concentre sur des sujets avancés liés à la mise en production de modèles de prévision de séries temporelles. Il aborde des aspects cruciaux tels que le suivi des modèles en production, l'interprétabilité des modèles et les techniques de quantification pour optimiser les performances.

**Date de dernière mise à jour :** 2026-02-16 16:05:59

---

## Présentation des Notebooks

### 1. `Decision-Trees-Interpretability.ipynb` - Interprétabilité des Arbres de Décision

Ce notebook est un projet complet qui se concentre sur l'interprétabilité des modèles basés sur les arbres de décision. Il utilise un flux de travail robuste et prêt pour la production, incluant la génération de données synthétiques, des pipelines de prétraitement avancés et l'explicabilité des modèles à l'aide de SHAP.

- **Concepts Clés :**
    - **Génération de Données Industrielles Synthétiques :** Création d'un jeu de données réaliste avec des caractéristiques industrielles.
    - **Pipelines de Prétraitement Avancés :** Mise en œuvre de pipelines pour le nettoyage et la préparation des données.
    - **Wrappers de Modèles Personnalisés :** Utilisation de wrappers pour ajouter des fonctionnalités de logging et de suivi.
    - **Évaluation Approfondie des Performances :** Analyse détaillée des performances du modèle.
    - **Explicabilité des Modèles (SHAP) :** Utilisation de la bibliothèque SHAP pour interpréter les prédictions du modèle.

### 2. `ML-Model-Monitoring-Prometheus.ipynb` - Suivi de Modèles avec des Métriques de type Prometheus

Ce notebook se concentre sur les stratégies de suivi des modèles en production en utilisant des métriques similaires à celles de Prometheus. Il montre comment mettre en place un système de surveillance pour suivre les performances des modèles et détecter les dérives.

- **Concepts Clés :**
    - **Monitoring de Modèles :** Importance du suivi des modèles en production.
    - **Métriques de type Prometheus :** Simulation de la collecte de métriques de performance.
    - **Détection de Dérive :** Techniques pour identifier les changements dans les performances du modèle ou dans la distribution des données.

### 3. `Model-Quantization-TensorRT.ipynb` - Quantification de Modèles (focus sur TensorRT)

Ce notebook explore les techniques de quantification post-entraînement, avec une discussion sur TensorRT. La quantification est une méthode d'optimisation qui réduit la taille des modèles et accélère l'inférence, ce qui est particulièrement utile pour le déploiement sur des appareils à ressources limitées.

- **Concepts Clés :**
    - **Quantification Post-Entraînement :** Réduction de la précision des poids du modèle après l'entraînement.
    - **TensorRT :** Discussion sur l'utilisation de TensorRT pour l'optimisation de l'inférence sur les GPU NVIDIA (avec une solution de repli sur CPU dans le notebook).
    - **Optimisation des Performances :** Comparaison des performances avant et après la quantification.
