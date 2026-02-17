# 10 - MLOps

Ce dossier est axé sur le MLOps (Machine Learning Operations), qui vise à déployer, surveiller et maintenir des modèles de machine learning en production de manière fiable et efficace. Il aborde des concepts cruciaux pour l'industrialisation des modèles, tels que l'optimisation pour les appareils de périphérie (edge), la quantification des modèles pour l'inférence à haute performance, et la surveillance continue.

**Date de dernière mise à jour :** 2026-02-16 07:18:50

## Prérequis

- Python 3.10+
- Un environnement virtuel (`venv`) est fortement recommandé. (Voir `docs/INSTALLATION.md`)
- Les dépendances de base incluent `scikit-learn` et `pandas`. Des dépendances plus lourdes comme `tensorflow`, `pytorch`, et `ultralytics` sont nécessaires pour les exemples spécifiques. (Voir `docs/OPTIONAL_HEAVY_DEPS.md`)

---

## Présentation des Notebooks (.ipynb)

Ce dossier réutilise des exemples de modèles des dossiers précédents mais les contextualise sous l'angle du MLOps. L'accent est mis sur les techniques permettant de rendre ces modèles plus efficaces, rapides et surveillables en production.

### 1. `Edge-Computing-TinyML.ipynb` - Edge Computing et TinyML

Ce notebook, bien que structurellement similaire à d'autres, est conçu pour illustrer les principes du déploiement de modèles sur des appareils de périphérie (Edge Computing) et le TinyML.

- **Objectif Conceptuel :** Démontrer comment un modèle de régression peut être simplifié et optimisé pour fonctionner sur des appareils à faibles ressources (comme des microcontrôleurs, des capteurs industriels, ou des smartphones).
- **Techniques MLOps Associées :**
    - **TinyML :** C'est un domaine du machine learning qui se concentre sur l'exécution de modèles sur du matériel de très faible puissance. Cela implique des techniques agressives de réduction de la taille du modèle, de la consommation d'énergie et de la latence.
    - **Pruning (Élagage) :** Retirer les poids ou les neurones non essentiels du modèle pour réduire sa taille sans impacter significativement sa performance.
    - **Quantification :** Utiliser des types de données moins précis pour les poids du modèle (par exemple, passer de `float32` à `int8`). Cela réduit la taille du modèle et accélère l'inférence, en particulier sur le matériel qui supporte l'arithmétique sur les entiers.
    - **Compilation de Modèles :** Utiliser des compilateurs spécifiques (comme TensorFlow Lite, ONNX Runtime) pour convertir le modèle entraîné en un format optimisé pour une plateforme matérielle cible.
- **Workflow Illustré (implicitement) :**
    1. **Entraînement :** Entraîner un modèle robuste (ici, un `GradientBoostingRegressor`) sur des données complètes.
    2. **Optimisation :** Appliquer des techniques de quantification et de pruning pour créer une version "tiny" du modèle.
    3. **Déploiement :** Intégrer ce modèle léger dans une application destinée à fonctionner sur un appareil de périphérie.
    4. **Inférence Locale :** Les prédictions sont faites directement sur l'appareil, sans nécessiter de connexion à un serveur central, ce qui garantit une faible latence et une meilleure confidentialité des données.

### 2. `Model-Quantization-TensorRT.ipynb` - Quantification de Modèle avec TensorRT

Ce notebook met en lumière la quantification, une étape clé de l'optimisation de l'inférence des modèles de deep learning, en se référant conceptuellement à TensorRT de NVIDIA.

- **Objectif Conceptuel :** Expliquer comment la quantification réduit la taille et la latence d'un modèle en utilisant une arithmétique de plus faible précision, ce qui est essentiel pour le déploiement en temps réel.
- **Techniques MLOps Associées (TensorRT) :**
    - **TensorRT :** C'est un SDK de NVIDIA pour l'inférence haute performance de modèles de deep learning. Il optimise les modèles pour les GPU NVIDIA en appliquant des fusions de couches, en sélectionnant les meilleurs noyaux (kernels) et, surtout, en appliquant la quantification.
    - **Quantification Post-Entraînement (Post-Training Quantization - PTQ) :** C'est la méthode la plus simple. Un modèle déjà entraîné en `float32` est converti en `int8`. TensorRT calibre le modèle en observant la plage de valeurs des activations sur un petit échantillon de données de validation pour déterminer comment mapper les valeurs `float32` en `int8` de manière optimale.
    - **Quantization-Aware Training (QAT) :** Une méthode plus avancée où la quantification est simulée pendant l'entraînement. Le modèle apprend à être robuste à la perte de précision, ce qui permet souvent d'obtenir une meilleure performance que le PTQ.
- **Workflow Illustré (implicitement) :**
    1. **Modèle Entraîné :** Partir d'un modèle de régression standard (`GradientBoostingRegressor` dans le code, mais conceptuellement un modèle de deep learning).
    2. **Conversion ONNX :** Exporter le modèle entraîné (par exemple, depuis PyTorch ou TensorFlow) au format ONNX (Open Neural Network Exchange).
    3. **Optimisation TensorRT :** Utiliser l'outil `trtexec` ou l'API Python de TensorRT pour parser le modèle ONNX et générer un "moteur" d'inférence optimisé. C'est à cette étape que la quantification (par exemple, en `INT8`) est appliquée.
    4. **Déploiement :** Utiliser ce moteur TensorRT dans l'application de production pour une inférence à faible latence et à haut débit sur les GPU NVIDIA.

### 3. `Prometheus-ML-Monitoring.ipynb` - Surveillance de Modèles ML avec Prometheus

Ce notebook aborde un aspect critique du MLOps : la surveillance continue des modèles en production pour détecter les dégradations de performance, la dérive des données, et d'autres problèmes.

- **Objectif Conceptuel :** Montrer comment intégrer la surveillance (monitoring) dans un service de prédiction en exposant des métriques qui peuvent être collectées par un système comme Prometheus.
- **Techniques MLOps Associées (Prometheus et Grafana) :**
    - **Exposition de Métriques :** Les applications de machine learning en production doivent exposer des métriques via un endpoint HTTP (souvent `/metrics`). Ces métriques peuvent inclure :
        - **Métriques Opérationnelles :** Latence des prédictions (temps de réponse), taux de requêtes, taux d'erreurs.
        - **Métriques de Données :** Distribution statistique des caractéristiques d'entrée (pour détecter la dérive des données ou *data drift*), distribution des prédictions de sortie (pour détecter la dérive du concept ou *concept drift*).
        - **Métriques de Performance du Modèle :** Si des étiquettes de vérité terrain deviennent disponibles (même avec un certain délai), on peut suivre l'exactitude, le F1-score, le MSE, etc., au fil du temps.
    - **Prometheus :** Un système de surveillance et d'alerte open-source qui "scrape" (collecte) périodiquement ces métriques à partir des endpoints exposés. Il stocke ces données sous forme de séries temporelles et dispose d'un puissant langage de requête (PromQL) pour analyser les données et configurer des alertes.
    - **Grafana :** Un outil de visualisation qui se connecte à Prometheus (parmi d'autres sources de données) pour créer des tableaux de bord (dashboards) dynamiques. Ces tableaux de bord permettent de visualiser les métriques en temps réel, de suivre les tendances et de diagnostiquer rapidement les problèmes.
- **Workflow Illustré (implicitement) :**
    1. **Modèle comme Service :** Le modèle (`AdvancedRegressor`) est enveloppé dans une API (par exemple, avec Flask ou FastAPI).
    2. **Instrumentation :** Des bibliothèques clientes Prometheus (comme `prometheus-client` pour Python) sont utilisées pour définir et mettre à jour des métriques (compteurs, jauges, histogrammes) à chaque appel de prédiction.
    3. **Déploiement :** L'API du modèle est déployée (par exemple, dans un conteneur Docker sur Kubernetes).
    4. **Configuration de Prometheus :** Une instance de Prometheus est configurée pour découvrir et scraper automatiquement les endpoints `/metrics` des instances du service de modèle.
    5. **Visualisation et Alerte :** Grafana est utilisé pour visualiser les métriques collectées par Prometheus, et des règles d'alerte sont mises en place dans Prometheus pour notifier les équipes en cas de détection d'anomalies (par exemple, si la latence moyenne dépasse un seuil ou si la distribution d'une caractéristique change de manière significative).
