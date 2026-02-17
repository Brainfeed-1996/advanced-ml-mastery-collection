# 03 - Clustering

Ce dossier est dédié à l'apprentissage non supervisé, et plus particulièrement aux techniques de clustering. Le clustering vise à regrouper des données non étiquetées en fonction de leurs similarités. Les notebooks de ce dossier explorent trois algorithmes de clustering populaires : K-Means, le clustering hiérarchique et DBSCAN.

**Date de dernière mise à jour :** 2026-02-08 17:42:46

---

## Présentation des Notebooks

### 1. `K-Means-Segmentation.ipynb` - Segmentation de la Clientèle avec K-Means

Ce notebook est un guide complet sur l'utilisation de l'algorithme K-Means pour la segmentation de la clientèle, une tâche courante en marketing. Il illustre comment identifier des segments de clientèle distincts à partir de leurs comportements d'achat.

- **Concepts Clés :**
    - **Génération de Données Synthétiques :** Création d'un jeu de données client réaliste avec des segments latents.
    - **Analyse Exploratoire des Données (EDA) :** Visualisation des distributions et des relations entre les variables.
    - **Sélection du Nombre de Clusters (K) :** Utilisation de la méthode du coude (Elbow) et du score de silhouette pour déterminer le nombre optimal de clusters.
    - **Profilage des Clusters :** Analyse des caractéristiques moyennes de chaque cluster pour créer des personas de clientèle.
    - **Visualisation avec PCA :** Utilisation de l'analyse en composantes principales (PCA) pour visualiser les clusters en 2D.

### 2. `Hierarchical-Clustering-Genes.ipynb` - Clustering Hiérarchique de Gènes

Ce notebook montre comment appliquer le clustering hiérarchique pour regrouper des gènes en fonction de leurs profils d'expression. Cette technique est largement utilisée en bio-informatique pour identifier des groupes de gènes ayant des fonctions similaires.

- **Concepts Clés :**
    - **Clustering Agglomératif :** Utilisation de l'algorithme de clustering hiérarchique ascendant pour construire un dendrogramme.
    - **Dendrogramme :** Visualisation de la structure hiérarchique des clusters et aide à la sélection du nombre de clusters.
    - **Découpage du Dendrogramme :** Utilisation de la fonction `fcluster` pour extraire un nombre spécifique de clusters à partir du dendrogramme.
    - **Évaluation du Clustering :** Utilisation du score de silhouette pour évaluer la qualité de la partition des clusters.

### 3. `DBSCAN-Anomaly-Detection.ipynb` - Détection d'Anomalies avec DBSCAN

Ce notebook utilise DBSCAN (Density-Based Spatial Clustering of Applications with Noise) pour identifier des anomalies dans un jeu de données. Contrairement à K-Means, DBSCAN n'a pas besoin que le nombre de clusters soit spécifié à l'avance et peut découvrir des clusters de formes arbitraires.

- **Concepts Clés :**
    - **Clustering Basé sur la Densité :** Compréhension des concepts de points centraux, de points frontières et de bruit (anomalies).
    - **Détection d'Anomalies :** Identification des points de données qui n'appartiennent à aucun cluster (points de bruit).
    - **Hyperparamètres de DBSCAN :** Exploration de l'impact des paramètres `eps` (distance maximale) et `min_samples` (nombre minimum de points).
    - **Visualisation des Clusters et des Anomalies :** Visualisation des clusters de formes non sphériques et des points classés comme anomalies.
