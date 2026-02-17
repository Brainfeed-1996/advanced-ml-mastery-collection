# 05 - Vision par Ordinateur

Ce dossier explore une gamme variée de workflows en vision par ordinateur, allant des techniques classiques comme les auto-encodeurs aux modèles de pointe pour la classification, la détection d'objets et l'apprentissage par transfert.

**Date de dernière mise à jour :** 2026-02-16 16:05:59

## Prérequis

- Python 3.10+
- Un environnement virtuel (`venv`) est fortement recommandé. (Voir `docs/INSTALLATION.md`)
- Des dépendances lourdes comme `pytorch`, `torchvision`, `ultralytics` et `scikit-learn` sont nécessaires pour certains notebooks. (Voir `docs/OPTIONAL_HEAVY_DEPS.md`)

---

## Présentation des Notebooks (.ipynb)

### 1. `Autoencoders-Denoising.ipynb` - Débruitage d'Images avec des Auto-Encodeurs

Ce notebook offre une introduction pratique aux auto-encodeurs, un type de réseau de neurones non supervisé, en se concentrant sur une tâche de débruitage d'images.

- **Objectif :** Entraîner un auto-encodeur à reconstruire des images de chiffres manuscrits (dataset `digits` de scikit-learn) après leur avoir ajouté un bruit gaussien.
- **Architecture :** L'auto-encodeur est composé de deux parties :
    - **Encodeur :** Une série de couches linéaires avec des fonctions d'activation ReLU qui compressent l'image d'entrée (64 pixels) en une représentation latente de faible dimension (16 dimensions).
    - **Décodeur :** Une autre série de couches qui prend la représentation latente et tente de reconstruire l'image originale. La fonction d'activation finale est une sigmoïde pour garantir que les valeurs des pixels de sortie sont comprises entre 0 et 1.
- **Techniques Abordées :**
    - **Entraînement non supervisé :** Le modèle apprend à partir des données elles-mêmes (images bruitées en entrée, images propres en sortie) sans avoir besoin d'étiquettes explicites.
    - **Perte de reconstruction (MSE) :** L'entraînement vise à minimiser l'erreur quadratique moyenne (Mean Squared Error) entre l'image reconstruite et l'image propre originale.
    - **Visualisation :** Le notebook se termine par une comparaison visuelle côte à côte des images bruitées, des images originales (propres) et des images reconstruites par le modèle, démontrant ainsi sa capacité à "apprendre" à enlever le bruit.

### 2. `CNN-Medical-Imaging.ipynb` - Classification d'Images avec un CNN

Ce notebook sert de pipeline de base complet pour l'entraînement d'un Réseau de Neurones Convolutif (CNN) pour la classification d'images. Il utilise le dataset `digits` comme un proxy simple pour illustrer le processus, qui serait applicable à des images médicales plus complexes.

- **Objectif :** Classifier des images de chiffres manuscrits (10 classes, de 0 à 9).
- **Architecture (`SmallCNN`) :** Un CNN simple est construit avec `PyTorch`, comprenant :
    - **Couches Convolutives :** Deux blocs de `Conv2d`, `ReLU`, et `MaxPool2d` pour extraire des caractéristiques hiérarchiques des images.
    - **Couche de Classification :** Une couche linéaire (`Linear`) finale qui prend les caractéristiques extraites et produit des scores pour chaque classe.
- **Techniques Abordées :**
    - **Préparation des données :** Redimensionnement des images et utilisation de `DataLoader` de PyTorch pour créer des lots (batches) pour l'entraînement.
    - **Boucle d'entraînement standard :** Itération sur les époques et les lots, calcul de la perte (`CrossEntropyLoss`), rétropropagation de l'erreur et mise à jour des poids du modèle avec un optimiseur (`Adam`).
    - **Évaluation du modèle :** Calcul de la précision (accuracy) sur un ensemble de test non vu pendant l'entraînement.
    - **Matrice de Confusion :** Utilisation de `seaborn` pour visualiser la matrice de confusion, un outil essentiel pour comprendre quelles classes le modèle confond le plus souvent.

### 3. `GAN-Synthetic-Data-Gen.ipynb` - Génération de Données Synthétiques avec un GAN

Ce notebook est une introduction minimale mais fonctionnelle aux Réseaux Antagonistes Génératifs (GANs) pour la génération de données synthétiques. Il entraîne un petit GAN à reproduire une distribution de données 2D simple.

- **Objectif :** Générer de nouvelles données qui ressemblent à une distribution de mélange de gaussiennes 2D (un nuage de points formant une croix).
- **Architecture (GAN) :** Le modèle est composé de deux réseaux en compétition :
    - **Générateur (G) :** Un simple réseau de neurones (MLP) qui prend un vecteur de bruit aléatoire en entrée et tente de produire une sortie qui ressemble aux vraies données (un point 2D).
    - **Discriminateur (D) :** Un autre MLP qui prend en entrée soit une vraie donnée, soit une donnée générée par G, et doit décider si elle est réelle ou fausse.
- **Techniques Abordées :**
    - **Jeu antagoniste :** Le générateur apprend à tromper le discriminateur, tandis que le discriminateur apprend à mieux distinguer le vrai du faux. Cet équilibre conduit le générateur à produire des données de plus en plus réalistes.
    - **Perte Binaire Croisée-Entropie (`BCEWithLogitsLoss`) :** Utilisée pour entraîner les deux réseaux.
    - **Visualisation :** Le résultat final est une dispersion (scatterplot) qui compare la distribution des vraies données avec celle des données générées par le GAN, montrant dans quelle mesure le modèle a réussi à apprendre la structure des données originales.

### 4. `Isolation-Forest-Cybersecurity.ipynb` - Détection d'Anomalies en Cybersécurité

Ce notebook est une démonstration de bout en bout de la détection d'anomalies sur des données tabulaires, un cas d'usage très courant en cybersécurité.

- **Objectif :** Identifier des événements réseau anormaux dans un jeu de données synthétiques simulant du trafic réseau.
- **Données Synthétiques :** Génération d'événements réseau "normaux" (par ex., trafic web sur les ports 80/443) et injection d'anomalies rares (par ex., pics de trafic entrant, connexions sur des ports inhabituels, tentatives de connexion échouées).
- **Modèle (`IsolationForest`) :** Utilisation de l'algorithme `Isolation Forest` de scikit-learn. Cet algorithme non supervisé est particulièrement efficace pour la détection d'anomalies car il fonctionne en "isolant" les observations. Les anomalies, étant rares et différentes, sont plus faciles à isoler que les points normaux.
- **Techniques Abordées :**
    - **Détection non supervisée :** Le modèle n'a pas besoin que les anomalies soient étiquetées pendant l'entraînement.
    - **Score d'Anomalie :** Le modèle ne donne pas une classification binaire (normal/anomalie) mais un score. Un score plus élevé indique une plus grande probabilité d'être une anomalie.
    - **Évaluation avec ROC-AUC :** La performance du modèle est évaluée à l'aide de l'aire sous la courbe ROC (Receiver Operating Characteristic). Un score ROC-AUC proche de 1.0 indique que le modèle est excellent pour distinguer les événements normaux des anomalies.
    - **Distribution des Scores :** Visualisation de la distribution des scores d'anomalie pour les points normaux et anormaux, montrant que le modèle assigne effectivement des scores beaucoup plus élevés aux anomalies.

### 5. `ResNet-Transfer-Learning.ipynb` - Apprentissage par Transfert avec ResNet

Ce notebook illustre la puissance de l'apprentissage par transfert (transfer learning), une technique fondamentale en vision par ordinateur où un modèle pré-entraîné sur une tâche à grande échelle est adapté pour une nouvelle tâche spécifique.

- **Objectif :** Classifier des images de chiffres (dataset `digits`) en utilisant les caractéristiques extraites par un modèle `ResNet18` pré-entraîné sur le dataset `ImageNet` (qui contient des millions d'images de 1000 classes différentes).
- **Techniques Abordées :**
    - **Chargement d'un modèle pré-entraîné :** Utilisation de `torchvision.models` pour charger un `ResNet18` avec ses poids pré-entraînés. Les poids sont téléchargés automatiquement lors de la première exécution.
    - **Extraction de Caractéristiques (Embeddings) :** La dernière couche de classification du ResNet (`fc`) est remplacée par une couche identité (`nn.Identity`). Le modèle est ensuite utilisé en mode évaluation (`.eval()`) pour ne pas mettre à jour ses poids. En faisant passer les images à travers ce "backbone", on obtient des vecteurs de caractéristiques de haute qualité (embeddings de 512 dimensions).
    - **Entraînement d'une "Tête" de Classification :** Seule une nouvelle couche linéaire simple (`head`), placée après le backbone, est entraînée sur la nouvelle tâche. Cela est beaucoup plus rapide et nécessite beaucoup moins de données que d'entraîner un CNN complet à partir de zéro.
    - **Prétraitement des images :** Les petites images de chiffres (8x8) sont redimensionnées en 224x224 et converties en 3 canaux (RGB) pour correspondre au format d'entrée attendu par ResNet.

### 6. `YOLOv8-Object-Detection.ipynb` - Détection d'Objets avec YOLOv8

Ce notebook montre comment utiliser `YOLOv8` (You Only Look Once), l'un des modèles de détection d'objets les plus performants et les plus rapides, pour faire de l'inférence sur une image.

- **Objectif :** Détecter la présence et l'emplacement d'objets dans une image.
- **Outils :** Utilisation de la bibliothèque `ultralytics`.
- **Techniques Abordées :**
    - **Inférence simple :** Le modèle `yolov8n.pt` (la plus petite et rapide des variantes de YOLOv8) est chargé et appliqué directement à une image. Les poids sont téléchargés automatiquement.
    - **Image Synthétique :** Pour éviter de dépendre de fichiers externes, une image simple contenant un rectangle est générée à l'aide de la bibliothèque `Pillow`.
    - **Résultats de Détection :** L'objet `results` retourné par le modèle contient une multitude d'informations, notamment les boîtes englobantes (`boxes`) des objets détectés, leurs classes et leurs scores de confiance.
    - **Visualisation :** La méthode `.plot()` intégrée à l'objet de résultats permet de visualiser très facilement l'image originale avec les boîtes de détection superposées.

---

## Comment Exécuter les Projets

### Exécution Interactive (Recommandé)

```bash
jupyter notebook 05-Computer-Vision/<nom_du_notebook>.ipynb
```

### Exécution Headless

```bash
python -m jupyter nbconvert --to notebook --execute \
  05-Computer-Vision/<nom_du_notebook>.ipynb --output <nom_du_notebook>.ipynb --output-dir 05-Computer-Vision
```
