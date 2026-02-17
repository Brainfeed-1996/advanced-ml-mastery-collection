# 04 - Traitement du Langage Naturel (NLP)

Ce dossier se concentre sur des workflows de Traitement du Langage Naturel (NLP), allant des approches de base comme TF-IDF aux modèles de pointe basés sur les Transformers comme BERT et GPT.

**Date de dernière mise à jour :** 2026-02-16 16:05:59

## Prérequis

- Python 3.10+
- Un environnement virtuel (`venv`) est fortement recommandé. (Voir `docs/INSTALLATION.md`)
- Des dépendances lourdes comme `pytorch`, `transformers` et `spacy` sont nécessaires pour certains notebooks. (Voir `docs/OPTIONAL_HEAVY_DEPS.md`)

---

## Présentation des Notebooks (.ipynb)

### 1. `BERT-Sentiment-Analysis.ipynb` - Analyse de Sentiments avec BERT

Ce notebook illustre comment utiliser un modèle Transformer pré-entraîné pour l'analyse de sentiments et le compare à une méthode de base.

- **Objectif :** Classifier une phrase comme ayant un sentiment positif ou négatif.
- **Techniques Abordées :**
    - **Modèle de base :** Pipeline `scikit-learn` avec `TfidfVectorizer` et `LogisticRegression`.
    - **Inférence avec Transformers :** Utilisation du pipeline `sentiment-analysis` de Hugging Face avec le modèle `distilbert-base-uncased-finetuned-sst-2-english` pour une classification de sentiment sémantique.
    - **Gestion des Erreurs de Cache :** Le code inclut une gestion d'erreur pour détecter et nettoyer un cache corrompu de Hugging Face, un problème courant lors du téléchargement de modèles.

### 2. `GPT-Fine-Tuning-Basics.ipynb` - Initiation au Fine-Tuning de GPT

Ce notebook offre une introduction pratique aux mécanismes de fine-tuning d'un modèle de type GPT (Generative Pre-trained Transformer). Il implémente un modèle causal de taille réduite à partir de zéro.

- **Objectif :** Entraîner un modèle de langage causal capable de générer du texte qui imite le style d'un corpus donné.
- **Données :** Un court texte sur les bonnes pratiques en Machine Learning industriel est utilisé comme corpus.
- **Techniques Abordées :**
    - **Tokenisation au niveau du caractère :** Création d'un vocabulaire simple basé sur les caractères uniques du texte.
    - **Préparation des données pour un modèle causal :** Création de blocs de séquences où le modèle apprend à prédire le caractère suivant.
    - **Implémentation d'un `TinyGPT` avec PyTorch :** Construction d'un Transformer simple avec des couches d'embedding (token et position), un encodeur Transformer et une tête de classification (Language Model head).
    - **Masque Causal :** Utilisation d'un masque triangulaire pour s'assurer que le modèle ne peut "voir" que les tokens précédents lors de la prédiction (mécanisme clé de l'auto-attention dans les modèles causaux).
    - **Entraînement et Échantillonnage (Sampling) :** Fine-tuning du modèle sur le corpus et génération de nouveau texte en utilisant un échantillonnage multinomial pour introduire de la variabilité.

### 3. `Named-Entity-Recognition-Spacy.ipynb` - Reconnaissance d'Entités Nommées avec spaCy

Ce notebook montre comment effectuer de la Reconnaissance d'Entités Nommées (NER), une tâche fondamentale du NLP qui consiste à identifier et à catégoriser des entités comme des personnes, des organisations ou des lieux dans un texte.

- **Objectif :** Extraire des informations structurées à partir d'un texte non structuré.
- **Outils :** Utilisation de `spaCy`, une bibliothèque de NLP très populaire et performante.
- **Techniques Abordées :**
    - **Chargement de modèle pré-entraîné :** Chargement automatique du modèle anglais `en_core_web_sm` de spaCy, avec un mécanisme de téléchargement s'il n'est pas déjà installé.
    - **Extraction d'entités :** Itération sur les entités (`doc.ents`) reconnues par le modèle pour extraire le texte et le label de chaque entité (par ex., `ORG` pour organisation, `GPE` pour entité géopolitique).
    - **Visualisation avec `displaCy` :** Utilisation de l'outil de visualisation de spaCy, `displaCy`, pour afficher le texte avec les entités mises en évidence et colorées selon leur type, ce qui facilite grandement l'interprétation.

### 4. `TFIDF-ErrorAnalysis-Classifier.ipynb` - Classification de Texte et Analyse d'Erreurs

Ce projet complet sert de modèle de base robuste pour tout problème de classification de texte. Il va au-delà de la simple mesure de performance en se concentrant sur l'analyse qualitative des erreurs du modèle.

- **Objectif :** Classifier des messages courts en différentes catégories (facturation, technique, sécurité) et comprendre pourquoi le modèle se trompe.
- **Données :** Un jeu de données synthétique est généré avec un vocabulaire qui se chevauche entre les catégories pour rendre la tâche de classification non triviale.
- **Techniques Abordées :**
    - **Vectorisation TF-IDF :** Utilisation de `TfidfVectorizer` avec des n-grammes (unigrammes et bigrammes) pour capturer des expressions simples.
    - **Classification avec `LogisticRegression` :** Entraînement d'un modèle linéaire avec une pondération des classes (`class_weight='balanced'`) pour gérer un éventuel déséquilibre.
    - **Analyse Quantitative :** Génération d'un rapport de classification complet et d'une matrice de confusion pour évaluer la performance globale.
    - **Analyse Qualitative des Erreurs :** Identification des prédictions incorrectes sur l'ensemble de test. Le notebook trie ces erreurs par **confiance** (la probabilité la plus élevée donnée par le modèle à sa prédiction incorrecte) afin de prioriser l'analyse des erreurs les plus "surprenantes" pour le modèle. C'est une étape cruciale pour l'amélioration itérative du modèle.

---

## Comment Exécuter les Projets

### Exécution Interactive (Recommandé)

```bash
jupyter notebook 04-NLP/<nom_du_notebook>.ipynb
```

### Exécution Headless

```bash
python -m jupyter nbconvert --to notebook --execute \
  04-NLP/<nom_du_notebook>.ipynb --output <nom_du_notebook>.ipynb --output-dir 04-NLP
```
