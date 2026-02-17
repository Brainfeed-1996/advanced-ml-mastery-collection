# 09 - IA Générative

Ce dossier est consacré à l'Intelligence Artificielle (IA) générative, un domaine fascinant du machine learning où les modèles sont capables de créer de nouveaux contenus, qu'il s'agisse de texte, d'images ou d'autres données. Il explore des concepts allant des modèles de langage de base aux réseaux antagonistes génératifs (GAN) plus complexes.

**Date de dernière mise à jour :** 2026-02-16 07:18:50

## Prérequis

- Python 3.10+
- Un environnement virtuel (`venv`) est fortement recommandé. (Voir `docs/INSTALLATION.md`)
- Les dépendances varient, allant de `scikit-learn` et `spacy` à des bibliothèques plus lourdes comme `pytorch`, `tensorflow` et `ultralytics`. (Voir `docs/OPTIONAL_HEAVY_DEPS.md`)

---

## Présentation des Notebooks (.ipynb)

### 1. `CharNGram-LanguageModel.ipynb` - Modèle de Langage N-Gramme au Niveau Caractère

Ce notebook est une excellente introduction aux modèles de langage génératifs, construit à partir de zéro sans dépendre de frameworks complexes de deep learning. Il est léger et conçu pour fonctionner rapidement sur CPU.

- **Objectif :** Construire un modèle de langage capable de générer du texte en apprenant les probabilités de séquences de caractères dans un corpus.
- **Technique (Modèle N-Gramme) :**
    - **Principe :** Un modèle N-gramme de caractères prédit le caractère suivant en se basant sur les `N-1` caractères précédents (le "contexte"). Il calcule la probabilité `P(caractère | contexte)`.
    - **Entraînement :** Le modèle parcourt le texte d'entraînement et compte les occurrences de chaque contexte (`ctx_counts`) et de chaque N-gramme complet (`ng_counts`).
    - **Lissage de Laplace (Smoothing) :** Pour gérer les contextes ou les N-grammes jamais vus dans les données d'entraînement (ce qui conduirait à des probabilités nulles), un lissage de Laplace (avec un paramètre `alpha`) est ajouté. Cela consiste à ajouter une petite valeur à tous les comptes, évitant ainsi les zéros.
- **Évaluation (Perplexité) :**
    - **Définition :** La perplexité est une mesure standard pour évaluer les modèles de langage. Elle peut être interprétée comme le facteur de branchement moyen du modèle. Une perplexité plus faible indique que le modèle est plus "sûr" de ses prédictions et donc meilleur. Une perplexité de 10, par exemple, signifie qu'à chaque étape, le modèle hésite en moyenne entre 10 choix possibles.
    - **Calcul :** Elle est calculée comme l'exponentielle de l'entropie croisée moyenne de l'ensemble de test.
- **Génération de Texte :**
    - Le notebook implémente une fonction de génération qui produit du nouveau texte caractère par caractère.
    - **Température de Sampling :** Un paramètre de "température" est introduit lors de l'échantillonnage du caractère suivant. Une température basse rend la génération plus déterministe et conservatrice (le modèle choisit les caractères les plus probables), tandis qu'une température élevée augmente l'aléatoire et la créativité, au risque de produire des séquences moins cohérentes.

### 2. `GAN-Face-Generation.ipynb` - Génération de Données Synthétiques avec GAN

Ce notebook offre une introduction pratique aux Réseaux Antagonistes Génératifs (GANs), un des concepts les plus puissants de l'IA générative, en utilisant un exemple simple en 2D.

- **Objectif :** Entraîner un GAN à générer des données synthétiques qui ressemblent à une distribution de données réelles (ici, un mélange de Gaussiennes en 2D).
- **Technique (GAN) :**
    - **Architecture :** Un GAN se compose de deux réseaux de neurones qui s'affrontent dans un jeu à somme nulle :
        1. **Le Générateur (G) :** Prend en entrée un bruit aléatoire (un vecteur latent) et essaie de produire des données qui ressemblent aux données réelles.
        2. **Le Discriminateur (D) :** Prend en entrée des données (soit réelles, soit générées par G) et essaie de déterminer si elles sont authentiques ou fausses.
    - **Processus d'Entraînement :**
        - **Entraînement du Discriminateur :** On montre au discriminateur un lot de données réelles (étiquetées comme "vraies") et un lot de données générées (étiquetées comme "fausses"). On met à jour ses poids pour qu'il s'améliore dans cette tâche de classification.
        - **Entraînement du Générateur :** On génère un nouveau lot de fausses données et on les passe au discriminateur. Cette fois, on met à jour les poids du *générateur* pour qu'il devienne meilleur à "tromper" le discriminateur, c'est-à-dire à générer des données que le discriminateur classera comme "vraies".
    - **Équilibre :** Au fil du temps, le générateur apprend à produire des échantillons de plus en plus réalistes, et le discriminateur devient de plus en plus compétent pour les distinguer, jusqu'à ce qu'un équilibre soit atteint.
- **Visualisation :** Le notebook se termine en générant un lot de nouveaux points de données à partir du générateur entraîné et en les comparant visuellement aux données réelles. Le graphique montre comment le GAN a appris à reproduire la structure en croix du mélange de Gaussiennes d'origine.

### 3. `Named-Entity-Recognition-Spacy.ipynb` - Reconnaissance d'Entités Nommées avec spaCy

Ce notebook est une démonstration rapide et efficace de la Reconnaissance d'Entités Nommées (NER), une tâche fondamentale du Traitement du Langage Naturel (NLP).

- **Objectif :** Identifier et classifier des entités (comme des personnes, des organisations, des lieux, des dates) dans un texte brut.
- **Technique (spaCy) :**
    - **Bibliothèque :** spaCy est une bibliothèque Python de pointe pour le NLP, conçue pour être rapide, efficace et prête pour la production.
    - **Modèles Pré-entraînés :** Le notebook utilise `en_core_web_sm`, un modèle de langue anglaise pré-entraîné par spaCy. Ce modèle inclut un pipeline de traitement complet avec un composant de NER déjà entraîné sur un grand corpus de texte.
    - **Processus :**
        1. Le texte est passé à l'objet `nlp` chargé, qui l'analyse.
        2. Le résultat, un objet `Doc`, contient une collection `ents` qui liste toutes les entités trouvées.
        3. Pour chaque entité, on peut accéder à son texte (`ent.text`) et à son type (`ent.label_`), comme `GPE` (Geopolitical Entity), `ORG` (Organization), `PRODUCT`, etc.
- **Visualisation :** spaCy fournit un outil de visualisation très pratique, `displacy`, qui peut générer un rendu HTML du texte où les entités sont surlignées et étiquetées avec leurs types respectifs, rendant l'analyse facile à interpréter.

### 4. `Object-Detection-YOLOv8.ipynb` - Détection d'Objets avec YOLOv8

Ce notebook montre comment utiliser YOLOv8, un modèle de détection d'objets de pointe, pour identifier des objets dans une image.

- **Objectif :** Appliquer un modèle YOLO pré-entraîné pour détecter la présence et l'emplacement d'objets dans une image synthétique simple.
- **Technique (YOLOv8) :**
    - **Bibliothèque :** `ultralytics` est la bibliothèque officielle pour utiliser les modèles YOLO. Le notebook utilise `yolov8n.pt` (la version "nano"), qui est le plus petit et le plus rapide des modèles YOLOv8, idéal pour une exécution rapide même sans GPU.
    - **Modèle Pré-entraîné :** Le modèle a été pré-entraîné sur le vaste jeu de données COCO, qui contient des milliers d'images avec des annotations pour 80 classes d'objets différentes.
    - **Inférence :** L'inférence est très simple : on charge le modèle et on lui passe le chemin de l'image. Le modèle renvoie une liste de résultats, où chaque résultat contient les boîtes englobantes (`boxes`) des objets détectés, ainsi que leur classe et leur score de confiance.
- **Visualisation :** Le notebook utilise la méthode `plot()` intégrée aux résultats de YOLO, qui génère une image avec les boîtes englobantes et les étiquettes dessinées directement dessus. Bien que l'image synthétique ne contienne aucun objet que le modèle COCO puisse reconnaître, le notebook illustre le flux de travail complet.

### 5. `Stable-Diffusion-Prompt-Eng.ipynb` - Ingénierie de Prompts avec Stable Diffusion

Ce notebook est une introduction à la génération d'images à partir de texte (text-to-image) en utilisant Stable Diffusion, un des modèles de diffusion les plus populaires.

- **Objectif :** Générer une image à partir d'une description textuelle (un "prompt") en utilisant un modèle de diffusion.
- **Technique (Stable Diffusion) :**
    - **Bibliothèque :** `diffusers` de Hugging Face est la bibliothèque standard pour interagir avec les modèles de diffusion comme Stable Diffusion.
    - **Modèles :** Pour des raisons de sécurité et de performance sur CPU, le notebook utilise par défaut une version "tiny" de Stable Diffusion (`hf-internal-testing/tiny-stable-diffusion-pipe`). Il est également configurable (via une variable d'environnement `SD_FULL=1`) pour utiliser le modèle complet `runwayml/stable-diffusion-v1-5` si des ressources suffisantes (GPU) sont disponibles.
    - **Processus de Diffusion :** Les modèles de diffusion génèrent des images en partant d'un bruit aléatoire et en le "déniègrant" de manière itérative. À chaque étape (`num_inference_steps`), le modèle affine l'image en se basant sur les instructions du prompt textuel pour se rapprocher de la description souhaitée.
- **Ingénierie de Prompts :** Le choix du prompt est crucial pour la qualité et le style de l'image générée. Le notebook utilise un prompt descriptif ("a cyberpunk city skyline at night, neon lights, cinematic") pour guider le modèle vers une image avec une esthétique spécifique.

### 6. `Transfer-Learning-ResNet.ipynb` - Apprentissage par Transfert avec ResNet

Ce notebook est un exemple classique de computer vision qui illustre la puissance de l'apprentissage par transfert (transfer learning).

- **Objectif :** Entraîner rapidement un classifieur d'images performant pour une nouvelle tâche (ici, la reconnaissance de chiffres manuscrits) en réutilisant un modèle puissant pré-entraîné sur une tâche beaucoup plus vaste (ImageNet).
- **Données :** Le jeu de données `digits` de scikit-learn (images de chiffres de 8x8 pixels) est utilisé. Pour être compatibles avec ResNet, ces petites images sont sur-échantillonnées à 224x224 pixels et converties en 3 canaux (RGB).
- **Technique (Transfer Learning) :**
    - **Modèle Pré-entraîné (Backbone) :** Un modèle `ResNet18`, pré-entraîné sur le jeu de données ImageNet (des millions d'images, 1000 classes), est chargé depuis `torchvision`. Ce modèle a déjà appris des caractéristiques visuelles très riches (bords, textures, formes, etc.) dans ses couches profondes.
    - **Extraction de Caractéristiques (Embeddings) :** On "gèle" les poids du backbone (pour ne pas les ré-entraîner) et on supprime sa couche de classification finale (`fc`). Le modèle est ensuite utilisé comme un simple extracteur de caractéristiques : on lui passe une image et il renvoie un vecteur de 512 dimensions (un "embedding") qui est une représentation riche de l'image.
    - **Nouvelle Tête de Classification (Head) :** Un nouveau classifieur très simple (une seule couche linéaire) est créé. Ce classifieur prend en entrée les embeddings de 512 dimensions et a 10 sorties (une pour chaque chiffre de 0 à 9).
    - **Entraînement :** Seuls les poids de cette nouvelle "tête" de classification sont entraînés. Comme le backbone fait déjà tout le travail lourd d'extraction de caractéristiques, l'entraînement de la tête est extrêmement rapide et ne nécessite que très peu de données.
- **Évaluation :** Le notebook montre qu'avec seulement 3 petites époques d'entraînement, cette approche atteint une excellente précision (autour de 76% dans cet exemple rapide) sur l'ensemble de test, démontrant l'efficacité de l'apprentissage par transfert.

### 7. `Variational-Autoencoders-MNIST.ipynb` - Autoencodeurs Variationnels (VAE) sur MNIST

Ce notebook explore les Autoencodeurs Variationnels (VAE), une version plus avancée et probabiliste des autoencodeurs classiques, capable non seulement de reconstruire des données mais aussi de générer de nouveaux échantillons plausibles.

- **Objectif :** Entraîner un VAE sur le jeu de données MNIST (chiffres manuscrits) pour apprendre une représentation latente de ces chiffres et générer de nouvelles images de chiffres.
- **Technique (VAE) :**
    - **Différence avec les AE :** Alors qu'un autoencodeur classique apprend à mapper une entrée vers un point unique dans l'espace latent, un VAE apprend à mapper une entrée vers une *distribution de probabilité* (généralement une Gaussienne) dans l'espace latent. L'encodeur prédit la moyenne (`mu`) et la variance (`log_var`) de cette distribution.
    - **Échantillonnage Latent :** Pour générer une reconstruction, on n'utilise pas directement la moyenne, mais on *échantillonne* un point (`z`) à partir de cette distribution. Cette étape d'échantillonnage est ce qui donne au VAE ses propriétés génératives.
    - **Fonction de Perte (Loss) :** La perte d'un VAE est composée de deux termes :
        1. **Perte de Reconstruction :** Comme pour un AE classique, elle mesure la différence entre l'image d'entrée et l'image reconstruite (souvent une erreur quadratique moyenne ou une entropie croisée binaire).
        2. **Divergence KL (Kullback-Leibler) :** C'est le terme de régularisation. Il mesure à quel point la distribution latente apprise s'écarte d'une distribution a priori simple (généralement une Gaussienne standard, de moyenne 0 et de variance 1). Ce terme force l'espace latent à être bien structuré et continu, ce qui est essentiel pour une génération cohérente.
- **Génération :** Une fois le VAE entraîné, on peut générer de nouvelles images en échantillonnant simplement un vecteur aléatoire `z` à partir de la distribution a priori (la Gaussienne standard) et en le passant au décodeur. Le notebook montre une grille d'images de chiffres générées par le modèle, illustrant sa capacité à créer de nouvelles données plausibles.
