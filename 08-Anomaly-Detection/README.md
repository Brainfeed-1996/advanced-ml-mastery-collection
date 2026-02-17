# 08 - Détection d'Anomalies

Ce dossier explore diverses techniques de détection d'anomalies, un domaine essentiel du machine learning qui vise à identifier des événements rares ou des observations suspectes qui s'écartent de la majorité des données. Les applications vont de la détection de fraude et de la cybersécurité à la maintenance prédictive.

**Date de dernière mise à jour :** 2026-02-15 23:43:08

## Prérequis

- Python 3.10+
- Un environnement virtuel (`venv`) est fortement recommandé. (Voir `docs/INSTALLATION.md`)
- Les dépendances de base de `scikit-learn` sont suffisantes pour la plupart des notebooks, mais certains peuvent nécessiter des bibliothèques plus lourdes comme `pytorch`. (Voir `docs/OPTIONAL_HEAVY_DEPS.md`)

---

## Présentation des Notebooks (.ipynb)

### 1. `Autoencoders-Network-Security.ipynb` - Détection d'Anomalies Réseau avec Autoencodeurs (proxy PCA)

Ce notebook simule l'utilisation d'un autoencodeur pour la détection d'anomalies dans des données de télémétrie réseau. Pour une exécution rapide, il utilise une Analyse en Composantes Principales (PCA) comme un proxy linéaire d'un autoencodeur neuronal.

- **Objectif :** Identifier des activités réseau suspectes (par exemple, des attaques) en se basant sur des métriques comme le nombre de connexions, les octets transférés, etc.
- **Données :** Un jeu de données synthétique est généré avec `make_blobs` pour représenter le trafic normal (inliers), et des points de données sont ajoutés en injectant du bruit pour simuler des anomalies (outliers).
- **Technique (Autoencodeur via PCA) :**
    - **Principe :** Un autoencodeur est un réseau de neurones entraîné à reconstruire son entrée. L'idée fondamentale est que s'il est entraîné uniquement sur des données normales, il apprendra à bien reconstruire ce type de données. Lorsqu'il sera confronté à une anomalie, l'erreur de reconstruction (la différence entre l'entrée et la sortie) sera élevée.
    - **Proxy PCA :** La PCA est une technique de réduction de dimension qui trouve les axes principaux de variance dans les données. En projetant les données sur un sous-espace de plus faible dimension, puis en les reprojetant dans l'espace d'origine, on obtient une reconstruction. L'erreur de reconstruction peut être utilisée de la même manière que pour un autoencodeur.
    - **Entraînement Semi-Supervisé :** Le modèle (ici, la PCA) est entraîné uniquement sur les données normales (inliers), ce qui est une approche semi-supervisée courante en détection d'anomalies.
    - **Détection :** L'erreur de reconstruction est calculée pour toutes les données (normales et anormales). Un seuil est ensuite défini (par exemple, le 99ème percentile des erreurs sur l'ensemble d'entraînement) pour classer une observation comme anormale si son erreur de reconstruction dépasse ce seuil.
- **Évaluation :** Le notebook évalue la performance du détecteur en utilisant la métrique AUC-ROC (Area Under the Receiver Operating Characteristic Curve), qui mesure la capacité du modèle à distinguer les classes. Une visualisation des distributions des erreurs de reconstruction pour les données normales et anormales montre clairement comment le modèle parvient à les séparer.

### 2. `Gradient-Boosting-Insurance.ipynb` - Prédiction de la Sévérité des Sinistres avec Gradient Boosting

Ce notebook est un exemple complet de régression utilisant des modèles de Gradient Boosting pour prédire la sévérité (le coût) de sinistres d'assurance. Il utilise le jeu de données `diabetes` de scikit-learn comme proxy.

- **Objectif :** Prédire une variable continue (la progression de la maladie, servant de proxy pour le coût d'un sinistre) à partir d'un ensemble de caractéristiques physiologiques.
- **Techniques Abordées :**
    - **Analyse Exploratoire (EDA) :** Visualisation de la distribution de la variable cible et des corrélations entre les caractéristiques pour comprendre les relations dans les données.
    - **Pipeline de Prétraitement :** Mise en place d'un pipeline robuste avec `scikit-learn` incluant l'imputation de valeurs manquantes (avec la médiane) et la standardisation des données (mise à l'échelle pour avoir une moyenne de 0 et une variance de 1).
    - **Modèles de Référence et Avancés :** Comparaison de plusieurs modèles :
        - `DummyRegressor` : Un modèle de référence très simple qui prédit toujours la moyenne de l'ensemble d'entraînement. Il sert de point de comparaison de base.
        - `Ridge` : Une régression linéaire avec régularisation L2, un modèle de référence plus solide.
        - `GradientBoostingRegressor` et `HistGradientBoostingRegressor` : Des modèles d'ensemble puissants basés sur des arbres de décision, connus pour leurs excellentes performances.
    - **Validation Croisée (Cross-Validation) :** Utilisation de la validation croisée K-Fold pour obtenir une estimation robuste de la performance des modèles en les entraînant et les évaluant sur différentes partitions des données.
    - **Optimisation d'Hyperparamètres :** Recherche aléatoire (`RandomizedSearchCV`) des meilleurs hyperparamètres pour le `HistGradientBoostingRegressor` (par exemple, le taux d'apprentissage, la profondeur maximale des arbres, etc.) afin d'améliorer ses performances.
    - **Interprétabilité (Permutation Importance) :** Après avoir entraîné le meilleur modèle, une analyse d'importance par permutation est effectuée. Cette technique mesure l'importance de chaque caractéristique en évaluant la baisse de performance du modèle lorsque les valeurs de cette caractéristique sont mélangées aléatoirement. C'est une méthode agnostique du modèle pour comprendre quelles variables ont le plus d'influence sur les prédictions.

### 3. `Isolation-Forest-Cyber.ipynb` - Détection d'Anomalies en Cybersécurité avec Isolation Forest

Ce notebook présente l'algorithme Isolation Forest, une technique de détection d'anomalies non supervisée efficace, appliquée à un cas d'usage de cybersécurité.

- **Objectif :** Identifier des activités réseau anormales à partir de données synthétiques sans avoir besoin d'étiquettes préalables.
- **Technique (Isolation Forest) :**
    - **Principe :** L'algorithme part de l'idée que les anomalies sont "peu nombreuses et différentes". Il construit une forêt d'arbres de décision aléatoires (Isolation Trees). Pour chaque arbre, les données sont partitionnées de manière récursive en sélectionnant une caractéristique et un point de coupe aléatoires. Les anomalies, étant différentes, nécessiteront en moyenne moins de partitions pour être isolées que les points normaux.
    - **Score d'Anomalie :** Le score d'un point est basé sur la longueur moyenne du chemin pour l'isoler à travers tous les arbres de la forêt. Un chemin plus court conduit à un score d'anomalie plus élevé.
    - **Détection Non Supervisée :** L'algorithme est entraîné sur l'ensemble des données (normales et anormales) sans utiliser les étiquettes. Le paramètre `contamination` est utilisé pour indiquer au modèle la proportion attendue d'anomalies dans les données, ce qui aide à définir le seuil de décision interne.
- **Évaluation :** Le notebook évalue les performances avec une matrice de confusion, un rapport de classification et l'AUC-ROC. La distribution des scores d'anomalie est également visualisée pour les deux classes, montrant l'efficacité de l'algorithme à séparer les observations normales des observations anormales.

### 4. `KNN-Recommender-Systems.ipynb` - Système de Recommandation avec k-NN

Ce notebook construit un système de recommandation simple basé sur la méthode des k plus proches voisins (k-NN), une approche de filtrage collaboratif basé sur les items.

- **Objectif :** Recommander de nouveaux items (par exemple, des films, des produits) à un utilisateur en se basant sur les items qu'il a déjà aimés.
- **Données :** Un jeu de données synthétique d'interactions utilisateur-item (ratings) est généré, simulant un scénario de recommandation.
- **Technique (k-NN Item-Item Collaborative Filtering) :**
    - **Matrice Utilisateur-Item :** Les données sont transformées en une matrice où les lignes représentent les utilisateurs et les colonnes les items. Les valeurs de la matrice sont les notes (ratings) données par les utilisateurs aux items.
    - **Similarité entre Items :** L'algorithme `NearestNeighbors` de scikit-learn est utilisé pour calculer la similarité entre les items. La similarité cosinus est une métrique couramment utilisée, mesurant l'angle entre les vecteurs de notation des items.
    - **Génération de Recommandations :** Pour un utilisateur donné :
        1. On identifie les items qu'il a le mieux notés.
        2. Pour chacun de ces items, on trouve les `k` items les plus similaires (les voisins).
        3. On agrège ces voisins pour former un ensemble d'items candidats.
        4. On calcule un score de recommandation pour chaque item candidat en faisant la somme pondérée des similarités entre le candidat et les items que l'utilisateur a aimés, pondérée par les notes de l'utilisateur.
        5. Les items avec les scores les plus élevés sont recommandés, en excluant ceux que l'utilisateur a déjà vus.

### 5. `Local-Outlier-Factor-Fraud.ipynb` - Détection de Fraude avec Local Outlier Factor (LOF)

Ce notebook utilise l'algorithme Local Outlier Factor (LOF) pour la détection d'anomalies, appliqué à un scénario de détection de fraude.

- **Objectif :** Identifier des transactions ou des activités frauduleuses (anomalies) dans un jeu de données.
- **Technique (Local Outlier Factor) :**
    - **Principe :** LOF est un algorithme non supervisé qui mesure l'écart de densité locale d'un point de données par rapport à ses voisins. Les anomalies sont des points qui ont une densité de voisins significativement plus faible que leurs propres voisins, ce qui signifie qu'ils sont situés dans des régions plus isolées de l'espace des caractéristiques.
    - **Score d'Anomalie :** Le score LOF d'un point est le ratio de la densité moyenne de ses voisins à sa propre densité locale. Un score supérieur à 1 indique que le point est dans une région moins dense que ses voisins et est donc potentiellement une anomalie.
- **Évaluation :** Similaire à l'Isolation Forest, le notebook évalue le modèle avec une matrice de confusion, un rapport de classification et l'AUC-ROC. La distribution des scores LOF est également tracée pour montrer comment les anomalies sont identifiées.

### 6. `Naive-Bayes-Spam-Filter.ipynb` - Filtre Anti-Spam avec Naive Bayes

Ce notebook est un exemple classique de classification de texte qui construit un filtre anti-spam en utilisant l'algorithme Naive Bayes.

- **Objectif :** Classifier des messages texte (SMS, e-mails) comme étant du "spam" (indésirable) ou du "ham" (légitime).
- **Données :** Pour être autonome, le notebook intègre et génère un petit jeu de données de messages de type SMS, avec des exemples clairs de spam et de ham.
- **Techniques Abordées :**
    - **Vectorisation de Texte (TF-IDF) :** Le texte brut ne peut pas être utilisé directement par les modèles de machine learning. Il est d'abord converti en une représentation numérique. Le notebook utilise `TfidfVectorizer` :
        - **TF (Term Frequency) :** Mesure la fréquence d'un mot dans un document.
        - **IDF (Inverse Document Frequency) :** Diminue le poids des mots qui apparaissent très fréquemment dans l'ensemble du corpus (comme "le", "un", "de") et augmente le poids des mots plus rares et plus informatifs.
        - **N-grammes :** Le vectoriseur est configuré pour considérer non seulement les mots individuels (unigrammes) mais aussi les paires de mots (bigrammes), ce qui permet de capturer plus de contexte.
    - **Algorithme Naive Bayes Multinomial :** C'est un classifieur probabiliste basé sur le théorème de Bayes, avec une hypothèse "naïve" d'indépendance entre les caractéristiques (les mots). Il est particulièrement bien adapté et très efficace pour la classification de texte.
    - **Validation Croisée Stratifiée :** La validation croisée est utilisée pour évaluer la performance, et elle est "stratifiée" pour s'assurer que chaque partition (fold) conserve la même proportion de spam/ham que l'ensemble de données initial, ce qui est important pour les données déséquilibrées.
    - **Optimisation d'Hyperparamètres (Grid Search) :** Une recherche par grille est effectuée sur le paramètre `alpha` du classifieur Naive Bayes. `alpha` est un paramètre de lissage (lissage de Laplace) qui gère le problème des mots qui n'apparaissent pas dans l'ensemble d'entraînement, évitant ainsi des probabilités nulles.
    - **Interprétation du Modèle :** Le notebook inspecte les probabilités logarithmiques des mots apprises par le modèle pour chaque classe. En calculant la différence, il identifie les mots et n-grammes les plus indicatifs du spam (par exemple, "http", "claim", "prize", "now").

### 7. `Streaming-Drift-Anomaly-Calibration.ipynb` - Détection d'Anomalies et de Dérive en Streaming

Ce notebook avancé aborde le défi de la détection d'anomalies dans un contexte de streaming, où les données arrivent en continu et leur distribution peut changer avec le temps (concept drift).

- **Objectif :** Construire un système capable de détecter des anomalies dans un flux de données de télémétrie tout en étant conscient que la définition de ce qui est "normal" peut évoluer.
- **Données :** Un flux de données synthétique est généré pour simuler des métriques de serveur (connexions, octets, entropie). Le flux est conçu pour inclure :
    - **Une phase de référence (normale).**
    - **Une dérive (drift) :** Après un certain temps, les caractéristiques statistiques des données normales changent (par exemple, le nombre moyen de connexions augmente).
    - **Des anomalies sporadiques :** Des pics soudains et importants sont injectés à la fois avant et après la dérive.
- **Techniques Abordées :**
    - **Ingénierie de Caractéristiques en Streaming :** Puisque le modèle doit traiter les données point par point sans voir le futur, les caractéristiques sont calculées en utilisant des statistiques mobiles (rolling statistics). Pour chaque point de données, le notebook calcule son Z-score (nombre d'écarts-types par rapport à la moyenne) en se basant sur la moyenne et l'écart-type d'une fenêtre de données passées.
    - **Détection d'Anomalies (Isolation Forest) :** L'algorithme Isolation Forest est utilisé pour calculer un score d'anomalie pour chaque point de données.
    - **Calibration du Seuil :** C'est une étape cruciale. Au lieu d'utiliser un seuil fixe ou le paramètre `contamination`, le modèle est d'abord entraîné sur une fenêtre de données de référence "propres" (connues pour ne pas contenir d'anomalies). Le seuil de détection est ensuite calibré sur cette même fenêtre pour atteindre un taux de fausses alertes souhaité (par exemple, 0.3%). Cela permet de définir un seuil statistiquement significatif avant de déployer le modèle sur le reste du flux.
    - **Évaluation en Streaming :** Le modèle et le seuil calibré sont appliqués à l'ensemble du flux. Le notebook évalue ensuite la capacité du modèle à identifier correctement les anomalies injectées (précision, rappel, F1-score).
- **Analyse de la Dérive :** La visualisation des scores d'anomalie dans le temps montre que le modèle, entraîné uniquement sur la période de référence, commence à générer plus d'alertes après le début de la dérive, car la nouvelle "normalité" s'écarte de ce qu'il a appris. Cela met en évidence la nécessité de mettre en place des stratégies de recalibration périodique ou des détecteurs de dérive de concept pour maintenir les performances dans un environnement de production.
