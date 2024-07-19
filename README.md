# Data-Mining

Ce projet utilise Streamlit pour créer une interface utilisateur permettant d'explorer, prétraiter, visualiser, regrouper et évaluer des données à partir d'un fichier CSV chargé par l'utilisateur.

## Lien Github 
https://github.com/loanerabillard/Data-Mining


## Installation

1. Accédez au répertoire du projet où les fichiers sont stockés.
2. Installez les dépendances nécessaires en exécutant la commande suivante dans votre terminal :
   pip install -r requirements.txt

## Utilisation

1. Lancez l'application Streamlit en exécutant la commande suivante dans votre terminal :
   streamlit run app.py

2. Accédez à l'application dans votre navigateur à l'adresse fournie par Streamlit. Vous verrez une interface pour charger un fichier CSV et sélectionner le type d'analyse que vous souhaitez effectuer.

### Navigation dans l'application

#### Chargement des données
- Sur la page d'accueil de Streamlit, chargez votre fichier CSV en utilisant le bouton de chargement de fichier.
- Le fichier CSV doit être propre et bien formaté pour une analyse optimale.

#### Exploration des Données
- Sélectionnez l'option "Data Exploration" dans la barre latérale.
- Visualisez des statistiques descriptives et des graphiques de base pour comprendre la distribution et les caractéristiques de vos données.

#### Pré-traitement des Données
- Sélectionnez l'option "Data Pre-processing" dans la barre latérale.
- Nettoyez les valeurs manquantes et normalisez les données.
- Cette étape prépare les données pour les analyses ultérieures.

#### Visualisation des Données
- Sélectionnez l'option "Data Visualization" dans la barre latérale.
- Créez des visualisations avancées pour mieux comprendre vos données.
- Utilisez différents types de graphiques pour explorer les relations entre les variables.

#### Clustering
- Sélectionnez l'option "Clustering" dans la barre latérale.
- Effectuez le clustering des données en utilisant des algorithmes comme KMeans et DBSCAN.
- Visualisez les clusters et comprenez comment les données sont regroupées.

#### Évaluation
- Sélectionnez l'option "Evaluation" dans la barre latérale.
- Évaluez la performance des clusters créés en utilisant des métriques comme le score de silhouette.
- Visualisez les résultats de l'évaluation pour comprendre la qualité des clusters.
