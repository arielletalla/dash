# Dashboard d'Optimisation des Dons de Sang

## Description
Cette application Dash permet d'analyser et d'optimiser le processus de don de sang en fournissant des insights détaillés sur les donneurs, l'éligibilité, la fidélisation et l'efficacité des campagnes de sensibilisation. L'application inclut des analyses géographiques et sentimentales ainsi qu'un modèle de prédiction d'éligibilité. voici le lien ver l'application (https://dash-0zuk.onrender.com) ou (https://dash-talla.onrender.com) . j'ai deploye 2 fois pour garantir l'accessibilite
## NB : une fois l'application ouverte,il faut appuyer sur la fleche en bleu (en au a gauche) pour que le Dash se fixe bien

## Fonctionnalités
- **Analyse de l'Efficacité des Campagnes** : Visualisation des tendances et des indicateurs clés liés aux campagnes de don de sang et visualisation des données géographiques à travers des cartes dynamiques.
.
- **Conditions de Santé & Éligibilité** : Exploration des critères d'éligibilité et des raisons d'indisponibilité.
- **Profilage des Donneurs & Volontaires** : Segmentation des donneurs à l'aide de clustering et analyse démographique détaillée.
- **Fidélisation des Donneurs** : Étude des comportements de retour des donneurs et identification des facteurs influents.
- **Analyse des Sentiments** : Extraction et visualisation des opinions des donneurs sur le don de sang grâce au traitement de texte.
- **Modèle de Prédiction** : Estimation de l'éligibilité d'un donneur potentiel en fonction de divers critères avec un modèle de machine learning.
- **Caractérisation des Dons Effectifs** : Analyse approfondie des dons effectués selon divers paramètres socio-démographiques et médicaux.


## Installation
### Prérequis
- Python 3.8+
- pip
- Un environnement virtuel (recommandé)

### Installation des dépendances
```bash
pip install -r requirements.txt
```

## Lancement de l'application
```bash
python app.py
```

L'application sera accessible via : [http://127.0.0.1:8050](http://127.0.0.1:8050 , via le lien :https://dash-talla.onrender.com) ou (https://dash-0zuk.onrender.com) vous changer le compiler en local

## Fichiers et Données
- `app.py` : Script principal contenant la logique du dashboard.
- `challenge.csv` et `Copie de Challenge_dataset(1).xlsx` : Données principales des donneurs. le traitement de copie de challenge donne le fichier challence qui et un fichier propre
- `modele_prediction.pkl` et `modele_equilibre.pkl` : Modèles de prédiction d'éligibilité au don de sang.
- `Region_cameroun.geojson` et `arrondissement_douala.geojson` : Données géographiques pour l'analyse spatiale.
- `assets/` : Contient les fichiers CSS et images pour l'interface utilisateur.

## Technologies Utilisées
- **Dash & Plotly** : Interface utilisateur interactive et visualisation avancée.
- **Pandas & NumPy** : Manipulation et traitement efficace des données.
- **Scikit-learn** : Clustering (K-Means) et modèle de prédiction (Decision Tree).
- **Folium & Geopandas** : Cartographie interactive pour l'analyse spatiale.
- **NLTK & WordCloud** : Analyse de texte et visualisation des mots les plus fréquents.
- **Bootstrap & Dash Bootstrap Components** : Amélioration de l'expérience utilisateur avec des composants stylisés.

## Auteurs
Développé par DZOGOUNG TALLA Arielle et son equipe.



