import dash
from dash import dash_table
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html
import dash_bootstrap_components as dbc
import geopandas as gpd
import folium
from itertools import combinations
from dash.exceptions import PreventUpdate
from branca.colormap import LinearColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64
from sklearn.tree import DecisionTreeClassifier
import requests
import pickle
import nltk
from collections import Counter
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string  # Pour charger un modèle ML
# Chargement des données (remplacez par votre fichier CSV)
import os



df = pd.read_csv("challenge.csv")  # Assurez-vous que le fichier est dans le même répertoire
comments = df['Si autres raison préciser '].dropna().tolist() + df['Autre raisons,  preciser '].dropna().tolist()
   
df1 = pd.read_excel("Copie de Challenge_dataset(1).xlsx",sheet_name='2020') 


df1["Age "] = df1["Age "].astype(str)
df1['Age ']=df1["Age "].str.replace('00','0')
df1['Age ']=df1["Age "].str.replace('4&','48')
df1['Age ']=df1["Age "].str.replace('2O','20')
df1['Age ']=df1["Age "].str.replace('25673051888','0')
df1['Age ']=df1["Age "].astype(int)
df1['Age ']=df1["Age "].replace(0,df1["Age "].median())

df1["Type de donation"]=df1["Type de donation"].fillna('non specifie')
# Conversion de la date du dernier don
df['Date du dernier don. '] = pd.to_datetime(df['Si oui preciser la date du dernier don. '], errors='coerce')

# Initialisation de l'application Dash avec un thème Bootstrap moderne

options_niveau_etude = [{"label": cat, "value": cat.replace(" ", "_")} for cat in df["Niveau d'etude"].unique()]

options_genre = [{"label": cat, "value": cat.replace(" ", "_")} for cat in df['Genre '].unique()]


options_situation_matrimoniale = [{"label": cat, "value": cat.replace(" ", "_")} for cat in df['Situation Matrimoniale (SM)'].unique()]

options_deja_donne = [{"label": cat, "value": cat.replace(" ", "_")} for cat in df['A-t-il (elle) déjà donné le sang '].unique()]

options_arrondissement = [
    {"label": str(cat), "value": str(cat)}
    for cat in df["Arrondissement_final"].dropna().unique()
]

options_religion = [{"label": cat, "value": cat} for cat in df["New_Religion"].unique()]


options_categorie = [{"label": cat, "value": cat} for cat in df["Categorie_final"].unique()]



options_eligibilite = [
    {"label": "Éligible", "value": "eligible"},
    {"label": "Non éligible", "value": "non_eligible"}
]


options_indisponibilite = [
    {"label": "Est sous anti-biothérapie", "value": "Raison indisponibilité  [Est sous anti-biothérapie  ]"},
    {"label": "date de dernier Don < 3 mois", "value": "Raison indisponibilité  [date de dernier Don < 3 mois ]"},
    {"label": "IST récente (Exclu VIH, Hbs, Hcv)", "value": "Raison indisponibilité  [IST récente (Exclu VIH, Hbs, Hcv)]"}
]

# Raisons d’indisponibilité de la femme
options_indisponibilite_femme = [
    {"label": "La DDR est mauvais si <14 jours avant le don", "value": 'Raison de l’indisponibilité de la femme [La DDR est mauvais si <14 jour avant le don]'},
    {"label": "Allaitement", "value": 'Raison de l’indisponibilité de la femme [Allaitement ]'},
    {"label": "A accouché ces 6 derniers mois", "value": 'Raison de l’indisponibilité de la femme [A accoucher ces 6 derniers mois  ]'},
    {"label": "Interruption de grossesse ces 06 derniers mois", "value": 'Raison de l’indisponibilité de la femme [Interruption de grossesse  ces 06 derniers mois]'},
    {"label": "Est enceinte", "value": 'Raison de l’indisponibilité de la femme [est enceinte ]'},
]

# =========================
# Raisons de non-eligibilité totale
# =========================
options_non_eligibilite_totale = [
    {"label": "Antécédent de transfusion", "value": 'Raison de non-eligibilité totale  [Antécédent de transfusion]'},
    {"label": "Porteur(HIV,hbs,hcv)", "value": 'Raison de non-eligibilité totale  [Porteur(HIV,hbs,hcv)]'},
    {"label": "Opéré", "value": 'Raison de non-eligibilité totale  [Opéré]'},
    {"label": "Drepanocytaire", "value": 'Raison de non-eligibilité totale  [Drepanocytaire]'},
    {"label": "Diabétique", "value": 'Raison de non-eligibilité totale  [Diabétique]'},
    {"label": "Hypertendus", "value": 'Raison de non-eligibilité totale  [Hypertendus]'},
    {"label": "Asthmatiques", "value": 'Raison de non-eligibilité totale  [Asthmatiques]'},
    {"label": "Cardiaque", "value": 'Raison de non-eligibilité totale  [Cardiaque]'},
    {"label": "Tatoue", "value": 'Raison de non-eligibilité totale  [Tatoué]'},
    {"label": "Scarifié", "value": 'Raison de non-eligibilité totale  [Scarifié]'},
]

# Charger le modèle et les encodeurs

with open("modele_prediction.pkl", "rb") as file:
    data_model = pickle.load(file)

model = data_model["model"]
encoders = data_model["encoders"]
target_encoder = data_model["target_encoder"]
def faire_prediction(data, retour_proba=False):
    """
    Fait une prédiction pour un problème multiclasse avec règles métier
    
    Args:
        data: dict des données d'entrée
        retour_proba: bool pour retourner les probabilités brutes
    
    Returns:
        str: Prédiction ou dict avec probabilités si retour_proba=True
        ou "définitivement inéligible"/"temporairement inéligible" selon règles métier
    """
    try:
        # Chargement du modèle
        with open("modele_equilibre.pkl", "rb") as file:
            model_data = pickle.load(file)
        
        # 1. Vérification des règles métier en premier
        # Règle 1: Non-éligibilité (définitive)
        if data.get('non_eligible_field') == 1:  # Adaptez au nom réel de votre champ
            return "définitivement inéligible"
        
        # Règle 2: Indisponibilité (temporaire)
        if data.get('indisponible_field') == 1:  # Adaptez au nom réel de votre champ
            return "temporairement inéligible"
        
        # Si aucune règle métier ne s'applique, continuer avec la prédiction normale
        pipeline = model_data["model"]
        encoders = model_data["encoders"]
        le_target = model_data["target_encoder"]
        expected_features = model_data["feature_names"]
        
        # Création du DataFrame d'entrée
        df_input = pd.DataFrame(columns=expected_features)
        
        # Remplissage des valeurs
        for feature in expected_features:
            if feature in data:
                value = data[feature]
                if value is None or pd.isna(value):
                    if feature in model_data.get("indispo_femme_vars", []):
                        df_input[feature] = ["Non concerné"]
                    else:
                        df_input[feature] = ["Non"]
                else:
                    df_input[feature] = [str(value)]
            else:
                if feature in model_data.get("indispo_femme_vars", []):
                    df_input[feature] = ["Non concerné"]
                else:
                    df_input[feature] = ["Non"]
        
        # Encodage des variables
        for col, encoder in encoders.items():
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)
                df_input[col] = df_input[col].apply(
                    lambda x: x if x in encoder.classes_ else (
                        'Non concerné' if col in model_data.get("indispo_femme_vars", []) else 'Non'
                    )
                )
                df_input[col] = encoder.transform(df_input[col])
        
        # Prédiction
        if retour_proba:
            probas = pipeline.predict_proba(df_input[expected_features])[0]
            return {cls: proba for cls, proba in zip(le_target.classes_, probas)}
        else:
            class_idx = pipeline.predict(df_input[expected_features])[0]
            return le_target.inverse_transform([class_idx])[0]
    
    except Exception as e:
        print(f"Erreur de prédiction: {str(e)}")
        return "Erreur de prédiction"

 



# Télécharger les stopwords si ce n'est pas déjà fait
nltk.download('stopwords')

# Fonction pour nettoyer le texte (convertir en minuscules, retirer la ponctuation et les stopwords)
def clean_text(text):
    stop_words = set(stopwords.words('french'))  # Liste des mots vides en français
    cleaned_text = []
    
    # Nettoyage du texte
    for phrase in text:
        # Convertir la phrase en minuscules
        phrase = phrase.lower()
        
        # Supprimer les caractères spéciaux, les chiffres et les guillemets
        phrase = re.sub(r'[^a-zA-Zéàèùêâôîç\s]', '', phrase)  # Garde seulement les lettres, les accents et les espaces
        
        # Découper en mots
        words = phrase.split()
        
        # Retirer les stopwords
        words = [word for word in words if word not in stop_words]
        
        # Ajouter les mots nettoyés à la liste
        cleaned_text.extend(words)
    
    return cleaned_text

# Fonction pour analyser la fréquence des mots et générer un nuage de mots
def generate_wordcloud(text):
    cleaned_tokens = clean_text(text)
    word_freq = Counter(cleaned_tokens)

    most_common_words = word_freq.most_common(10)

    # Séparer les mots et leurs fréquences
    words = [word for word, freq in most_common_words]
    freqs = [freq for word, freq in most_common_words]

   
    
    
    return words, freqs

# Obtenir les mots et les fréquences
words, freqs = generate_wordcloud(comments)


# Créer un nuage de mots avec Plotly

# Créer un graphique de type barplot pour le nuage de mots
wordcloud_data = {
    "data": [
        go.Bar(
            x=words,
            y=freqs,
            text=words,
            hoverinfo='x+y+text',
            marker=dict(color='blue')  # Vous pouvez personnaliser la couleur ici
        )
    ],
    "layout": go.Layout(
        title="top 10 des mots les plus frequents ",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        hovermode="closest",
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    ),
}


external_stylesheets = [
    "/assets/bootstrap.min.css",  # Thème Bootstrap en local
    "/assets/icons/bootstrap-icons.css"  # Icônes Bootstrap en local
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)



# Charger le fichier GeoJSON
gdf = gpd.read_file("Region cameroun.geojson")

# Vérifier le système de coordonnées

# Reprojeter en WGS84 (EPSG:4326) si nécessaire
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")
    

# Charger le fichier GeoJSON
gdf2 = gpd.read_file("arrondissement_douala.geojson")
 #Vérifier et convertir les colonnes de type Timestamp en chaînes de caractères
for col in gdf2.columns:
    if pd.api.types.is_datetime64_any_dtype(gdf2[col]):  # Vérifier si la colonne contient des dates
        gdf2[col] = gdf2[col].astype(str)  # Convertir en chaîne de caractères


# Vérifier le système de coordonnées


# Reprojeter en WGS84 (EPSG:4326) si nécessaire
if gdf2.crs != "EPSG:4326":
    gdf2 = gdf2.to_crs("EPSG:4326")
   



# Style pour l'icône en arrière-plan
icon_style = {
    "position": "absolute",  # Position absolue par rapport au conteneur parent
    "top": "50%",  # Centrage vertical
    "left": "50%",  # Centrage horizontal
    "transform": "translate(-50%, -50%)",  # Ajustement pour centrer parfaitement
    "font-size": "4rem",  # Taille de l'icône
    "color": "black",  # Couleur de l'icône
    "opacity": "0.2",  # Transparence pour un effet discret
    "z-index": "1",  # Place l'icône en arrière-plan
}


# Fonction de filtrage
def filter_dataframe(df, genre, arrondissement, age_range):
    filtered_df = df.copy()
    if genre != "Tous":
        filtered_df = filtered_df[filtered_df['Genre '] == genre]
    if arrondissement != "Tous":
        filtered_df = filtered_df[filtered_df['Arrondissement de résidence '] == arrondissement]
    filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
    return filtered_df

def filter_dataframe1(df, genre, age_range):
    filtered_df = df.copy()
    if genre != "Tous":
        filtered_df = filtered_df[filtered_df['Genre '] == genre]
    filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
    return filtered_df

data=pd.read_csv("challenge.csv")  

data.columns = data.columns.str.strip()

# Sélectionner les caractéristiques
features = [
    'Age', "Niveau d'etude", 'Genre', 'Taille_imputé', 'Poids_imputé', 'Categorie_final',
    'Situation Matrimoniale (SM)', 'quartier_final', 'Arrondissement_final', 'Quartier de Résidence',
    'New_Religion', 'A-t-il (elle) déjà donné le sang', 'Taux d’hémoglobine', 'ÉLIGIBILITÉ AU DON.'
]

# Vérifier si toutes les colonnes existent dans le DataFrame
missing_columns = [col for col in features if col not in data.columns]
if missing_columns:
    raise KeyError(f"Les colonnes suivantes sont manquantes : {missing_columns}")

df2 = data[features]

# Séparer les colonnes quantitatives et catégorielles
quantitative_cols = ['Age', 'Taille_imputé', 'Poids_imputé', 'Taux d’hémoglobine']
categorical_cols = [
    "Niveau d'etude", 'Genre', 'Categorie_final', 'Situation Matrimoniale (SM)',
    'quartier_final', 'Arrondissement_final', 'Quartier de Résidence', 'New_Religion',
    'A-t-il (elle) déjà donné le sang', 'ÉLIGIBILITÉ AU DON.'
]

# Encodage des variables catégorielles (one-hot encoding)
df_encoded = pd.get_dummies(df2, columns=categorical_cols, drop_first=False)

# Gestion des valeurs manquantes (suppression des lignes avec des valeurs manquantes)
df_encoded = df_encoded.dropna()

# Normalisation des variables quantitatives (uniquement pour le clustering)
scaler = StandardScaler()
df_encoded[quantitative_cols] = scaler.fit_transform(df_encoded[quantitative_cols])

# Déterminer le nombre de clusters avec la méthode du coude
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_encoded)
    distortions.append(kmeans.inertia_)


# Appliquer K-Means avec k=3 (par exemple)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_encoded)

# Ajouter les clusters au DataFrame original (non standardisé)
df2['Cluster'] = clusters
df_encoded['Cluster'] = clusters  # Ajouter également à df_encoded pour la visualisation

# Réduction de dimensionnalité avec ACP (2 composantes principales)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_encoded.drop(columns=['Cluster']))

# Ajouter les composantes principales au DataFrame
df_encoded['PCA1'] = df_pca[:, 0]
df_encoded['PCA2'] = df_pca[:, 1]

# Visualiser les clusters dans l'espace 2D de l'ACP avec Plotly
fig_pca = px.scatter(
    df_encoded,
    x='PCA1',
    y='PCA2',
    color='Cluster',  # Utiliser la colonne 'Cluster' pour la couleur
    title='Visualisation des Clusters ',
    labels={'PCA1': 'Composante Principale 1 (PCA1)', 'PCA2': 'Composante Principale 2 (PCA2)'},
    color_continuous_scale=px.colors.qualitative.Pastel,
    hover_data=df_encoded.columns
)
fig_pca.update_layout(
    xaxis=dict(range=[-4, 5]),  # Ajouter une marge de 1
    yaxis=dict(range=[-5, 3]),
    
    
    template='plotly_white')


# Analyser les clusters sur les données non standardisées
cluster_stats = df2.groupby('Cluster')[quantitative_cols].median()

# Visualiser les caractéristiques des clusters avec Plotly
fig_cluster_stats = go.Figure()
for cluster in cluster_stats.index:
    fig_cluster_stats.add_trace(go.Bar(
        x=cluster_stats.columns,
        y=cluster_stats.loc[cluster],
        name=f'Cluster {cluster}'
    ))

fig_cluster_stats.update_layout(
    title='Caractéristiques des Clusters selon le variables quantitatives',
    xaxis_title='Variables',
    yaxis_title='Valeur Moyenne',
    barmode='group',
    template='plotly_white'
)


encoded_columns = [col_encoded for col_encoded in df_encoded.columns if col_encoded.startswith("Niveau d'etude" + '_')]
    

    

    # Convertir les colonnes encodées en valeurs numériques
df_encoded[encoded_columns] = df_encoded[encoded_columns].apply(pd.to_numeric, errors='coerce')

    # Transformer les colonnes encodées en une seule variable catégorielle
df_encoded["Niveau d'etude" + '_modalite'] = df_encoded[encoded_columns].idxmax(axis=1)  # Trouver la modalité dominante

    # Créer un tableau croisé entre le cluster et toutes les modalités de la variable catégorielle
cluster_cross = pd.crosstab(df_encoded['Cluster'], df_encoded["Niveau d'etude" + '_modalite'], normalize='index') * 100

    # S'assurer que toutes les modalités sont bien présentes
all_modalities = ["Niveau d'etude" + '_' + modality for modality in df2["Niveau d'etude"].unique()]
for modality in all_modalities:
    if modality not in cluster_cross.columns:
        cluster_cross[modality] = 0  # Ajouter la modalité manquante avec une fréquence de 0%

    # Trier les colonnes pour respecter l'ordre des modalités
cluster_cross = cluster_cross[all_modalities]


    # Préparer les données pour Plotly
cluster_cross_melted = cluster_cross.reset_index().melt(id_vars='Cluster', value_name='Pourcentage', var_name='Modalité')
cluster_cross_melted['Modalité'] = cluster_cross_melted['Modalité'].str.split('_').str[-1]
    # Créer un graphique interactif avec Plotly
fig_clus_etude = px.bar(cluster_cross_melted, 
                 x='Cluster', 
                 y='Pourcentage', 
                 color='Modalité', 
                 title=f"Distribution du niveau d'etude par cluster",
                 labels={'Pourcentage': 'Pourcentage (%)', 'Cluster': 'Cluster'},
                 color_discrete_sequence=px.colors.qualitative.Pastel)  # Utilisation d'une palette de couleurs agréable
    
    # Personnaliser le layout
fig_clus_etude.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Pourcentage (%)',
        legend_title='Modalité',
        barmode='stack',
        template='plotly_white',  # Utilisation d'un template clair et moderne
        hovermode='x unified'  # Affichage des informations au survol
    )


    # Filtrer les colonnes encodées correspondant à la variable catégorielle
encoded_columns = [col_encoded for col_encoded in df_encoded.columns if col_encoded.startswith('Genre' + '_')]

   
    
    # Convertir les colonnes encodées en valeurs numériques
df_encoded[encoded_columns] = df_encoded[encoded_columns].apply(pd.to_numeric, errors='coerce')

    # Transformer les colonnes encodées en une seule variable catégorielle
df_encoded['Genre' + '_modalite'] = df_encoded[encoded_columns].idxmax(axis=1)  # Trouver la modalité dominante

    # Créer un tableau croisé entre le cluster et toutes les modalités de la variable catégorielle
cluster_cross = pd.crosstab(df_encoded['Cluster'], df_encoded['Genre' + '_modalite'], normalize='index') * 100

    # S'assurer que toutes les modalités sont bien présentes
all_modalities = ['Genre' + '_' + modality for modality in df2['Genre'].unique()]
for modality in all_modalities:
    if modality not in cluster_cross.columns:
        cluster_cross[modality] = 0  # Ajouter la modalité manquante avec une fréquence de 0%

    # Trier les colonnes pour respecter l'ordre des modalités
cluster_cross = cluster_cross[all_modalities]


    # Préparer les données pour Plotly
cluster_cross_melted = cluster_cross.reset_index().melt(id_vars='Cluster', value_name='Pourcentage', var_name='Modalité')
cluster_cross_melted['Modalité'] = cluster_cross_melted['Modalité'].str.split('_').str[-1]
    # Créer un graphique interactif avec Plotly
fig_clus_genre = px.bar(cluster_cross_melted, 
                 x='Cluster', 
                 y='Pourcentage', 
                 color='Modalité', 
                 title=f"Distribution du genre par cluster",
                 labels={'Pourcentage': 'Pourcentage (%)', 'Cluster': 'Cluster'},
                 color_discrete_sequence=px.colors.qualitative.Pastel)  # Utilisation d'une palette de couleurs agréable
    
    # Personnaliser le layout
fig_clus_genre.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Pourcentage (%)',
        legend_title='Modalité',
        barmode='stack',
        template='plotly_white',  # Utilisation d'un template clair et moderne
        hovermode='x unified'  # Affichage des informations au survol
    )


    # Filtrer les colonnes encodées correspondant à la variable catégorielle
encoded_columns = [col_encoded for col_encoded in df_encoded.columns if col_encoded.startswith('Situation Matrimoniale (SM)' + '_')]
    

   
  
    # Convertir les colonnes encodées en valeurs numériques
df_encoded[encoded_columns] = df_encoded[encoded_columns].apply(pd.to_numeric, errors='coerce')

    # Transformer les colonnes encodées en une seule variable catégorielle
df_encoded['Situation Matrimoniale (SM)' + '_modalite'] = df_encoded[encoded_columns].idxmax(axis=1)  # Trouver la modalité dominante

    # Créer un tableau croisé entre le cluster et toutes les modalités de la variable catégorielle
cluster_cross = pd.crosstab(df_encoded['Cluster'], df_encoded['Situation Matrimoniale (SM)' + '_modalite'], normalize='index') * 100

    # S'assurer que toutes les modalités sont bien présentes
all_modalities = ['Situation Matrimoniale (SM)' + '_' + modality for modality in df2['Situation Matrimoniale (SM)'].unique()]
for modality in all_modalities:
    if modality not in cluster_cross.columns:
        cluster_cross[modality] = 0  # Ajouter la modalité manquante avec une fréquence de 0%

    # Trier les colonnes pour respecter l'ordre des modalités
cluster_cross = cluster_cross[all_modalities]


    # Préparer les données pour Plotly
cluster_cross_melted = cluster_cross.reset_index().melt(id_vars='Cluster', value_name='Pourcentage', var_name='Modalité')
cluster_cross_melted['Modalité'] = cluster_cross_melted['Modalité'].str.split('_').str[-1]
    # Créer un graphique interactif avec Plotly
fig_clus_stat = px.bar(cluster_cross_melted, 
                 x='Cluster', 
                 y='Pourcentage', 
                 color='Modalité', 
                 title=f"Distribution de la Situation Matrimoniale (SM) par cluster",
                 labels={'Pourcentage': 'Pourcentage (%)', 'Cluster': 'Cluster'},
                 color_discrete_sequence=px.colors.qualitative.Pastel)  # Utilisation d'une palette de couleurs agréable
    
    # Personnaliser le layout
fig_clus_stat.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Pourcentage (%)',
        legend_title='Modalité',
        barmode='stack',
        template='plotly_white',  # Utilisation d'un template clair et moderne
        hovermode='x unified'  # Affichage des informations au survol
    )
    

    # Filtrer les colonnes encodées correspondant à la variable catégorielle
encoded_columns = [col_encoded for col_encoded in df_encoded.columns if col_encoded.startswith('Arrondissement_final' + '_')]
    

   
  
    # Convertir les colonnes encodées en valeurs numériques
df_encoded[encoded_columns] = df_encoded[encoded_columns].apply(pd.to_numeric, errors='coerce')

    # Transformer les colonnes encodées en une seule variable catégorielle
df_encoded['Arrondissement_final' + '_modalite'] = df_encoded[encoded_columns].idxmax(axis=1)  # Trouver la modalité dominante

    # Créer un tableau croisé entre le cluster et toutes les modalités de la variable catégorielle
cluster_cross = pd.crosstab(df_encoded['Cluster'], df_encoded['Arrondissement_final' + '_modalite'], normalize='index') * 100

    # S'assurer que toutes les modalités sont bien présentes
#all_modalities = ['Arrondissement_final' + '_' + modality for modality in df['Arrondissement_final'].unique()]
all_modalities = ['Arrondissement_final' + '_' + str(modality) for modality in df2['Arrondissement_final'].unique() if pd.notna(modality)]
for modality in all_modalities:
    if modality not in cluster_cross.columns:
        cluster_cross[modality] = 0  # Ajouter la modalité manquante avec une fréquence de 0%

    # Trier les colonnes pour respecter l'ordre des modalités
cluster_cross = cluster_cross[all_modalities]


    # Préparer les données pour Plotly
cluster_cross_melted = cluster_cross.reset_index().melt(id_vars='Cluster', value_name='Pourcentage', var_name='Modalité')
cluster_cross_melted['Modalité'] = cluster_cross_melted['Modalité'].str.split('_').str[-1]


    # Créer un graphique interactif avec Plotly
fig_clus_ar= px.bar(cluster_cross_melted, 
                 x='Cluster', 
                 y='Pourcentage', 
                 color='Modalité', 
                 title=f"Distribution des Arrondissements par cluster",
                 labels={'Pourcentage': 'Pourcentage (%)', 'Cluster': 'Cluster'},
                 color_discrete_sequence=px.colors.qualitative.Pastel)  # Utilisation d'une palette de couleurs agréable
    
    # Personnaliser le layout
fig_clus_ar.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Pourcentage (%)',
        legend_title='Modalité',
        barmode='stack',
        template='plotly_white',  # Utilisation d'un template clair et moderne
        hovermode='x unified'  # Affichage des informations au survol
    )
    


    # Filtrer les colonnes encodées correspondant à la variable catégorielle
encoded_columns = [col_encoded for col_encoded in df_encoded.columns if col_encoded.startswith('ÉLIGIBILITÉ AU DON.' + '_')]
    

 

    # Convertir les colonnes encodées en valeurs numériques
df_encoded[encoded_columns] = df_encoded[encoded_columns].apply(pd.to_numeric, errors='coerce')

    # Transformer les colonnes encodées en une seule variable catégorielle
df_encoded['ÉLIGIBILITÉ AU DON.' + '_modalite'] = df_encoded[encoded_columns].idxmax(axis=1)  # Trouver la modalité dominante

    # Créer un tableau croisé entre le cluster et toutes les modalités de la variable catégorielle
cluster_cross = pd.crosstab(df_encoded['Cluster'], df_encoded['ÉLIGIBILITÉ AU DON.'+ '_modalite'], normalize='index') * 100

    # S'assurer que toutes les modalités sont bien présentes
all_modalities = ['ÉLIGIBILITÉ AU DON.' + '_' + modality for modality in df2['ÉLIGIBILITÉ AU DON.'].unique()]
for modality in all_modalities:
    if modality not in cluster_cross.columns:
        cluster_cross[modality] = 0  # Ajouter la modalité manquante avec une fréquence de 0%

    # Trier les colonnes pour respecter l'ordre des modalités
cluster_cross = cluster_cross[all_modalities]


    # Préparer les données pour Plotly
cluster_cross_melted = cluster_cross.reset_index().melt(id_vars='Cluster', value_name='Pourcentage', var_name='Modalité')
cluster_cross_melted['Modalité'] = cluster_cross_melted['Modalité'].str.split('_').str[-1]
    # Créer un graphique interactif avec Plotly
fig_clus_eli = px.bar(cluster_cross_melted, 
                 x='Cluster', 
                 y='Pourcentage', 
                 color='Modalité', 
                 title=f"Distribution de l'eligibilite par cluster",
                 labels={'Pourcentage': 'Pourcentage (%)', 'Cluster': 'Cluster'},
                 color_discrete_sequence=px.colors.qualitative.Pastel)  # Utilisation d'une palette de couleurs agréable
    
    # Personnaliser le layout
fig_clus_eli.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Pourcentage (%)',
        legend_title='Modalité',
        barmode='stack',
        template='plotly_white',  # Utilisation d'un template clair et moderne
        hovermode='x unified'  # Affichage des informations au survol
    )
    
    # Filtrer les colonnes encodées correspondant à la variable catégorielle
encoded_columns = [col_encoded for col_encoded in df_encoded.columns if col_encoded.startswith('New_Religion' + '_')]
    

    
    

    # Convertir les colonnes encodées en valeurs numériques
df_encoded[encoded_columns] = df_encoded[encoded_columns].apply(pd.to_numeric, errors='coerce')

    # Transformer les colonnes encodées en une seule variable catégorielle
df_encoded['New_Religion' + '_modalite'] = df_encoded[encoded_columns].idxmax(axis=1)  # Trouver la modalité dominante

    # Créer un tableau croisé entre le cluster et toutes les modalités de la variable catégorielle
cluster_cross = pd.crosstab(df_encoded['Cluster'], df_encoded['New_Religion' + '_modalite'], normalize='index') * 100

    # S'assurer que toutes les modalités sont bien présentes
all_modalities = ['New_Religion' + '_' + modality for modality in df2['New_Religion'].unique()]
for modality in all_modalities:
    if modality not in cluster_cross.columns:
        cluster_cross[modality] = 0  # Ajouter la modalité manquante avec une fréquence de 0%

    # Trier les colonnes pour respecter l'ordre des modalités
cluster_cross = cluster_cross[all_modalities]


    # Préparer les données pour Plotly
cluster_cross_melted = cluster_cross.reset_index().melt(id_vars='Cluster', value_name='Pourcentage', var_name='Modalité')
cluster_cross_melted['Modalité'] = cluster_cross_melted['Modalité'].str.split('_').str[-1]
    
    # Créer un graphique interactif avec Plotly
fig_clus_rel = px.bar(cluster_cross_melted, 
                 x='Cluster', 
                 y='Pourcentage', 
                 color='Modalité', 
                 title=f"Distribution de la religion par cluster",
                 labels={'Pourcentage': 'Pourcentage (%)', 'Cluster': 'Cluster'},
                 color_discrete_sequence=px.colors.qualitative.Pastel)  # Utilisation d'une palette de couleurs agréable
    
    # Personnaliser le layout
fig_clus_rel.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Pourcentage (%)',
        legend_title='Modalité',
        barmode='stack',
        template='plotly_white',  # Utilisation d'un template clair et moderne
        hovermode='x unified'  # Affichage des informations au survol
    )


    #### profil du donneur ideale

# Charger les données
data = pd.read_csv('challenge.csv') 
dat = pd.read_csv('challenge.csv') # Remplacer par votre fichier
data.columns = data.columns.str.strip()  # Nettoyer les noms de colonnes

# Préparer les données
features = [
    'Age', "Niveau d'etude", 'Genre', 'Taille_imputé', 'Poids_imputé', 'Categorie_final',
    'Situation Matrimoniale (SM)', 'quartier_final', 'Arrondissement_final', 'Quartier de Résidence',
    'New_Religion', 'A-t-il (elle) déjà donné le sang', 'Taux d’hémoglobine', 'ÉLIGIBILITÉ AU DON.'
]
df2 = data[features]

# Séparer les colonnes quantitatives et catégorielles
quantitative_cols = ['Age', 'Taille_imputé', 'Poids_imputé', 'Taux d’hémoglobine']
categorical_cols = [
    "Niveau d'etude", 'Genre', 'Categorie_final', 'Situation Matrimoniale (SM)',
    'quartier_final', 'Arrondissement_final', 'Quartier de Résidence', 'New_Religion',
    'A-t-il (elle) déjà donné le sang', 'ÉLIGIBILITÉ AU DON.'
]

# Encodage des variables catégorielles
df_encoded = pd.get_dummies(df2, columns=categorical_cols, drop_first=False)

# Gestion des valeurs manquantes
df_encoded = df_encoded.dropna()

# Appliquer KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_encoded)

# Ajouter les clusters au DataFrame
df2['Cluster'] = clusters
df_encoded['Cluster'] = clusters

# Calculer les moyennes des caractéristiques pour chaque cluster
cluster_profiles = df_encoded.groupby('Cluster').mean()

# Sélectionner le cluster idéal (par exemple, Cluster 0)
ideal_cluster = cluster_profiles.loc[1]  # Vous pouvez ajuster selon le cluster souhaité

# Créer un profil lisible pour le donneur idéal
donneur_ideal = {
    "Âge moyen": f"{round(ideal_cluster['Age'], 1)} ans",
    "Poids moyen": f"{round(ideal_cluster['Poids_imputé'], 1)} kg",
    "Taille moyenne": f"{round(ideal_cluster['Taille_imputé'], 1)} m",
    "Taux d'Hémoglobine moyen": f"{round(ideal_cluster['Taux d’hémoglobine'], 1)} g/dL",
    "Éligibilité au Don": (
        "Définitivement Non-eligible" if ideal_cluster['ÉLIGIBILITÉ AU DON._Définitivement non-eligible'] > 0.5 else
        "Temporairement Non-eligible" if ideal_cluster['ÉLIGIBILITÉ AU DON._Temporairement Non-eligible'] > 0.5 else
        "Eligible" if ideal_cluster['ÉLIGIBILITÉ AU DON._Eligible'] > 0.5 else
        "Non défini"
    ),
    "Niveau d'étude": "Universitaire" if ideal_cluster["Niveau d'etude_Universitaire"] >= 0.4 else
                      ("Secondaire" if ideal_cluster["Niveau d'etude_Secondaire"] >= 0.4 else
                       ("Primaire" if ideal_cluster["Niveau d'etude_Primaire"] >=0.4 else "Aucun")),
    "Genre": "Homme" if ideal_cluster['Genre_Homme'] > 0.5 else "Femme",
    "Statut matrimonial": "Marié (e)" if ideal_cluster['Situation Matrimoniale (SM)_Marié (e)'] > 0.5 else
                         ("Célibataire" if ideal_cluster['Situation Matrimoniale (SM)_Célibataire'] > 0.5 else
                          "Divorcé(e)" if ideal_cluster['Situation Matrimoniale (SM)_Divorcé(e)'] > 0.5 else "Veuf (veuve)")
}










# Style pour la barre latérale
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "250px",
    "padding": "20px",
    "background-color": "#ff69b4",  # Rose élégant
    "transition": "all 0.3s",
    "overflow-y": "auto"  # Permettre le défilement vertical
}
SIDEBAR_HIDDEN = {
    "width": "50px",
    "padding": "10px",
    "transition": "all 0.3s",
    "overflow-y": "auto"  # Permettre le défilement vertical
}

# Style pour le contenu principal
CONTENT_STYLE = {
    "margin": "0",
    "width": "100%",
    "height": "calc(100vh - 40px)",  # Ajuster la hauteur pour éviter le débordement
    "padding": "20px",
    "transition": "all 0.3s",
    "overflow-y": "auto",  # Permettre le défilement vertical
}

# Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Container([
        # En-tête avec photo
        dbc.Row([
            dbc.Col([
                html.Div(
                    style={
                        'background-image': 'url("/assets/sang.PNG")',  # Chemin vers l'image
                        'background-size': 'cover',  # Ajuste l'image à la taille de l'en-tête
                        'background-position': 'center',  # Centre l'image
                        'height': '200px',  # Hauteur de l'en-tête
                        'display': 'flex',
                        'align-items': 'center',
                        'justify-content': 'center',
                        'color': 'blue',  # Couleur du texte
                        'font-size': '24px',  # Taille du texte
                        'font-weight': 'bold',  # Gras
                    },
                    children=[
                        #html.H1("Tableau de bord pour l'optimisation des dons de sang", className="text-center text-white")
                    ]
                )
            ], width=12)
        ]),

        # Barre latérale et contenu principal
        dbc.Row([
            dbc.Col([
                html.Div(id="sidebar", children=[
                    dbc.Button("⬅", id="toggle-sidebar", color="light", className="mb-3"),
                    html.Div([
                        html.H3("DASHBOARD", className="text-center text-white"),
                        html.Hr(),
                        dbc.Tabs([
                    dbc.Tab(label="Analyse de l’Efficacité des Campagnes", tab_id="tab-1"),
                    dbc.Tab(label="Conditions de Santé & Éligibilité", tab_id="tab-2"),
                    dbc.Tab(label="Profilage des donneurs & volontaires", tab_id="tab-3"),
                    
                    dbc.Tab(label="Fidélisation des Donneurs", tab_id="tab-7"),
                    dbc.Tab(label="Analyse des sentiments", tab_id="tab-4"),
                    dbc.Tab(label="Modèle de Prédiction", tab_id="tab-5"),
                    dbc.Tab(label="Caractérisation des dons effectif", tab_id="tab-6"),
                            

                        ], id="tabs"),
                        
                        html.Hr(),
                        # Filtres ajoutés ici
                        html.Div([
                            html.Label("Genre", className="text-white"),
                            dcc.Dropdown(
                                id='genre-filter',
                                options=[{'label': 'Tous', 'value': 'Tous'}] + 
                                        [{'label': genre, 'value': genre} for genre in df['Genre '].unique()],
                                value='Tous',
                                clearable=False
                            ),
                            html.Label("Arrondissement", className="text-white mt-3"),
                            dcc.Dropdown(
                                id='arrondissement-filter',
                                options=[{'label': 'Tous', 'value': 'Tous'}] + 
                                        [{'label': arr, 'value': arr} for arr in df['New_Arrondissement'].unique()],
                                value='Tous',
                                clearable=False
                            ),
                            html.Label("Âge", className="text-white mt-3"),
                            dcc.RangeSlider(
                                id='age-slider',
                                min=df['Age'].min(),
                                max=df['Age'].max(),
                                step=1,
                                marks={i: str(i) for i in range(int(df['Age'].min()), int(df['Age'].max()) + 1, 10)},
                                value=[df['Age'].min(), df['Age'].max()]
                            ),
                            html.Br(),
                            dbc.Button("Appliquer les filtres", id="apply-filters", color="light", className="mt-2")
                        ])
                    ], id="sidebar-content"),
                ], style=SIDEBAR_STYLE)
            ], width=2, id="sidebar-col"),

            dbc.Col([
                
                html.Div(id="tabs-content", style=CONTENT_STYLE),
                html.Div([
    dbc.Row([
        dbc.Col([html.H2(id='fig-a')], width=6),
        dbc.Col([html.H2(id='fig-b')], width=6),
        dbc.Col([html.H2(id='fig-c')], width=6),
        dbc.Col([html.H2(id='fig-d')], width=6),
        dbc.Col([dcc.Graph(id='fig-genre')], width=6),
        dbc.Col([dcc.Graph(id='fig-matrimoniale'), ], width=6),
        dbc.Col([dcc.Graph(id='fig-religion')], width=6),
        dbc.Col([dcc.Graph(id='fig-profession')], width=6),
        dbc.Col([dcc.Graph(id='fig-etude')], width=6),
        dbc.Col([dcc.Graph(id='fig-age')], width=6),
        dbc.Col([html.Iframe(id='fig-map')], width=6),
        dbc.Col([html.Iframe(id='fig-map1')], width=6),
        dbc.Col([dcc.Graph(id='fig-quartier')], width=6),
        
        dbc.Col([dcc.Graph(id='fig-mois')], width=6),
        dbc.Col([dcc.Graph(id='fig-periode')], width=6),
        dbc.Col([dcc.Graph(id='fig-semaine')], width=6)

    ], id="tab-1-content", style={"display": "none"}),  # Content for tab 1, initially hidden

    dbc.Row([


        dbc.Col([dcc.Graph(id='fig-eligibilite')], width=6),
        
    dbc.Col([dcc.Graph(id='fig-bi')], width=6),
    dbc.Col([dcc.Graph(id='fig-hemo')], width=6),
   
    dbc.Col([dcc.Graph(id='fig-dte-don')], width=6),
    dbc.Col([dcc.Graph(id='fig-ist')], width=6),
    #dbc.Col([dcc.Graph(id='fig-treemap')], width=6),
    dbc.Col([dcc.Graph(id='fig-heat')], width=6),
    dbc.Col([dcc.Graph(id='fig-ddr')], width=6),
    dbc.Col([dcc.Graph(id='fig-al')], width=6),
    dbc.Col([dcc.Graph(id='fig-acc')], width=6),
    dbc.Col([dcc.Graph(id='fig-intgro')], width=6),
    dbc.Col([dcc.Graph(id='fig-enc')], width=6),
    dbc.Col([dcc.Graph(id='fig-trans')], width=6),
    dbc.Col([dcc.Graph(id='fig-Ope')], width=6),
    dbc.Col([dcc.Graph(id='fig-Drepa')], width=6),
    dbc.Col([dcc.Graph(id='fig-Hyper')], width=6),

    dbc.Col([dcc.Graph(id='fig-Asthma')], width=6),
    dbc.Col([dcc.Graph(id='fig-Card')], width=6),
    dbc.Col([dcc.Graph(id='fig-Tat')], width=6),
    dbc.Col([dcc.Graph(id='fig-Scar')], width=6)
    

    
    ], id="tab-2-content", style={"display": "none"}),  # Content for tab 2, initially hidden

    dbc.Row([
        dbc.Col([dcc.Graph(id='fig-ecart')], width=6),
        dbc.Col([dcc.Graph(id='fig-type')], width=6),
        dbc.Col([dcc.Graph(id='fig-genre-f')], width=6),
        dbc.Col([dcc.Graph(id='fig-etude-f')], width=6),

        dbc.Col([dcc.Graph(id='fig-statut-f')], width=6),
        dbc.Col([dcc.Graph(id='fig-tranche-f')], width=6),
        dbc.Col([dcc.Graph(id='fig-religion-f')], width=6),

        dbc.Col([dcc.Graph(id='fig-arrond-f')], width=6)


        
    ], id="tab-7-content", style={"display": "none"}), 
     
     dbc.Row([
        
        dbc.Col([dcc.Graph(id='fig-sentiment')], width=6),
        dbc.Col([dcc.Graph(id='fig-label')], width=6),
        dbc.Col([dcc.Graph(id='fig-sent-tem')], width=6)



        
    ], id="tab-4-content", style={"display": "none"}),  # Content for tab 3, initially hidden
])
             


                
                
               
                
                
                
            ], width=12, id="content-col")
        ])
    ], fluid=True),
    # Définition des graphiques
    
    
])









# Callbacks pour afficher/masquer la barre latérale
@app.callback(
    Output("sidebar", "style"),
    Output("sidebar-content", "style"),
    Output("tabs-content", "style"),
    Input("toggle-sidebar", "n_clicks"),
    prevent_initial_call=True)
   
def toggle_sidebar(n):
    if n and n % 2 == 1:
        return SIDEBAR_HIDDEN, {"display": "none"}, {"margin-left": "50px", "padding": "20px", "transition": "all 0.3s", "overflow-y": "auto", "height": "calc(100vh - 40px)"}
    return SIDEBAR_STYLE, {"display": "block"}, {"margin-left": "250px", "padding": "20px", "transition": "all 0.3s", "overflow-y": "auto", "height": "calc(100vh - 40px)"}


# Indicateurs
total_donneurs = len(df1)
portion_femmes = round(100 * len(df1[df1["Sexe"] == "F"]) / total_donneurs, 2)
median_age = df1["Age "].median()

fig_donneurs = go.Figure()

fig_donneurs.add_trace(go.Indicator(
    mode="number",
    value=total_donneurs,
    title={"text": "<b>Nombre total de donneurs</b>", "font": {"size": 20, "color": "white"}},
    number={"font": {"size": 40, "color": "white"}},
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_donneurs.update_layout(
    paper_bgcolor="royalblue",  # Fond bleu élégant
    plot_bgcolor="royalblue",
    margin=dict(l=20, r=20, t=40, b=20),
    height=180,  # Taille réduite
    width=300,  # Largeur ajustée
    font=dict(color="white"),
    shapes=[  # Ajout d'un glow effect
        dict(
            type="rect",
            x0=0, x1=1, y0=0, y1=1,
            xref="paper", yref="paper",
            fillcolor="rgba(255, 255, 255, 0.1)",  # Effet de glow
            layer="below",
            line=dict(width=0),
        )
    ]
)

# URL d'une icône transparente (symbole féminin par exemple)

fig_femmes = go.Figure(go.Indicator(
    mode="gauge+number",
    value=portion_femmes,
   
    number={"font": {"size": 36, "color": "#FF007F"},
        "suffix": " %"  # Ajout du pourcentage après la valeur
    },  # Taille du texte réduite
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
        'bar': {'color': "#FF007F", 'thickness': 0.3},
        'bgcolor': "white",
        'borderwidth': 0,
        'steps': [{'range': [0, 100], 'color': "rgba(255, 182, 193, 0.4)"}],
        'shape': "angular",
    }
))



fig_femmes.update_layout(
     title={"text": "<b>Nombre total de donneurs</b>", "font": {"size": 20, "color": "#FF007F"}},
    paper_bgcolor="whitesmoke",
    plot_bgcolor="whitesmoke",
    height=170,  # Taille réduite
    width=300,  # Taille plus compacte
    margin=dict(l=10, r=10, t=30, b=10),
    font=dict(color="black"),
)



fig_age = go.Figure(go.Indicator(
    mode="number",
    value=median_age,
    title={"text": "<b>Âge médian</b>", "font": {"size": 22, "color": "white"}},
    number={"font": {"size": 48, "color": "white"}},  # Orange brillant
    gauge={
        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
        "bar": {"color": "#FFA500"},
        "steps": [
            {"range": [0, 50], "color": "rgba(255, 165, 0, 0.4)"},
            {"range": [50, 100], "color": "rgba(255, 69, 0, 0.4)"}  # Dégradé orange-rouge
        ],
        "borderwidth": 0,
    }
))

fig_age.update_layout(
    paper_bgcolor="lightseagreen",
    plot_bgcolor="lightseagreen",
    height=180,  # Taille réduite
    width=300,
    margin=dict(l=20, r=20, t=40, b=20),
    font=dict(color="white"),
)



# Créer la figure du pie chart
sang_counts = df1['Groupe Sanguin ABO / Rhesus '].value_counts().sort_values(ascending=True)

fig_sang= px.histogram(df1, x='Groupe Sanguin ABO / Rhesus ', title='Répartition par groupe sanguin',color='Groupe Sanguin ABO / Rhesus ',category_orders={'Groupe Sanguin ABO / Rhesus ': sang_counts.index.tolist()} )
    
# Créer la figure du pie chart
phenotype_counts = df1['Phenotype '].value_counts().sort_values(ascending=True)

fig_phenotype= px.histogram(df1, x='Phenotype ', title='Répartition par groupe phenotype',color='Phenotype ',category_orders={'Phenotype ': phenotype_counts.index.tolist()} )


values_g = df1['Type de donation'].value_counts().reset_index()
values_g.columns = ['Type de donation', 'Count']
fig_donnation = px.pie(
        values_g,
        names='Type de donation',
        values='Count',
        title='Répartition par Type de don',
        color='Type de donation',
        color_discrete_sequence=px.colors.qualitative.Plotly,  # Palette de couleurs modernes
        hole=0.4  # Créer un donut chart
    )

    # Améliorations esthétiques
fig_donnation.update_traces(
        textinfo='percent+label', 
        textfont_size=14,
        marker=dict(line=dict(width=2, color='white')))







# Callback pour mettre à jour le contenu en fonction des onglets et des filtres
@app.callback(
    Output("tabs-content", "children"),
    
    [Input("tabs", "active_tab"),
     ],
    [State("genre-filter", "value"),
     State("arrondissement-filter", "value"),
     State("age-slider", "value")],
    prevent_initial_call=True
)


def render_content(active_tab, genre, arrondissement, age_range):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    else:
        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Filtrage des données
    filtered_df = filter_dataframe(df, genre, arrondissement, age_range)

    


    if active_tab== "tab-1":
        # Caracterisation des donneurs
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Indicateurs clés ", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        
                                        html.H4("Donneurs", className="card-title"),
                                        html.H2(id='fig-a', className="card-text"),

                                    ], className="text-center", style={
                                        "background-color": "rgba(144, 238, 144, 0.7)",
                                        "border-radius": "10px",
                                        "padding": "20px",
                                    })
                                ], className="text-center")
                            ], width=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Age median", className="card-title"),
                                        html.H2(id='fig-b', className="card-text"),
                                    ])
                                ], className="text-center",style={
                                        "background-color": "rgba(147, 112, 219, 0.7)",
                                        "border-radius": "10px"
                                        
                                    }),
                            ], width=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Taux Premier don", className="card-title"),
                                        html.H2(id='fig-c', className="card-text"),
                                    ])
                                ], className="text-center",style={
                                        "background-color": "rgba(255, 192, 203, 0.7)",
                                        "border-radius": "10px"
                                        
                                    }),
                            ], width=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Taux d'etrangers", className="card-title"),
                                        html.H2(id='fig-d', className="card-text"),
                                    ])
                                ], className="text-center",style={
                                        "background-color": "rgba(173, 216, 230, 0.7)",
                                        "border-radius": "10px"
                                       
                                    }),
                            ], width=3),
                        ])
                    ])
                ])
            ], width=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Caracterisation sociaux demographiques des donneurs", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([dcc.Graph(id='fig-genre')], width=6),
                            dbc.Col([dcc.Graph(id='fig-matrimoniale')], width=6)
                            
                            
                        ])
                    ])
                ])
            ], width=12),


            dbc.Col([
                dbc.Card([
                    #dbc.CardHeader("Graphiques supplémentaires", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            #dbc.Col([dcc.Graph(id='fig-religion')], width=12)
                            dbc.Col([dcc.Graph(id='fig-religion')], width=12)
                            
                            
                        ])
                    ])
                ])
            ], width=12),

            dbc.Col([
                dbc.Card([
                    #dbc.CardHeader("Graphiques supplémentaires", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            #dbc.Col([dcc.Graph(id='fig-religion')], width=12)
                            dbc.Col([dcc.Graph(id='fig-profession')], width=12)
                            
                            
                        ])
                    ])
                ])
            ], width=12),

            dbc.Col([
                dbc.Card([
                    #dbc.CardHeader("Graphiques supplémentaires", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([dcc.Graph(id='fig-etude')], width=6),
                            dbc.Col([dcc.Graph(id='fig-age')], width=6)
                            
                            
                        ])
                    ])
                ])
            ], width=12),

            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analyse geographique(cartographie)", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Iframe(
                                    id='fig-map',
                                    style={"width": "100%", "height": "300px", "border": "none"}
                                )
                            ], width=6),
                            dbc.Col([
                                html.Iframe(
                                    id='fig-map1',
                                    style={"width": "100%", "height": '300px', "border": "none"}
                                )
                            ], width=6)


        
                            #dbc.Col([dcc.Graph(id='fig-age')], width=6)
                            
                            
                        ])
                    ])
                ])
            ], width=12),
            dbc.Col([
                dbc.Card([
                    #dbc.CardHeader("Graphiques supplémentaires", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([dcc.Graph(id='fig-quartier')], width=12),
                            #dbc.Col([dcc.Graph(id='fig-age')], width=6)
                            
                            
                        ])
                    ])
                ])
            ], width=12),
            dbc.Col([
                 dbc.Card([
                    dbc.CardHeader("Analyse temporelle", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([

                        



                            dbc.Col([dcc.Graph(id='fig-mois')], width=12),
                            dbc.Col([dcc.Graph(id='fig-periode')], width=12),

                            #dbc.Col([dcc.Graph(id='fig-semaine')], width=12)
                        ]),
                        dbc.Row([

                        



                            

                            dbc.Col([dcc.Graph(id='fig-semaine')], width=12)
                        ])


                        ])
                 ])
                    

            ])

        ])
    elif active_tab == "tab-6":
        # caracterisation des donneurs de sang
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Indicateurs cles", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                             dbc.Col(dbc.Card([
            dbc.CardBody(dcc.Graph(figure=fig_donneurs))
        ] ), width=4),
        
        dbc.Col(
    dbc.Card([
        
        dbc.CardBody(dcc.Graph(figure=fig_femmes, config={"displayModeBar": False}))  
    ]), 
    width=4
),
        
        dbc.Col(dbc.Card([
            dbc.CardBody(dcc.Graph(figure=fig_age ))
        ] ), width=4
                   
                    ),
   

        dbc.Col(dbc.Card([
            dbc.CardBody(dcc.Graph(figure=fig_donnation))
        ]),width=8 ),
    ], className="mb-4"),
                           
                        
                    ])
                ])
            ],width=12),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Graphique caracterisant les dons effectifs", className="bg-secondary text-white"),
                    dbc.CardBody([
                        
                            
                                 dbc.Col([dcc.Graph(figure=fig_sang)], width=12),
                                 
                            
                ])
            
        
      
        
       
                           
                        
                    
                ])
            ],width=12),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Indicateur et graphique caracterisant les donneurs", className="bg-secondary text-white"),
                    dbc.CardBody([
                        
                            
                                 dbc.Col([dcc.Graph(figure=fig_phenotype)], width=12),
                                 
                            
                ])
            
        
      
        
       
                           
                        
                    
                ])
            ],width=12)


        ])
    elif active_tab== "tab-2":
        # Illigibilite au don de sang
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                                    dbc.CardHeader("Distribution des donneurs selon l'eligibilite au don", className="bg-secondary text-white"),
                                        dbc.CardBody([


                dbc.Col([dcc.Graph(id='fig-eligibilite')], width=12)
                                        ])
                ])
                
            ],width=12),
                   dbc.Col([
                            
                                dbc.Card([
                                    dbc.CardHeader("Taux d'indisponibilite par raison d'indisponibilite", className="bg-secondary text-white"),
                                        dbc.CardBody([
                                            dbc.Row([

                                            dbc.Col([  dcc.Graph(id='fig-bi')], width=3),
                                            dbc.Col([  dcc.Graph(id='fig-hemo')], width=3),

                                            dbc.Col([dcc.Graph(id='fig-dte-don')], width=3),
                                            dbc.Col([dcc.Graph(id='fig-ist')], width=3),
                                            #dbc.Col([dcc.Graph(id='fig-heat')], width=3),
                                            #dbc.Col([dcc.Graph(id='fig-treemap')], width=12)

                                            ])
                                
                            
                                    ])
                                ])
                    ]),


            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Taux d'indisponibilite par raison d'indisponibilite de la femmes", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([dcc.Graph(id='fig-ddr')], width=3),
                            dbc.Col([dcc.Graph(id='fig-al')], width=3),
                            dbc.Col([dcc.Graph(id='fig-acc')], width=3),
                            dbc.Col([dcc.Graph(id='fig-intgro')], width=3),
                            #dbc.Col([dcc.Graph(id='fig-heat')], width=3),
                            dbc.Col([dcc.Graph(id='fig-enc')], width=3)
                            
                            ])
                    ])
                ])
            ],width=12),
             dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Taux de non eligibilite par raison", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([

                                dbc.Col([dcc.Graph(id='fig-trans')], width=3),
                                dbc.Col([dcc.Graph(id='fig-Ope')], width=3),
                                dbc.Col([dcc.Graph(id='fig-Drepa')], width=3),
                                dbc.Col([dcc.Graph(id='fig-Hyper')], width=3),


                                dbc.Col([dcc.Graph(id='fig-Asthma')], width=3),
                                dbc.Col([dcc.Graph(id='fig-Card')], width=3),
                                dbc.Col([dcc.Graph(id='fig-Tat')], width=3),
                                dbc.Col([dcc.Graph(id='fig-Scar')], width=3)
    
    

                            
                            
                            ])
                    ])
                ])
            ],width=12),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Association entre les raisons d'indisponibilite", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Col([dcc.Graph(id='fig-heat')], width=12),

                    ])
                ])


            ],width=12)


        ])
    elif active_tab == "tab-3":
        # Efficacité des Campagnes
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Segmentation des volontaires", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dbc.Row([
                       dbc.Col([ dcc.Graph(figure=fig_pca,style={'height': '400px'} )],width=12) ,
                        ]),
                        dbc.Row([
                            html.Div([
                            html.H1("Profil du Donneur Idéal basé sur le Clustering", style={'textAlign': 'center', 'color': '#5D3FD3'}),

                            # Tableau interactif avec style
                            dash_table.DataTable(
                                id='tableau-profil',
                                columns=[
                                    {'name': 'Attribut', 'id': 'attribut'},
                                    {'name': 'Valeur', 'id': 'valeur'}
                                ],
                                data=[{'attribut': key, 'valeur': value} for key, value in donneur_ideal.items()],
                                style_table={'height': '400px', 'overflowY': 'auto', 'border': '2px solid #5D3FD3'},
                                style_header={'backgroundColor': '#5D3FD3', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
                                style_cell={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#f9f9f9', 'fontSize': '16px'},
                                style_data={'backgroundColor': '#f4f6f9', 'borderBottom': '1px solid #ddd'},
                                style_data_conditional=[
                                    {
                                        'if': {'column_id': 'valeur', 'filter_query': '{valeur} contains "Non-eligible"'},
                                        'backgroundColor': '#FF6F61', 'color': 'white'
                                    },
                                    {
                                        'if': {'column_id': 'valeur', 'filter_query': '{valeur} contains "Eligible"'},
                                        'backgroundColor': '#4CAF50', 'color': 'white'
                                    },
                                    {
                                        'if': {'column_id': 'valeur', 'filter_query': '{valeur} contains "Homme"'},
                                        'backgroundColor': '#B0E0E6', 'color': 'black'
                                    },
                                    {
                                        'if': {'column_id': 'valeur', 'filter_query': '{valeur} contains "Femme"'},
                                        'backgroundColor': '#F8C8DC', 'color': 'black'
                                    }
                                ]
                            ),
                            ])

                        ]),
                         dbc.Row([
                             dbc.Col([
                             dbc.Card([
                                dbc.CardHeader("Profil de chaque cluster", className="bg-secondary text-white"),
                                dbc.CardBody([
                                    dbc.Row([

                            dbc.Col([ dcc.Graph(figure=fig_cluster_stats,style={'height': '400px'} )],width=12),
                            dbc.Col([dcc.Graph(figure=fig_clus_genre,style={'height': '400px'} )],width=6),
                            dbc.Col([dcc.Graph(figure=fig_clus_etude,style={'height': '400px'} )],width=6),
                            dbc.Col([dcc.Graph(figure=fig_clus_stat,style={'height': '400px'} )],width=6),
                            dbc.Col([dcc.Graph(figure=fig_clus_etude,style={'height': '400px'} )],width=6),
                            dbc.Col([dcc.Graph(figure=fig_clus_rel,style={'height': '400px'} )],width=6),
                            dbc.Col([dcc.Graph(figure=fig_clus_ar,style={'height': '400px'} )],width=6),
                            dbc.Col([dcc.Graph(figure=fig_clus_eli,style={'height': '400px'} )],width=12),
                            
                        ])
                             ])
                             ])
                             ])
                         
                        
                        

                        
                    ])
                ])
            ])
            ])

        ])
    elif active_tab == "tab-5":
        # Modèle de Prédiction (exemple placeholder)
        return dbc.Row([
            dbc.Col([
                dbc.Card([
        dbc.CardHeader("Caractristique sociaux demograpiques", className="bg-light fw-semibold"),
        dbc.CardBody([
            # 1ère ligne : Âge + Niveau d'étude + Genre
            dbc.Row([
                dbc.Col([
                    dbc.Label("Âge", className="fw-bold"),
                    dbc.Input(id="input_age", type="number", placeholder="Ex: 30", min=0),
                ], width=4),
                dbc.Col([
                    dbc.Label("Niveau d'étude", className="fw-bold"),
                    dcc.Dropdown(
                        id="dropdown_niveau_etude",
                        options=options_niveau_etude,
                        placeholder="Sélectionnez un niveau d'étude",
                        clearable=True
                    )
                ], width=4),
                dbc.Col([
                    dbc.Label("Genre", className="fw-bold"),
                    dcc.Dropdown(
                        id="dropdown_genre",
                        options=options_genre,
                        placeholder="Sélectionnez un genre",
                        clearable=True
                    )
                ], width=4)
            ], className="mb-3"),

            # 2ème ligne : Situation Matrimoniale + A-t-il déjà donné le sang ? + Taux d’hémoglobine
            dbc.Row([
                dbc.Col([
                    dbc.Label("Situation Matrimoniale (SM)", className="fw-bold"),
                    dcc.Dropdown(
                        id="dropdown_situation_mat",
                        options=options_situation_matrimoniale,
                        placeholder="Sélectionnez une situation",
                        clearable=True
                    )
                ], width=4),
                dbc.Col([
                    dbc.Label("A-t-il (elle) déjà donné le sang ?", className="fw-bold"),
                    dcc.Dropdown(
                        id="dropdown_deja_donne",
                        options=options_deja_donne,
                        placeholder="Oui / Non",
                        clearable=True
                    )
                ], width=4),
                dbc.Col([
                    dbc.Label("Taux d’hémoglobine (g/dL)", className="fw-bold"),
                    dbc.Input(id="input_taux_hb", type="number", placeholder="Ex: 12.5", min=0, step=0.1),
                ], width=4)
            ], className="mb-3"),

            # 3ème ligne : ÉLIGIBILITÉ AU DON + Arrondissement + Religion
            dbc.Row([
                
                dbc.Col([
                    dbc.Label("Arrondissement", className="fw-bold"),
                    dcc.Dropdown(
                        id="dropdown_arrondissement",
                        options=options_arrondissement,
                        placeholder="Sélectionnez un arrondissement",
                        clearable=True
                    )
                ], width=4),
                dbc.Col([
                    dbc.Label("Religion", className="fw-bold"),
                    dcc.Dropdown(
                        id="dropdown_religion",
                        options=options_religion,
                        placeholder="Sélectionnez une religion",
                        clearable=True
                    )
                ], width=4),
                    dbc.Col([
                    dbc.Label("profession", className="fw-bold"),
                    dcc.Dropdown(
                        id="dropdown_categorie",
                        options=options_categorie,
                        placeholder="Ex: Étudiant, Employé, etc.",
                        clearable=True
                    )
                ], width=4)
            ], className="mb-3"),

            
        ])
    ], className="shadow mb-4"),
    dbc.Col([
    dbc.Card([
        dbc.CardHeader("Situation de santé", className="bg-light fw-semibold"),
        dbc.CardBody([
            # Section 1 : Raisons générales d'indisponibilité
            dbc.Row([
                dbc.Col([
                    dbc.Label("Cochez les éléments qui vous concernent :", className="fw-bold mb-3"),
                    dbc.Checklist(
                        id="checklist_indisponibilite",
                        options=options_indisponibilite,
                        value=[],
                        switch=True,
                        className="mb-4"
                    )
                ], width=12)
            ]),
            
            # Section 2 : Raisons spécifiques aux femmes
            dbc.Row([
                dbc.Col([
                    dbc.Label("Pour les donneuses uniquement :", className="fw-bold mb-3 text-primary"),
                    *[
                        dbc.Row([
                            dbc.Col(dbc.Label(opt["label"], width=4, className="pe-0"), width=4),
                            dbc.Col(
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Oui", "value": f"{opt['value']}_oui"},
                                        {"label": "Non", "value": f"{opt['value']}_non"},
                                        {"label": "Non concerné", "value": f"{opt['value']}_nc"}
                                    ],
                                    value=f"{opt['value']}_nc",
                                    inline=True,
                                    id=f"radio_{opt['value']}",
                                    className="btn-group",
                                    inputClassName="btn-check",
                                    labelClassName="btn btn-outline-primary",
                                    labelCheckedClassName="active"
                                ),
                                width=8
                            )
                        ], className="g-2 mb-3 align-items-center")
                        for opt in [
                            {"label": "DDR <14 jours", "value": "ddr_recente"},
                            {"label": "Allaitement", "value": "allaitement"},
                            {"label": "Grossesse en cours", "value": "enceinte"},
                            {"label": "Accouchement <6 mois", "value": "accouchement_recent"}
                        ]
                    ]
                ], width=12)
            ])
        ])
    ], className="shadow-sm mb-4"),
    
    # Bouton de validation principal
    dbc.Button(
        "Valider toutes les réponses",
        id="btn-validate-main",
        color="primary",
        className="w-100",
        size="lg"
    )
], width=6),


 

        dbc.Row([
            dbc.Col([
                dbc.Label("Cochez toute raison applicable :", className="fw-bold"),
                dbc.Checklist(
                    id="checklist_non_eligibilite",
                    options=options_non_eligibilite_totale,  # Liste des options à définir
                    value=[],
                    inline=False
                )
            ], width=12),

            dbc.Col([
                dbc.Button("Prédire", id="btn_predire", color="primary", className="me-2 mt-4"),
            ], width=2),

            dbc.Col([
                html.Div(id="prediction_result", className="text-center text-success fw-bold mt-4")
            ], width=6),
        ], className="mb-5"),
    ])
], className="shadow mb-4")

    # =========================
    # Validation finale
    # =========================
    
    elif active_tab == "tab-7":
        # Fidelisation
        return dbc.Row([
            dbc.Col([
                dbc.Card([
        dbc.CardHeader("Caracteristion des donneurs relativement a l'ecart entre deux dons consecutifs", className="bg-light fw-semibold"),
        dbc.CardBody([
            # 1ère ligne : Âge + Niveau d'étude + Genre
             dbc.Row([

                            dbc.Col([dcc.Graph(id='fig-ecart')], width=6),
                            dbc.Col([dcc.Graph(id='fig-type')], width=6),
                            
                                        
                        ])
            
        ])

                ])

            ], width=12),
             dbc.Col([
                dbc.Card([
        dbc.CardHeader("", className="bg-light fw-semibold"),
        dbc.CardBody([
            # 1ère ligne : Âge + Niveau d'étude + Genre
             dbc.Row([

                            dbc.Col([dcc.Graph(id='fig-genre-f')], width=12),
                            dbc.Col([dcc.Graph(id='fig-etude-f')], width=12),

                            dbc.Col([dcc.Graph(id='fig-statut-f')], width=12),
                            dbc.Col([dcc.Graph(id='fig-tranche-f')], width=12),
                            dbc.Col([dcc.Graph(id='fig-religion-f')], width=12),

                            dbc.Col([dcc.Graph(id='fig-arrond-f')], width=12)
                                        
                        ])
            
        ])

                ])

            ], width=12),


        ])
    
    elif active_tab == "tab-4":
        # Fidelisation
        return dbc.Row([
            dbc.Col([
                dbc.Card([
        dbc.CardHeader("Analyse des mots frequents et des sentiments sur les observations des donneurs", className="bg-light fw-semibold"),
        dbc.CardBody([
            # 1ère ligne : Âge + Niveau d'étude + Genre
             dbc.Row([
                 
                 dbc.Col([
                      dcc.Graph(
                                
                                figure=wordcloud_data  # Utilisez wordcloud_data ici
                            )
                 ]),
                        # Graphique de sentiment
                        dbc.Col([
                            dcc.Graph(id='fig-sentiment')
                        ], width=12),

               

            ]),
        ])
                ])
            ]),
             dbc.Col([
                dbc.Card([
        dbc.CardHeader("", className="bg-light fw-semibold"),
        dbc.CardBody([
            # 1ère ligne : Âge + Niveau d'étude + Genre
             dbc.Row([
                
                dbc.Col([dcc.Graph(id='fig-label')], width=12),
                dbc.Col([dcc.Graph(id='fig-sent-tem')], width=12)

                                        
                        ])
            
        ])

                ])

            ], width=12),


        ])








# Callback pour mettre à jour les graphiques
@app.callback(
        
    [
    Output('fig-a', 'children'),  # Total donneurs
        Output('fig-b', 'children'),  # Taux d'éligibilité
        Output('fig-c', 'children'),  # Taux premier don
        Output('fig-d', 'children'),  # Taux étrangers
        
    Output('fig-genre', 'figure'),
    Output('fig-matrimoniale', 'figure'),
    Output('fig-religion', 'figure'),
    Output('fig-profession', 'figure'),
    Output('fig-etude', 'figure'),
    Output('fig-age', 'figure'),
    Output('fig-map', 'srcDoc'),
    Output('fig-map1', 'srcDoc'),
    Output('fig-quartier', 'figure'),
    
    Output('fig-mois', 'figure'),
    Output('fig-periode', 'figure'),
    Output('fig-semaine', 'figure'),
    

      
     ]
    ,
    [Input("tabs", "active_tab"),  # 🔥 Ajout de cet input pour déclencher la callback
     Input("apply-filters", "n_clicks")],  # 🔥 Ajout d'un bouton pour filtrer
    
   
    [State('genre-filter', 'value'),
     State('arrondissement-filter', 'value'),
     State('age-slider', 'value')]
    
)
def update_graphs(active_tab, n_clicks, genre, arrondissement, age_range):
    

    filtered_df = filter_dataframe(df, genre, arrondissement, age_range)
    

    total = len(filtered_df)
    mediane=filtered_df['Age'].median()

   # eligible = round(100 * len(filtered_df[filtered_df['ÉLIGIBILITÉ AU DON.'] == 'Eligible']) / total, 2) if total > 0 else 0

    premier_don = round(100 * len(filtered_df[filtered_df['A-t-il (elle) déjà donné le sang '] == 'Non']) / total, 2) if total > 0 else 0

    etrangers = round(100 * len(filtered_df[filtered_df['Nationalité '] != 'Camerounaise']) / total, 2) if total > 0 else 0
    


    profession_counts = filtered_df['Categorie_final'].value_counts().sort_values(ascending=True)
    etude_counts = filtered_df["Niveau d'etude"].value_counts().sort_values(ascending=True)
    religion_counts = filtered_df['New_Religion'].value_counts().sort_values(ascending=True)

    #profession_counts = filtered_df['Categorie_final'].value_counts().sort_values(ascending=True)

    values_g = filtered_df['Genre '].value_counts().reset_index()
    values_g.columns = ['Genre ', 'Count']
    fig_genre = px.pie(
            values_g,
            names='Genre ',
            values='Count',
            title='Répartition par Genre',
            color='Genre ',
            color_discrete_sequence=px.colors.qualitative.Plotly,  # Palette de couleurs modernes
            hole=0.4  # Créer un donut chart
        )

        # Améliorations esthétiques
    fig_genre.update_traces(
            textinfo='percent+label', 
            textfont_size=14,
            marker=dict(line=dict(width=2, color='white')))  # Bordure blanche autour des segments

    fig_genre.update_layout(
            title_font_size=24,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)  # Positionnement de la légende
        )
        
        #fig_matrimoniale = px.histogram(filtered_df, x='Situation Matrimoniale (SM)', title='Répartition par situation matrimoniale', color='Situation Matrimoniale (SM)')
        
    fig_profession = px.histogram(filtered_df, x='Categorie_final', title='Répartition par profession',color='Categorie_final',category_orders={"Categorie_final": profession_counts.index.tolist()} )
        
        
    fig_etude = px.histogram(filtered_df, x="Niveau d'etude", title='Répartition par niveau d’étude', color="Niveau d'etude",category_orders={"Niveau d'etude": etude_counts.index.tolist()} )
    
    
    fig_religion = px.histogram(filtered_df, x='New_Religion', title='Répartition par religion', color='New_Religion',category_orders={"New_Religion": religion_counts.index.tolist()} )
        # Calculer la valeur de chaque situation matrimoniale
    values = filtered_df['Situation Matrimoniale (SM)'].value_counts().reset_index()
    values.columns = ['Situation Matrimoniale (SM)', 'Count']

    # Créer la figure du pie chart
    fig_matrimoniale = px.pie(
            values,
            names='Situation Matrimoniale (SM)',
            values='Count',
            title='Répartition par situation matrimoniale',
            color='Situation Matrimoniale (SM)',
            color_discrete_sequence=px.colors.qualitative.Set3,  # Changer la palette de couleurs
            template='plotly',  # Utiliser le template Plotly pour améliorer le style
            hole=0.3  # Créer un graphique de type donut (optionnel)
        )
    


        # Améliorations esthétiques
    fig_matrimoniale.update_traces(textinfo='percent+label', textfont_size=14)
    fig_matrimoniale.update_layout(title_font_size=20)

            # Créer la figure de l'histogramme des âges
    fig_age = px.histogram(
            filtered_df,
            x='Age',
            title='Répartition des âges',
            nbins=10,  # Nombre de bacs (bins) pour l'histogramme
            color_discrete_sequence=px.colors.qualitative.Set2,  # Choisir une palette de couleurs
        )

        # Améliorations esthétiques de l'histogramme
    fig_age.update_layout(
            xaxis_title='Âge',
            yaxis_title='Nombre de personnes',
            title_font_size=20,
            xaxis=dict(showgrid=True),  # Afficher les lignes de grille
            yaxis=dict(showgrid=True),  # Afficher les lignes de grille
        )
    

    fig_age.update_traces(marker=dict(line=dict(width=1, color='black')))  # Bordure des barres

    grouped = filtered_df.groupby('Region')['ID']
    result = grouped.count().reset_index(name='Count')
    result_dict = result.set_index('Region')['Count'].to_dict()



            # Ajouter les effectifs comme une nouvelle colonne dans le GeoDataFrame
    gdf['EFFECTIFS'] = gdf['Nom_Région'].map(result_dict).fillna(0)

        # Créer une carte centrée sur le Cameroun
    m = folium.Map(location=[7.3697, 12.3547], zoom_start=6, control_scale=True)

        # Définir une échelle de couleurs pour les effectifs
    colormap = LinearColormap(
            colors=["yellow", "orange", "red"],
            vmin=gdf["EFFECTIFS"].min(),
            vmax=gdf["EFFECTIFS"].max(),
            caption="Effectifs par région"
        )

        # Ajouter les régions du Cameroun avec une couleur basée sur les effectifs
    folium.GeoJson(
            gdf,
            name="Effectifs par région",
            style_function=lambda x: {
                "fillColor": colormap(x["properties"]["EFFECTIFS"]),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["Nom_Région", "EFFECTIFS"],
                aliases=["Région", "Effectifs"]
            ),
        ).add_to(m)

        # Ajouter l'échelle de couleurs et le contrôle des couches
    colormap.add_to(m)
    folium.LayerControl().add_to(m)

        # Stocker la carte dans une variable
    fig_map = m
    fig_map_html = fig_map._repr_html_()



    grouped = filtered_df.groupby('Arrondissement_final')['ID']
    result = grouped.count().reset_index(name='Count')
    result_dict = result.set_index('Arrondissement_final')['Count'].to_dict()



            # Ajouter les effectifs comme une nouvelle colonne dans le GeoDataFrame
    gdf2['EFFECTIFS'] = gdf2["ADM3_FR"].map(result_dict).fillna(0)

        # Créer une carte centrée sur Douala
    douala_coords = [4.0511, 9.7679]
    m1 = folium.Map(location=douala_coords, zoom_start=12, control_scale=True)

        # Définir une échelle de couleurs pour les effectifs
    colormap = LinearColormap(
            colors=["yellow", "orange", "red"],  # Définir les couleurs de l'échelle
            vmin=gdf2["EFFECTIFS"].min(),       # Valeur minimale des effectifs
            vmax=gdf2["EFFECTIFS"].max(),       # Valeur maximale des effectifs
            caption="Effectifs par commune de Douala"  # Légende de l'échelle de couleurs
        )

        # Ajouter les régions du Cameroun avec une couleur basée sur les effectifs
    folium.GeoJson(
            gdf2,
            name="Effectifs par commune",
            style_function=lambda x: {
                "fillColor": colormap(x["properties"]["EFFECTIFS"]),  # Couleur en fonction des effectifs
                "color": "black",  # Couleur des bordures
                "weight": 1,       # Épaisseur des bordures
                "fillOpacity": 0.7  # Opacité de remplissage
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["ADM3_FR", "EFFECTIFS"],  # Champs à afficher dans le tooltip
                aliases=["Arrondissement", "Effectifs"]  # Alias pour les champs
            ),
        ).add_to(m1)

        # Ajouter l'échelle de couleurs à la carte
    colormap.add_to(m1)

        # Ajouter un contrôle des couches pour activer/désactiver les couches
    folium.LayerControl().add_to(m1)

        # Stocker la carte dans une variable
    fig_map1 = m1
    fig_map1_html = fig_map1._repr_html_() 

        # Retourner la carte avec les autres figures
    values_g = filtered_df['quartier_final'].value_counts().reset_index()
    values_g.columns = ['quartier_final', 'Count']
    values_g=pd.DataFrame(values_g)
    values_g=values_g.sort_values(by='Count', ascending=False)
                                                          
                                                        
        # Sélectionner les 15 premiers quartiers (ou moins s'il y en a moins de 15)
    top_quartiers = values_g=values_g.head(20)
        
    
        
        # Créer l'histogramme avec Plotly
    fig_quartier = px.bar(

            top_quartiers,
            x="quartier_final",
            y="Count",
            title=f"Top {len(top_quartiers)} quartiers avec les plus grands effectifs à",
            labels={"Effectifs": "Effectifs", "Quartier": "Quartier"},
            color_discrete_sequence=["blue"]  # Couleur des barres

        )
        
        # Personnaliser l'affichage
    fig_quartier.update_layout(

            xaxis_title="Quartier",
            yaxis_title="Effectifs",
            showlegend=False

        )
  


    don_par_mois = filtered_df.groupby('Mois').size().reset_index(name='Nombre de dons')
    fig_mois = px.area(
    don_par_mois, 
    x='Mois', 
    y='Nombre de dons',
    title="📊 Évolution du Nombre de Dons par Mois",
    labels={'Mois': "Date (Annee-Mois)", 'Nombre de dons': "Nombre de dons"},
    line_shape="spline",  # Courbe fluide
    color_discrete_sequence=["deepskyblue"],  # Couleur douce
    template="plotly_white",  # Fond clair
)

    # Personnalisation du layout
    fig_mois.update_layout(
        xaxis_title="📅 Date (Année-Mois)",
        yaxis_title="🩸 Nombre de dons",
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14),
        xaxis=dict(showgrid=False),  # Pas de grille verticale
        yaxis=dict(showgrid=True, gridcolor="lightgray"),  # Grille légère pour lecture facile
        margin=dict(l=40, r=40, t=60, b=40),  # Ajustement des marges
        paper_bgcolor="#F8F9FA",  # Fond léger
        plot_bgcolor="#F8F9FA",  # Fond graphique léger
    )
    

    # Animation pour un effet fluide
    fig_mois.update_traces(fill='tozeroy', mode='lines+markers', marker=dict(size=6, color="deepskyblue"))

        # Compter le nombre de dons par jour du mois
    don_par_jour = filtered_df.groupby('periode_du_mois').size().reset_index(name='Nombre de dons')

    # Création du diagramme en barres
    # 🎨 Nouvelle palette de couleurs (Vert clair, Rose, Violet clair)
    couleurs_douces = ["#98FB98", "#FFB6C1", "#D8BFD8"]  # Vert clair, Rose clair, Violet clair

    # Création du diagramme à barres
    fig_periode = px.bar(
        don_par_jour, 
        x='periode_du_mois', 
        y='Nombre de dons',
        title="📊 Fréquence des Dons de Sang par Jour du Mois",
        text_auto=True,  # Affichage des valeurs sur les barres
        color='periode_du_mois',  # Colorer en fonction de la période du mois
        color_discrete_sequence=couleurs_douces,  # ✅ Nouvelle palette de couleurs
        labels={'periode_du_mois': "Jour du Mois", 'Nombre de dons': "Nombre de dons"},
        template="plotly_white",  # ✅ Fond blanc
    )

    # Personnalisation du layout
    fig_periode.update_layout(
        xaxis_title="📅 Jour du Mois",
        yaxis_title="🩸 Nombre de dons",
        font=dict(family="Arial, sans-serif", size=14),
        xaxis=dict(
            tickmode="linear",  # Affichage des jours 1, 2, 3, ... 31
            tick0=1,
            dtick=1,
            showgrid=False
        ),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),  # Grille légère
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="white",  # ✅ Fond totalement blanc
        plot_bgcolor="white",  
        coloraxis_showscale=False,  # ✅ Masquer la légende des couleurs (inutile ici)
    )

    # Augmenter la taille des labels pour une meilleure visibilité
    fig_periode.update_traces(
        textfont_size=14,  # ✅ Valeurs plus grandes
        textfont_family="Arial Black",  # ✅ Police plus lisible et élégante
        textfont_color="black",  # ✅ Texte en noir pour un bon contraste
    )
        

    # Compter le nombre de dons par jour de la semaine
    don_par_jour = filtered_df.groupby('Jour de la semaine').size().reset_index(name='Nombre de dons')

    # Ordonner les jours de la semaine correctement
    jours_ordre = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    don_par_jour['Jour de la semaine'] = pd.Categorical(don_par_jour['Jour de la semaine'], categories=jours_ordre, ordered=True)
    don_par_jour = don_par_jour.sort_values("Jour de la semaine")

        # 🌈 Palette avec des couleurs plus douces et vives
    couleurs_vives = ["#87CEEB", "#FFB6C1", "#98FB98", "#FFA07A", "#FFD700", "#FF69B4", "#40E0D0"]

    fig_semaine = px.treemap(
    don_par_jour,
    path=['Jour de la semaine'],
    values='Nombre de dons',
    title="🩸 Nombre de Dons par Jour de la Semaine",
    color='Nombre de dons',
    color_continuous_scale=couleurs_vives,  # 🎨 Couleurs douces et vives
    labels={'Nombre de dons': "Nombre de dons"},
)

    # Amélioration du layout
    fig_semaine.update_layout(
        font=dict(family="Arial, sans-serif", size=16),  # ✅ Police plus grande et plus lisible
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",  # ✅ Arrière-plan totalement blanc
    )

    # Amélioration des tracés
    fig_semaine.update_traces(
        marker=dict(cornerradius=10),  # ✅ Bords arrondis pour un effet moderne
        textinfo="label+value",  # ✅ Afficher le jour + nombre de dons
        textfont_size=18,  # ✅ Taille du texte augmentée pour une meilleure visibilité
        textfont_family="Arial Black",  # ✅ Police plus élégante et professionnelle
        textfont_color="black",  # ✅ Texte en noir pour un bon contraste
    )





        

        
        
    


        

        
    return str(total),  f"{mediane}",  f"{premier_don}%",  f"{etrangers}%",  fig_genre,fig_matrimoniale, fig_religion, fig_profession, fig_etude, fig_age, fig_map_html, fig_map1_html,fig_quartier,fig_mois,fig_periode,fig_semaine
    


# Couleurs professionnelles harmonisées
COLORS = {
    'primary': '#2E86AB',  # Bleu profond
    'secondary': '#A23B72',  # Violet-rouge
    'tertiary': '#F18F01',  # Orange vif
    'quaternary': '#C73E1D',  # Rouge brique
    'quinary': '#3B1F2B',  # Violet foncé
    'senary': '#48A9A6',  # Turquoise
    'light': '#F7F7F7',  # Gris très clair
    'dark': '#333333'   # Gris foncé
}

# Fonction pour créer un donut chart standardisé
def create_donut_chart(value, title, main_color):
    fig = go.Figure(go.Pie(
        values=[value, 100 - value],
        marker=dict(
            colors=[main_color, 'lightgray'],
            line=dict(color='white', width=3)
        ),
        hole=0.7,
        textinfo='none',
        hoverinfo='none',
        showlegend=False,
        rotation=90
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, family="Arial, sans-serif", color="#2a3f5f"),
            x=0.5, xanchor='center', y=0.95, yanchor='top'
        ),
        annotations=[
            dict(
                text=f"<b>{value:.2f}%</b>",
                x=0.5, y=0.5,
                font=dict(size=28, color=main_color, family="Arial, sans-serif"),
                showarrow=False
            )
        ],
        margin=dict(t=60, b=40, l=40, r=40),
        width=250, height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_traces(pull=[0.05 if value > 50 else 0, 0])
    
    # Badge d'objectif (30% comme seuil)
    if value > 30:
        fig.add_annotation(
            text=f"<span style='background-color: #ffcccc; padding: 2px 8px; border-radius: 10px'>+{(value-30):.2f}%</span>",
            x=0.5, y=0.3, showarrow=False, font=dict(size=12, color="darkred")
        )
    else:
        fig.add_annotation(
            text="<span style='background-color: #ccffcc; padding: 2px 8px; border-radius: 10px'>✓ Objectif</span>",
            x=0.5, y=0.3, showarrow=False, font=dict(size=12, color="darkgreen")
        )
    
    return fig

# Palette de couleurs web standard avec guillemets doubles
COLORS = {
    "blue": "#3498DB",     # Bleu vif
    "red": "#E74C3C",      # Rouge vif
    "green": "#2ECC71",    # Vert vif
    "orange": "#F39C12",   # Orange vif
    "purple": "#9B59B6",   # Violet
    "teal": "#1ABC9C",     # Turquoise
    "navy": "#34495E",     # Bleu marine
    "gray": "#95A5A6",     # Gris
    "lightgray": "#ECF0F1",# Gris clair
    "white": "#FFFFFF"     # Blanc
}
couleur = {
    "bleu": "#3498DB",
    "rouge": "#E74C3C", 
    "vert": "#2ECC71",
    "orange": "#F39C12",
    "violet": "#9B59B6",
    "turquoise": "#1ABC9C",
    "bleu_fonce": "#34495E",
    "gris": "#95A5A6",
    "gris_clair": "#ECF0F1",
    "blanc": "#FFFFFF",
    "vert_pale": "#6A8D73"  # Pour les scarifications
}

# Fonction pour créer les indicateurs avec guillemets doubles
def create_indicator(value, title, color):
    return {
        "mode": "number+gauge",
        "value": value,
        "domain": {"x": [0, 1], "y": [0, 1]},
        "title": {
            "text": f"<b>{title}</b>", 
            "font": {"size": 14, "color": COLORS["navy"], "family": "Arial"}
        },
        "gauge": {
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["gray"]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": COLORS["lightgray"]},
                {"range": [50, 100], "color": COLORS["gray"]}
            ],
            "borderwidth": 1.5,
            "bordercolor": COLORS["gray"]
        },
        "number": {
            "font": {"size": 18, "color": color, "family": "Arial"},
            "suffix": "%"
        }
    }

   
# Callbacks pour afficher/masquer la barre latérale
@app.callback(
    [
     Output('fig-eligibilite', 'figure'),
    Output('fig-bi', 'figure'),
    Output('fig-hemo', 'figure'),
    Output('fig-dte-don', 'figure'),
    Output('fig-ist', 'figure'),
    #Output('fig-treemap', 'figure'),
    Output('fig-heat', 'figure'),
    Output('fig-ddr', 'figure'),
    Output('fig-al', 'figure'),
    Output('fig-acc', 'figure'),
    Output('fig-intgro', 'figure'),
    Output('fig-enc', 'figure'),
    
    Output('fig-trans', 'figure'),
    Output('fig-Ope', 'figure'),
    Output('fig-Drepa', 'figure'),
    Output('fig-Hyper', 'figure'),
    Output('fig-Asthma', 'figure'),
    Output('fig-Card', 'figure'),
    Output('fig-Tat', 'figure'),
    Output('fig-Scar', 'figure')


      
     ]
    ,
    [Input("tabs", "active_tab"),  # 🔥 Ajout de cet input pour déclencher la callback
     Input("apply-filters", "n_clicks")],  # 🔥 Ajout d'un bouton pour filtrer
    
    [State('genre-filter', 'value'),
     State('arrondissement-filter', 'value'),
     State('age-slider', 'value')]
    
)
def update_graphs2(active_tab, n_clicks, genre, arrondissement, age_range):
    filtered_df = filter_dataframe(df, genre, arrondissement, age_range)
    data1=filtered_df[filtered_df["ÉLIGIBILITÉ AU DON."]=='Temporairement Non-eligible']
    fig_biothe = go.Figure(go.Indicator(
        mode="number+gauge",
        value=100*round(len(data1[data1['Raison indisponibilité  [Est sous anti-biothérapie  ]']=='Oui'])/len(data1),2),
        
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": 'anti-biothérapie', "font": {"size": 16, "color": "#2C3E50"}},  
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "lightgrey"},
            "bar": {"color": "#FFB6C1"},  
            "steps": [
                {"range": [0, 50], "color": "#AED6F1"},  
                {"range": [50, 100], "color": "#85C1E9"}  
            ],
            "borderwidth": 1.2,  
            "bordercolor": "#D5DBDB"  
        },
        number={"font": {"size": 16, "color": "#2C3E50"},
                "suffix": "%"
                }  
    ))

    fig_biothe.update_layout(
            margin={"l": 5, "r": 5, "t": 50, "b": 5},  # Ajustement pour bien afficher le titre
            height=110,  
            paper_bgcolor="white"
        )


    fig_hemo = go.Figure(go.Indicator(
        mode="number+gauge",
        value=100*round(len(data1[data1['Raison indisponibilité  [Taux d’hémoglobine bas ]']=='Oui'])/len(data1),2),

        
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": 'Taux d’hémoglobine bas', "font": {"size": 16, "color": "#2C3E50"}},  
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "lightgrey"},
            "bar": {"color": "#90EE90"},  
            "steps": [
                {"range": [0, 50], "color": "#AED6F1"},  
                {"range": [50, 100], "color":"#85C1E9"}  
            ],
            "borderwidth": 1.2,  
            "bordercolor": "#D5DBDB"  
        },
        number={"font": {"size": 16, "color": "#2C3E50"},
                "suffix": "%"
                }  
    ))

    fig_hemo.update_layout(
            margin={"l": 5, "r": 5, "t": 50, "b": 5},  # Ajustement pour bien afficher le titre
            height=110,  
            paper_bgcolor="white"
        )

    
    fig_dte_don = go.Figure(go.Indicator(
        mode="number+gauge",
        value=100*round(len(data1[data1['Raison indisponibilité  [date de dernier Don < 3 mois ]']=='Oui'])/len(data1),2),
        
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": 'date de dernier Don < 3 mois', "font": {"size": 16, "color": "#2C3E50"}},  
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "lightgrey"},
            "bar": {"color": "#FFB74D"},  
            "steps": [
                {"range": [0, 50], "color": "#AED6F1"},  
                {"range": [50, 100], "color": "#85C1E9"}  
            ],
            "borderwidth": 1.2,  
            "bordercolor": "#D5DBDB"  
        },
        number={"font": {"size": 16, "color": "#2C3E50"},
                "suffix": "%"
                }  
    ))

    fig_dte_don.update_layout(
            margin={"l": 5, "r": 5, "t": 50, "b": 5},  # Ajustement pour bien afficher le titre
            height=110,  
            paper_bgcolor="white"
        )
    


    fig_ist = go.Figure(go.Indicator(
        mode="number+gauge",
        value=100 * round(len(data1[data1['Raison indisponibilité  [IST récente (Exclu VIH, Hbs, Hcv)]']=='Oui']) / len(data1), 2),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": 'IST récente (Exclu VIH, Hbs, Hcv)', "font": {"size": 16, "color": "#2C3E50"}},  
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "lightgrey"},
            "bar": {"color": "#D8BFD8"},  
            "steps": [
                {"range": [0, 50], "color": "#AED6F1"},  
                {"range": [50, 100], "color": "#85C1E9"}  
            ],
            "borderwidth": 1.2,  
            "bordercolor": "#D5DBDB"  
        },
        number={"font": {"size": 16, "color": "#2C3E50"},
                "suffix": "%"
                }  
    ))

    fig_ist.update_layout(
            margin={"l": 5, "r": 5, "t": 50, "b": 5},  # Ajustement pour bien afficher le titre
            height=110,  
            paper_bgcolor="white"
        )
    

    data1=filtered_df[filtered_df["ÉLIGIBILITÉ AU DON."]=='Temporairement Non-eligible']
    data11=data1[data1['Genre ']=='Femme']
    valeur=0
    if len(data11) == 0:
           
        valeur = 0
    else:
        valeur=100*round(len(data11[data11['Raison de l’indisponibilité de la femme [La DDR est mauvais si <14 jour avant le don]']=='Oui'])/len(data11),2)
        
    

    fig_ddr = go.Figure(go.Indicator(
        mode="number+gauge",
        value=valeur,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": 'La DDR est mauvais si <14 jour', "font": {"size": 16, "color": "#2C3E50"}},  
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "lightgrey"},
            "bar": {"color": "#FFB6C1"},  
            "steps": [
                {"range": [0, 50], "color": "#AED6F1"},  
                {"range": [50, 100], "color": "#85C1E9"}  
            ],
            "borderwidth": 1.2,  
            "bordercolor": "#D5DBDB"  
        },
        number={"font": {"size": 16, "color": "#2C3E50"},
                "suffix": "%"
                }  
    ))

    fig_ddr.update_layout(
            margin={"l": 5, "r": 5, "t": 50, "b": 5},  # Ajustement pour bien afficher le titre
            height=110,  
            paper_bgcolor="white"
        )
    


    
    if len(data11) == 0:
           
        valeur = 0
    else:
        valeur=100*round(len(data11[data11['Raison de l’indisponibilité de la femme [Allaitement ]']=='Oui'])/len(data11),2)



    
    
    fig_al = go.Figure(go.Indicator(
        mode="number+gauge",
        value=valeur,
        
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": 'Allaitement', "font": {"size": 16, "color": "#2C3E50"}},  
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "lightgrey"},
            "bar": {"color": "#90EE90"},  
            "steps": [
                {"range": [0, 50], "color": "#AED6F1"},  
                {"range": [50, 100], "color":"#85C1E9"}  
            ],
            "borderwidth": 1.2,  
            "bordercolor": "#D5DBDB"  
        },
        number={"font": {"size": 16, "color": "#2C3E50"},
                "suffix": "%"
                }  
    ))

    fig_al.update_layout(
            margin={"l": 5, "r": 5, "t": 50, "b": 5},  # Ajustement pour bien afficher le titre
            height=110,  
            paper_bgcolor="white"
        )
    
    if len(data11) == 0:
           
        valeur = 0
    else:
        valeur=100*round(len(data11[data11['Raison de l’indisponibilité de la femme [A accoucher ces 6 derniers mois  ]']=='Oui'])/len(data11),2)
        
    

    
    fig_acc = go.Figure(go.Indicator(
        mode="number+gauge",
        value=valeur,
        
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": 'accoucher ces 6 derniers mois', "font": {"size": 16, "color": "#2C3E50"}},  
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "lightgrey"},
            "bar": {"color": "#FFB74D"},  
            "steps": [
                {"range": [0, 50], "color": "#AED6F1"},  
                {"range": [50, 100], "color": "#85C1E9"}  
            ],
            "borderwidth": 1.2,  
            "bordercolor": "#D5DBDB"  
        },
        number={"font": {"size": 16, "color": "#2C3E50"},
                "suffix": "%"
                }  

    )
    )

    fig_acc.update_layout(
            margin={"l": 5, "r": 5, "t": 50, "b": 5},  # Ajustement pour bien afficher le titre
            height=110,  
            paper_bgcolor="white"
        )
    if len(data11) == 0:
           
        valeur = 0
    else:
         valeur=100 * round(len(data11[data11['Raison de l’indisponibilité de la femme [Interruption de grossesse  ces 06 derniers mois]']=='Oui']) / len(data11), 2)
       
    
    


    fig_intgro = go.Figure(go.Indicator(
        mode="number+gauge",
        value=valeur,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": 'Interruption de grossesse', "font": {"size": 16, "color": "#2C3E50"}},  
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "lightgrey"},
            "bar": {"color": "#D8BFD8"},  
            "steps": [
                {"range": [0, 50], "color": "#AED6F1"},  
                {"range": [50, 100], "color": "#85C1E9"}  
            ],
            "borderwidth": 1.2,  
            "bordercolor": "#D5DBDB"  
        },
        number={"font": {"size": 16, "color": "#2C3E50"},
                "suffix": "%"
                }  
    ))

    fig_intgro.update_layout(
            margin={"l": 5, "r": 5, "t": 50, "b": 5},  # Ajustement pour bien afficher le titre
            height=110,  
            paper_bgcolor="white"
        )
    if len(data11) == 0:
           
        valeur = 0
    else:
         valeur=100 * round(len(data11[data11['Raison de l’indisponibilité de la femme [est enceinte ]']=='Oui']) / len(data11), 2)
        



    fig_enc = go.Figure(go.Indicator(
        mode="number+gauge",
        value=valeur,
       domain={"x": [0, 1], "y": [0, 1]},
        title={"text": 'enceinte', "font": {"size": 16, "color": "#2C3E50"}},  
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "lightgrey"},
            "bar": {"color": "#D8BFD8"},  
            "steps": [
                {"range": [0, 50], "color": "#AED6F1"},  
                {"range": [50, 100], "color": "#85C1E9"}  
            ],
            "borderwidth": 1.2,  
            "bordercolor": "#D5DBDB"  
        },
        number={"font": {"size": 16, "color": "#2C3E50"},
                "suffix": "%"
                }  
    ))

    fig_enc.update_layout(
            margin={"l": 5, "r": 5, "t": 50, "b": 5},  # Ajustement pour bien afficher le titre
            height=110,  
            paper_bgcolor="white"
        )


        
  





    for col in ['Raison indisponibilité  [Est sous anti-biothérapie  ]', 
                'Raison indisponibilité  [Taux d’hémoglobine bas ]', 
                'Raison indisponibilité  [date de dernier Don < 3 mois ]', 
                'Raison indisponibilité  [IST récente (Exclu VIH, Hbs, Hcv)]']:
        data1[col] = data1[col].map({'Oui': 1, 'Non': 0})
            
    renaming_dict = {
        'Raison indisponibilité  [Est sous anti-biothérapie  ]': 'Sous antibiothérapie',
        'Raison indisponibilité  [Taux d’hémoglobine bas ]': 'Hémoglobine basse',
        'Raison indisponibilité  [date de dernier Don < 3 mois ]': 'Dernier don < 3 mois',
        'Raison indisponibilité  [IST récente (Exclu VIH, Hbs, Hcv)]': 'IST récente'
    }

        # Renommer les colonnes
    data1.rename(columns=renaming_dict, inplace=True)


    choix = ['Sous antibiothérapie', 
            'Hémoglobine basse', 
            'Dernier don < 3 mois', 
            'IST récente']

    

        # Calculer les combinaisons et leurs fréquences
    combinaisons_frequences = {}
    for r in range(2, len(choix) + 1):
        for comb in combinations(choix, r):
            freq = data1[list(comb)].all(axis=1).sum()
            if freq > 0:
                combinaisons_frequences[' + '.join(comb)] = freq

        # Convertir en DataFrame
    combinaisons_frequences_df = pd.DataFrame(list(combinaisons_frequences.items()), columns=['Combinaison', 'Fréquence'])

        # Vérifier la structure du DataFrame
   

        # Créer un DataFrame avec les bonnes colonnes
    treemap_data = combinaisons_frequences_df

        # Convertir 'Fréquence' en numérique et remplir les valeurs manquantes par 0
    treemap_data['Fréquence'] = pd.to_numeric(treemap_data['Fréquence'], errors='coerce')
    treemap_data['Fréquence'].fillna(0, inplace=True)

        # Couleurs personnalisées
    couleurs = {
            'Raison indisponibilité  [Est sous anti-biothérapie  ]': 'skyblue',
            'Raison indisponibilité  [Taux d’hémoglobine bas ]': 'pink',
            'Raison indisponibilité  [date de dernier Don < 3 mois ]': 'lightgreen',
            'Raison indisponibilité  [IST récente (Exclu VIH, Hbs, Hcv)]': 'violet'
        }
    data2=data1[['Sous antibiothérapie', 
            'Hémoglobine basse', 
            'Dernier don < 3 mois', 
            'IST récente']]
    len(data2)
    data2_binary = data2.notna().astype(int)  # Convertir les valeurs en 0/1

   

        # Calcul de la matrice de co-occurrence
    #co_occurrence_matrix = data2_binary.T @ data2_binary
    co_occurrence_matrix = pd.DataFrame(0, index=data2_binary.columns, columns=data2_binary.columns)

    # Calcul des co-occurrences de 1
    for col1 in data2_binary.columns:
        for col2 in data2_binary.columns:
            # Compter les cas où les deux colonnes ont la valeur 1 en même temps
            co_occurrence_matrix.loc[col1, col2] = ((data2_binary[col1] == 1) & (data2_binary[col2] == 1)).sum()
    #intersection_matrix =pd.DataFrame(0, index=data2_binary.columns, columns=data2_binary.columns)
    intersection_matrix=co_occurrence_matrix

    
   
   
        

     # ✅ Vérifier que la matrice est bien remplie
    # Création de la heatmap avec Plotly Express
    fig_heat = px.imshow(
    intersection_matrix,
    labels=dict(x="Raison", y="Raison", color="Apparition Simultanée"),
    x=intersection_matrix.columns,
    y=intersection_matrix.columns,
    color_continuous_scale=[(0, "#FFD1DC"), (1, "#ADD8E6")],  # Rose clair et bleu clair
    text_auto=True,  # Affiche les valeurs directement sur la heatmap
    title="<b>🌡️ Heatmap des Apparitions Simultanées entre les Raisons d'Indisponibilité</b>",
    width=900, height=700
)
    fig_heat.update_layout(
    title_font=dict(size=16, color="black", family="Arial Black"),  # Titre en gras et moderne
    font=dict(size=14, color="black"),  # Texte plus lisible
    xaxis=dict(
        title="Raisons d'Indisponibilité",  # Titre de l'axe X
        title_font=dict(size=18, color="black", family="Arial"),
        tickfont=dict(size=14, color="black"),
        tickangle=-45  # Inclinaison des labels pour une meilleure lisibilité
    ),
    yaxis=dict(
        title="Raisons d'Indisponibilité",  # Titre de l'axe Y
        title_font=dict(size=18, color="black", family="Arial"),
        tickfont=dict(size=14, color="black")
    ),
    coloraxis_colorbar=dict(
        title="Apparition Simultanée",  # Titre de la barre de couleur
        title_font=dict(size=16, color="black", family="Arial"),
        tickfont=dict(size=14, color="black"),
        tickvals=[0, 1],  # Valeurs discrètes (0 ou 1)
        ticktext=["Non", "Oui"]  # Texte explicite pour les valeurs
    ),
    margin=dict(t=120, b=80, l=80, r=80),  # Marges ajustées
    paper_bgcolor="white",  # Fond blanc
    plot_bgcolor="white"   # Fond de la zone de tracé blanc
)

    # Personnalisation des annotations (valeurs dans les cellules)
    fig_heat.update_traces(
        textfont=dict(size=14, color="black", family="Arial"),  # Taille et couleur du texte dans les cellules
        hovertemplate="<b>%{x}</b> et <b>%{y}</b><br>Apparition Simultanée: %{z}<extra></extra>"  # Texte au survol
    )
  

    



        # Générer le Treemap avec Plotly Express
    fig_treemap = px.treemap(
            treemap_data, 
            path=['Combinaison'],  # Colonne pour hiérarchiser les données
            values='Fréquence',    # Colonne pour les valeurs des segments
            color='Combinaison',   # Colonne pour la couleur des segments
            color_discrete_map=couleurs,  # Mapping des couleurs
            title="🌿 Treemap des Raisons d'Inéligibilité des Volontaires",  # Titre du graphique
            width=1200,  # Largeur du graphique
            height=800   # Hauteur du graphique
        )

        # Générer le Treemap avec Plotly Express
        # Amélioration du style

    fig_treemap.update_layout(
            title_font=dict(size=24, color="black", family="Arial"),
                # Taille, couleur et police du titre
            title_x=0.5,
            margin=dict(t=80, l=50, r=50, b=50),  # Marges autour du graphique
            paper_bgcolor="white",  # Couleur de fond du graphique
            plot_bgcolor="white",   # Couleur de fond de la zone de tracé
            hoverlabel=dict(        # Style des étiquettes au survol
                bgcolor="white",    # Couleur de fond
                font_size=16,       # Taille de la police
                font_family="Arial"
            )
        )
        

        # Amélioration des labels
    fig_treemap.update_traces(
            textinfo="label+percent entry",  # Afficher le label et le pourcentage
            textfont=dict(size=16, color="white", family="Arial"),  # Taille, couleur et police des labels
            marker=dict(
                line=dict(width=1, color="black")  # Bordure des éléments du treemap
            ),
            hovertemplate="<b>%{label}</b><br>Fréquence: %{value}<br>Part: %{percentParent:.2%}<extra></extra>",  # Format du texte au survol
            insidetextfont=dict(size=70, color="white"),  # Taille et couleur du texte à l'intérieur des segments
            outsidetextfont=dict(size=20, color="black")  # Taille et couleur du texte à l'extérieur des segments
        )
    
  


   
        
    values_g = filtered_df["ÉLIGIBILITÉ AU DON."].value_counts().reset_index()
    values_g.columns = ["ÉLIGIBILITÉ AU DON.", 'Count']

    fig_eligibilite = px.pie(
            values_g,
            names="ÉLIGIBILITÉ AU DON.",
            values='Count',
            title='Répartition par Genre',
            color="ÉLIGIBILITÉ AU DON.",
            color_discrete_sequence=px.colors.qualitative.Plotly,  # Palette de couleurs modernes
            hole=0.4  # Créer un donut chart
        )
     # Mise en forme du layout
    fig_eligibilite.update_layout(
        title_font=dict(size=22, color='black'),  
        legend_title="Statut d'éligibilité",
        legend=dict(x=1, y=0.9),  # Positionnement de la légende
        margin=dict(t=50, b=20, l=20, r=20),  
        showlegend=True  
    )
    renaming1_dict = {
    "Raison de non-eligibilité totale  [Antécédent_de_transfusion] ": "Antécédent_de_transfusion",
    "Raison de non-eligibilité totale  [Porteur(HIV,hbs,hcv)]": "Porteur(HIV,hbs,hcv)",
    "Raison de non-eligibilité totale  [Opéré]": "Opéré",
    "Raison de non-eligibilité totale  [Drepanocytaire]": "Drepanocytaire",
    "Raison de non-eligibilité totale  [Diabétique]": "Diabétique",
    "Raison de non-eligibilité totale  [Hypertendus]": "Hypertendus",
    "Raison de non-eligibilité totale  [Asthmatiques]": "Asthmatiques",
    "Raison de non-eligibilité totale  [Cardiaque]": "Cardiaque",
    "Raison de non-eligibilité totale  [Tatoué]": "Tatoué",
    "Raison de non-eligibilité totale  [Scarifié]": "Scarifié"
}

    


    data2=filtered_df[filtered_df["ÉLIGIBILITÉ AU DON."]=='Définitivement non-eligible'] # Renommer les colonnes
    data2.rename(columns=renaming1_dict, inplace=True)

    trans=100*round(len(data2[data2["Porteur(HIV,hbs,hcv)"]=='Oui'])/len(data2),2)
    Ope=100*round(len(data2[data2["Opéré"]=='Oui'])/len(data2),2)
    Drepa=100*round(len(data2[data2["Drepanocytaire"]=='Oui'])/len(data2),2)
    Hyper=100*round(len(data2[data2["Hypertendus"]=='Oui'])/len(data2),2)
    Asthma=100*round(len(data2[data2["Asthmatiques"]=='Oui'])/len(data2),2)
    Card=100*round(len(data2[data2["Cardiaque"]=='Oui'])/len(data2),2)
    Tat=100*round(len(data2[data2["Tatoué"]=='Oui'])/len(data2),2)
    Scar=100*round(len(data2[data2["Scarifié"]=='Oui'])/len(data2),2)


        # Définition des couleurs
    filled_color = 'deepskyblue'  # Partie remplie
    empty_color = 'lightgray'  # Partie vide
    background_color = 'whitesmoke'  # Fond léger

    fig_trans = go.Figure(go.Pie(
    values=[trans, 100 - trans],
    marker=dict(
        colors=['#1f77b4', 'lightgray'],
        line=dict(color='white', width=3)  # J'ai retiré 'dash' et augmenté la largeur
    ),
    hole=0.7,
    textinfo='none',
    hoverinfo='none',
    showlegend=False,
    rotation=90
))

    # Texte central avec formatage à 2 décimales
    fig_trans.update_layout(
        title=dict(
            text="<b>Taux de Scarification</b>",
            font=dict(
                size=18,
                family="Arial, sans-serif",
                color="#2a3f5f"
            ),
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top'
        ),
        annotations=[
            dict(
                text=f"<b>{trans:.2f}%</b>",  # Format à 2 décimales
                x=0.5,
                y=0.5,
                font=dict(
                    size=28,
                    color='#1f77b4',
                    family="Arial, sans-serif"
                ),
                showarrow=False
            )
        ],
        margin=dict(t=60, b=40, l=40, r=40),
        width=250,
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Effet de tirage conditionnel (sans la propriété dash qui causait l'erreur)
    fig_trans.update_traces(
        pull=[0.05 if trans > 50 else 0, 0]  # Effet de "tirage" si taux élevé
    )

    # Ajout conditionnel d'un badge d'objectif
    if trans > 30:
        fig_trans.add_annotation(
            text=f"<span style='background-color: #ffcccc; padding: 2px 8px; border-radius: 10px'>+{(trans-30):.2f}%</span>",
            x=0.5,
            y=0.3,
            showarrow=False,
            font=dict(size=12, color="darkred")
        )
    else:
        fig_trans.add_annotation(
            text="<span style='background-color: #ccffcc; padding: 2px 8px; border-radius: 10px'>✓ Objectif</span>",
            x=0.5,
            y=0.3,
            showarrow=False,
            font=dict(size=12, color="darkgreen")
        )
 
    fig_Ope = create_donut_chart(Ope, "Opérations", '#1f77b4')
    fig_Drepa = create_donut_chart(Drepa, "Drépanocytose", '#ff7f0e') 
    fig_Hyper = create_donut_chart(Hyper, "Hypertension", '#2ca02c')
    fig_Asthma = create_donut_chart(Asthma, "Asthme", '#d62728')
    fig_Card = create_donut_chart(Card, "Cardiopathies", '#9467bd')
    fig_Tat = create_donut_chart(Tat, "Tatouages", '#8c564b')
    fig_Scar = create_donut_chart(Scar, "Scarifications", '#e377c2')
    fig_trans = create_donut_chart(trans, "Taux de Scarification", '#1f77b4')
       


    return fig_eligibilite,fig_biothe,fig_hemo,fig_dte_don,fig_ist,fig_heat,fig_ddr,fig_al,fig_acc,fig_intgro,fig_enc,fig_trans,fig_Ope,fig_Drepa,fig_Hyper,fig_Asthma,fig_Card,fig_Tat,fig_Scar


indispo_vars = [
    'Raison indisponibilité  [Est sous anti-biothérapie  ]',
    'Raison indisponibilité  [date de dernier Don < 3 mois ]',
    'Raison indisponibilité  [IST récente (Exclu VIH, Hbs, Hcv)]'
]

indispo_femme_vars = [
    'Raison de l’indisponibilité de la femme [La DDR est mauvais si <14 jour avant le don]',
       'Raison de l’indisponibilité de la femme [Allaitement ]',
       'Raison de l’indisponibilité de la femme [A accoucher ces 6 derniers mois  ]',
       'Raison de l’indisponibilité de la femme [Interruption de grossesse  ces 06 derniers mois]',
       'Raison de l’indisponibilité de la femme [est enceinte ]',
]

non_eligible_vars = [
    'Raison de non-eligibilité totale  [Antécédent de transfusion]',
       'Raison de non-eligibilité totale  [Porteur(HIV,hbs,hcv)]',
       'Raison de non-eligibilité totale  [Opéré]',
       'Raison de non-eligibilité totale  [Drepanocytaire]',
       'Raison de non-eligibilité totale  [Diabétique]',
       'Raison de non-eligibilité totale  [Hypertendus]',
       'Raison de non-eligibilité totale  [Asthmatiques]',
       'Raison de non-eligibilité totale  [Cardiaque]',
       'Raison de non-eligibilité totale  [Tatoué]',
       'Raison de non-eligibilité totale  [Scarifié]',

]

























   
# Callbacks pour afficher/masquer la barre latérale
@app.callback(
    [
     Output('fig-ecart', 'figure'),
    Output('fig-type', 'figure'),
    Output('fig-genre-f', 'figure'),
    Output('fig-statut-f', 'figure'),
    Output('fig-etude-f', 'figure'),
    Output('fig-tranche-f', 'figure'),
    Output('fig-religion-f', 'figure'),
    Output('fig-arrond-f', 'figure'),
   

      
     ]
    ,
    [Input("tabs", "active_tab"),  # 🔥 Ajout de cet input pour déclencher la callback
     Input("apply-filters", "n_clicks")],  # 🔥 Ajout d'un bouton pour filtrer
    
    [State('genre-filter', 'value'),
     State('arrondissement-filter', 'value'),
     State('age-slider', 'value')]
    
)
def update_graphs3(active_tab, n_clicks, genre, arrondissement, age_range):
    filtered_df = filter_dataframe(df, genre, arrondissement, age_range)

    data1=filtered_df[filtered_df['A-t-il (elle) déjà donné le sang ']=='Oui']

    fig_ecart = px.histogram(data1, x='mois_ecart', 
                   nbins=50,
                   color_discrete_sequence=['#2b8cbe'],
                   opacity=0.9,
                   title="<b>Distribution de ceux qui ont donnee le sang au moins une fois selon l'ecar entre les deux derniers don</b>",
                   template='plotly_white')

    fig_ecart.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12, color="#333333"),
        title_x=0.5,
        title_font=dict(size=16),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            linecolor='#ddd',
            title='Écart en mois'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            linecolor='#ddd',
            title='Nombre de donneurs'
        ),
        bargap=0.1
    )
    fig_type = px.pie(data1, names='Type Donneur', 
             color_discrete_sequence=px.colors.qualitative.Pastel,
             hole=0.4,
             title='<b>RÉPARTITION DES DONNEURS PAR FRÉQUENCE</b>')

    fig_type.update_traces(textposition='inside', 
                    textinfo='percent+label',
                    marker=dict(line=dict(color='white', width=1)))

    fig_type.update_layout(
        font=dict(size=12, family="Arial"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_x=0.5
    )
     

    # Filtrer les donneurs ciblés
    donneurs_cibles = data1[data1['Type Donneur'].isin(['Hyper-Régulier', 'Régulier', 'Occasionnel'])]

    # Calcul du nombre de donneurs par type et niveau d'étude
    study_counts = donneurs_cibles.groupby(["Niveau d'etude", 'Type Donneur']).size().unstack().fillna(0)

    # Calcul du total par niveau d'étude
    study_totals = study_counts.sum(axis=1)

    # Calcul des pourcentages pour chaque type de donneur par niveau d'étude
    study_percent = study_counts.div(study_totals, axis=0) * 100

    # Définition des couleurs pastel
    colors = {'Hyper-Régulier': '#A7C7E7',  # Bleu clair
            'Régulier': '#D8BFD8',        # Violet clair
            'Occasionnel': '#FFB6C1'}     # Rose clair

    # Création de la figure
    fig_etude_f= go.Figure()

    for donor_type in ['Hyper-Régulier', 'Régulier', 'Occasionnel']:
        fig_etude_f.add_trace(go.Bar(
            x=study_percent[donor_type],  # Proportion en pourcentage
            y=study_percent.index,  # Niveau d’étude
            name=donor_type,
            orientation='h',
            marker=dict(
                color=colors[donor_type],
                line=dict(color='white', width=1.5),  # Contours fins blancs
                opacity=0.95  # Opacité douce
            ),
            hoverinfo='x+name'  # Affichage du pourcentage lors du survol
        ))

    # Mise en forme visuelle
    fig_etude_f.update_layout(
        title='<b>Répartition des Donneurs par Niveau d\'Étude (en %)</b>',
        title_x=0.5,
        barmode='stack',  # Utilisation de stack pour un total de 100% par niveau d’étude
        bargap=0.2,
        xaxis=dict(
            title="Proportion (%)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1.2,
            gridcolor='rgba(0,0,0,0.05)',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        ),
        yaxis=dict(title="Niveau d'Étude", categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=16),
        legend=dict(
            title="Type de Donneur", 
            orientation="h", 
            x=0.3, y=-0.3, 
            bgcolor="rgba(255,255,255,0.7)",  # Fond semi-transparent
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(l=120, r=30, t=50, b=80)  # Ajustement des marges pour éviter les superpositions
    )

     # Calcul du nombre de donneurs par type et niveau d'étude
    genre_counts = donneurs_cibles.groupby(["Genre ", 'Type Donneur']).size().unstack().fillna(0)

    # Calcul du total par niveau d'étude
    genre_totals = genre_counts.sum(axis=1)

    # Calcul des pourcentages pour chaque type de donneur par niveau d'étude
    genre_percent = genre_counts.div(genre_totals, axis=0) * 100

    # Définition des couleurs pastel
    colors = {'Hyper-Régulier': '#A7C7E7',  # Bleu clair
            'Régulier': '#D8BFD8',        # Violet clair
            'Occasionnel': '#FFB6C1'}     # Rose clair

    # Création de la figure
    fig_genre_f = go.Figure()

    for donor_type in ['Hyper-Régulier', 'Régulier', 'Occasionnel']:
            fig_genre_f .add_trace(go.Bar(
            x=genre_percent[donor_type],  # Proportion en pourcentage
            y=genre_percent.index,  # Niveau d’étude
            name=donor_type,
            orientation='h',
            marker=dict(
                color=colors[donor_type],
                line=dict(color='white', width=1.5),  # Contours fins blancs
                opacity=0.95  # Opacité douce
            ),
            hoverinfo='x+name'  # Affichage du pourcentage lors du survol
        ))

    # Mise en forme visuelle
    fig_genre_f.update_layout(
        title='<b>Répartition des Donneurs par Genre (en %)</b>',
        title_x=0.5,
        barmode='stack',  # Utilisation de stack pour un total de 100% par niveau d’étude
        bargap=0.2,
        xaxis=dict(
            title="Proportion (%)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1.2,
            gridcolor='rgba(0,0,0,0.05)',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        ),
        yaxis=dict(title="Genre", categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=16),
        legend=dict(
            title="Type de Donneur", 
            orientation="h", 
            x=0.3, y=-0.3, 
            bgcolor="rgba(255,255,255,0.7)",  # Fond semi-transparent
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(l=120, r=30, t=50, b=80)  # Ajustement des marges pour éviter les superpositions
    )


     # Calcul du nombre de donneurs par type et niveau d'étude
    religion_counts = donneurs_cibles.groupby(['New_Religion', 'Type Donneur']).size().unstack().fillna(0)

    # Calcul du total par niveau d'étude
    religion_totals = religion_counts.sum(axis=1)

    # Calcul des pourcentages pour chaque type de donneur par niveau d'étude
    religion_percent = religion_counts.div(religion_totals, axis=0) * 100

    # Définition des couleurs pastel
    colors = {'Hyper-Régulier': '#A7C7E7',  # Bleu clair
            'Régulier': '#D8BFD8',        # Violet clair
            'Occasionnel': '#FFB6C1'}     # Rose clair

    # Création de la figure
    fig_religion_f= go.Figure()

    for donor_type in ['Hyper-Régulier', 'Régulier', 'Occasionnel']:
        fig_religion_f.add_trace(go.Bar(
            x=religion_percent[donor_type],  # Proportion en pourcentage
            y=religion_percent.index,  # Niveau d’étude
            name=donor_type,
            orientation='h',
            marker=dict(
                color=colors[donor_type],
                line=dict(color='white', width=1.5),  # Contours fins blancs
                opacity=0.95  # Opacité douce
            ),
            hoverinfo='x+name'  # Affichage du pourcentage lors du survol
        ))

    # Mise en forme visuelle
    fig_religion_f.update_layout(
        title='<b>Répartition des Donneurs par religion (en %)</b>',
        title_x=0.5,
        barmode='stack',  # Utilisation de stack pour un total de 100% par niveau d’étude
        bargap=0.2,
        xaxis=dict(
            title="Proportion (%)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1.2,
            gridcolor='rgba(0,0,0,0.05)',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        ),
        yaxis=dict(title="Religion", categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=16),
        legend=dict(
            title="Type de Donneur", 
            orientation="h", 
            x=0.3, y=-0.3, 
            bgcolor="rgba(255,255,255,0.7)",  # Fond semi-transparent
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(l=120, r=30, t=50, b=80)  # Ajustement des marges pour éviter les superpositions
    )




     # Calcul du nombre de donneurs par type et niveau d'étude
    statut_counts = donneurs_cibles.groupby(['Situation Matrimoniale (SM)', 'Type Donneur']).size().unstack().fillna(0)

    # Calcul du total par niveau d'étude
    statut_totals = statut_counts.sum(axis=1)

    # Calcul des pourcentages pour chaque type de donneur par niveau d'étude
    statut_percent = statut_counts.div(statut_totals, axis=0) * 100

    # Définition des couleurs pastel
    colors = {'Hyper-Régulier': '#A7C7E7',  # Bleu clair
            'Régulier': '#D8BFD8',        # Violet clair
            'Occasionnel': '#FFB6C1'}     # Rose clair

    # Création de la figure
    fig_statut_f = go.Figure()

    for donor_type in ['Hyper-Régulier', 'Régulier', 'Occasionnel']:
        fig_statut_f.add_trace(go.Bar(
            x=statut_percent[donor_type],  # Proportion en pourcentage
            y=statut_percent.index,  # Niveau d’étude
            name=donor_type,
            orientation='h',
            marker=dict(
                color=colors[donor_type],
                line=dict(color='white', width=1.5),  # Contours fins blancs
                opacity=0.95  # Opacité douce
            ),
            hoverinfo='x+name'  # Affichage du pourcentage lors du survol
        ))

    # Mise en forme visuelle
    fig_statut_f.update_layout(
        title='<b>Répartition des Donneurs par Situation Matrimoniale (SM) (en %)</b>',
        title_x=0.5,
        barmode='stack',  # Utilisation de stack pour un total de 100% par niveau d’étude
        bargap=0.2,
        xaxis=dict(
            title="Proportion (%)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1.2,
            gridcolor='rgba(0,0,0,0.05)',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        ),
        yaxis=dict(title='Situation Matrimoniale (SM)', categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=16),
        legend=dict(
            title="Type de Donneur", 
            orientation="h", 
            x=0.3, y=-0.3, 
            bgcolor="rgba(255,255,255,0.7)",  # Fond semi-transparent
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(l=120, r=30, t=50, b=80)  # Ajustement des marges pour éviter les superpositions
    )


    


     # Calcul du nombre de donneurs par type et niveau d'étude
    tranche_counts = donneurs_cibles.groupby(['Tranche_Age', 'Type Donneur']).size().unstack().fillna(0)

    # Calcul du total par niveau d'étude
    tranche_totals = tranche_counts.sum(axis=1)

    # Calcul des pourcentages pour chaque type de donneur par niveau d'étude
    tranche_percent = tranche_counts.div(tranche_totals, axis=0) * 100

    # Définition des couleurs pastel
    colors = {'Hyper-Régulier': '#A7C7E7',  # Bleu clair
            'Régulier': '#D8BFD8',        # Violet clair
            'Occasionnel': '#FFB6C1'}     # Rose clair

    # Création de la figure
    fig_tranche_f = go.Figure()

    for donor_type in ['Hyper-Régulier', 'Régulier', 'Occasionnel']:
        fig_tranche_f.add_trace(go.Bar(
            x=tranche_percent[donor_type],  # Proportion en pourcentage
            y=tranche_percent.index,  # Niveau d’étude
            name=donor_type,
            orientation='h',
            marker=dict(
                color=colors[donor_type],
                line=dict(color='white', width=1.5),  # Contours fins blancs
                opacity=0.95  # Opacité douce
            ),
            hoverinfo='x+name'  # Affichage du pourcentage lors du survol
        ))

    # Mise en forme visuelle
    fig_tranche_f.update_layout(
        title='<b>Répartition des Donneurs par Tranche_Age(en %)</b>',
        title_x=0.5,
        barmode='stack',  # Utilisation de stack pour un total de 100% par niveau d’étude
        bargap=0.2,
        xaxis=dict(
            title="Proportion (%)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1.2,
            gridcolor='rgba(0,0,0,0.05)',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        ),
        yaxis=dict(title='Tranche_Age', categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=16),
        legend=dict(
            title="Type de Donneur", 
            orientation="h", 
            x=0.3, y=-0.3, 
            bgcolor="rgba(255,255,255,0.7)",  # Fond semi-transparent
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(l=120, r=30, t=50, b=80)  # Ajustement des marges pour éviter les superpositions
    )



    


     # Calcul du nombre de donneurs par type et niveau d'étude
    arrond_counts = donneurs_cibles.groupby(['Arrondissement_final', 'Type Donneur']).size().unstack().fillna(0)

    # Calcul du total par niveau d'étude
    arrond_totals = arrond_counts.sum(axis=1)

    # Calcul des pourcentages pour chaque type de donneur par niveau d'étude
    arrond_percent = arrond_counts.div(arrond_totals, axis=0) * 100

    # Définition des couleurs pastel
    colors = {'Hyper-Régulier': '#A7C7E7',  # Bleu clair
            'Régulier': '#D8BFD8',        # Violet clair
            'Occasionnel': '#FFB6C1'}     # Rose clair

    # Création de la figure
    fig_arrond_f = go.Figure()

    for donor_type in ['Hyper-Régulier', 'Régulier', 'Occasionnel']:
        fig_arrond_f.add_trace(go.Bar(
            x=arrond_percent[donor_type],  # Proportion en pourcentage
            y=arrond_percent.index,  # Niveau d’étude
            name=donor_type,
            orientation='h',
            marker=dict(
                color=colors[donor_type],
                line=dict(color='white', width=1.5),  # Contours fins blancs
                opacity=0.95  # Opacité douce
            ),
            hoverinfo='x+name'  # Affichage du pourcentage lors du survol
        ))

    # Mise en forme visuelle
    fig_arrond_f.update_layout(
        title='<b>Répartition des Donneurs par Arrondissement_final (en %)</b>',
        title_x=0.5,
        barmode='stack',  # Utilisation de stack pour un total de 100% par niveau d’étude
        bargap=0.2,
        xaxis=dict(
            title="Proportion (%)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1.2,
            gridcolor='rgba(0,0,0,0.05)',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"]
        ),
        yaxis=dict(title='Arrondissement_final', categoryorder='total ascending'),
        plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=16),
        legend=dict(
            title="Type de Donneur", 
            orientation="h", 
            x=0.3, y=-0.3, 
            bgcolor="rgba(255,255,255,0.7)",  # Fond semi-transparent
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(l=120, r=30, t=50, b=80)  # Ajustement des marges pour éviter les superpositions
    )











    return fig_ecart,fig_type,fig_genre_f,fig_statut_f,fig_etude_f,fig_tranche_f,fig_religion_f,fig_arrond_f

            










   
# Callbacks pour afficher/masquer la barre latérale
@app.callback(
    [
     
    Output('fig-sentiment', 'figure'),
    Output('fig-label', 'figure'),
    Output('fig-sent-tem', 'figure'),
    
       
     ]
    ,
    [Input("tabs", "active_tab"),  # 🔥 Ajout de cet input pour déclencher la callback
     Input("apply-filters", "n_clicks")],  # 🔥 Ajout d'un bouton pour filtrer
    
    [State('genre-filter', 'value'),
     State('arrondissement-filter', 'value'),
     State('age-slider', 'value')]
    
)

def update_graphs4(active_tab, n_clicks, genre, arrondissement, age_range):
    #analyse des sentiments
    filtered_df = filter_dataframe(dat, genre, arrondissement, age_range)

    sentiment_counts = filtered_df['sentiment'].value_counts().sort_values(ascending=True)

    label_counts = filtered_df['label'].value_counts().sort_values(ascending=True)

    comments = filtered_df['Si autres raison préciser '].dropna().tolist() + filtered_df['Autre raisons,  preciser '].dropna().tolist()
    
    
    
   
    #fig_mots=generate_wordcloud(comments)
    fig_sentiment = px.histogram(filtered_df, x="sentiment", title='Répartition par sentimnt', color="sentiment",category_orders={"sentiment": sentiment_counts.index.tolist()} )
    fig_label= px.histogram(filtered_df, x="label", title='Répartition par label', color="label",category_orders={"label": label_counts.index.tolist()} )
    

    
    

        # Agréger les scores par mois pour voir l'évolution moyenne des sentiments
    df_monthly = filtered_df.groupby('Mois').agg({'score': 'mean'}).reset_index()

    # Créer un graphique interactif avec Plotly
    fig_sent_tem = go.Figure()

    # Ajouter la courbe des scores moyens par mois
    fig_sent_tem.add_trace(go.Scatter(
        x=df_monthly['Mois'].astype(str),
        y=df_monthly['score'],
        mode='lines+markers',
        name='Score Moyen',
        line=dict(color='blue', width=4),
        marker=dict(symbol='circle', size=8, color='blue'),
        hovertemplate='Mois: %{x}<br>Score moyen: %{y:.2f}<extra></extra>'
    ))

    # Ajouter des titres et des labels
    fig_sent_tem.update_layout(
        title="Évolution des Scores des Sentiments au Cours du Temps",
        xaxis_title="Mois",
        yaxis_title="Score moyen",
        template='plotly_white',
        xaxis=dict(tickangle=45),
        plot_bgcolor='rgb(30,30,30)',
        paper_bgcolor='rgb(30,30,30)',
        font=dict(color='white'),
        hovermode='closest',
        showlegend=False
    )

    return fig_sentiment,fig_label,fig_sent_tem




        # Générer et afficher le nuage de mots
    


 



    return fig_mots,fig_sentiment

            






@app.callback(
    Output("prediction_result", "children"),
    Input("btn_predire", "n_clicks"),
    State("input_age", "value"),
    State("dropdown_niveau_etude", "value"),
    State("dropdown_genre", "value"),
    State("dropdown_situation_mat", "value"),
    State("dropdown_deja_donne", "value"),
    State("input_taux_hb", "value"),
    State("dropdown_arrondissement", "value"),
    State("dropdown_religion", "value"),
    State("dropdown_categorie", "value"),
    State("checklist_indisponibilite", "value"),
    State("radio_ddr_recente", "value"),  # Changed from checklist_indisponibilite_femme
    State("radio_allaitement", "value"),  # Changed from checklist_indisponibilite_femme
    State("radio_enceinte", "value"),    # Changed from checklist_indisponibilite_femme
    State("radio_accouchement_recent", "value"),  # Changed from checklist_indisponibilite_femme
    State("checklist_non_eligibilite", "value"),
    prevent_initial_call=True
)
def update_prediction(n_clicks, age, niveau_etude, genre, situation_mat, deja_donne, taux_hb,
                     arrondissement, religion, categorie, indispo, ddr_recente, 
                     allaitement, enceinte, accouchement_recent, non_eligible):

    if n_clicks is None:
        return ""

    # Construire le dictionnaire de données
    data = {
        "Age": age,
        "Niveau d'etude": niveau_etude,
        "Genre ": genre,
        "Situation Matrimoniale (SM)": situation_mat,
        "A-t-il (elle) déjà donné le sang ": deja_donne,
        "Taux d’hémoglobine ": taux_hb,
        "Arrondissement_final": arrondissement,
        "New_Religion": religion,
        "Categorie_final": categorie,
    }

    # 1. Gestion standard des cases à cocher (Oui/Non)
    for col in indispo_vars + non_eligible_vars:
        data[col] = "Oui" if col in (indispo + non_eligible) else "Non"

    # 2. Gestion spéciale pour les indisponibilités femmes
     # Map radio button values to exact feature names
    women_mapping = {
        'Raison de l’indisponibilité de la femme [La DDR est mauvais si <14 jour avant le don]': ddr_recente.split('_')[-1],
        'Raison de l’indisponibilité de la femme [Allaitement ]': allaitement.split('_')[-1],
        'Raison de l’indisponibilité de la femme [est enceinte ]': enceinte.split('_')[-1],
        'Raison de l’indisponibilité de la femme [A accoucher ces 6 derniers mois  ]': accouchement_recent.split('_')[-1],
        # Set default value for interruption de grossesse if radio button doesn't exist
        'Raison de l’indisponibilité de la femme [Interruption de grossesse  ces 06 derniers mois]': "Non"  # Default value
    }
    

    # Add all women-specific variables
    for col in indispo_femme_vars:
        if genre == 'Femme':
            data[col] = women_mapping.get(col, "Non")
        else:
            data.loc[:, 'col'] = "Non concerné"
    
    
   

    resultat_prediction = faire_prediction(data)
    return f"Résultat : {resultat_prediction}"



#host = '0.0.0.0'  # Utilise '0.0.0.0' pour accepter les connexions externes
port = int(os.environ.get('PORT', 8050))  # Utilise la variable d'environnement PORT ou 8050 par défaut

port = int(os.environ.get('PORT', 8050))
    server.run(host='0.0.0.0', port=port, debug=False)  # Run the Flask server
