import streamlit as st
import numpy as np
import cv2
import os
from skimage.feature import graycomatrix, graycoprops
from skimage import color
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from joblib import dump, load
import pandas as pd

# Importation des modules externes si disponibles
try:
    from Descripteur import descripteur_GLCM
except ImportError:
    descripteur_GLCM = None

try:
    from Distance import calculer_distance
except ImportError:
    calculer_distance = None

try:
    from Extraction import concatener as hsv_concatener
except ImportError:
    hsv_concatener = None

# Vérification de l'authentification de l'utilisateur
if not st.session_state.get("logged_in", False):
    st.error("Vous devez être connecté pour accéder à cette page.")
    st.warning("Veuillez vous connecter via la page de connexion.")
    st.stop()

# Configuration des chemins
DATASET_PATH = os.path.abspath("dataset/")  # Chemin absolu
FEATURES_DIR = os.path.abspath("features/")

# Création des dossiers si nécessaire
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Chargement du modèle BiT une seule fois
@st.cache_resource
def load_bit_model():
    return hub.load("https://tfhub.dev/google/bit/m-r50x1/1")

bit_model = load_bit_model()

# Fonctions d'extraction internes
def extract_glcm_features(image):
    try:
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image
        gray = (gray * 255).astype(np.uint8)
        glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135],
                             levels=256, symmetric=True, normed=True)
        features = np.array([np.mean(graycoprops(glcm, prop)) for prop in 
                             ['contrast', 'correlation', 'energy', 'homogeneity']])
        return features
    except Exception as e:
        st.error(f"Erreur GLCM: {str(e)}")
        return np.zeros(4)

def extract_haralick_features(image):
    try:
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image
        gray = (gray * 255).astype(np.uint8)
        glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135],
                             levels=256, symmetric=True, normed=True)
        features = np.array([np.mean(graycoprops(glcm, prop)) for prop in 
                             ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity', 'ASM']])
        return features
    except Exception as e:
        st.error(f"Erreur Haralick: {str(e)}")
        return np.zeros(6)

def extract_bit_features(image):
    try:
        image = cv2.resize(image, (224, 224))
        image = tf.image.convert_image_dtype(image, tf.float32)
        features = bit_model(tf.expand_dims(image, axis=0)).numpy().flatten()
        return features
    except Exception as e:
        st.error(f"Erreur BiT: {str(e)}")
        return np.zeros(2048)

def extract_concatenated_features(image):
    return np.concatenate([
        extract_glcm_features(image),
        extract_haralick_features(image),
        extract_bit_features(image)
    ])

# Dictionnaire des fonctions d'extraction à utiliser
extraction_functions = {
    "GLCM": extract_glcm_features,
    "Haralick": extract_haralick_features,
    "BiT": extract_bit_features,
    "Concatenated": extract_concatenated_features
}
if descripteur_GLCM is not None:
    extraction_functions["GLCM_Module"] = descripteur_GLCM
if hsv_concatener is not None:
    extraction_functions["HSV_Histogram"] = hsv_concatener

# Fonctions de distance internes
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))

def canberra_distance(a, b):
    with np.errstate(invalid='ignore'):
        return np.sum(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12))

distance_functions = {
    "Euclidean": euclidean_distance,
    "Manhattan": manhattan_distance,
    "Chebyshev": chebyshev_distance,
    "Canberra": canberra_distance
}

# Option pour utiliser la fonction de distance du module externe
use_distance_module = st.sidebar.checkbox("Utiliser le module distance externe", value=False)
if use_distance_module and calculer_distance is not None:
    distance_functions["Module_Distance"] = calculer_distance

# Chargement et éventuellement précalcul des caractéristiques
def load_features(descriptor):
    features_path = os.path.join(FEATURES_DIR, f"{descriptor}_features.npy")
    paths_path = os.path.join(FEATURES_DIR, "image_paths.npy")
    scaler_path = os.path.join(FEATURES_DIR, f"{descriptor}_scaler.joblib")

    if not os.path.exists(features_path) or not os.path.exists(paths_path) or not os.path.exists(scaler_path):
        st.info(f"Précalcul des caractéristiques avec le descripteur {descriptor.upper()}...")
        image_paths = []
        features = []
        
        # Utiliser la fonction d'extraction choisie
        extraction_function = extraction_functions.get(descriptor)
        if extraction_function is None:
            st.error("Fonction d'extraction introuvable pour ce descripteur.")
            return None, None, None
        
        for root, _, files in os.walk(DATASET_PATH):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    try:
                        img = cv2.imread(path)
                        if img is None:
                            continue
                        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        feat = extraction_function(image)
                        features.append(feat)
                        image_paths.append(path)
                    except Exception as e:
                        continue

        scaler = MinMaxScaler()
        features = scaler.fit_transform(np.array(features))
        
        np.save(features_path, features)
        np.save(paths_path, image_paths)
        dump(scaler, scaler_path)
    
    features = np.load(features_path)
    image_paths = np.load(paths_path)
    scaler = load(scaler_path)
    
    if len(features) == 0 or len(image_paths) == 0:
        st.error("Aucune caractéristique trouvée ! Vérifiez le dataset.")
        return None, None, None
    
    return features, image_paths, scaler

def main():
    st.title("🔍 Système de Recherche d'Images par Contenu")
    
    if not os.path.exists(DATASET_PATH) or len(os.listdir(DATASET_PATH)) == 0:
        st.error("Dataset introuvable ou vide ! Placez des images dans le dossier 'dataset/'")
        return

    # Choix du descripteur via la barre latérale
    available_descriptors = list(extraction_functions.keys())
    descriptor = st.sidebar.selectbox("Descripteur", available_descriptors)
    
    distance_metric = st.sidebar.selectbox("Mesure de distance", list(distance_functions.keys()))
    top_n = st.sidebar.slider("Nombre de résultats", 1, 20, 5)
    
    query_file = st.file_uploader("Téléversez une image requête", type=["jpg", "jpeg", "png"])
    
    if query_file:
        try:
            query_img = Image.open(query_file).convert('RGB')
            st.image(query_img, caption="Image Requête", width=300)
            query_image = np.array(query_img)
            
            # Extraction des caractéristiques de la requête avec la fonction choisie
            extraction_function = extraction_functions.get(descriptor)
            if extraction_function is None:
                st.error("Fonction d'extraction introuvable pour ce descripteur.")
                return
            
            query_features = extraction_function(query_image)
            
            dataset_features, image_paths, scaler = load_features(descriptor)
            if dataset_features is None:
                return
                
            # Normalisation de la caractéristique de la requête
            query_features_scaled = scaler.transform([query_features])[0]
            
            # Calcul des distances
            if distance_metric == "Module_Distance" and calculer_distance is not None:
                distances = calculer_distance(query_features_scaled, dataset_features, distance_type="euclidienne")
            else:
                distance_func = distance_functions.get(distance_metric)
                distances = [distance_func(query_features_scaled, feat) for feat in dataset_features]
                distances = np.nan_to_num(distances, nan=np.inf)
            
            # Tri des indices
            sorted_indices = np.argsort(distances)[:top_n]
            
            st.subheader(f"Top {top_n} Images Similaires")
            if len(sorted_indices) == 0:
                st.warning("Aucun résultat trouvé - Essayez d'ajuster les paramètres")
            else:
                # Création d'un DataFrame pour résumer les résultats
                result_data = []
                cols = st.columns(3)
                for i, idx in enumerate(sorted_indices):
                    try:
                        filename = os.path.basename(image_paths[idx])
                        dist_val = distances[idx]
                        result_data.append({"Fichier": filename, "Distance": f"{dist_val:.2f}"})
                        with cols[i % 3]:
                            img_disp = Image.open(image_paths[idx])
                            st.image(img_disp, use_container_width=True)
                            st.caption(f"Distance: {dist_val:.2f}\nFichier: {filename}")
                    except Exception as e:
                        st.error(f"Erreur d'affichage: {str(e)}")
                # Affichage du résumé des résultats sous forme de tableau
                st.markdown("### Résumé des images similaires")
                st.dataframe(pd.DataFrame(result_data))
            
            # Section détaillée d'analyse
            with st.expander("Détails de l'analyse"):
                # Tableau du vecteur de caractéristiques normalisé de la requête
                st.markdown("**Vecteur de caractéristiques de la requête (normalisé) :**")
                feat_df = pd.DataFrame(query_features_scaled.reshape(1, -1),
                                       columns=[f"f{i+1}" for i in range(len(query_features_scaled))])
                st.dataframe(feat_df)
                
                # Affichage des distances dans un tableau détaillé
                st.markdown("**Distances calculées avec chaque image du dataset :**")
                all_data = []
                for i, (path, dist) in enumerate(zip(image_paths, distances)):
                    all_data.append({"Index": i,
                                     "Fichier": os.path.basename(path),
                                     "Distance": f"{dist:.2f}"})
                st.dataframe(pd.DataFrame(all_data))
                
        except Exception as e:
            st.error(f"Erreur générale: {str(e)}")

if __name__ == "__main__":
    main()
