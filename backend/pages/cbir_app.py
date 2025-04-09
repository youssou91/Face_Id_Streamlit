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

# Configuration des chemins (à vérifier absolument)
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

# Fonctions d'extraction de caractéristiques avec gestion d'erreur
def extract_glcm_features(image):
    try:
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image
        gray = (gray * 255).astype(np.uint8)
        glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
        return np.array([np.mean(graycoprops(glcm, prop)) for prop in ['contrast', 'correlation', 'energy', 'homogeneity']])
    except Exception as e:
        st.error(f"Erreur GLCM: {str(e)}")
        return np.zeros(4)  # Valeur par défaut

def extract_haralick_features(image):
    try:
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image
        gray = (gray * 255).astype(np.uint8)
        glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
        return np.array([np.mean(graycoprops(glcm, prop)) for prop in ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity', 'ASM']])
    except Exception as e:
        st.error(f"Erreur Haralick: {str(e)}")
        return np.zeros(6)  # Valeur par défaut

def extract_bit_features(image):
    try:
        image = cv2.resize(image, (224, 224))
        image = tf.image.convert_image_dtype(image, tf.float32)
        return bit_model(tf.expand_dims(image, axis=0)).numpy().flatten()
    except Exception as e:
        st.error(f"Erreur BiT: {str(e)}")
        return np.zeros(2048)  # Dimension par défaut pour BiT-m-r50x1

def extract_concatenated_features(image):
    return np.concatenate([
        extract_glcm_features(image),
        extract_haralick_features(image),
        extract_bit_features(image)
    ])

# Fonctions de distance avec gestion de division par zéro
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))

def canberra_distance(a, b):
    with np.errstate(invalid='ignore'):
        return np.sum(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12))

# Chargement des caractéristiques avec normalisation
def load_features(descriptor):
    features_path = os.path.join(FEATURES_DIR, f"{descriptor}_features.npy")
    paths_path = os.path.join(FEATURES_DIR, "image_paths.npy")

    if not os.path.exists(features_path) or not os.path.exists(paths_path):
        st.info(f"Précalcul des caractéristiques {descriptor}...")
        image_paths = []
        features = []
        
        for root, _, files in os.walk(DATASET_PATH):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    try:
                        # Chargement d'image avec vérification
                        img = cv2.imread(path)
                        if img is None:
                            st.error(f"Image non lue: {path}")
                            continue
                        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Extraction des caractéristiques
                        if descriptor == "glcm":
                            feat = extract_glcm_features(image)
                        elif descriptor == "haralick":
                            feat = extract_haralick_features(image)
                        elif descriptor == "bit":
                            feat = extract_bit_features(image)
                        else:
                            feat = extract_concatenated_features(image)
                            
                        features.append(feat)
                        image_paths.append(path)
                        
                    except Exception as e:
                        st.error(f"Erreur avec {path}: {str(e)}")
                        continue

        # Normalisation des caractéristiques
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        
        np.save(features_path, features)
        np.save(paths_path, image_paths)
    
    features = np.load(features_path)
    image_paths = np.load(paths_path)
    
    # Vérification finale
    if len(features) == 0 or len(image_paths) == 0:
        st.error("Aucune caractéristique trouvée! Vérifiez le dataset")
        return None, None
    
    return features, image_paths

# Interface Streamlet
def main():
    st.title("🔍 Système de Recherche d'Images par Contenu")
    
    # Vérification initiale du dataset
    if not os.path.exists(DATASET_PATH) or len(os.listdir(DATASET_PATH)) == 0:
        st.error("Dataset introuvable ou vide! Placez des images dans le dossier 'dataset/'")
        return

    # Paramètres
    st.sidebar.header("Paramètres")
    descriptor = st.sidebar.selectbox("Descripteur", ["GLCM", "Haralick", "BiT", "Concatenated"])
    distance_metric = st.sidebar.selectbox("Mesure de distance", ["Euclidean", "Manhattan", "Chebyshev", "Canberra"])
    top_n = st.sidebar.slider("Nombre de résultats", 1, 20, 5)
    
    # Téléversement d'image
    query_file = st.file_uploader("Téléversez une image requête", type=["jpg", "jpeg", "png"])
    
    if query_file:
        try:
            # Traitement de l'image requête
            query_img = Image.open(query_file).convert('RGB')
            st.image(query_img, caption="Image Requête", width=300)
            query_image = np.array(query_img)
            query_image = cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR)
            
            # Extraction des caractéristiques
            descriptor = descriptor.lower()
            if descriptor == "glcm":
                query_features = extract_glcm_features(query_image)
            elif descriptor == "haralick":
                query_features = extract_haralick_features(query_image)
            elif descriptor == "bit":
                query_features = extract_bit_features(query_image)
            else:
                query_features = extract_concatenated_features(query_image)
            
            # Chargement du dataset
            dataset_features, image_paths = load_features(descriptor)
            
            if dataset_features is None:
                return
                
            # Vérification de compatibilité
            if query_features.shape[0] != dataset_features.shape[1]:
                st.error(f"Incompatibilité de dimensions: {query_features.shape} vs {dataset_features.shape[1:]}")
                return
                
            # Calcul des distances
            distance_func = globals()[f"{distance_metric.lower()}_distance"]
            distances = [distance_func(query_features, feat) for feat in dataset_features]
            distances = np.nan_to_num(distances, nan=np.inf)
            
            # Tri des résultats
            sorted_indices = np.argsort(distances)[:top_n]
            
            # Affichage
            st.subheader(f"Top {top_n} Images Similaires")
            
            if len(sorted_indices) == 0:
                st.warning("Aucun résultat trouvé - Essayez d'ajuster les paramètres")
            else:
                cols = st.columns(3)
                for i, idx in enumerate(sorted_indices):
                    with cols[i % 3]:
                        try:
                            img = Image.open(image_paths[idx])
                            st.image(img, use_column_width=True)
                            st.caption(f"Distance: {distances[idx]:.2f} - {os.path.basename(image_paths[idx])}")
                        except Exception as e:
                            st.error(f"Erreur d'affichage: {str(e)}")
            
            # Section debug
            with st.expander("Informations de débogage"):
                st.write("Dimensions des caractéristiques requête:", query_features.shape)
                st.write("Dimensions du dataset:", dataset_features.shape)
                st.write("Exemple de chemins:", image_paths[:3])
                st.write("Distances calculées:", distances[:5])
                
        except Exception as e:
            st.error(f"Erreur générale: {str(e)}")

if __name__ == "__main__":
    main()
