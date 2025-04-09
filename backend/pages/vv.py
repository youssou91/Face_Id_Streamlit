import streamlit as st
import numpy as np
import os
import glob
from skimage import io
import cv2

# Chargement des fichiers de caractéristiques extraites
def load_features(descriptor_type):
    file_path = f"features/{descriptor_type}_features.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        st.error(f"Les caractéristiques pour {descriptor_type} ne sont pas disponibles.")
        return None

# Calcul des descripteurs GLCM (Haralick) avec OpenCV
def extract_glcm_features_cv(image):
    # Convertir l'image en niveaux de gris
    grey_image = np.array(image.convert('L'), dtype=np.uint8)
    
    # Calculer la matrice GLCM pour un décalage (1 pixel) à un angle (0°)
    glcm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey_image)
    
    # Calculez les propriétés de GLCM
    contrast = cv2.calcHist([glcm], [0], None, [256], [0, 256])
    correlation = np.corrcoef(glcm.flatten())
    energy = np.sum(glcm**2)
    homogeneity = np.sum(glcm / (1 + np.abs(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1]))))
    
    return np.concatenate([contrast.flatten(), correlation.flatten(), [energy], [homogeneity]])

# Distance Euclidienne
def euclidean_distance(features1, features2):
    return np.linalg.norm(features1 - features2)

# Distance Manhattan
def manhattan_distance(features1, features2):
    return np.sum(np.abs(features1 - features2))

# Distance Tchebychev
def chebyshev_distance(features1, features2):
    return np.max(np.abs(features1 - features2))

# Distance Canberra
def canberra_distance(features1, features2):
    return np.sum(np.abs(features1 - features2) / (np.abs(features1) + np.abs(features2)))

# Calcul de la similarité
def calculate_similarities_cv(query_features, all_features, distance_metric):
    distances = []
    for features in all_features:
        if distance_metric == 'euclidean':
            distances.append(euclidean_distance(query_features, features))
        elif distance_metric == 'manhattan':
            distances.append(manhattan_distance(query_features, features))
        elif distance_metric == 'chebyshev':
            distances.append(chebyshev_distance(query_features, features))
        elif distance_metric == 'canberra':
            distances.append(canberra_distance(query_features, features))
    return distances

# Fonction de recherche d'images
def search_images_cv(query_image, num_results, distance_metric, descriptor_type):
    query_features = extract_glcm_features_cv(query_image)
    all_features = load_features(descriptor_type)

    if all_features is not None:
        similarities = calculate_similarities_cv(query_features, all_features, distance_metric)
        sorted_indexes = np.argsort(similarities)[:num_results]

        return sorted_indexes, similarities
    return None, None

# Configuration de la page
st.set_page_config(page_title="CBIR - Recherche d'Images", layout="wide")
st.title("🔍 Recherche d'Images Basée sur le Contenu (CBIR)")

# Interface utilisateur
st.sidebar.header("Paramètres de Recherche")
image_file = st.sidebar.file_uploader("Télécharger une image de requête", type=["jpg", "jpeg", "png"])

num_results = st.sidebar.slider("Nombre de résultats à afficher", 1, 10, 5)
distance_metric = st.sidebar.selectbox("Choisir la mesure de distance", ["euclidean", "manhattan", "chebyshev", "canberra"])
descriptor_type = st.sidebar.selectbox("Choisir le descripteur", ["glcm", "haralick", "bit"])

if image_file:
    query_image = io.imread(image_file)
    st.image(query_image, caption="Image de requête", use_column_width=True)

    if st.button("Rechercher des images similaires"):
        st.spinner("Recherche des images similaires...")

        # Recherche d'images similaires
        sorted_indexes, similarities = search_images_cv(query_image, num_results, distance_metric, descriptor_type)

        if sorted_indexes is not None:
            st.subheader(f"Résultats : {num_results} images similaires")

            # Afficher les résultats
            for i, idx in enumerate(sorted_indexes):
                image_path = f"dataset_images/{idx}.jpg"
                result_image = io.imread(image_path)
                st.image(result_image, caption=f"Résultat {i+1} - Similarité: {similarities[idx]:.4f}", use_column_width=True)
        else:
            st.error("Erreur dans la recherche d'images.")
else:
    st.warning("Veuillez télécharger une image pour effectuer la recherche.")
