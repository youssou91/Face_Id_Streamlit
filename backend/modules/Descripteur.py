import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops

def descripteur_GLCM(image):
    """
    Calcule un descripteur basé sur la matrice de cooccurrence des niveaux de gris (GLCM).
    L'image doit être un tableau numpy. Si l'image est en couleur, elle sera convertie en niveaux de gris.

    Args:
        image (np.array): Image en format numpy (RGB ou niveaux de gris).

    Returns:
        np.array: Vecteur de caractéristiques composé des propriétés du GLCM.
    """
    # Conversion en niveau de gris si nécessaire
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Assurer que l'image est en uint8
    if gray.dtype != 'uint8':
        gray = (255 * (gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-10)).astype('uint8')

    # Calcul du GLCM
    glcm = greycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Calcul des propriétés : contraste, dissimilarité, homogénéité, énergie, corrélation et ASM
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    ASM = greycoprops(glcm, 'ASM')[0, 0]

    # Création du vecteur de caractéristiques
    features = np.array([contrast, dissimilarity, homogeneity, energy, correlation, ASM])
    return features
