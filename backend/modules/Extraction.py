import numpy as np
import cv2

def concatener(image):
    """
    Extrait un vecteur de caractéristiques en concaténant les histogrammes de couleur HSV.
    Cette méthode calcule un histogramme pour chaque canal (H, S, V) et les normalise,
    puis concatène les trois histogrammes en un vecteur de caractéristiques.

    Args:
        image (np.array): Image en format numpy (RGB).

    Returns:
        np.array: Vecteur de caractéristiques concaténé.
    """
    # Convertir l'image de RGB à HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculer l'histogramme pour chaque canal avec 32 bins
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    
    # Normaliser et aplatir chaque histogramme
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    # Concaténer les histogrammes en un seul vecteur de caractéristiques
    feature_vector = np.concatenate([h_hist, s_hist, v_hist])
    return feature_vector
