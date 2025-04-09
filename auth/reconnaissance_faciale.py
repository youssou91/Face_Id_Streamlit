import face_recognition
import numpy as np
import cv2

def process_face_image(image):
    
    try:
        # Détecter les visages dans l'image
        face_locations = face_recognition.face_locations(image)
        
        # Vérifier qu'il y a exactement un visage
        if len(face_locations) == 0:
            return {
                'valid': False,
                'message': 'Aucun visage détecté dans l\'image'
            }
        elif len(face_locations) > 1:
            return {
                'valid': False,
                'message': 'Plusieurs visages détectés dans l\'image'
            }
        
        # Calculer l'encodage du visage
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            return {
                'valid': False,
                'message': 'Impossible d\'extraire les caractéristiques du visage'
            }
        
        # Vérifier la qualité de l'image
        face_location = face_locations[0]
        face_image = image[face_location[0]:face_location[2], 
                         face_location[3]:face_location[1]]
        
        # Convertir en niveaux de gris pour le calcul de netteté
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < 100:  # Seuil arbitraire, à ajuster selon les besoins
            return {
                'valid': False,
                'message': 'Image trop floue. Veuillez prendre une photo plus nette'
            }
        
        return {
            'valid': True,
            'message': 'Visage détecté avec succès',
            'encoding': face_encodings[0]
        }
        
    except Exception as e:
        return {
            'valid': False,
            'message': f'Erreur lors du traitement de l\'image: {str(e)}'
        }

def verify_face(face_encoding, stored_encoding, tolerance=0.6):
    
    try:
        # Convertir les encodages en tableaux numpy si nécessaire
        if isinstance(stored_encoding, bytes):
            stored_encoding = np.frombuffer(stored_encoding)
        
        # Comparer les visages
        matches = face_recognition.compare_faces(
            [stored_encoding],
            face_encoding,
            tolerance=tolerance
        )
        
        # Calculer la distance entre les visages
        face_distances = face_recognition.face_distance(
            [stored_encoding],
            face_encoding
        )
        
        return {
            'match': bool(matches[0]),
            'confidence': float(1 - face_distances[0])
        }
        
    except Exception as e:
        return {
            'match': False,
            'confidence': 0.0,
            'error': str(e)
        }

def extract_face_features(image):
    """
    Extrait les points caractéristiques du visage pour une visualisation.
    """
    try:
        # Détecter le visage
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return {
                'valid': False,
                'message': 'Aucun visage détecté'
            }
        
        # Extraire les points caractéristiques
        face_landmarks = face_recognition.face_landmarks(image, face_locations)
        
        if not face_landmarks:
            return {
                'valid': False,
                'message': 'Impossible d\'extraire les points caractéristiques'
            }
        
        return {
            'valid': True,
            'landmarks': face_landmarks[0],
            'face_location': face_locations[0]
        }
        
    except Exception as e:
        return {
            'valid': False,
            'message': f'Erreur lors de l\'extraction: {str(e)}'
        }

def draw_face_landmarks(image, landmarks):
    """
    Dessine les points caractéristiques sur une image.
    """
    img_copy = image.copy()
    
    # Couleurs pour différentes parties du visage
    colors = {
        'chin': (255, 0, 0),
        'left_eyebrow': (0, 255, 0),
        'right_eyebrow': (0, 255, 0),
        'nose_bridge': (0, 0, 255),
        'nose_tip': (0, 0, 255),
        'left_eye': (255, 255, 0),
        'right_eye': (255, 255, 0),
        'top_lip': (255, 0, 255),
        'bottom_lip': (255, 0, 255)
    }
    
    # Dessiner chaque partie du visage
    for facial_feature, points in landmarks.items():
        color = colors.get(facial_feature, (255, 255, 255))
        points = np.array(points)
        cv2.polylines(img_copy, [points], True, color, 2)
    
    return img_copy