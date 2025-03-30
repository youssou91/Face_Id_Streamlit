import face_recognition
import numpy as np
from PIL import Image
import io

def extraction_caracteristiques_visage(photo_file):
    """
    Extrait les caractéristiques du visage d'une photo
    """
    try:
        # Lire l'image depuis le FileStorage
        image_stream = io.BytesIO(photo_file.read())
        image = face_recognition.load_image_file(image_stream)
        
        # Détecter les visages
        face_locations = face_recognition.face_locations(image)
        
        # Vérifier qu'il y a exactement un visage
        if len(face_locations) != 1:
            return None
        
        # Calculer l'encodage du visage
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return face_encodings[0] if face_encodings else None
        
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques: {str(e)}")
        return None

def comparaison_reussi(photo_file, stored_encoding):
    """
    Compare le visage de la photo avec l'encodage stocké
    """
    try:
        # Extraire les caractéristiques du nouveau visage
        nouveau_encodage = extraction_caracteristiques_visage(photo_file)
        if nouveau_encodage is None:
            return False
        
        # Convertir l'encodage stocké en tableau numpy
        stored_encoding_array = np.frombuffer(stored_encoding)
        
        # Comparer les visages
        matches = face_recognition.compare_faces(
            [stored_encoding_array],
            nouveau_encodage,
            tolerance=0.6
        )
        
        return matches[0] if matches else False
        
    except Exception as e:
        print(f"Erreur lors de la comparaison: {str(e)}")
        return False