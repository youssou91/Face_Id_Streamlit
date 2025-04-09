import numpy as np

def calculer_distance(vecteur, data, distance_type='euclidienne'):
    """
    Calcule la distance entre un vecteur de requête et chacun des vecteurs dans la collection 'data'.
    
    Args:
        vecteur (np.array): Vecteur de caractéristiques de la requête.
        data (np.array): Tableau contenant les vecteurs de caractéristiques de la base de données.
        distance_type (str): Type de distance à utiliser parmi : 'euclidienne', 'manhattan', 'tchebychev', 'canberra'.
        
    Returns:
        np.array: Tableau des distances calculées.
    """
    distances = []
    for d in data:
        if distance_type == 'euclidienne':
            dist = np.linalg.norm(vecteur - d)
        elif distance_type == 'manhattan':
            dist = np.sum(np.abs(vecteur - d))
        elif distance_type == 'tchebychev':
            dist = np.max(np.abs(vecteur - d))
        elif distance_type == 'canberra':
            # Pour éviter la division par zéro, on ajoute une petite valeur
            num = np.abs(vecteur - d)
            denom = np.abs(vecteur) + np.abs(d) + 1e-10
            dist = np.sum(num / denom)
        else:
            raise ValueError(f"Le type de distance '{distance_type}' n'est pas supporté.")
        distances.append(dist)
    return np.array(distances)
