import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
import numpy as np
import pickle
import face_recognition
import bcrypt
import io

# Configuration de la base de données
DATABASE_URL = 'mysql://root:@localhost/db_ia'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Définition du modèle User
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128))
    face_encoding = Column(LargeBinary)
    google_id = Column(String(100), unique=True)
    facebook_id = Column(String(100), unique=True)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email
        }

# Création des tables si elles n'existent pas
Base.metadata.create_all(engine)

# Fonctions d'aide pour le hachage des mots de passe
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def get_face_encoding_from_image(image_data):
    try:
        file_bytes = np.frombuffer(image_data.read(), np.uint8)
        image = face_recognition.load_image_file(io.BytesIO(file_bytes))
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) != 1:
            return None, "L'image doit contenir exactement un visage."
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        return face_encoding, None
    except Exception as e:
        return None, f"Erreur lors du traitement de l'image : {e}"

# Interface Streamlit
st.title("Application d'Authentification Streamlit")
menu = st.sidebar.radio("Menu", options=["Connexion", "Inscription", "Connexion par visage"])

if menu == "Inscription":
    st.header("Inscription")
    username = st.text_input("Nom d'utilisateur")
    email = st.text_input("Email")
    password = st.text_input("Mot de passe", type="password")
    confirm_password = st.text_input("Confirmer le mot de passe", type="password")
    
    st.subheader("Ajouter une photo de votre visage")
    # Choix de la méthode : téléchargement ou capture par caméra
    image_source = st.radio("Méthode de capture", ("Télécharger une image", "Utiliser la caméra"))
    
    image_data = None
    if image_source == "Télécharger une image":
        image_data = st.file_uploader("Télécharger une photo", type=["jpg", "jpeg", "png"])
    else:
        image_data = st.camera_input("Capturez votre image")
    
    if st.button("S'inscrire"):
        if not (username and email and password and confirm_password):
            st.error("Veuillez remplir tous les champs.")
        elif password != confirm_password:
            st.error("Les mots de passe ne correspondent pas.")
        else:
            face_encoding = None
            if image_data is not None:
                encoding, error = get_face_encoding_from_image(image_data)
                if error:
                    st.error(error)
                else:
                    face_encoding = encoding
            
            try:
                new_user = User(
                    username=username,
                    email=email,
                    password_hash=hash_password(password)
                )
                if face_encoding is not None:
                    new_user.face_encoding = pickle.dumps(face_encoding)
                session.add(new_user)
                session.commit()
                st.success("Inscription réussie!")
            except IntegrityError:
                session.rollback()
                st.error("Nom d'utilisateur ou email déjà utilisé.")
            except Exception as e:
                session.rollback()
                st.error(f"Erreur : {e}")

elif menu == "Connexion":
    st.header("Connexion")
    email = st.text_input("Email")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        user = session.query(User).filter_by(email=email).first()
        if user and check_password(password, user.password_hash):
            st.success(f"Bienvenue, {user.username}!")
            st.write("Informations utilisateur :", user.to_dict())
        else:
            st.error("Identifiants invalides.")

elif menu == "Connexion par visage":
    st.header("Connexion par reconnaissance faciale")
    
    st.subheader("Fournissez votre image")
    # Choix de la méthode : téléchargement ou capture par caméra
    image_source = st.radio("Méthode de capture", ("Télécharger une image", "Utiliser la caméra"))
    
    image_data = None
    if image_source == "Télécharger une image":
        image_data = st.file_uploader("Télécharger une photo", type=["jpg", "jpeg", "png"])
    else:
        image_data = st.camera_input("Capturez votre image")
    
    threshold = st.number_input("Seuil de correspondance", value=0.6)
    if st.button("Se connecter"):
        if image_data is None:
            st.error("Veuillez fournir une image.")
        else:
            encoding, error = get_face_encoding_from_image(image_data)
            if error:
                st.error(error)
            else:
                face_encoding = encoding
                users = session.query(User).filter(User.face_encoding != None).all()
                best_match = None
                min_distance = 1.0
                for user in users:
                    stored_encoding = pickle.loads(user.face_encoding)
                    distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
                    if distance < min_distance and distance < threshold:
                        min_distance = distance
                        best_match = user
                if best_match:
                    st.success(f"Visage reconnu! Bienvenue, {best_match.username}. Confiance : {round(1 - min_distance, 2)}")
                    st.write("Informations utilisateur :", best_match.to_dict())
                else:
                    st.error("Visage non reconnu.")
