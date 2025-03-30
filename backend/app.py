import streamlit as st
import face_recognition
import bcrypt
import numpy as np
import pickle
import io
import requests
import os
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from google_auth_oauthlib.flow import Flow
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la base de données
DATABASE_URL = 'mysql://root:@localhost/db_ia'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Modèle Utilisateur
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

Base.metadata.create_all(engine)

# Configuration OAuth
GOOGLE_CLIENT_ID = os.getenv('1078916682099-pnqp77d6jbnb0ut8i0u6efprcmr9am6i.apps.googleusercontent.com')
GOOGLE_CLIENT_SECRET = os.getenv('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')
FACEBOOK_CLIENT_ID = os.getenv('FACEBOOK_CLIENT_ID')
FACEBOOK_CLIENT_SECRET = os.getenv('FACEBOOK_CLIENT_SECRET')
REDIRECT_URI = 'http://localhost:8501/'

# Fonctions utilitaires
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def get_face_encoding(image_data):
    try:
        img = face_recognition.load_image_file(io.BytesIO(image_data))
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) != 1:
            return None, "L'image doit contenir exactement un visage"
        return face_recognition.face_encodings(img, face_locations)[0], None
    except Exception as e:
        return None, str(e)

# Interface Streamlit
st.set_page_config(page_title="Auth App", layout="wide")
st.title("🛡️ Application d'Authentification Multimodale")

# Gestion de l'état de session
if 'user' not in st.session_state:
    st.session_state.user = None

# Menu principal
menu_options = [
    "Connexion Standard",
    "Inscription",
    "Connexion Faciale", 
    "Connexion Google",
    "Connexion Facebook"
]

# Définir le menu avec des paramètres stables
choice = st.sidebar.selectbox(
    "Menu",
    options=menu_options,
    key="main_navigation",  
    index=0 
)
# Section Inscription
if choice == "Inscription":
    st.header("📝 Créer un compte")
    
    with st.form("register_form"):
        username = st.text_input("Nom d'utilisateur")
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type="password")
        confirm_password = st.text_input("Confirmer le mot de passe", type="password")
        image_source = st.radio("Source de l'image", ["Téléchargement", "Caméra"])
        
        image_data = None
        if image_source == "Téléchargement":
            image_data = st.file_uploader("Télécharger une photo", type=["jpg", "jpeg", "png"])
        else:
            image_data = st.camera_input("Prendre une photo")
        
        submitted = st.form_submit_button("S'inscrire")
        
        if submitted:
            if password != confirm_password:
                st.error("Les mots de passe ne correspondent pas")
            else:
                face_encoding = None
                if image_data:
                    encoding, error = get_face_encoding(image_data.getvalue())
                    if error:
                        st.error(error)
                    else:
                        face_encoding = pickle.dumps(encoding)
                
                try:
                    new_user = User(
                        username=username,
                        email=email,
                        password_hash=hash_password(password),
                        face_encoding=face_encoding
                    )
                    session.add(new_user)
                    session.commit()
                    st.success("Compte créé avec succès!")
                except IntegrityError:
                    session.rollback()
                    st.error("Nom d'utilisateur ou email déjà utilisé")

# Section Connexion Standard
elif choice == "Connexion Standard":
    st.header("🔑 Connexion Standard")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")
        
        if submitted:
            user = session.query(User).filter_by(email=email).first()
            if user and check_password(password, user.password_hash):
                st.session_state.user = user
                st.success(f"Bienvenue {user.username}!")
            else:
                st.error("Identifiants incorrects")

# Section Connexion Faciale
elif choice == "Connexion Faciale":
    st.header("📸 Connexion Faciale")
    
    with st.form("face_login"):
        image_source = st.radio("Source", ["Téléchargement", "Caméra"])
        threshold = st.slider("Seuil de confiance", 0.3, 0.9, 0.6)
        
        image_data = None
        if image_source == "Téléchargement":
            image_data = st.file_uploader("Télécharger une photo", type=["jpg", "jpeg", "png"])
        else:
            image_data = st.camera_input("Prendre une photo")
        
        submitted = st.form_submit_button("Vérifier l'identité")
        
        if submitted and image_data:
            encoding, error = get_face_encoding(image_data.getvalue())
            if error:
                st.error(error)
            else:
                users = session.query(User).filter(User.face_encoding.isnot(None)).all()
                best_match = None
                min_distance = 1.0
                
                for user in users:
                    stored_enc = pickle.loads(user.face_encoding)
                    distance = face_recognition.face_distance([stored_enc], encoding)[0]
                    if distance < min_distance and distance < threshold:
                        min_distance = distance
                        best_match = user
                
                if best_match:
                    st.session_state.user = best_match
                    st.success(f"Bienvenue {best_match.username}! (Confiance: {(1 - min_distance)*100:.1f}%)")
                else:
                    st.error("Aucun utilisateur reconnu")

# Section Connexion Google
elif choice == "Connexion Google":
    st.header("🔵 Connexion avec Google")
    
    flow = Flow.from_client_config(
        client_config={
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=["openid", "https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"],
        redirect_uri=REDIRECT_URI
    )
    
    # Génération de l'URL d'autorisation
    auth_url, _ = flow.authorization_url(prompt="consent")
    st.markdown(f"[Se connecter avec Google]({auth_url})")
    
    # Gestion du callback avec la nouvelle syntaxe
    query_params = st.query_params
    if 'code' in query_params:
        try:
            flow.fetch_token(code=query_params['code'])
            credentials = flow.credentials
            user_info = requests.get(
                "https://www.googleapis.com/oauth2/v1/userinfo",
                headers={"Authorization": f"Bearer {credentials.token}"}
            ).json()
            
            user = session.query(User).filter_by(google_id=user_info['id']).first()
            if not user:
                user = User(
                    username=user_info['name'],
                    email=user_info['email'],
                    google_id=user_info['id']
                )
                session.add(user)
                session.commit()
            
            st.session_state.user = user
            st.success(f"Bienvenue {user.username}!")
        except Exception as e:
            st.error(f"Erreur d'authentification: {str(e)}")

# Section Connexion Facebook
elif choice == "Connexion Facebook":
    st.header("🔵 Connexion avec Facebook")
    
    facebook = OAuth2Session(
        client_id=FACEBOOK_CLIENT_ID,
        redirect_uri=REDIRECT_URI
    )
    
    auth_url, _ = facebook.authorization_url(
        "https://www.facebook.com/v12.0/dialog/oauth"
    )
    st.markdown(f"[Se connecter avec Facebook]({auth_url})")
    
    # Gestion du callback avec la nouvelle syntaxe
    query_params = st.query_params
    if 'code' in query_params:
        try:
            token = facebook.fetch_token(
                "https://graph.facebook.com/v12.0/oauth/access_token",
                client_secret=FACEBOOK_CLIENT_SECRET,
                code=query_params['code']
            )
            user_info = facebook.get(
                "https://graph.facebook.com/me?fields=id,name,email"
            ).json()
            
            user = session.query(User).filter_by(facebook_id=user_info['id']).first()
            if not user:
                user = User(
                    username=user_info['name'],
                    email=user_info.get('email', ''),
                    facebook_id=user_info['id']
                )
                session.add(user)
                session.commit()
            
            st.session_state.user = user
            st.success(f"Bienvenue {user.username}!")
        except Exception as e:
            st.error(f"Erreur d'authentification: {str(e)}")

# Affichage de l'utilisateur connecté
if st.session_state.user:
    st.sidebar.success(f"Connecté en tant que {st.session_state.user.username}")
    if st.sidebar.button("Déconnexion"):
        st.session_state.user = None
        st.rerun()