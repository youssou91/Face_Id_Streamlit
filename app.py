import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from database.db import init_db, add_user, verify_user, get_face_encodings, verify_face_login
from auth.reconnaissance_faciale import process_face_image, verify_face
from auth.social_auth import init_google_auth, google_callback
import cv2
import numpy as np
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Authentification Intelligente CBIR",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
    }
    .auth-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .webcam-box {
        border: 3px solid #4CAF50;
        border-radius: 15px;
        overflow: hidden;
    }
    .quality-bar {
        height: 8px;
        border-radius: 4px;
        background: #eee;
        margin: 10px 0;
    }
    .quality-fill {
        height: 100%;
        border-radius: 4px;
        background: #4CAF50;
        transition: width 0.5s ease;
    }
    .social-auth-button {
        width: 100%;
        margin: 5px 0;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .google-button {
        background-color: white;
        color: #757575;
    }
    .facebook-button {
        background-color: #1877f2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Gestion de l'état
if "auth_data" not in st.session_state:
    st.session_state.auth_data = {
        "authenticated": False,
        "user": None,
        "face_image": None,
        "face_encoding": None,
        "current_step": 1,
        "auth_method": "credentials"  # credentials, face, google
    }

class FaceCaptureTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return img

def face_login_interface():
    st.markdown("### Connexion par reconnaissance faciale")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        ctx = webrtc_streamer(
            key="face_login",
            video_transformer_factory=FaceCaptureTransformer,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        if ctx.video_transformer and st.button("Vérifier mon visage"):
            frame = ctx.video_transformer.frame
            if frame is not None:
                process_face_login(frame)

    with col2:
        uploaded_file = st.file_uploader("Ou uploadez une photo", type=["jpg", "png"])
        if uploaded_file:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            process_face_login(img)

def process_face_login(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_data = process_face_image(img_rgb)
    
    if face_data["valid"]:
        user = verify_face_login(face_data["encoding"])
        if user:
            st.session_state.auth_data.update({
                "authenticated": True,
                "user": user
            })
            st.success("Authentification réussie!")
            st.experimental_rerun()
        else:
            st.error("Visage non reconnu")
    else:
        st.error(f"Erreur : {face_data['message']}")

def social_auth_buttons():
    st.markdown("### Ou connectez-vous avec")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Google", key="google_auth", 
                    help="Se connecter avec Google",
                    use_container_width=True):
            auth_url = init_google_auth()
            st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', 
                       unsafe_allow_html=True)
    
    with col2:
        if st.button("Facebook", key="facebook_auth",
                    help="Se connecter avec Facebook",
                    use_container_width=True):
            st.warning("Connexion Facebook à implémenter")

def capture_interface():
    with st.container():
        st.markdown("### Étape 1/2 : Capture de votre visage")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.expander("📷 Utiliser la webcam", expanded=True):
                ctx = webrtc_streamer(
                    key="webcam",
                    video_transformer_factory=FaceCaptureTransformer,
                    media_stream_constraints={"video": True, "audio": False},
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
                
                if ctx.video_transformer and st.button("Prendre une photo"):
                    frame = ctx.video_transformer.frame
                    if frame is not None:
                        process_captured_image(frame)

        with col2:
            with st.expander("📁 Uploader une photo", expanded=True):
                uploaded_file = st.file_uploader("Choisir un fichier", type=["jpg", "png"])
                if uploaded_file:
                    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    process_captured_image(img)

def process_captured_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_data = process_face_image(img_rgb)
    
    if face_data["valid"]:
        st.session_state.auth_data.update({
            "face_image": img_rgb,
            "face_encoding": face_data["encoding"],
            "current_step": 2
        })
        st.experimental_rerun()
    else:
        st.error(f"Erreur : {face_data['message']}")

def registration_form():
    with st.form("registration_form"):
        st.markdown("### Étape 2/2 : Informations du compte")
        
        username = st.text_input("Nom d'utilisateur*")
        email = st.text_input("Email*")
        password = st.text_input("Mot de passe*", type="password")
        confirm_pwd = st.text_input("Confirmer mot de passe*", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.auth_data["face_image"], 
                    caption="Photo capturée", 
                    use_column_width=True)
            
            # Indicateur de qualité
            blur_score = cv2.Laplacian(cv2.cvtColor(
                st.session_state.auth_data["face_image"], 
                cv2.COLOR_RGB2GRAY)).var()
            
            st.markdown(f"**Netteté** : {blur_score:.1f}/200")
            st.markdown("""
                <div class="quality-bar">
                    <div class="quality-fill" style="width:{}%"></div>
                </div>
            """.format(min(int(blur_score/2), 100)), unsafe_allow_html=True)
            
            if st.button("🔄 Reprendre la photo"):
                st.session_state.auth_data.update({
                    "face_image": None,
                    "face_encoding": None,
                    "current_step": 1
                })
                st.experimental_rerun()

        with col2:
            if st.form_submit_button("✅ Créer le compte"):
                if all([username, email, password]) and (password == confirm_pwd):
                    if add_user(username, email, password, 
                              st.session_state.auth_data["face_encoding"]):
                        st.success("Compte créé avec succès!")
                        st.session_state.auth_data.update({
                            "face_image": None,
                            "face_encoding": None,
                            "current_step": 1
                        })
                        st.experimental_rerun()
                    else:
                        st.error("Ce nom d'utilisateur existe déjà")
                else:
                    st.error("Veuillez remplir correctement tous les champs")

def main():
    if not st.session_state.auth_data["authenticated"]:
        with st.container():
            st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🔐 Authentification CBIR</h1>", 
                       unsafe_allow_html=True)
            
            # Méthodes d'authentification
            auth_method = st.radio(
                "Choisissez une méthode d'authentification :",
                ["Identifiants", "Reconnaissance faciale", "Réseaux sociaux"],
                horizontal=True,
                key="auth_method"
            )
            
            if auth_method == "Identifiants":
                with st.expander("#### Connexion classique", expanded=True):
                    tab1, tab2 = st.tabs(["Se connecter", "S'inscrire"])
                    
                    with tab1:
                        with st.form("login_form"):
                            username = st.text_input("Nom d'utilisateur")
                            password = st.text_input("Mot de passe", type="password")
                            
                            if st.form_submit_button("Se connecter"):
                                user = verify_user(username, password)
                                if user:
                                    st.session_state.auth_data.update({
                                        "authenticated": True,
                                        "user": user
                                    })
                                    st.experimental_rerun()
                                else:
                                    st.error("Identifiants incorrects")
                    
                    with tab2:
                        if st.session_state.auth_data["current_step"] == 1:
                            capture_interface()
                        else:
                            registration_form()
            
            elif auth_method == "Reconnaissance faciale":
                face_login_interface()
            
            else:  # Réseaux sociaux
                social_auth_buttons()

    else:
        st.title(f"Bienvenue, {st.session_state.auth_data['user']['username']}!")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("assets/dashboard_icon.png", width=150)
        with col2:
            st.header("Recherche CBIR")
            st.write("Cette fonctionnalité sera disponible dans le Projet 2")
        
        if st.button("Déconnexion", type="primary"):
            st.session_state.auth_data["authenticated"] = False
            st.experimental_rerun()

if __name__ == "__main__":
    try:
        init_db()
        main()
    except Exception as e:
        st.error(f"Erreur critique : {str(e)}")