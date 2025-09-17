import streamlit as st
import json
import os
from style import aplicar_estilos_generales, set_background
import base64





# --- Estilos generales para login ---
def aplicar_estilos_login():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .title {
            text-align: center;
            font-size: 36px;
            font-weight: 800;
            color: white;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: white;
            margin-bottom: 2rem;
        }

        .login-box {
            background-color: rgba(255, 255, 255, 0.10);
            padding: 2rem;
            border-radius: 16px;
            max-width: 360px;
            margin: 0 auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(8px);
        }

        .stTextInput > div > input {
            border: none !important;
            border-bottom: 2px solid white !important;
            background-color: transparent !important;
            color: white !important;
            padding: 0.5rem;
        }

        .stTextInput > label {
            color: white !important;
            font-weight: 500;
        }

        .stRadio > div > label {
            color: white !important;
        }

        .stButton > button {
            background-color: #1976d2;
            color: white;
            border-radius: 6px;
            font-weight: bold;
            padding: 0.5rem 1.5rem;
            width: 100%;
            margin-top: 1rem;
        }

        .stButton > button:hover {
            background-color: #125ea5;
            transition: 0.3s;
        }

        /* ESTILO APLICADO AL span INTERNO */
        div[data-testid="stRadio"] label span {
            color: white !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            background-color: transparent !important;
        }
        /* Fondo blanco translúcido detrás del st.radio() */
        div[data-testid="stRadio"] {
        background-color: rgba(255, 255, 255, 0.5);
        padding: -0rem 0.7rem;              /* Menos espacio */
        border-radius: 10px;
        width: max-content;
        margin: 2rem 0 -0.8rem -0.1rem;     /* Alineado con "Usuario" */
        }







        </style>
    """, unsafe_allow_html=True)




# --- Carga de usuarios registrados ---
def cargar_usuarios():
    if os.path.exists("usuarios.json"):
        with open("usuarios.json", "r") as f:
            return json.load(f)
    return {}

# --- Guardar nuevo usuario ---
def registrar_usuario(username, password):
    usuarios = cargar_usuarios()
    if username in usuarios:
        return False
    usuarios[username] = password
    with open("usuarios.json", "w") as f:
        json.dump(usuarios, f)
    return True

# --- Comprobar credenciales ---
def comprobar_credenciales(username, password):
    usuarios = cargar_usuarios()
    return usuarios.get(username) == password

# --- Render del login ---
def render_login():
    set_background("assets/fondoweb23.png")
    aplicar_estilos_login()

    # Espaciado superior
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # Logos superiores
    col1, col2, col3, col4, col5 = st.columns([1, 0.9, 0.8, 1, 0.5])

    with col2:
        st.markdown("<div style='margin-top:30px'></div>", unsafe_allow_html=True)
        st.image("assets/deusto.png", width=210)

    with col3:
        st.image("assets/ma.png", width=180)

    with col4:
        st.image("assets/ml.png", width=150)

    # Título principal
    st.markdown('<div class="title">Welcome to MountAdapt</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Prediction and Analysis of Health and climate change</div>', unsafe_allow_html=True)

    # --- Caja del login ---
    with st.container():
        

        accion = st.radio("", ["Login", "Register"], horizontal=True, key="login_action_radio")
        username = st.text_input("User", key="login_username_input")
        password = st.text_input("Password", type="password", key="login_password_input")

        if st.button("Enter", key="login_submit_button"):
            if accion == "Login":
                if comprobar_credenciales(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success("Login correcto.")
                else:
                    st.error("Credenciales inválidas.")
            elif accion == "Register":
                if registrar_usuario(username, password):
                    st.success("Usuario registrado correctamente.")
                else:
                    st.error("Ese nombre de usuario ya existe.")

        st.markdown('</div>', unsafe_allow_html=True)

