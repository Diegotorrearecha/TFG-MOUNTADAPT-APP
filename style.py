import streamlit as st
import base64

import base64

def set_background_main(image_path="assets/fondoweb2.png"):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
          

            </style>
        """, unsafe_allow_html=True)



def aplicar_estilos_generales():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;600;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Títulos */
        h1, h2, h3 {
            font-weight: 800 !important;
            color: #1c3f60 !important;
        }

        /* Botones */
        .stButton > button {
            background-color: #1976d2;
            color: white;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1.2rem;
        }

        .stButton > button:hover {
            background-color: #1565c0;
            transition: 0.3s;
        }

        /* Cuadros visuales */
        .stDataFrame, .stTable {
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 0.5rem;
            background-color: rgba(255,255,255,0.95);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f6f9fc;
            border-right: 1px solid #dee2e6;
        }

        /* Área central */
        .main {
            background-color: #fdfdfd;
            padding: 2rem;
        }

        /* Márgenes de inputs */
        .stSelectbox, .stRadio, .stTextInput, .stSlider {
            margin-bottom: 1rem;
        }


                



/* Efecto al hacer clic */
input:focus, input:active {
    border: none !important;
    outline: none !important;
    box-shadow: 0 0 0 2px #1976d2 !important;
}

/* Autofill */
input:-webkit-autofill {
    box-shadow: 0 0 0 1000px white inset !important;
    -webkit-text-fill-color: #333 !important;
}

/* Botón centrado */
.stButton > button {
    max-width: 500px;
   
}
.main-content {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem 3rem;
    border-radius: 18px;
    max-width: 1000px;
    margin: auto;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}



        </style>
    """, unsafe_allow_html=True)

def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)