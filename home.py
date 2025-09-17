import streamlit as st

def render_home():
    st.markdown("""
        <style>
            .home-container {
                text-align: center;
                padding: 4rem 2rem;
                font-family: 'Arial', sans-serif;
            }

            .home-title {
                font-size: 3rem;
                font-weight: bold;
                margin-bottom: 1rem;
                color: #003049;
            }

            .home-subtitle {
                font-size: 1.5rem;
                margin-bottom: 2rem;
                color: #555;
            }

            .home-card {
                background-color: #f0f0f0;
                padding: 2rem;
                border-radius: 1rem;
                margin: 1rem;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }

            .home-button {
                margin-top: 1.5rem;
            }
        </style>

        <div class="home-container">
            <div class="home-title">Bienvenido al Dashboard MountAdapt</div>
            <div class="home-subtitle">Analiza el impacto del cambio climático en la salud cardiovascular</div>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("🔍 Módulos disponibles:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="home-card">', unsafe_allow_html=True)
        st.markdown("### IA Models")
        st.write("Modelos predictivos con datos originales y aumentados.")
        if st.button("Ir a IA Models"):
            st.session_state.view = "IA Models"
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="home-card">', unsafe_allow_html=True)
        st.markdown("### Explainability")
        st.write("Explicabilidad de las predicciones con SHAP y LIME.")
        if st.button("Ir a Explainability"):
            st.session_state.view = "Explainability"
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="home-card">', unsafe_allow_html=True)
        st.markdown("### Forecasting")
        st.write("Predicción temporal con modelos como SARIMAX.")
        if st.button("Ir a Forecasting"):
            st.session_state.view = "Time Series Forecasting"
        st.markdown('</div>', unsafe_allow_html=True)
