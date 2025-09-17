import streamlit as st

# ⚠️ Esto va siempre lo primero
st.set_page_config(page_title="Health & Climate Analysis", layout="wide")

# ⬇️ Solo importamos esto antes del login
from login import render_login
from style import aplicar_estilos_generales, set_background_main

# Aplicamos estilos generales (sin fondo todavía)
aplicar_estilos_generales()
set_background_main()

# ⚠️ Mostrar login antes de cualquier cosa
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.get("logged_in", False):
    render_login()
    st.stop()

from utils import load_data, analysis, graphs, export_pdf
from utils.augment_utils import augment_all_dfs
import plotly.express as px
import pandas as pd
from utils.models import prepare_dataset, train_models, feature_selection
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from utils.augment_utils import augment_mixup_with_noise
from utils.augment_utils import augment_with_interpolation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
from utils.export_pdf import create_pdf
import datetime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from utils import load_data, analysis, graphs, export_pdf
from fpdf import FPDF
import base64
from utils.graphs import plot_coverage_gantt






# Load all data
try:
    df_cases     = load_data.load_cases()
    df_patients  = load_data.load_patients()
    df_temp      = load_data.load_temperatures()
    df_mortality = load_data.load_mortality()
    df_lifexp    = load_data.load_health_expectancy()
    df_maxtemp   = load_data.load_max_temps()
    df_medtemp   = load_data.load_medi_temp()

    # Datasets originales
    df_dict = {
        "cases":     df_cases,
        "patients":  df_patients,
        "temp":      df_temp,
        "mortality": df_mortality,
        "life expectancy": df_lifexp,
        "max temp": df_maxtemp,
        "average temp summer": df_medtemp
    }

    # Añadir datasets personalizados si existen
    custom_dfs = st.session_state.get("custom_dfs", {})
    df_dict.update(custom_dfs)

except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()


from fpdf import FPDF
from io import BytesIO

def crear_pdf_en_memoria(summary_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for key, value in summary_data.items():
        pdf.set_text_color(0, 0, 0)
        safe_text = f"{key}:\n{value}\n".replace("–", "-")  # Sustituir guion largo
        pdf.multi_cell(0, 10, safe_text, align="L")


    # Creamos buffer de memoria
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)  # volvemos al inicio del archivo

    return buffer.read()  # devolvemos bytes listos para st.download_button











st.title("Predicting Health Indicators from Climate Data")



with st.sidebar.expander("🗓️ years available", expanded=False):
    for name, df in df_dict.items():
        if df is not None and "Year_Num" in df.columns:
            years = sorted(df["Year_Num"].dropna().unique())
            if len(years) > 5:
                st.write(f"{name.capitalize()}: {years[0]} – {years[-1]}  ({len(years)} years)")
            else:
                st.write(f"{name.capitalize()}: {years}")

# Navegación
st.sidebar.title("Navigation")
view = st.sidebar.radio("Go to", [
    "Correlation analysis",
    "Variable Impact",
    "IA models",
    "Augmented Correlations",
    "Explainability",
    "Export report",
    "Time Series Forecasting",
    "Upload New Variable" 
])

if view == "Correlation analysis":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("🌍 Health & Climate Explorer")
    st.markdown("""
        Welcome to the **Health & Climate Explorer**. This tool allows you to explore how changes in climate may relate to public health indicators such as life expectancy, disease rates, and more.

        🔍 In this section, you can:
        - Select two variables (e.g., temperature and cardiovascular disease cases)
        - See how strongly they are correlated over time
        - View the time series side-by-side

        This helps identify possible patterns or relationships that could be investigated further.
        """)



    # SELECTORES ÚNICOS
    # Etiquetado por colores para facilitar la selección
    ambiental_vars = {
    "Temperature (°C)": "temp",
    "Max Temp (°C)": "max temp",
    "Average Temperature in Summer (°C)": "average temp summer"
    }

    salud_vars = {
    "Illness Cases (Cardiovascular)": "cases",
    "Out-of-hospital Patients (Circulatory)": "patients",
    "Mortality (Cardiovascular)": "mortality",
    "Life Expectancy": "life expectancy"
        }

# Combina con etiquetas de color
    colored_options = {
        f"🔵 {k}": v for k, v in ambiental_vars.items()
    }
    colored_options.update({
        f"🔴 {k}": v for k, v in salud_vars.items()
    })


    var_x_label = st.selectbox("Select Variable X (Ambiental Variable Recommended)", list(colored_options.keys()))
    var_y_label = st.selectbox("Select Variable Y (Health Variable Recommended)", list(colored_options.keys()))

    if colored_options[var_x_label] == colored_options[var_y_label]:
        st.warning("Please select two different variables.")
        st.stop()

    # Calcula merged
    df_x = df_dict[colored_options[var_x_label]]
    df_y = df_dict[colored_options[var_y_label]]
    x = df_x.groupby("Year_Num")[df_x.columns[-1]].sum()
    y = df_y.groupby("Year_Num")[df_y.columns[-1]].sum()
    merged = (
        x.rename("X")
         .to_frame()
         .join(y.rename("Y"), how="inner")
         .reset_index()
    )

    # Estadísticos
    corr, pval = pearsonr(merged["X"], merged["Y"])
    st.markdown(f"**Period analyzed:** {int(merged.Year_Num.min())} - {int(merged.Year_Num.max())}")
    st.markdown(f"**Pearson correlation**(measures the strength of the relationship.): `{corr:.4f}`")
    st.markdown(f"**p-value**(tells you if the correlation is statistically significant): `{pval:.4f}`")
    # Guardar variables seleccionadas y valor de correlación
    st.session_state["selected_corr"] = (colored_options[var_x_label], colored_options[var_y_label])
    st.session_state["last_corr_value"] = round(corr, 4)
    st.session_state["last_p_value"] = round(pval, 4)
    st.session_state["corr_years"] = (int(merged.Year_Num.min()), int(merged.Year_Num.max()))
    
    # Mostrar tabla
    with st.expander("Show merged data", expanded=False):
        st.dataframe(merged.set_index("Year_Num"))


    # Etiquetas para los ejes
    x_label = var_x_label
    y_label = var_y_label


# Crear gráfico
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged["Year_Num"],
        y=merged["X"],
        mode='lines+markers',
        name=x_label,
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=merged["Year_Num"],
        y=merged["Y"],
        mode='lines+markers',
        name=y_label,
        line=dict(color='red', dash='dot'),
        yaxis="y2"
    ))

    fig.update_layout(
    title="Time series of merged data",
    xaxis=dict(
        title="Year_Num",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title=x_label,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    yaxis2=dict(
        title=y_label,
        overlaying='y',
        side='right',
        showgrid=False
    ),
    legend=dict(
        orientation='h',
        yanchor='top',
        y=1.05,
        xanchor='center',
        x=0.5
    ),
    width=1000,
    height=600
)
    



    st.plotly_chart(fig)
    st.markdown('</div>', unsafe_allow_html=True)


elif view == "Export report":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.subheader("📄 Export Report")

    # 1. Correlación
    selected_corr = st.session_state.get("selected_corr", ("Mortality", "temp"))
    corr_value = st.session_state.get("last_corr_value", "N/A")

    # 2. Mejor modelo IA y métricas
    best_model = st.session_state.get("mejor_modelo_nombre", "MLPRegressor")
    best_r2 = st.session_state.get("mejor_r2", "N/A")
    best_mae = st.session_state.get("mejor_mae", "N/A")
    best_rmse = st.session_state.get("mejor_rmse", "N/A")

    # Asegurar que los valores están en formato string correctamente
    def safe_round(x):
        try:
            return round(float(x), 4)
        except:
            return "N/A"

    best_r2 = safe_round(best_r2)
    best_mae = safe_round(best_mae)
    best_rmse = safe_round(best_rmse)

    # 3. Mejor método de data augmentation
    best_aug = st.session_state.get("mejor_augmentacion", "Mixup + Gaussian Noise")

    # 4. Variable de forecasting
    forecast_var = st.session_state.get("forecast_variable", "cases")

    # 5. Fecha
    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    p_value = st.session_state.get("last_p_value", "N/A")
    corr_years = st.session_state.get("corr_years", ("N/A", "N/A"))

    # Forecasting
    forecast_result = st.session_state.get("forecast_result", None)
    forecast_summary = "No forecasting data available."

    if forecast_result is not None:
        try:
            forecast_summary = ""
            for year, row in forecast_result.iterrows():
                año = year.year
                valor = round(row["Predicción"], 2)
                ci_low = round(row["CI_low"], 2)
                ci_high = round(row["CI_high"], 2)
                forecast_summary += f"{año}: {valor} (CI: {ci_low}–{ci_high})\n"
        except Exception as e:
            forecast_summary = f"⚠️ Error parsing forecast: {e}"
    # EXPLAINABILITY
    shap_mean = st.session_state.get("shap_mean", [])
    shap_top3 = st.session_state.get("shap_top3", [])
    lime_exp = st.session_state.get("lime_exp", "No LIME explanation available")

# Construimos texto explicativo
    shap_text = "No SHAP data available."
    if shap_mean:
        lines = [f"- {d['Variable']}: {round(d['Importance'], 4)}" for d in shap_mean]
        shap_text = "\n".join(lines)

    top3_text = ", ".join(shap_top3) if shap_top3 else "N/A"
    # VARIABLE IMPACT
    selectkbest_scores = st.session_state.get("selectkbest_scores", [])
    top3_selectkbest = st.session_state.get("top3_selectkbest", [])
    perm_importance = st.session_state.get("perm_importance", [])

# Texto SelectKBest
    selectkbest_text = "No data available."
    if selectkbest_scores:
        selectkbest_text = "\n".join([
            f"- {item['Variable']}: {round(item['Score'], 4)}"
            for item in selectkbest_scores
        ])

# Top 3
    top3_text = ", ".join(top3_selectkbest) if top3_selectkbest else "N/A"

# Texto Permutation Importance
    perm_text = "No data available."
    if perm_importance:
        perm_text = "\n".join([
            f"- {item['Variable']}: {round(item['Importance'], 4)}"
            for item in perm_importance
        ])



    # 6. Datos para el PDF
    summary_data = {
        "Project": "MountAdapt: Climate and Cardiovascular Analysis",
        "Selected Correlation": f"{selected_corr[0]} vs {selected_corr[1]}",
        "Correlation value": f"{corr_value}",
        "p-value": f"{p_value}",
        "Period Analyzed": f"{corr_years[0]} - {corr_years[1]}",
        "Best IA Model": f"{best_model} (R² = {best_r2}, MAE = {best_mae}, RMSE = {best_rmse})",
        "Best Augmentation Technique": best_aug,
        "Forecasting Target Variable": forecast_var,
        "Forecast Results (Next 5 Years)": forecast_summary,
        "Report Generated": fecha,
        "SHAP Mean Feature Importance": shap_text,
        "Top 3 Influential Variables (SHAP)": top3_text,
        "LIME Explanation (Instance 0)": lime_exp,
        "SelectKBest Scores": selectkbest_text,
        "Top 3 Variables (SelectKBest)": top3_text,
        "Permutation Importance (Random Forest)": perm_text


    }

    pdf_bytes = crear_pdf_en_memoria(summary_data)

    st.download_button(
        label="📥 Download PDF",
        data=pdf_bytes,
        file_name="mountadapt_report.pdf",
        mime="application/pdf"
    )
    pretty = {
    "cases": "Cases",
    "patients": "Patients",
    "mortality": "CVD mortality",
    "life expectancy": "Healthy life expectancy (years)",
    "health_life_expectancy": "Healthy life expectancy (years)",
    "temp": "Mean temperature (°C)",
    "temperature": "Mean temperature (°C)",
    "max temp": "Annual maximum temperature (°C)",
    "max_temp": "Annual maximum temperature (°C)",
    "average temp summer": "Summer mean temperature (°C)",
    "medi_temp": "Summer mean temperature (°C)",
}
    from utils.load_data import load_all_data
    from utils.analysis import compute_basic_descriptives
    # Mostrar PDF embebido
    b64 = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    df_dict = load_all_data()
    core_vars = ["cases","patients","mortality","health_life_expectancy","temp","max_temp","average temp summer"]
    core_vars = [v for v in core_vars if v in df_dict]

    desc = compute_basic_descriptives(
    df_dict, use_vars=core_vars, eff_start=2014, eff_end=2020, pretty=pretty
)

    st.subheader("Basic descriptives (2014–2020)")
    st.dataframe(desc.style.format({
    "Mean": "{:,.2f}", "SD": "{:,.2f}", "Min": "{:,.2f}", "Max": "{:,.2f}"
    }))

    csv = desc.to_csv(index=False).encode("utf-8")
    st.download_button("Download Table 9.2 (CSV)", csv, "table_9_2_descriptives.csv", "text/csv")

    # opcional: guarda para el PDF
    st.session_state["table_9_2_csv"] = csv




elif view == "Variable Impact":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("🌿 Variable Impact Analysis")
    st.markdown("""
    This section helps you understand which climate variables have the most influence on specific health indicators.

    We use three techniques:
    - **SelectKBest**: Measures statistical correlation.
    - **RFE**: Identifies which variables improve prediction when used in linear models.
    - **Permutation Importance**: Evaluates how much model accuracy drops when a variable is shuffled.
    """)

    # Preparar el dataset combinado
    try:
        data = prepare_dataset(df_dict)

        if data.empty or data.shape[0] < 2:
            st.error("❌ The combined dataset is empty or has fewer than two samples. Check that the variables share common years.")
            st.stop()

        st.write("🔍 Dataset shape:", data.shape)
        st.dataframe(data.head())

        # Extraer variables posibles (todas menos Year_Num)
        posibles_objetivos = [col for col in data.columns if col != "Year_Num"]

        target = st.selectbox("Select a health variable", posibles_objetivos)

        if target not in data.columns:
            st.error(f"❌ La variable '{target}' no está presente en el dataset combinado.")
            st.stop()

        # Separar X e Y
        X = data.drop(columns=["Year_Num", target])
        Y = data[target]
        # Imputar NaNs para SelectKBest (por ejemplo, con la media)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)


# 🔄 Combinar X e Y para imputar ambos de forma consistente
        df_model = pd.concat([X, Y], axis=1)

# ❌ Eliminar filas si la variable objetivo (Y) tiene NaN (no se puede predecir)
        df_model = df_model.dropna(subset=[target])

# ✅ Imputar X con la media (solo si quedan NaN)
        X = df_model.drop(columns=[target])
        Y = df_model[target]
        X = X.fillna(X.mean())



        if X.empty or Y.empty or len(X) < 2:
            st.error("❌ No hay suficientes datos para entrenar los modelos. Verifica la calidad del dataset.")
            st.stop()

        st.subheader("📊 Variable Relevance Rankings")

        # Feature Selection
        fs_dict, rfe_obj = feature_selection(X, Y)

        selectkbest_df = pd.DataFrame({
            "Variable": X.columns,
            "Score": fs_dict["SelectKBest"]
        }).sort_values(by="Score", ascending=False)

        rfe_df = pd.DataFrame({
            "Variable": X.columns,
            "Ranking": fs_dict["RFE"]
        }).sort_values(by="Ranking")

        st.write("🏆 **SelectKBest scores** (higher = more related):")
        st.dataframe(selectkbest_df)

        st.write("📌 **RFE rankings** (lower = more important):")
        st.dataframe(rfe_df)

        selected_rfe = [var for var, rank in zip(X.columns, rfe_obj.ranking_) if rank == 1]
        st.success(f"✅ Selected by RFE (Top {len(selected_rfe)}): {', '.join(selected_rfe)}")

        # -----------------------------
        # 🎯 Permutation Importance
        # -----------------------------
        st.subheader("📉 Permutation Importance (RandomForest)")
        st.markdown("""
        This analysis shows how much each variable affects the model’s accuracy.
        More important variables cause a bigger drop in performance when randomly shuffled.
        """)

        rf = RandomForestRegressor(random_state=42)
        rf.fit(X, Y)

        result = permutation_importance(rf, X, Y, n_repeats=10, random_state=42)
        perm_df = pd.DataFrame({
            "Variable": X.columns,
            "Importance": result.importances_mean
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(perm_df)

        fig = px.bar(perm_df, x="Variable", y="Importance",
                     title="Permutation Importance (mean drop in model score)",
                     labels={"Importance": "Mean Importance"})

        st.plotly_chart(fig)
        # Guardar resultados en sesión para exportar
        try:
    # SelectKBest (todas)
            st.session_state["selectkbest_scores"] = selectkbest_df.to_dict(orient="records")

    # Top 3 (puedes usar RFE si prefieres)
            st.session_state["top3_selectkbest"] = selectkbest_df.head(3)["Variable"].tolist()

    # Permutation importance
            st.session_state["perm_importance"] = perm_df.to_dict(orient="records")

        except Exception as e:
            st.session_state["selectkbest_scores"] = []
            st.session_state["top3_selectkbest"] = []
            st.session_state["perm_importance"] = []


    except Exception as e:
        st.error(f"❌ Error analyzing variable relevance: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


elif view == "IA models":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.subheader("Comparative of models with different augmentation techniques")
    st.markdown("""
        In this section, we train and compare machine learning models to estimate health indicators 
        based on selected climate variables.

         Use the selector below to choose the variable you want to predict.
        """)

    # Lista de métodos a probar
    
    st.title("Predictive models")
    # Tabla comparativa
    
    data = prepare_dataset(df_dict)

    if data is not None:
        # Usamos solo columnas del dataframe final combinado
        posibles_objetivos = [col for col in data.columns if col != "Year_Num"]


        target = st.selectbox("Selecciona variable a predecir", posibles_objetivos)

        if target not in data.columns:
            st.error(f"La variable '{target}' no está disponible en el conjunto de datos combinado. Revisa tus CSV o la selección.")
            st.stop()

            
        try:
                
                X = data.drop(columns=["Year_Num", target])
                Y = data[target]

                
                
                
                st.markdown("""
| Model            | Type                  | Key Hyperparameters                       | Notes                                                              |
|------------------|------------------------|--------------------------------------------|---------------------------------------------------------------------|
| LinearRegression | Linear                 | None (default model)                       | Baseline model, useful for detecting simple linear relationships   |
| RandomForest     | Ensemble (bagging)     | n_estimators=100, random_state=42      | Robust to noise and overfitting                                    |
| GradientBoosting | Ensemble (boosting)    | n_estimators=100, learning_rate=0.1    | Strong performance, but sensitive to noise                         |
| SVR              | Support Vector Machine | kernel='rbf', C=1.0, epsilon=0.1      | Effective in non-linear spaces, computationally expensive          |
| MLPRegressor     | Deep Neural Network    | hidden_layer_sizes=(100, 75, 50, 25)     | Can capture complex patterns; requires more data and tuning        |
""")

                st.markdown("""
💡 **Note on MLPRegressor**:  
The MLPRegressor used in this project was configured with **4 hidden layers** of sizes: (100, 75, 50, 25).  
This deep architecture allows the model to capture complex and non-linear relationships between climate variables and health indicators.
""")



                st.markdown("---")
                tabla_resultados = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                total = 4  # 4 métodos de aumento

                for i, metodo_nombre in enumerate(["Original", "Gaussian Noise", "Mixup", "Interpolation"]):
                    status_text.text(f"🔄 Training models with data: {metodo_nombre}...")

    #  Selección del dataset correspondiente
                    if metodo_nombre == "Original":
                        df_dict_usado = df_dict
                    elif metodo_nombre == "Gaussian Noise":
                        df_dict_usado = augment_all_dfs(df_dict, method="noise", noise_level=0.05)
                    elif metodo_nombre == "Mixup":
                        df_dict_usado = augment_all_dfs(df_dict, method="mixup", n_samples=10)  # Antes 100
                    elif metodo_nombre == "Interpolation":
                        df_dict_usado = augment_all_dfs(df_dict, method="interpolation", n_samples=10)

                    try:
                        df_aug = prepare_dataset(df_dict_usado)

                        if target not in df_aug.columns:
                            st.warning(f"⚠️ In the method {metodo_nombre}, the column '{target}'is not in the data.")
                            continue

                        X_aug = df_aug.drop(columns=["Year_Num", target])
                        y_aug = df_aug[target]

                        resultados = train_models(X_aug, y_aug)
                        resultados["Dataset"] = metodo_nombre
                        tabla_resultados.append(resultados)

                    except Exception as e:
                        st.warning(f"❌ Error in method {metodo_nombre}: {e}")

                    progress_bar.progress((i + 1) / total)

                status_text.text("✅ Finished Training.")




                # Mo   strar comparativa
                if tabla_resultados:
                    comparativa_df = pd.concat(tabla_resultados, ignore_index=True)
                    comparativa_df = pd.concat(tabla_resultados, ignore_index=True)

# Guardar el mejor modelo global en session_state
                    if not comparativa_df.empty:
                        mejor_fila = comparativa_df.loc[comparativa_df["R2"].idxmax()]
                        st.session_state["mejor_modelo_nombre"] = mejor_fila["Model"]
                        st.session_state["mejor_r2"] = mejor_fila["R2"]
                        st.session_state["mejor_mae"] = mejor_fila["MAE"]
                        st.session_state["mejor_rmse"] = mejor_fila["RMSE"]
                        st.session_state["mejor_augmentacion"] = mejor_fila["Dataset"]

                    

                    with st.expander("Show model comparison table", expanded=False):
                        st.dataframe(comparativa_df)
                        st.markdown("### 📈 R² Comparison Between Augmentation Methods")
                        fig = px.bar(comparativa_df, x="Model", y="R2", color="Dataset", barmode="group",
                                title="R² Comparison Between Augmentation Methods")
                        st.plotly_chart(fig)

                    

                else:
                    st.warning("⚠️ No se pudieron generar resultados para los métodos aumentados.")
                
                # 📊 Tabla comparativa entre modelos con y sin Data Augmentation (Gaussiano)
                # 📊 Tabla comparativa entre modelos con y sin Data Augmentation (Gaussiano)
                with st.expander("📊 Model Results (Original vs Best Augmentation)"):
                    st.markdown("""
    This section compares model performance using original data and the best **data augmentation techniques**.  
    It displays **R²**, **MAE**, and **RMSE** values for each method.
    """)

                    if "comparativa_df" in locals() and not comparativa_df.empty:
        # Datos filtrados por método
                        df_original = comparativa_df[comparativa_df["Dataset"] == "Original"]
                        df_noise = comparativa_df[comparativa_df["Dataset"] == "Ruido Gaussiano"]
                        df_mixup = comparativa_df[comparativa_df["Dataset"] == "Mixup"]
                        df_interp = comparativa_df[comparativa_df["Dataset"] == "Interpolación"]

        # Mejor método por modelo (mayor R2 entre augmentations)
                        best_per_model = pd.concat([df_noise, df_mixup, df_interp])
                        best_per_model = best_per_model.loc[best_per_model.groupby("Model")["R2"].idxmax()].reset_index(drop=True)

        # Tabla comparativa original + mejor método
                        comparison_data = pd.concat([
                            df_original[["Model", "R2", "MAE", "RMSE", "Dataset"]],
                            best_per_model[["Model", "R2", "MAE", "RMSE", "Dataset"]]
                        ], ignore_index=True)

                        st.markdown("### 🔄 Original vs Best Augmented Result per Model")
                        st.dataframe(comparison_data, use_container_width=True)

                        st.markdown("### 🏆 Best Augmentation Method per Model")
                        st.dataframe(best_per_model[["Model", "Dataset", "R2", "MAE", "RMSE"]], use_container_width=True)


                        

                    else:
                        st.warning("No results found to display comparison.")



                # Gráfico de mejores predicciones vs valores reales
                    st.markdown("### 🎯 Prediction vs Real (Best Model)")

# Elegimos el modelo con mayor R2
                    mejor_fila = comparativa_df.loc[comparativa_df["R2"].idxmax()]
                    modelo_nombre = mejor_fila["Model"]
                    dataset_origen = mejor_fila["Dataset"]
                    st.session_state["mejor_modelo_nombre"] = modelo_nombre
                    st.session_state["mejor_r2"] = mejor_fila["R2"]
                    st.session_state["mejor_mae"] = mejor_fila["MAE"]
                    st.session_state["mejor_rmse"] = mejor_fila["RMSE"]
                    st.session_state["mejor_augmentacion"] = dataset_origen


# Seleccionar el dataset correspondiente
                    if dataset_origen == "Original":
                        df_dict_usado = df_dict
                    elif dataset_origen == "Ruido Gaussiano":
                        df_dict_usado = augment_all_dfs(df_dict, method="noise", noise_level=0.05)
                    elif dataset_origen == "Mixup":
                        df_dict_usado = augment_all_dfs(df_dict, method="mixup", n_samples=10)
                    elif dataset_origen == "Interpolación":
                        df_dict_usado = augment_all_dfs(df_dict, method="interpolation", n_samples=10)

# Preparar los datos
                    df_aug = prepare_dataset(df_dict_usado)
                    X = df_aug.drop(columns=["Year_Num", target])
                    y = df_aug[target]

# Dividir para evaluación
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo correspondiente


                    modelos = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100, 75, 50, 25), max_iter=500, random_state=42)
                    }

                    modelo = modelos[modelo_nombre]
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)

# Graficar y_test vs y_pred
                    fig_pred = px.scatter(x=y_test, y=y_pred,
                      labels={'x': 'Real Values', 'y': 'Predicted Values'},
                      title=f"{modelo_nombre} on {dataset_origen} Data")

                    fig_pred.add_shape(type="line", x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
                        line=dict(color="green", dash="dash"))

                    st.plotly_chart(fig_pred, use_container_width=True)

                try:
                    df_dict_aug = augment_all_dfs(df_dict, method="noise", noise_level=0.05)
                    print(df_dict_aug["cases"]["Year_Num"].value_counts())
                    data_aug = prepare_dataset(df_dict_aug)
                    X_aug = data_aug.drop(columns=["Year_Num", target])
                    y_aug = data_aug[target]

                    

                except Exception as e:
                    st.error(f"❌ Error al generar datos aumentados: {e}")


        except Exception as e:
                st.error(f"Error construyendo el modelo: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "Augmented Correlations":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("🧬 Augmented Correlation Analysis")

    # Generar datos aumentados
    df_dict_aug = augment_mixup_with_noise(df_dict, n_samples=10, noise_level=0.05)   
    df_dict_interp = augment_with_interpolation(df_dict)

    # Selección de variable
    available_vars = list(df_dict_aug.keys())
    selected_var = st.selectbox("Select Variable", options=available_vars)

    # Checkboxes para seleccionar qué mostrar
    show_mixup = st.checkbox("Mostrar Gaussian + Mixup", value=True)
    show_interp = st.checkbox("Mostrar Interpolación", value=True)

    # Cargar los datos
    df_y_orig = df_dict[selected_var]
    df_y_aug = df_dict_aug[selected_var]
    df_y_interp = df_dict_interp[selected_var]

    # Detectar columna de valores
    value_col = [col for col in df_y_orig.columns if col != "Year_Num"][0]

    # Agrupar por año
    y_orig = df_y_orig.groupby("Year_Num")[value_col].mean().reset_index()
    y_aug = df_y_aug.groupby("Year_Num")[value_col].mean().reset_index()
    y_interp = df_y_interp.groupby("Year_Num")[value_col].mean().reset_index()

    # Crear figura
    fig = go.Figure()

    # Original
    fig.add_trace(go.Scatter(
        x=y_orig["Year_Num"],
        y=y_orig[value_col],
        mode="lines+markers",
        name=f"{selected_var} (Original)",
        line=dict(color="blue")
    ))

    # Aumentado con Gaussian + Mixup
    if show_mixup:
        fig.add_trace(go.Scatter(
            x=y_aug["Year_Num"],
            y=y_aug[value_col],
            mode="lines+markers",
            name=f"{selected_var} (Gaussian + Mixup)",
            line=dict(color="red", dash="dot")
        ))

    # Aumentado por Interpolación
    if show_interp:
        fig.add_trace(go.Scatter(
            x=y_interp["Year_Num"],
            y=y_interp[value_col],
            mode="lines+markers",
            name=f"{selected_var} (Interpolación)",
            line=dict(color="green", dash="dash")
        ))

    # Layout
    fig.update_layout(
        title="Augmented Correlation Analysis",
        xaxis_title="Year",
        yaxis_title=selected_var,
        template="plotly_white",
        width=1000,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "Explainability":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.markdown("## 🔍 Explainability: SHAP Analysis")
    st.markdown("""
    This section allows you to **interpret model predictions** using SHAP (SHapley Additive exPlanations). You
    can analyze which features contribute most to the prediction of a selected health variable.

    **Steps performed:**
    - A Random Forest Regressor is trained to predict the selected variable.
    - SHAP values are computed to explain the model's output.
    - Visualizations show the importance and behavior of each feature.
    """)

    data = prepare_dataset(df_dict)

    if data.empty or data.shape[0] < 2:
        st.error("❌The combined dataset is empty or has fewer than two samples. Check that the variables share common years.")
        st.stop()

    # Variables posibles como objetivo (salud)
    posibles_objetivos = [col for col in data.columns if col != "Year_Num"]
    target = st.selectbox("Select a health variable to explain", posibles_objetivos)

    if target not in data.columns:
            st.error(f"La variable '{target}' no está disponible en el conjunto de datos combinado.")
            st.stop()

        # Prepara datos
    X = data.drop(columns=["Year_Num", target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    st.subheader("📌 SHAP Summary Plot")
    st.markdown("This plot shows the **distribution of SHAP values** per feature. Color indicates feature value (red=high, blue=low).")
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("📊 SHAP Feature Importance (Mean Absolute Value)")
    st.markdown("This bar plot shows the **average importance** of each feature.")
    shap.plots.bar(shap_values, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    # 🔍 LIME Explanation for a Single Prediction (usar la MISMA escala que el modelo)
    st.subheader("🔍 LIME Explanation for a Single Prediction")

    idx = st.slider(
        "Selecciona una observación del conjunto de TEST para explicar",
        0, len(X_test_scaled) - 1, 0
    )

    lime_explainer = LimeTabularExplainer(
        training_data=X_train_scaled.values,               # 👈 training con escala del modelo
        feature_names=X_train_scaled.columns.tolist(),
        mode="regression",
        discretize_continuous=False
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=X_test_scaled.iloc[idx].values,           # 👈 fila de TEST con la misma escala
        predict_fn=lambda A: model.predict(np.array(A))    # el modelo espera esa escala
    )

    fig = lime_exp.as_pyplot_figure()
    st.pyplot(fig)

    # Guardar resumen SHAP y LIME en sesión
    try:
    # SHAP media de importancia
        shap_mean_importance = pd.DataFrame({
            "Variable": X.columns,
            "Importance": np.abs(shap_values.values).mean(axis=0)
            }).sort_values(by="Importance", ascending=False)

        st.session_state["shap_mean"] = shap_mean_importance.to_dict(orient="records")

    # Top 3 variables más influyentes
        top3 = shap_mean_importance.head(3)["Variable"].tolist()
        st.session_state["shap_top3"] = top3

    except Exception as e:
        st.session_state["shap_mean"] = []
        st.session_state["shap_top3"] = []

# LIME (guardar texto plano de explicación)
    try:
        lime_exp_text = ""
        for feat, val in lime_exp.as_list():
            lime_exp_text += f"{feat}: {val:.2f}\n"
        st.session_state["lime_exp"] = lime_exp_text.strip()
    except Exception as e:
        st.session_state["lime_exp"] = "Error generating LIME explanation"

    st.markdown('</div>', unsafe_allow_html=True)


elif view == "Time Series Forecasting":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("📈 Time Series Forecasting (5 years)")

    st.markdown("""
    Select a time variable (e.g., cases, temperature, etc.).
The model will fit a 5-year forecast using SARIMAX.
    """)

    # Extraer variables disponibles con una columna "Year_Num"
    temporal_vars = [k for k in df_dict.keys() if "Year_Num" in df_dict[k].columns]
    var_ts = st.selectbox("SSelect a  variable", options=temporal_vars)

    df_ts = df_dict[var_ts]
    date_col = "Year_Num"
    value_col = [col for col in df_ts.columns if col != date_col][0]

    # Convertimos el "Year_Num" a datetime si es necesario
    df_ts = df_ts.copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col], format="%Y")
    df_ts = df_ts.set_index(date_col).sort_index()

    # Ajustamos modelo SARIMAX
    try:
        model = SARIMAX(df_ts[value_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)

        # Predecir 5 años (asumiendo datos anuales)
        steps = 5
        forecast = results.get_forecast(steps=steps)
        forecast_index = pd.date_range(start=df_ts.index.max() + pd.DateOffset(years=1), periods=steps, freq='Y')
        forecast_values = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # Construir DataFrame para graficar
        df_forecast = pd.DataFrame({
            "Fecha": forecast_index,
            "Predicción": forecast_values.values,
            "CI_low": forecast_ci.iloc[:, 0].values,
            "CI_high": forecast_ci.iloc[:, 1].values
        }).set_index("Fecha")

        # Graficar
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_ts.index,
            y=df_ts[value_col],
            name="Histórico",
            mode="lines"
        ))

        fig.add_trace(go.Scatter(
            x=df_forecast.index,
            y=df_forecast["Predicción"],
            name="Predicción SARIMAX (5 years)",
            mode="lines",
            line=dict(dash="dash", color="orange")
        ))

        fig.add_trace(go.Scatter(
            x=df_forecast.index.tolist() + df_forecast.index[::-1].tolist(),
            y=df_forecast["CI_high"].tolist() + df_forecast["CI_low"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(255,165,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False
        ))

        fig.update_layout(
            title=f"Preiction of 5 years - {var_ts}",
            xaxis_title="Fecha",
            yaxis_title=value_col,
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
        # Guardar en sesión los resultados del forecasting
        st.session_state["forecast_result"] = df_forecast
        st.session_state["forecast_variable"] = var_ts


    except Exception as e:
        st.error(f"❌ Error al ajustar el modelo SARIMAX: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "Upload New Variable":
    
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.header("Upload new variable")

    if "custom_dfs" not in st.session_state:
        st.session_state["custom_dfs"] = {}

    # Paso 1: obtener todos los años base de los datasets originales
    original_years = set()
    for df in df_dict.values():
        if "Year_Num" in df.columns:
            original_years |= set(df["Year_Num"].dropna())
    original_years = sorted(original_years)
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        new_df_raw = pd.read_csv(uploaded_file)

        if "Year_Num" not in new_df_raw.columns:
            st.error("❌ El CSV debe tener una columna 'Year_Num'.")
        else:
            variable_cols = [col for col in new_df_raw.columns if col != "Year_Num"]
            if len(variable_cols) != 1:
                st.warning("⚠️ El CSV debe contener solo una variable adicional además de 'Year_Num'")
            else:
                var_name = variable_cols[0]

                # Paso 2: Adaptar al conjunto de años original (rellenar con NaNs si falta)
                new_df = pd.DataFrame({"Year_Num": original_years})
                new_df = new_df.merge(new_df_raw, on="Year_Num", how="left")

                # Guardar en sesión
                st.session_state["custom_dfs"][var_name] = new_df

                st.success(f"✅ Variable '{var_name}' adapted to years base and added correctly.")
                st.dataframe(new_df)
    st.markdown('</div>', unsafe_allow_html=True)
