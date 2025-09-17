# MountAdapt — Interactive Climate-Health Dashboard (Prototype)

## 🚀 Quickstart

### 1. Clona el repositorio
```bash
git clone https://github.com/<your-user>/mountadapt.git
cd mountadapt
```

### 2. Entorno virtual e instalación
```bash
python.exe -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## 📦 Requirements

### Minimal dependencies
- streamlit  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- statsmodels   # SARIMAX  
- shap  
- lime  
- reportlab     # PDF export  
- PyPDF2  

> **Nota:** guarda estas dependencias en `requirements.txt` para instalación rápida (`pip install -r requirements.txt`).

---

## 📊 Outputs (what the app produces)

La aplicación genera los siguientes artefactos y visualizaciones de forma dinámica, en función de los datos cargados por el usuario:

### Correlation Analysis
- Matriz de correlación (heatmap) sobre dataset original.
- **Augmented Correlations**: misma matriz aplicada al dataset con Data Augmentation (Gaussian Noise + Mixup), con comparación lado a lado.

### IA Models
- Tabla comparativa de modelos (R², MAE, RMSE) para:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - SVR
  - MLPRegressor (configuración mostrada)
- Visualización de **y_test vs y_pred** para el mejor modelo (scatter + línea ideal).
- Exportable a CSV/PNG.

### Explainability
- Importancia global de variables con **SHAP** (summary plot, bar plot).
- Explicación local de predicciones concretas con **LIME** (sustituye al SHAP dependence plot).
- Descarga de imágenes explicativas.

### Forecasting (series temporales)
- Modelo SARIMAX para la variable seleccionada.
- Predicción a 5 años vista con intervalos de confianza.
- Gráficos interactivos: histórico vs forecast + componentes (trend/seasonal/resid).

### Data handling & App features
- Upload de CSVs personalizados: fusión automática con dataset base y validaciones (missing values, formato de fecha).
- Login/registro de usuarios (con fondo configurable y logos).
- Export dinámico a **PDF**: informe generado con los resultados y figuras vistas en pantalla (personalizado por usuario y dataset).
- Logs y resumen reproducible (WBS/Gantt/risks) para entrega académica.

---

## 🛠 Uso (run)

- Para lanzar la app Streamlit:
```bash
streamlit run app.py --server.port 8501
```

- Para lanzar tests / notebooks:
```bash
jupyter notebook
# MountAdapt — Climate–Health Dashboard 🌍❤️

**Streamlit web app** for climate–health analytics and forecasting of cardiovascular risk.  
Prototype of my Final Year Project, integrating EU/Romania datasets with ML models, explainability, and forecasting.  
All end-user UI is in **English**.

---

##  Key Features
-  **Custom login** with branded background and logos.  
-  **Data ingestion**: base datasets + user-uploaded CSVs auto-integrated across tabs.  
-  **Correlation analysis**: original vs augmented (Gaussian Noise + Mixup).  
-  **Feature selection**: SelectKBest, RFE, Permutation Importance.  
-  **Machine learning models**: Linear Regression, Random Forest, Gradient Boosting, SVR, MLPRegressor.  
-  **Best model so far**: `MLPRegressor(hidden_layer_sizes=(100, 75, 50, 25))`.  
-  **Explainability**: global SHAP + local LIME.  
-  **Forecasting**: SARIMAX (5-year horizon).  
-  **Dynamic PDF export** with results from all tabs.  

---

##  Repository structure
.
├── app.py # Main Streamlit app
├── home.py # Homepage
├── login.py # Login page & auth
├── style.py # Custom CSS/theme
│
├── assets/ # Logos, images
├── data/ # Base datasets + user uploads (.gitignore for uploads)
├── scripts/ # Extra scripts/notebooks
├── utils/ # Core logic
│ ├── analysis.py
│ ├── augment_utils.py
│ ├── export_pdf.py
│ ├── graphs.py
│ ├── load_data.py
│ ├── models.py
│ └── init.py

yaml
Copiar código



# o
python -m pytest tests/
```

---

##  Estructura recomendada del repo (resumen)
```
mountadapt/
├─ app.py
├─ requirements.txt
├─ src/
│  ├─ models/
│  ├─ utils/
│  └─ forecasting/
├─ notebooks/
├─ outputs/           # figuras, CSVs, PDF generados
└─ README.md
```

---


