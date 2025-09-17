# MountAdapt — Climate–Health Dashboard 

**Streamlit web app** for climate–health analytics and forecasting of cardiovascular risk.  
This is the **prototype of my Final Year Project**, integrating multiple EU/Romania datasets and delivering ML models, explainability, and forecasting inside a single interactive dashboard.  
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
-  **Dynamic PDF export** with results from all tabs (metrics, correlations, plots, forecast).  

---

##  Repository structure
├── app.py # Main Streamlit app
├── home.py # Homepage
├── login.py # Login page & auth
├── style.py # Custom CSS/theme
│
├── assets/ # Logos, images
├── data/ # Base datasets + user uploads (.gitignore for uploads)
├── scripts/ # Extra scripts/notebooks
├── utils/ # Core logic
│ ├── analysis.py # Data processing + analytics
│ ├── augment_utils.py # Data augmentation (Gaussian Noise, Mixup)
│ ├── export_pdf.py # Dynamic PDF export with reportlab
│ ├── graphs.py # Custom plots & figures
│ ├── load_data.py # Dataset loading/merging
│ ├── models.py # ML models training & evaluation
│ └── init.py


---

##  Quickstart

###  Clone the repository
```bash
git clone https://github.com/<your-user>/mountadapt.git
cd mountadapt

---
##   Requirements

Minimal dependencies:

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
statsmodels       # SARIMAX
shap
lime
reportlab         # PDF export
PyPDF2

Install with:
pip install -r requirements.txt

---
##Outputs

Interactive correlations (original vs augmented).

Feature selection rankings.

Model comparison table (R², MAE, RMSE).

y_test vs y_pred scatter for best model.

SHAP importance (global) + LIME explanation (individual).

SARIMAX 5-year forecast plots.

Exportable PDF report.
