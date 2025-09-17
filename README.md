# MountAdapt — Climate–Health Dashboard (Prototype) 🌍❤️

**Streamlit app** for climate–health analytics and forecasting of cardiovascular risk.  
Designed as a reproducible **Final Year Project**: data integration (EU/Romania), ML models (Linear, RF, GB, SVR, **MLPRegressor**), **data augmentation** (Gaussian Noise + Mixup), **explainability** (global **SHAP**, local **LIME**), and **time series** (**SARIMAX**, 5-year forecast).  
All end-user UI **in English**.

---

##  Key Features
- **Data ingestion**: clean merge of base datasets with **user-uploaded CSVs** (auto-integrated across tabs).
- **Correlation analysis**: original vs **Augmented Correlations** (Gaussian Noise + Mixup).
- **Feature selection**: SelectKBest, RFE, Permutation Importance.
- **Model zoo**: Linear, Random Forest, Gradient Boosting, SVR, **MLPRegressor (best so far: (100,75,50,25))**.
- **Explainability**: **SHAP** for global importance; **LIME** for per-prediction explanations.
- **Forecasting**: **SARIMAX** with **5-year horizon** for a selected variable.
- **PDF export**: dynamic report reflecting the user’s actual session (models, metrics, correlations, forecasting, etc.).
- **Login page**: custom background + logos (easily swappable).

---

##  Repository Structure
