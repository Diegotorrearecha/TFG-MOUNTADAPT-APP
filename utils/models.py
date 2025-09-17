# utils/models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import plotly.express as px
import shap
from sklearn.metrics import mean_squared_error
# al inicio de utils/models.py
import numpy as np


def prepare_dataset(df_dict):
    merged_df = None

    for name, df in df_dict.items():
        if df is not None and "Year_Num" in df.columns:
            value_cols = [col for col in df.columns if col != "Year_Num"]
            if len(value_cols) == 1:
                df = df.rename(columns={value_cols[0]: name})
                if merged_df is None:
                    merged_df = df.copy()
                else:
                    merged_df = pd.merge(merged_df, df, on="Year_Num", how="outer")

    # Filtrar años comunes
    if merged_df is not None:
        merged_df = merged_df.fillna(merged_df.mean(numeric_only=True))

    return merged_df





def feature_selection(X, y, k=5):
    """Aplica SelectKBest y RFE para identificar las mejores variables."""
    results = {}

    # SelectKBest
    skb = SelectKBest(score_func=f_regression, k=k)
    skb.fit(X, y)
    results['SelectKBest'] = pd.Series(skb.scores_, index=X.columns).sort_values(ascending=False)

    # RFE
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    results['RFE'] = pd.Series(rfe.ranking_, index=X.columns).sort_values()

    return results, rfe


def train_models(X, y):
    """Entrena varios modelos de regresión y devuelve sus métricas."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(),
        "MLPRegressor":MLPRegressor(hidden_layer_sizes=(100, 75, 50, 25), max_iter=1000, random_state=42)
    }

    results = []

    for name, model in models.items():
        if name == "MLPRegressor":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)  # MSE
        rmse = float(np.sqrt(mse))                # RMSE
        results.append({
            "Model": name,
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": rmse
        })

    return pd.DataFrame(results)
