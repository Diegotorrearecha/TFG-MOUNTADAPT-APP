import streamlit as st
import pandas as pd
from scipy.stats import pearsonr

def custom_correlation(df_dict):
    st.write("### 🔍 Custom Correlation Analysis")

    variable_options = {
        "Cases (illness)": "cases",
        "Out-of-hospital Patients": "patients",
        "Temperature (°C)": "temp",
        "CVD Mortality": "mortality"
    }

    var_x = st.selectbox("Select Variable X", list(variable_options.keys()))
    var_y = st.selectbox("Select Variable Y", list(variable_options.keys()))

    if variable_options[var_x] == variable_options[var_y]:
        st.warning("Please select two different variables.")
        return

    try:
        df_x = df_dict[variable_options[var_x]]
        df_y = df_dict[variable_options[var_y]]

        x = df_x.groupby("Year_Num")[df_x.columns[-1]].sum()
        y = df_y.groupby("Year_Num")[df_y.columns[-1]].sum()

        merged = pd.merge(x, y, on="Year_Num")
        merged.columns = ["X", "Y"]
        if merged.shape[0] < 2:
            st.warning("Not enough data to compute correlation.")
            return

        corr, pval = pearsonr(merged["X"], merged["Y"])

        st.markdown(f"**Period analyzed:** {merged.index.min()} - {merged.index.max()}")
        st.markdown(
        f"**Pearson correlation:** `{round(corr, 4)}`  \n"
        f"<span style='font-size: 0.85em; color: grey'>→ Indica la fuerza de la relación lineal entre las dos variables.</span>",
        unsafe_allow_html=True
)

        st.markdown(
        f"**p-value:** `{round(pval, 4)}`  \n"
        f"<span style='font-size: 0.85em; color: grey'>→ Mide la significancia estadística: si es menor que 0.05, la correlación es relevante.</span>",
        unsafe_allow_html=True
)


        with st.expander("Show merged data"):
            st.dataframe(merged)
    except Exception as e:
        st.error(f"Error: {e}")

def show_correlations(df_cases, df_patients):
    st.write("### 📊 Correlation: Illness Cases vs Patients Outside Hospital")

    try:
        # Group both datasets by year
        cases_by_year = df_cases.groupby("Year_Num")["Cases"].sum()
        patients_by_year = df_patients.groupby("Year_Num")["Patients"].sum()

        # Merge them on year
        merged = pd.merge(cases_by_year, patients_by_year, on="Year_Num", how="inner")
        merged = merged.dropna()

        # Pearson correlation
        corr, p_value = pearsonr(merged["Cases"], merged["Patients"])

        # Output
        st.write("**Period analyzed:**", f"{merged.index.min()} - {merged.index.max()}")
        st.write("**Pearson correlation:**", round(corr, 4))
        st.write("**p-value:**", round(p_value, 4))

        with st.expander("Show data used in correlation"):
            st.dataframe(merged)

    except Exception as e:
        st.error(f"Error calculating correlation: {e}")


import pandas as pd
import numpy as np

def _value_col(df):
    # devuelve la primera columna numérica distinta de Year_Num
    for c in df.columns:
        if c.lower() != "year_num" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # fallback
    return [c for c in df.columns if c.lower() != "year_num"][0]

def compute_basic_descriptives(df_dict, use_vars=None, eff_start=2014, eff_end=2020, pretty=None):
    """
    df_dict: {name: DataFrame con Year_Num y columna de valor}
    use_vars: lista opcional para fijar orden
    eff_start/eff_end: ventana efectiva (incluyente)
    pretty: dict opcional {name: etiqueta+unidad}
    """
    items = use_vars if use_vars else list(df_dict.keys())
    rows = []
    for name in items:
        if name not in df_dict or df_dict[name] is None:
            continue
        df = df_dict[name].copy()
        if "Year_Num" not in df.columns:
            continue
        vcol = _value_col(df)
        mask = (pd.to_numeric(df["Year_Num"], errors="coerce") >= eff_start) & \
               (pd.to_numeric(df["Year_Num"], errors="coerce") <= eff_end)
        s = pd.to_numeric(df.loc[mask, vcol], errors="coerce").dropna()
        if s.empty:
            mean = sd = minv = maxv = np.nan
            n = 0
        else:
            mean = s.mean()
            sd   = s.std(ddof=1) if len(s) > 1 else 0.0
            minv = s.min()
            maxv = s.max()
            n    = int(s.shape[0])
        label = pretty.get(name, name) if pretty else name
        rows.append({
            "Variable": label,
            "Years [min–max]": f"{eff_start}–{eff_end}",
            "n": n,
            "Mean": mean,
            "SD": sd,
            "Min": minv,
            "Max": maxv
        })
    df_out = pd.DataFrame(rows)
    return df_out
