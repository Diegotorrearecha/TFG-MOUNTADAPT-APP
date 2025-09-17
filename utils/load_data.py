import pandas as pd


import pandas as pd
  ###
def load_cases():
    """Carga y devuelve los casos de enfermedades del sistema circulatorio (CVD) en formato largo."""
    df = pd.read_csv("data/new_cases_of_illness.csv", encoding="latin1")
    df = df[df.iloc[:, 0].str.strip() == "Disease of the circulatory system"]
    df_long = df.melt(
        id_vars=df.columns[0],
        var_name="Year_Num",
        value_name="CVD_Cases"
    )
    df_long["Year_Num"] = pd.to_numeric(df_long["Year_Num"], errors="coerce")
    df_long["CVD_Cases"] = pd.to_numeric(df_long["CVD_Cases"], errors="coerce")
    return df_long.dropna(subset=["Year_Num", "CVD_Cases"])\
                  .astype({"Year_Num": int})\
                  .loc[:, ["Year_Num", "CVD_Cases"]]

###
def load_patients():
    """Carga pacientes fuera del hospital para enfermedades del sistema circulatorio (CVD) en formato largo."""
    # Cargar el CSV
    df = pd.read_csv("data/patients_out_of_hospital.csv", encoding="latin1")
    df = df[df.iloc[:, 0].str.strip() == "Disease of the circulatory system"]
    df_long = df.melt(
        id_vars=df.columns[0],
        var_name="Year_Num",
        value_name="CVD_Patients"
    )
    df_long["Year_Num"] = pd.to_numeric(df_long["Year_Num"], errors="coerce")
    df_long["CVD_Patients"] = pd.to_numeric(df_long["CVD_Patients"], errors="coerce")
    return df_long.dropna(subset=["Year_Num", "CVD_Patients"])\
                  .astype({"Year_Num": int})\
                  .loc[:, ["Year_Num", "CVD_Patients"]]

def load_health_expectancy():
    """Carga y devuelve la esperanza de vida saludable anual (‘Total’) en formato long."""
    df = pd.read_csv("data/health_life_expectancy.csv", index_col=0)
    if "Total" not in df.index:
        raise ValueError("Fila 'Total' no encontrada en health_life_expectancy.csv")
    s = df.loc["Total"].astype(str).str.replace(",", ".").astype(float)
    df_long = s.reset_index()
    df_long.columns = ["Year_Num","Health_Life_Expectancy"]
    df_long["Year_Num"] = pd.to_numeric(df_long["Year_Num"], errors="coerce")
    return df_long.dropna(subset=["Year_Num","Health_Life_Expectancy"])\
                  .astype({"Year_Num": int})



def load_temperatures():
    df = pd.read_csv(
        "data/temp_edit.csv",  # o el nombre que le hayas dado
        sep=",",
        decimal=",",
        header=0
    )
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Year_Num", "media_anual": "Temp_C"})
    df["Year_Num"] = pd.to_numeric(df["Year_Num"], errors="coerce").astype("Int64")
    df["Temp_C"]   = pd.to_numeric(df["Temp_C"],   errors="coerce")
    df = df.dropna(subset=["Year_Num", "Temp_C"])
    df["Year_Num"] = df["Year_Num"].astype(int)
    df["Temp_C"]   = df["Temp_C"].round(2)

    return df[["Year_Num", "Temp_C"]]



import numpy as np



import pandas as pd

def load_mortality():
    df = pd.read_csv("data/mortality_rate_CVD.csv", sep=",", decimal=".", encoding="latin1")
    mask = df.iloc[:, 0].str.contains("Disease of the circulatory system", na=False)
    sel = df[mask & df.iloc[:, 1].str.strip().str.upper().eq("TOTAL")]

    if sel.empty:
        raise ValueError("No se encontró la fila 'Disease of the circulatory system - TOTAL'")

    years = sel.columns[2:]
    df_long = sel.melt(value_vars=years, var_name="Year_Num", value_name="Mortality")
    df_long["Year_Num"] = pd.to_numeric(df_long["Year_Num"], errors="coerce").astype("Int64")
    df_long["Mortality"] = pd.to_numeric(df_long["Mortality"], errors="coerce")

    return df_long.dropna(subset=["Year_Num", "Mortality"]).reset_index(drop=True)


def load_max_temps():

    df = pd.read_csv("data/max_temp.csv",
                     encoding="utf-8",
                     dtype={"Year_Num": int, "Max_Temp": float})
    df = df.rename(columns={
        "año": "Year_Num",        # si tu CSV usaba 'año'
        "media_anual": "Max_Temp" # si usaba 'media_anual'
    })
    df = df.dropna(subset=["Max_Temp"])
    return df[["Year_Num", "Max_Temp"]]

def load_medi_temp():
    df = pd.read_csv("data/medi_temp.csv")
    df = df.rename(columns={
        "año": "Year_Num",
        "media_verano": "Medi_Temp"  # ← esta línea era el problema
    })

    df["Year_Num"] = pd.to_numeric(df["Year_Num"], errors="coerce")
    df["Medi_Temp"] = pd.to_numeric(df["Medi_Temp"], errors="coerce")
    df = df.dropna(subset=["Year_Num", "Medi_Temp"])
    df = df.astype({"Year_Num": int})
    df = df[["Year_Num", "Medi_Temp"]].sort_values("Year_Num")

    return df

def load_all_data():
    return {
        "cases": load_cases(),
        "patients": load_patients(),
        "health_life_expectancy": load_health_expectancy(),
        "temperature": load_temperatures(),
        "mortality": load_mortality(),
        "max_temp": load_max_temps(),
        "medi_temp": load_medi_temp()
    }
