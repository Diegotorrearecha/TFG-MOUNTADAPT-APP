print("▶️ Arrancando script de preprocesado…")
import pandas as pd

# 1) Leemos el Excel original
df = pd.read_excel("data/long_temperatures.xlsx", header=None)

# 2) Nos quedamos solo con las filas que contienen datos (desde la fila 5 en adelante)
data = df.iloc[4:, :].copy()

# 3) Renombramos la primera columna a Year_Num
data = data.rename(columns={0: "Year_Num"})

# 4) Eliminamos columnas de texto (1 y 2) y convertimos todo lo demás a numérico
data_numeric = data.drop(columns=[1, 2], errors="ignore").apply(pd.to_numeric, errors="coerce")

# 5) Calculamos la media anual (columnas 3 en adelante)
data_numeric["Temp_C"] = data_numeric.loc[:, 3:].mean(axis=1)

# 6) Limpiamos filas sin año o sin temperatura
data_clean = data_numeric.dropna(subset=["Year_Num", "Temp_C"])

# 7) Ajustamos tipos
data_clean["Year_Num"] = data_clean["Year_Num"].astype(int)
data_clean["Temp_C"]  = data_clean["Temp_C"].round(2)

# 8) Nos quedamos solo con las dos columnas que nos interesan
df_annual = data_clean[["Year_Num", "Temp_C"]]

# 9) Guardamos a CSV
df_annual.to_csv("data/annual_temperatures.csv", index=False)
print("✔️  CSV generado: data/annual_temperatures.csv")

