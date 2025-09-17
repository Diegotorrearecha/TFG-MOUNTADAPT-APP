import numpy as np
import pandas as pd
import random


# --- Ruido gaussiano
def add_gaussian_noise(df, noise_level=0.05, n_copias=1):
    augmented = []
    cols = df.select_dtypes(include=[np.number]).columns.drop("Year_Num", errors="ignore")
    for _ in range(n_copias):
        noisy = df.copy()
        for col in cols:
            ruido = np.random.normal(0, noise_level * df[col].std(), size=len(df))
            noisy[col] += ruido
        augmented.append(noisy)
    return pd.concat([df] + augmented, ignore_index=True)

# --- Mixup
def mixup_rows(df, n_samples=None):
    if n_samples is None:
        n_samples = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_mix = []

    for _ in range(n_samples):
        rows = df.sample(2).reset_index(drop=True)
        row_mix = {}

        for col in df.columns:
            val_a = rows.loc[0, col]
            val_b = rows.loc[1, col]

            if col in numeric_cols:
                alpha = np.random.rand()
                row_mix[col] = alpha * val_a + (1 - alpha) * val_b
            else:
                row_mix[col] = val_a  # copiar directamente el valor categórico o string

        df_mix.append(row_mix)

    df_mix_df = pd.DataFrame(df_mix)
    return pd.concat([df, df_mix_df], ignore_index=True)

# --- Interpolación aleatoria
def interpolate_rows(df, n_samples=None):
    if n_samples is None:
        n_samples = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_interp = []

    for _ in range(n_samples):
        rows = df.sample(2).reset_index(drop=True)
        row_interp = {}

        for col in df.columns:
            val_a = rows.loc[0, col]
            val_b = rows.loc[1, col]

            if col in numeric_cols:
                row_interp[col] = (val_a + val_b) / 2
            else:
                row_interp[col] = val_a  # Copia directa si no es numérica

        df_interp.append(row_interp)

    df_interp_df = pd.DataFrame(df_interp)
    return pd.concat([df, df_interp_df], ignore_index=True)


# --- Aplicar técnica a todos los dfs
def augment_all_dfs(df_dict, method="noise", **kwargs):
    df_dict_aug = {}
    for name, df in df_dict.items():
        if df is None or "Year_Num" not in df.columns:
            continue
        if method == "noise":
            df_dict_aug[name] = add_gaussian_noise(df, **kwargs)
        elif method == "mixup":
            df_dict_aug[name] = mixup_rows(df, **kwargs)
        elif method == "interpolation":
            df_dict_aug[name] = interpolate_rows(df, **kwargs)
        else:
            raise ValueError(f"Método de aumento no reconocido: {method}")
    return df_dict_aug

def augment_mixup_with_noise(df_dict, n_samples=10, noise_level=0.05):

    augmented_dict = {}

    for key, df in df_dict.items():
        cols = df.columns
        df = df.copy()
        augmented_rows = []

        for _ in range(n_samples):
            i, j = np.random.randint(0, len(df), size=2)
            λ = np.random.rand()
            new_row = λ * df.iloc[i].values + (1 - λ) * df.iloc[j].values
            # Añadir ruido
            noise = np.random.normal(0, df.std().values * noise_level)
            new_row += noise
            augmented_rows.append(new_row)

        augmented_df = pd.DataFrame(augmented_rows, columns=cols)
        full_df = pd.concat([df, augmented_df], ignore_index=True)
        augmented_dict[key] = full_df


    return augmented_dict

def augment_with_interpolation(df_dict, missing_rate=0.3):
    """
    Elimina aleatoriamente valores y los interpola para crear una versión augmentada.
    """
   
    df_dict_interp = {}

    for var, df in df_dict.items():
        df_copy = df.copy()
        value_col = [col for col in df.columns if col != "Year_Num"][0]

        # Crear una copia con valores faltantes aleatorios
        df_copy.loc[df_copy.sample(frac=missing_rate).index, value_col] = np.nan

        # Interpolación lineal
        df_copy[value_col] = df_copy[value_col].interpolate(method='linear', limit_direction='both')

        df_dict_interp[var] = df_copy

    return df_dict_interp

