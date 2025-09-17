import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd



def plot_custom_correlation(df_x, df_y, label_x, label_y):
    st.write(f"### ✅ {label_x} vs {label_y}")

    try:
        # Agrupar por año
        x_by_year = df_x.groupby("Year_Num")[df_x.columns[-1]].sum().reset_index()
        y_by_year = df_y.groupby("Year_Num")[df_y.columns[-1]].sum().reset_index()

        # Unir por año
        merged = pd.merge(x_by_year, y_by_year, on="Year_Num", how="inner")
        merged.columns = ["Year_Num", "X", "Y"]

        if len(merged) < 2:
            st.warning("Not enough data to plot correlation.")
            return

        # Gráfico dual
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Year")
        ax1.set_ylabel(label_x, color="tab:blue")
        ax1.plot(merged["Year_Num"], merged["X"], color="tab:blue", label=label_x)
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel(label_y, color="tab:red")
        ax2.plot(merged["Year_Num"], merged["Y"], color="tab:red", label=label_y)
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Plotting error: {e}")

# utils/graphs.py
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

def plot_coverage_gantt(df_dict, use_vars=None, title="Coverage by variable", save_path=None):
    """
    df_dict: dict {nombre_variable: DataFrame con columnas ['Year_Num', <valor>]}
    use_vars: lista opcional de claves de df_dict a incluir (orden = el de la lista)
    Devuelve: fig, meta (dict con ventana efectiva y tabla resumen)
    """
    # Filtrar y ordenar variables
    data = {k: v for k, v in df_dict.items() if v is not None and "Year_Num" in v.columns}
    if use_vars:
        data = {k: data[k] for k in use_vars if k in data}

    # Coberturas individuales
    rows = []
    year_sets = []
    for name, df in data.items():
        years = sorted(pd.to_numeric(df["Year_Num"], errors="coerce").dropna().astype(int).unique())
        if len(years) == 0:
            continue
        rows.append({"variable": name, "min": years[0], "max": years[-1], "n": len(years)})
        year_sets.append(set(years))

    if not rows:
        raise ValueError("No hay variables válidas con Year_Num.")

    # Ventana efectiva común (intersección)
    effective_years = sorted(set.intersection(*year_sets)) if len(year_sets) > 1 else sorted(list(year_sets[0]))
    eff_start = effective_years[0] if effective_years else None
    eff_end   = effective_years[-1] if effective_years else None
    eff_n     = len(effective_years)

    # Figura
    rows_df = pd.DataFrame(rows)
    if use_vars:
        rows_df["order"] = rows_df["variable"].apply(lambda x: use_vars.index(x) if x in use_vars else 1e9)
        rows_df = rows_df.sort_values("order")
    else:
        rows_df = rows_df.sort_values("variable")

    fig_h = 1.2 + 0.5 * len(rows_df)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y_positions = range(len(rows_df))

    for y, r in zip(y_positions, rows_df.itertuples()):
        ax.broken_barh([(r.min, r.max - r.min + 1)], (y - 0.35, 0.7))
        ax.text(r.max + 0.3, y, f"[{r.min}-{r.max}]  n={r.n}", va="center", fontsize=9)

    # Banda de ventana efectiva común
    if eff_start is not None and eff_end is not None and eff_end >= eff_start:
        ax.add_patch(Rectangle((eff_start - 0.5, -0.6), eff_end - eff_start + 1, len(rows_df)+0.2,
                               alpha=0.15, edgecolor="none"))
        ax.text(eff_start, len(rows_df)-0.4, f"Effective window: {eff_start}–{eff_end} (n={eff_n})",
                fontsize=9)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(rows_df["variable"].tolist())
    ax.set_xlabel("Year")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    meta = {
        "effective_start": eff_start,
        "effective_end": eff_end,
        "effective_n": eff_n,
        "table": rows_df[["variable", "min", "max", "n"]]
    }
    return fig, meta
