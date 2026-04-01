# utils/helpers.py
# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares compartidas por todas las páginas.
# Sin dependencias de Streamlit — importable desde cualquier contexto.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import io
import os


# ── Paleta corporativa ────────────────────────────────────────────────────────
COLORS = {
    "teal":      "#1D9E75", "teal_lt":   "#E1F5EE",
    "coral":     "#D85A30", "coral_lt":  "#FAECE7",
    "purple":    "#7F77DD", "purple_lt": "#EEEDFE",
    "amber":     "#BA7517", "amber_lt":  "#FAEEDA",
    "gray":      "#888780", "gray_lt":   "#F1EFE8",
    "blue":      "#378ADD", "blue_lt":   "#E6F1FB",
    "green":     "#639922", "green_lt":  "#EAF3DE",
    "red":       "#E24B4A", "red_lt":    "#FCEBEB",
    "text":      "#2C2C2A", "text2":     "#5F5E5A",
    "bg":        "#FAFAF8", "border":    "#D3D1C7",
}

SEGMENT_COLORS = {
    "Enterprise": COLORS["teal"],
    "SMB":        COLORS["blue"],
    "Startup":    COLORS["amber"],
    "Individual": COLORS["coral"],
}

QUAD_COLORS = {
    "Máxima Prioridad":    COLORS["coral"],
    "Proteger Relación":   COLORS["teal"],
    "Intervención Ligera": COLORS["amber"],
    "Monitoreo Pasivo":    COLORS["gray"],
}

QUAD_ACTIONS = {
    "Máxima Prioridad":    "Account manager + oferta personalizada urgente",
    "Proteger Relación":   "Programa de fidelización y acceso anticipado a features",
    "Intervención Ligera": "Campaña automatizada de bajo costo",
    "Monitoreo Pasivo":    "Email automático mensual, sin acción activa",
}


# ── Carga de datos ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """
    Carga el dataset desde Excel.
    Limpia nombres de columnas y tipos básicos.
    """
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[()$%]", "", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.rstrip("_")
    )
    # Asegurar tipos numéricos en columnas clave
    numeric_cols = [
        "tenure_meses", "plan_precio", "num_usuarios",
        "logins_30d", "logins_30d_previos", "features_usados",
        "tickets_soporte", "tickets_resueltos",
        "nps_actual", "nps_30d_atras",
        "paginas_visitadas", "sesiones_30d", "sesiones_completadas",
        "api_calls_30d", "dias_sin_login", "dias_renovacion",
        "num_integraciones", "fallos_pago", "degradaciones",
        "referidos_dados", "tiempo_al_evento", "churn",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ── Exportar DataFrame a Excel en memoria ─────────────────────────────────────
def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Datos") -> bytes:
    """Convierte un DataFrame a bytes de Excel para descarga en Streamlit."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


# ── Métricas rápidas de resumen ────────────────────────────────────────────────
def resumen_rapido(df: pd.DataFrame) -> dict:
    """Devuelve KPIs básicos del dataset filtrado."""
    total = len(df)
    churners = int(df["churn"].sum()) if "churn" in df.columns else 0
    tasa = churners / total if total > 0 else 0
    mrr = (df["plan_precio"] * df["num_usuarios"]).sum() if "plan_precio" in df.columns else 0
    mrr_riesgo = (
        (df.loc[df["churn"] == 1, "plan_precio"] * df.loc[df["churn"] == 1, "num_usuarios"]).sum()
        if "churn" in df.columns else 0
    )
    return {
        "total_clientes":  total,
        "churners":        churners,
        "tasa_churn":      tasa,
        "mrr_total":       mrr,
        "mrr_en_riesgo":   mrr_riesgo,
    }


# ── Kaplan-Meier manual ────────────────────────────────────────────────────────
def kaplan_meier(times: np.ndarray, events: np.ndarray):
    """
    Calcula la curva de supervivencia Kaplan-Meier.
    Devuelve (tiempos, supervivencias).
    """
    df_km = pd.DataFrame({"t": times, "e": events}).sort_values("t")
    unique_t = np.sort(np.unique(df_km["t"][df_km["e"] == 1]))
    S, km_t = [1.0], [0.0]
    for t in unique_t:
        d = int(((df_km["t"] == t) & (df_km["e"] == 1)).sum())
        r = int((df_km["t"] >= t).sum())
        if r > 0:
            S.append(S[-1] * (1 - d / r))
            km_t.append(float(t))
    return np.array(km_t), np.array(S)


# ── PSI ────────────────────────────────────────────────────────────────────────
def calculate_psi(expected: np.ndarray, actual: np.ndarray,
                  buckets: int = 10) -> float:
    """
    Population Stability Index.
    < 0.10 → estable | 0.10-0.20 → leve | > 0.20 → reentrenar
    """
    breakpoints = np.linspace(0, 1, buckets + 1)
    exp_c, _ = np.histogram(expected, bins=breakpoints)
    act_c, _ = np.histogram(actual,   bins=breakpoints)
    exp_p = np.where(exp_c == 0, 1e-4, exp_c / len(expected))
    act_p = np.where(act_c == 0, 1e-4, act_c / len(actual))
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


# ── Etiqueta PSI ───────────────────────────────────────────────────────────────
def psi_label(psi: float) -> tuple:
    """Devuelve (emoji, texto, color) según el valor de PSI."""
    if psi < 0.10:
        return "✅", "Estable",     COLORS["teal"]
    elif psi < 0.20:
        return "⚠️",  "Leve",       COLORS["amber"]
    else:
        return "🔴", "Reentrenar", COLORS["red"]


# ── Clasificador de segmento RFM ───────────────────────────────────────────────
def clasificar_rfm(r: int, f: int, m: int) -> str:
    if r >= 4 and f >= 4 and m >= 4: return "VIP"
    elif r >= 4 and f >= 4:           return "Fiel activo"
    elif r >= 4:                       return "Reciente"
    elif m >= 4:                       return "Ballena"
    elif r >= 3:                       return "En riesgo"
    elif r <= 2 and f <= 2:            return "Dormido"
    else:                              return "Perdido"


# ── Asegurar carpetas de salida ────────────────────────────────────────────────
def ensure_dirs(base: str):
    for folder in ["models_saved", "exports"]:
        os.makedirs(os.path.join(base, folder), exist_ok=True)
