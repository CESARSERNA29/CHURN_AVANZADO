# utils/feature_engineering.py
# ─────────────────────────────────────────────────────────────────────────────
# Toda la lógica de Feature Engineering en funciones puras.
# Sin Streamlit. Reutilizable desde páginas y scripts externos.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np


# Lista canónica de features para el modelo
FEATURE_COLS = [
    "tenure_meses", "plan_precio", "num_usuarios",
    "logins_30d", "delta_logins", "pct_cambio_logins",
    "features_usados", "tickets_sin_resolver", "tasa_resolucion",
    "delta_nps", "nps_actual",
    "tasa_completacion", "dias_sin_login",
    "fallos_pago", "degradaciones", "num_integraciones",
    "en_periodo_renovacion", "contrato_vencido",
    "rfm_total", "engagement_score", "friccion_score",
    "profundidad_producto", "ltv_estimado", "referidos_dados",
    "api_calls_30d",
    "seg_Enterprise", "seg_SMB", "seg_Startup", "seg_Individual",
]


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el DataFrame crudo y devuelve uno enriquecido
    con todas las features derivadas.
    No modifica el DataFrame original (trabaja sobre copia).
    """
    d = df.copy()

    # ── RFM ───────────────────────────────────────────────────────────────────
    d["rfm_recency"]   = 1 / (1 + d["dias_sin_login"])
    d["rfm_frequency"] = d["logins_30d"] / (d["tenure_meses"] + 1)
    d["rfm_monetary"]  = d["plan_precio"] * d["num_usuarios"]

    for col in ["rfm_recency", "rfm_frequency", "rfm_monetary"]:
        d[f"{col}_score"] = pd.qcut(
            d[col], q=5, labels=[1, 2, 3, 4, 5],
            duplicates="drop"
        ).astype(float)

    d["rfm_total"] = (
        d["rfm_recency_score"] +
        d["rfm_frequency_score"] +
        d["rfm_monetary_score"]
    )

    # Etiqueta RFM legible
    d["rfm_r"] = d["rfm_recency_score"].fillna(1).astype(int)
    d["rfm_f"] = d["rfm_frequency_score"].fillna(1).astype(int)
    d["rfm_m"] = d["rfm_monetary_score"].fillna(1).astype(int)
    d["rfm_segmento"] = (
        d["rfm_r"].astype(str) +
        d["rfm_f"].astype(str) +
        d["rfm_m"].astype(str)
    )

    # ── Señales de cambio ─────────────────────────────────────────────────────
    d["delta_logins"]       = d["logins_30d"] - d["logins_30d_previos"]
    d["pct_cambio_logins"]  = d["delta_logins"] / (d["logins_30d_previos"] + 1)
    d["delta_nps"]          = d["nps_actual"] - d["nps_30d_atras"]

    # ── Señales de fricción ───────────────────────────────────────────────────
    d["tasa_resolucion"]       = d["tickets_resueltos"] / (d["tickets_soporte"] + 1)
    d["tasa_completacion"]     = d["sesiones_completadas"] / (d["sesiones_30d"] + 1)
    d["tickets_sin_resolver"]  = d["tickets_soporte"] - d["tickets_resueltos"]
    d["friccion_score"]        = (
        d["tickets_sin_resolver"] * 2 +
        d["fallos_pago"] * 3 +
        d["degradaciones"] * 4
    )

    # ── Profundidad de producto ───────────────────────────────────────────────
    d["profundidad_producto"] = (
        d["features_usados"] / 20 +
        d["num_integraciones"] / 10 +
        np.log1p(d["api_calls_30d"]) / 15
    )

    # ── Señales de contrato ───────────────────────────────────────────────────
    d["en_periodo_renovacion"] = (
        d["dias_renovacion"].between(-30, 60)
    ).astype(int)
    d["contrato_vencido"] = (d["dias_renovacion"] < 0).astype(int)

    # ── Engagement score ──────────────────────────────────────────────────────
    freq_max = d["rfm_frequency"].max() + 1e-6
    prof_max = d["profundidad_producto"].max() + 1e-6
    d["engagement_score"] = (
        0.25 * d["rfm_frequency"] / freq_max +
        0.20 * d["tasa_completacion"] +
        0.20 * d["profundidad_producto"] / prof_max +
        0.15 * d["nps_actual"] / 10 +
        0.10 * d["referidos_dados"] / (d["referidos_dados"].max() + 1e-6) +
        0.10 * d["rfm_recency"]
    )

    # ── LTV estimado ──────────────────────────────────────────────────────────
    d["ltv_estimado"] = d["plan_precio"] * d["num_usuarios"] * np.maximum(d["tenure_meses"], 1)
    d["mrr"]          = d["plan_precio"] * d["num_usuarios"]

    # ── One-hot encoding de segmento ──────────────────────────────────────────
    for seg in ["Enterprise", "SMB", "Startup", "Individual"]:
        col = f"seg_{seg}"
        if "segmento" in d.columns:
            d[col] = (d["segmento"] == seg).astype(int)
        else:
            d[col] = 0

    # ── Asegurar que todas las feature cols existen ───────────────────────────
    for col in FEATURE_COLS:
        if col not in d.columns:
            d[col] = 0

    # ── NPS clasificación ─────────────────────────────────────────────────────
    d["nps_categoria"] = pd.cut(
        d["nps_actual"],
        bins=[-1, 6, 8, 10],
        labels=["Detractor", "Pasivo", "Promotor"]
    )

    return d
