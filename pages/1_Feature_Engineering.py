# pages/1_Feature_Engineering.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from my_utils.helpers import COLORS, to_excel_bytes

st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")
st.title("⚙️ Feature Engineering")
st.caption("Variables derivadas construidas sobre el dataset crudo")

if "df_fe" not in st.session_state:
    st.warning("Vuelve a la página **Home** para cargar los datos primero.")
    st.stop()

df = st.session_state.get("df_filtered", st.session_state["df_fe"])

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 RFM", "📉 Señales de cambio", "🔧 Fricción",
    "🔗 Profundidad producto", "📋 Tabla completa"
])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Análisis RFM")
    col1, col2 = st.columns(2)

    with col1:
        # Distribución rfm_total
        fig = px.histogram(
            df, x="rfm_total", color="churn",
            color_discrete_map={0: COLORS["teal"], 1: COLORS["coral"]},
            nbins=13, barmode="overlay", opacity=0.75,
            labels={"rfm_total": "RFM Total Score", "churn": "Churn"},
            title="Distribución RFM Total por Churn"
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Heatmap segmento RFM
        if "rfm_segmento" in df.columns and "segmento" in df.columns:
            rfm_pivot = (
                df.groupby(["segmento", "rfm_total"])
                .size().reset_index(name="n")
            )
            fig2 = px.density_heatmap(
                df, x="rfm_total", y="segmento",
                color_continuous_scale="Teal",
                title="Densidad RFM por Segmento",
            )
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    # Tabla RFM por segmento
    st.subheader("Estadísticas RFM por segmento")
    rfm_stats = (
        df.groupby("segmento")[["rfm_total", "rfm_recency_score",
                                 "rfm_frequency_score", "rfm_monetary_score"]]
        .mean().round(2).reset_index()
    )
    st.dataframe(rfm_stats, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Señales de cambio de comportamiento")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.box(
            df, x="churn", y="delta_logins",
            color="churn",
            color_discrete_map={0: COLORS["teal"], 1: COLORS["coral"]},
            labels={"churn": "Churn", "delta_logins": "Δ Logins"},
            title="Delta Logins por Churn",
            points="outliers",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", height=320, margin=dict(t=40, b=20))
        fig.update_xaxes(tickvals=[0, 1], ticktext=["No churn", "Churn"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x="churn", y="pct_cambio_logins",
            color="churn",
            color_discrete_map={0: COLORS["teal"], 1: COLORS["coral"]},
            labels={"churn": "Churn", "pct_cambio_logins": "% Cambio Logins"},
            title="% Cambio Logins por Churn",
            points="outliers",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", height=320, margin=dict(t=40, b=20))
        fig.update_xaxes(tickvals=[0, 1], ticktext=["No churn", "Churn"])
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.box(
            df, x="churn", y="delta_nps",
            color="churn",
            color_discrete_map={0: COLORS["teal"], 1: COLORS["coral"]},
            labels={"churn": "Churn", "delta_nps": "Δ NPS"},
            title="Delta NPS por Churn",
            points="outliers",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", height=320, margin=dict(t=40, b=20))
        fig.update_xaxes(tickvals=[0, 1], ticktext=["No churn", "Churn"])
        st.plotly_chart(fig, use_container_width=True)

    # Correlación de señales de cambio con churn
    st.subheader("Correlación de variables de cambio con churn")
    change_vars = ["delta_logins", "pct_cambio_logins", "delta_nps",
                    "dias_sin_login", "logins_30d", "nps_actual"]
    corr_vals = df[[*change_vars, "churn"]].corr()["churn"].drop("churn").sort_values()
    fig_corr = px.bar(
        x=corr_vals.values, y=corr_vals.index, orientation="h",
        color=corr_vals.values,
        color_continuous_scale=["#D85A30", "#F1EFE8", "#1D9E75"],
        color_continuous_midpoint=0,
        labels={"x": "Correlación con churn", "y": "Variable"},
        title="Correlación de Pearson con Churn",
    )
    fig_corr.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            coloraxis_showscale=False, height=300,
                            margin=dict(t=40, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Señales de fricción")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="friccion_score", color="churn",
            color_discrete_map={0: COLORS["teal"], 1: COLORS["coral"]},
            nbins=20, barmode="overlay", opacity=0.75,
            title="Distribución Fricción Score por Churn",
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df, x="tasa_resolucion", y="tasa_completacion",
            color="churn",
            color_discrete_map={0: COLORS["teal"], 1: COLORS["coral"]},
            opacity=0.5,
            title="Tasa Resolución vs Tasa Completación",
            labels={"tasa_resolucion": "Tasa Resolución",
                    "tasa_completacion": "Tasa Completación"},
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Fricción por segmento y región
    st.subheader("Fricción media por segmento")
    fric_seg = df.groupby("segmento")[
        ["friccion_score", "tasa_resolucion", "tickets_sin_resolver"]
    ].mean().round(3).reset_index()
    st.dataframe(fric_seg, use_container_width=True)

    if "region" in df.columns:
        st.subheader("Fricción media por región")
        fric_reg = df.groupby("region")[
            ["friccion_score", "tasa_resolucion", "tickets_sin_resolver"]
        ].mean().round(3).reset_index().sort_values("friccion_score", ascending=False)
        fig_fr = px.bar(
            fric_reg, x="region", y="friccion_score",
            color="friccion_score",
            color_continuous_scale=["#E1F5EE", "#D85A30"],
            title="Fricción Score Promedio por Región",
        )
        fig_fr.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              height=320, margin=dict(t=40, b=20),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_fr, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Profundidad de producto y engagement")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df, x="profundidad_producto", y="engagement_score",
            color="churn",
            color_discrete_map={0: COLORS["teal"], 1: COLORS["coral"]},
            opacity=0.5,
            title="Profundidad vs Engagement",
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x="segmento", y="profundidad_producto",
            color="segmento",
            color_discrete_map={
                "Enterprise": COLORS["teal"], "SMB": COLORS["blue"],
                "Startup": COLORS["amber"], "Individual": COLORS["coral"],
            },
            title="Profundidad de Producto por Segmento",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # LTV estimado
    st.subheader("LTV estimado por segmento")
    ltv_stats = df.groupby("segmento")["ltv_estimado"].describe().round(0).reset_index()
    st.dataframe(ltv_stats, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Tabla completa de features")

    fe_cols = [c for c in [
        "cliente_id", "segmento", "region", "churn",
        "rfm_total", "rfm_segmento", "engagement_score",
        "delta_logins", "pct_cambio_logins", "delta_nps",
        "friccion_score", "tasa_resolucion", "tasa_completacion",
        "profundidad_producto", "ltv_estimado", "mrr",
        "en_periodo_renovacion", "contrato_vencido",
    ] if c in df.columns]

    st.dataframe(df[fe_cols].reset_index(drop=True), use_container_width=True, height=420)

    excel_fe = to_excel_bytes(df[fe_cols], "Feature_Engineering")
    st.download_button(
        "⬇️ Exportar features (Excel)",
        data=excel_fe,
        file_name="features_engineering.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
