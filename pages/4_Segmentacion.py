# pages/4_Segmentacion.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from my_utils.helpers import COLORS, QUAD_COLORS, QUAD_ACTIONS, to_excel_bytes

st.set_page_config(page_title="Segmentación", page_icon="🗺️", layout="wide")
st.title("🗺️ Segmentación y Priorización")
st.caption("Matriz LTV × Riesgo, microsegmentos y tablas de acción por región y categoría")
st.caption("MRR: Monthly Recurring Revenue (Ingresos Recurrentes Mensuales)")

if "df_fe" not in st.session_state:
    st.warning("Vuelve a la página **Home** para cargar los datos primero.")
    st.stop()

df   = st.session_state.get("df_filtered", st.session_state["df_fe"])
meta = st.session_state["meta"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Matriz de priorización", "🗂️ Por región",
    "📦 Por segmento", "🔬 Microsegmentos",
    "📋 Tabla exportable"
])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Matriz LTV × Riesgo de Churn")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(
            df,
            x="ltv_estimado", y="churn_score",
            color="cuadrante",
            color_discrete_map=QUAD_COLORS,
            hover_data=[c for c in ["cliente_id", "segmento", "region", "mrr"]
                        if c in df.columns],
            opacity=0.65, size_max=8,
            labels={"ltv_estimado": "LTV Estimado ($)",
                    "churn_score":  "Churn Score"},
        )
        fig.add_hline(y=meta["risk_threshold"], line_dash="dash",
                       line_color=COLORS["gray"], opacity=0.5,
                       annotation_text=f"Umbral riesgo ({meta['risk_threshold']})")
        fig.add_vline(x=meta["ltv_median"], line_dash="dash",
                       line_color=COLORS["gray"], opacity=0.5,
                       annotation_text=f"Mediana LTV")
        fig.update_layout(
            height=480, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Resumen cuadrantes")
        for quad, color in QUAD_COLORS.items():
            sub = df[df["cuadrante"] == quad]
            if len(sub) == 0:
                continue
            mrr_quad  = sub["mrr"].sum()
            churn_rate = sub["churn"].mean() if "churn" in sub.columns else 0
            st.markdown(f"""
            <div style="border-left:4px solid {color};padding:8px 12px;
                        margin-bottom:10px;border-radius:0 8px 8px 0;
                        background:#FAFAF8">
              <b style="color:{color}">{quad}</b><br>
              <small>{len(sub):,} donantes · MRR ${mrr_quad:,.0f}<br>
              Tasa churn real: {churn_rate:.1%}</small><br>
              <small style="color:#5F5E5A;font-style:italic">{QUAD_ACTIONS[quad]}</small>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if "region" not in df.columns:
        st.info("El dataset no contiene la columna 'region'.")
    else:
        st.subheader("Matriz de priorización por región")

        # Tabla pivot: región × cuadrante
        pivot = (
            df.groupby(["region", "cuadrante"])
            .agg(n_clientes=("churn_score", "count"),
                 score_medio=("churn_score", "mean"),
                 mrr_riesgo=("mrr_en_riesgo", "sum"))
            .reset_index()
        )

        # Heatmap: donantes por región y cuadrante
        pivot_wide = pivot.pivot_table(
            index="region", columns="cuadrante",
            values="n_clientes", fill_value=0
        )
        fig_heat = px.imshow(
            pivot_wide,
            color_continuous_scale=["#E1F5EE", "#D85A30"],
            text_auto=True,
            title="Nº Donates por Región y Cuadrante",
            aspect="auto",
        )
        fig_heat.update_layout(height=420, margin=dict(t=40, b=20),
                                coloraxis_showscale=False)
        st.plotly_chart(fig_heat, use_container_width=True)

        # MRR en riesgo por región
        st.subheader("MRR en riesgo por región")
        mrr_reg = (
            df.groupby("region")
            .agg(
                clientes=("churn_score", "count"),
                score_medio=("churn_score", "mean"),
                mrr_total=("mrr", "sum"),
                mrr_en_riesgo=("mrr_en_riesgo", "sum"),
                max_prioridad=("cuadrante",
                               lambda x: (x == "Máxima Prioridad").sum()),
            )
            .reset_index()
            .sort_values("mrr_en_riesgo", ascending=False)
        )
        mrr_reg["pct_mrr_riesgo"] = mrr_reg["mrr_en_riesgo"] / mrr_reg["mrr_total"]

        fig_mrr = px.bar(
            mrr_reg, x="region", y="mrr_en_riesgo",
            color="score_medio",
            color_continuous_scale=["#E1F5EE", "#D85A30"],
            text=mrr_reg["mrr_en_riesgo"].map(lambda x: f"${x:,.0f}"),
            title="MRR en Riesgo por Región",
        )
        fig_mrr.update_layout(
            height=360, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, b=20), coloraxis_showscale=False,
            xaxis_title="", yaxis_title="MRR en riesgo ($)",
        )
        st.plotly_chart(fig_mrr, use_container_width=True)

        st.dataframe(
            mrr_reg.style.format({
                "score_medio":      "{:.3f}",
                "mrr_total":        "${:,.0f}",
                "mrr_en_riesgo":    "${:,.0f}",
                "pct_mrr_riesgo":   "{:.1%}",
            }).background_gradient(subset=["mrr_en_riesgo"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

        excel_reg = to_excel_bytes(mrr_reg, "Riesgo_por_Region")
        st.download_button(
            "⬇️ Exportar tabla región (Excel)", excel_reg,
            "riesgo_por_region.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Análisis por segmento de cliente")

    seg_stats = (
        df.groupby("segmento")
        .agg(
            n=("churn_score", "count"),
            churn_score_medio=("churn_score", "mean"),
            churn_real=("churn", "mean"),
            ltv_medio=("ltv_estimado", "mean"),
            mrr_total=("mrr", "sum"),
            mrr_riesgo=("mrr_en_riesgo", "sum"),
            max_prioridad=("cuadrante",
                            lambda x: (x == "Máxima Prioridad").sum()),
        )
        .reset_index()
        .sort_values("churn_score_medio", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            seg_stats, x="segmento", y="churn_real",
            color="segmento",
            color_discrete_map={
                "Enterprise": COLORS["teal"], "SMB": COLORS["blue"],
                "Startup": COLORS["amber"], "Individual": COLORS["coral"],
            },
            text=seg_stats["churn_real"].map(lambda x: f"{x:.1%}"),
            title="Tasa de Churn Real por Segmento",
        )
        fig.update_layout(showlegend=False, height=320,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            seg_stats, x="segmento", y="mrr_riesgo",
            color="segmento",
            color_discrete_map={
                "Enterprise": COLORS["teal"], "SMB": COLORS["blue"],
                "Startup": COLORS["amber"], "Individual": COLORS["coral"],
            },
            text=seg_stats["mrr_riesgo"].map(lambda x: f"${x:,.0f}"),
            title="MRR en Riesgo por Segmento",
        )
        fig2.update_layout(showlegend=False, height=320,
                            plot_bgcolor="white", paper_bgcolor="white",
                            margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        seg_stats.style.format({
            "churn_score_medio": "{:.3f}",
            "churn_real":        "{:.1%}",
            "ltv_medio":         "${:,.0f}",
            "mrr_total":         "${:,.0f}",
            "mrr_riesgo":        "${:,.0f}",
        }).background_gradient(subset=["churn_score_medio"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

    excel_seg = to_excel_bytes(seg_stats, "Riesgo_por_Segmento")
    st.download_button(
        "⬇️ Exportar tabla segmento (Excel)", excel_seg,
        "riesgo_por_segmento.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Microsegmentos — zona de alto riesgo")
    hr_df = df[df["microsegmento"] >= 0].copy()

    if len(hr_df) == 0:
        st.info("No hay microsegmentos calculados para la selección actual.")
    else:
        micro_stats = (
            hr_df.groupby("microsegmento")
            .agg(
                n=("churn_score", "count"),
                score_medio=("churn_score", "mean"),
                ltv_medio=("ltv_estimado", "mean"),
                eng_medio=("engagement_score", "mean"),
                fric_medio=("friccion_score", "mean"),
                delta_nps_medio=("delta_nps", "mean"),
                churn_real=("churn", "mean"),
            )
            .reset_index()
        )

        micro_colors = [COLORS["coral"], COLORS["amber"], COLORS["purple"], COLORS["teal"]]
        fig = px.scatter(
            micro_stats, x="ltv_medio", y="score_medio",
            size="n", color="microsegmento",
            color_discrete_sequence=micro_colors,
            text="microsegmento",
            size_max=50,
            title="Microsegmentos: LTV Medio vs Score Medio",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=380, plot_bgcolor="white",
                           paper_bgcolor="white", margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            micro_stats.style.format({
                "score_medio":     "{:.3f}",
                "ltv_medio":       "${:,.0f}",
                "eng_medio":       "{:.3f}",
                "fric_medio":      "{:.1f}",
                "delta_nps_medio": "{:.2f}",
                "churn_real":      "{:.1%}",
            }).background_gradient(subset=["score_medio"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Tabla exportable — todos los donantes con score y cuadrante")

    export_cols = [c for c in [
        "cliente_id", "segmento", "region",
        "churn_score", "cuadrante", "decil_riesgo",
        "ltv_estimado", "mrr", "mrr_en_riesgo",
        "churn", "delta_nps", "friccion_score",
        "dias_renovacion", "contrato_vencido",
        "engagement_score", "profundidad_producto",
    ] if c in df.columns]

    # Filtro adicional dentro de la página
    quad_filter = st.multiselect(
        "Filtrar por cuadrante",
        options=list(QUAD_COLORS.keys()),
        default=list(QUAD_COLORS.keys()),
    )
    export_df = df[df["cuadrante"].isin(quad_filter)][export_cols].reset_index(drop=True)

    st.dataframe(
        export_df.style.format({
            "churn_score":    "{:.1%}",
            "ltv_estimado":   "${:,.0f}",
            "mrr":            "${:,.0f}",
            "mrr_en_riesgo":  "${:,.0f}",
        }).background_gradient(subset=["churn_score"], cmap="RdYlGn_r"),
        use_container_width=True, height=440,
    )
    st.caption(f"{len(export_df):,} donantes en la selección")

    excel_exp = to_excel_bytes(export_df, "Donantes_Segmentados")
    st.download_button(
        "⬇️ Exportar tabla completa (Excel)", excel_exp,
        "donantes_segmentados.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
