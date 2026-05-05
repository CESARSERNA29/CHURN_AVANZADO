# Home.py
# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada principal del dashboard de Churn.
# Carga datos, ejecuta Feature Engineering y entrena modelos UNA SOLA VEZ,
# guarda todo en st.session_state para que las páginas lo consuman sin
# recalcular.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

# Asegurar que la raíz del proyecto esté en el path
# sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(r"C:\Users\cesar\OneDrive\Escritorio\UNICEF\CHURN STREAMLIT\SEGUNDA PRUEBA")


# Estos no son librerías, se están llamado unos scripts de Python
from my_utils.helpers import load_data, resumen_rapido, ensure_dirs, COLORS, to_excel_bytes
from my_utils.feature_engineering import run_feature_engineering
from my_utils.models        import train_all_models

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Churn Intelligence Dashboard",
    page_icon   = "🔄",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Ruta base del proyecto ─────────────────────────────────────────────────────
BASE_DIR = os.getcwd()
#BASE_DIR     = os.path.dirname(__file__)
DATA_PATH    = os.path.join(BASE_DIR, "data", "02_Datos_Churn_Streamlit.xlsx")
MODELS_DIR   = os.path.join(BASE_DIR, "models_saved")
EXPORTS_DIR  = os.path.join(BASE_DIR, "exports")
ensure_dirs(BASE_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA Y PROCESAMIENTO (cached — se ejecuta solo cuando cambia el archivo)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Cargando y procesando datos…")
def load_and_process(path: str):
    df_raw = load_data(path)
    df_fe  = run_feature_engineering(df_raw)
    return df_raw, df_fe


@st.cache_resource(show_spinner="Entrenando modelos… (solo la primera vez)")
def train_models_cached(path: str):
    _, df_fe = load_and_process(path)
    results  = train_all_models(df_fe, models_dir=MODELS_DIR)
    return results


# ── Carga inicial ──────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error(
        f"**Archivo no encontrado:** `{DATA_PATH}`\n\n"
        "Por favor verifica que el archivo Excel esté en la carpeta `data/`."
    )
    st.stop()

df_raw, df_fe = load_and_process(DATA_PATH)
results       = train_models_cached(DATA_PATH)

# Adjuntar scores y cuadrantes al DataFrame enriquecido
meta = results["meta"]
df_fe["churn_score"]   = meta["ens_all"]
df_fe["cuadrante"]     = meta["cuadrante"]
df_fe["decil_riesgo"]  = meta["decil"]
df_fe["microsegmento"] = meta["microseg"]
df_fe["mrr_en_riesgo"] = df_fe["mrr"] * df_fe["churn_score"]

# Guardar en session_state para que las páginas lo usen
st.session_state["df_raw"]    = df_raw
st.session_state["df_fe"]     = df_fe
st.session_state["results"]   = results
st.session_state["meta"]      = meta
st.session_state["data_path"] = DATA_PATH
st.session_state["base_dir"]  = BASE_DIR
st.session_state["exports_dir"] = EXPORTS_DIR


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTROS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
# 🎨 ESTILO KPI + ajustes
st.markdown("""
<style>

/* KPI estilo tarjeta */
.kpi-box {
    background-color:#f1f5f9;
    padding:12px;
    border-radius:10px;
    text-align:center;
    margin-bottom:10px;
}

.kpi-title {
    font-size:12px;
    color:#64748b;
}

.kpi-value {
    font-size:20px;
    font-weight:700;
    color:#0f172a;
}

</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/000000/recurring-appointment.png", width=40)
    st.title("Churn Intelligence")
    st.caption("Dashboard de Análisis y Retención de Donantes")
    st.divider()

    st.subheader("🔎 Filtros globales")

    # =========================
    # SEGMENTO (con conteo)
    # =========================
    if "segmento" in df_fe.columns:
        seg_counts = df_fe["segmento"].value_counts()
        segmentos_disp = seg_counts.index.tolist()
        segmentos_labels = [f"{s} ({seg_counts[s]:,})" for s in segmentos_disp]

        seg_map = dict(zip(segmentos_labels, segmentos_disp))

        seg_sel_labels = st.segmented_control(
            "Segmento de donantes",
            options=segmentos_labels,
            selection_mode="multi",
            default=segmentos_labels,
        )

        seg_sel = [seg_map[s] for s in seg_sel_labels] if seg_sel_labels else []

    else:
        seg_sel = []

    # =========================
    # REGIÓN (con conteo)
    # =========================
    if "region" in df_fe.columns:
        reg_counts = df_fe["region"].value_counts()
        regiones_disp = reg_counts.index.tolist()
        regiones_labels = [f"{r} ({reg_counts[r]:,})" for r in regiones_disp]

        reg_map = dict(zip(regiones_labels, regiones_disp))

        reg_sel_labels = st.segmented_control(
            "Región",
            options=regiones_labels,
            selection_mode="multi",
            default=regiones_labels,
        )

        reg_sel = [reg_map[r] for r in reg_sel_labels] if reg_sel_labels else []
    else:
        reg_sel = []

    # =========================
    # SCORE
    # =========================
    st.markdown("**📊 Churn Score**")
    score_range = st.slider(
        "Rango",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.05,
    )

    # =========================
    # CUADRANTE
    # =========================
    st.markdown("**🎯 Estrategia**")
    cuads_disp = [
        "Máxima Prioridad",
        "Proteger Relación",
        "Intervención Ligera",
        "Monitoreo Pasivo"
    ]

    cuad_sel = st.segmented_control(
        "Cuadrante",
        options=cuads_disp,
        selection_mode="multi",
        default=cuads_disp,
    )

    # =========================
    # LTV
    # =========================
    st.markdown("**💰 Valor del Donante**")
    ltv_min = st.number_input(
        "LTV mínimo ($)",
        min_value=0, value=0, step=100,
    )

    st.divider()

    # =========================
    # FILTROS
    # =========================
    mask = pd.Series([True] * len(df_fe))

    if seg_sel:
        mask &= df_fe["segmento"].isin(seg_sel)

    if reg_sel and "region" in df_fe.columns:
        mask &= df_fe["region"].isin(reg_sel)

    mask &= df_fe["churn_score"].between(score_range[0], score_range[1])
    mask &= df_fe["cuadrante"].isin(cuad_sel)
    mask &= df_fe["ltv_estimado"] >= ltv_min

    df_filtered = df_fe[mask].copy()
    st.session_state["df_filtered"] = df_filtered

    # =========================
    # KPI PRO
    # =========================
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Donantes Seleccionados</div>
        <div class="kpi-value">{mask.sum():,} / {len(df_fe):,}</div>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # EXPORTAR
    # =========================
    if st.button("⬇️ Exportar selección", use_container_width=True):
        excel_bytes = to_excel_bytes(df_filtered, "Donantes_Filtrados")
        st.download_button(
            label="Descargar Excel",
            data=excel_bytes,
            file_name="donantes_filtrados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.divider()
    st.caption("Explora los análisis desde el menú lateral.")

# ══════════════════════════════════════════════════════════════════════════════
# CONTENIDO PRINCIPAL — HOME
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔄 Churn Intelligence Dashboard")
st.caption("Análisis predictivo de abandono de donantes · Actualizado con datos del archivo cargado   ·    (CAS)")
st.caption("MRR: Monthly Recurring Revenue (Ingresos Recurrentes Mensuales)")
st.divider()

kpis = resumen_rapido(df_filtered)

# ── Fila 1: KPIs principales ──────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Donantes totales",    f"{kpis['total_clientes']:,}")
c2.metric("Churners reales",     f"{kpis['churners']:,}")
c3.metric("Tasa de churn",       f"{kpis['tasa_churn']:.1%}")
c4.metric("MRR total",           f"${kpis['mrr_total']:,.0f}")
c5.metric("MRR en riesgo",       f"${kpis['mrr_en_riesgo']:,.0f}",
          delta=f"-{kpis['mrr_en_riesgo']/kpis['mrr_total']*100:.1f}% del MRR" if kpis['mrr_total'] > 0 else "",
          delta_color="inverse")

st.divider()

# ── Fila 2: Distribución de cuadrantes + Churn por segmento ──────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribución de cuadrantes estratégicos")
    quad_counts = df_filtered["cuadrante"].value_counts().reset_index()
    quad_counts.columns = ["Cuadrante", "Donantes"]
    #quad_colors_list = [
    #    {"Máxima Prioridad": COLORS["coral"],
    #     "Proteger Relación": COLORS["teal"],
    #     "Intervención Ligera": COLORS["amber"],
    #     "Monitoreo Pasivo": COLORS["gray"]}.get(q, COLORS["gray"])
    #    for q in quad_counts["Cuadrante"]
    #]
    # 🎨 Colores personalizados por cuadrante
    quad_colors_list = [
        {
            "Máxima Prioridad":    "#B60D07",  # 🔴 Rojo (nuevo)
            "Proteger Relación":   "#1490F5",  # 🔵 (reemplaza verde)
           "Intervención Ligera": "#FFAB3D",  # 🟠 Naranja
            "Monitoreo Pasivo":    "#FAF20F",  # 🟡 Amarillo
        }.get(q, "#BDC3C7")  # Gris fallback
        for q in quad_counts["Cuadrante"]
    ]
    
    fig_quad = px.bar(
        quad_counts, x="Cuadrante", y="Donantes",
        color="Cuadrante",
        color_discrete_sequence=quad_colors_list,
        text="Donantes",
    )
    fig_quad.update_traces(textposition="outside")
    fig_quad.update_layout(
        showlegend=False, height=340,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20),
        xaxis_title="", yaxis_title="Nro. de donantes",
    )
    st.plotly_chart(fig_quad, use_container_width=True)

with col_b:
    st.subheader("Churn real por segmento")
    if "segmento" in df_filtered.columns:
        seg_stats = (
            df_filtered.groupby("segmento")
            .agg(tasa_churn=("churn", "mean"), n=("churn", "count"))
            .reset_index()
        )
        seg_stats["tasa_pct"] = seg_stats["tasa_churn"] * 100
        fig_seg = px.bar(
            seg_stats, x="segmento", y="tasa_pct",
            color="segmento",
            color_discrete_map={
                "Enterprise": COLORS["teal"],
                "SMB":        COLORS["blue"],
                "Startup":    COLORS["amber"],
                "Individual": COLORS["coral"],
            },
            text=seg_stats["tasa_pct"].map(lambda x: f"{x:.1f}%"),
        )
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(
            showlegend=False, height=340,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20),
            xaxis_title="", yaxis_title="Tasa de churn (%)",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

# ── Fila 3: Scatter LTV × Riesgo + Distribución de score ─────────────────────
col_c, col_d = st.columns(2)




with col_c:
    st.subheader("Matriz LTV × Riesgo")
    fig_scatter = px.scatter(
        df_filtered,
        x="ltv_estimado", y="churn_score",
        color="cuadrante",
        color_discrete_map={
            "Máxima Prioridad":    "#B60D07",  # 🔴 Rojo (nuevo)
            "Proteger Relación":   "#1490F5",  # 🔵 (reemplaza verde)
            "Intervención Ligera": "#FFAB3D",  # 🟠 Naranja
            "Monitoreo Pasivo":    "#FAF20F",  # 🟡 Amarillo
        },
        hover_data=["cliente_id", "segmento", "mrr"] if "cliente_id" in df_filtered.columns else None,
        opacity=0.6, size_max=6,
    )
    fig_scatter.add_hline(y=meta["risk_threshold"], line_dash="dash",
                           line_color=COLORS["gray"], opacity=0.6)
    fig_scatter.add_vline(x=meta["ltv_median"], line_dash="dash",
                           line_color=COLORS["gray"], opacity=0.6)
    fig_scatter.update_layout(
        height=340, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20),
        xaxis_title="LTV estimado ($)", yaxis_title="Churn Score",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_d:
    st.subheader("Distribución del Churn Score")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_filtered.loc[df_filtered["churn"] == 0, "churn_score"],
        name="No churn", marker_color=COLORS["teal"],
        opacity=0.7, nbinsx=30,
    ))
    fig_hist.add_trace(go.Histogram(
        x=df_filtered.loc[df_filtered["churn"] == 1, "churn_score"],
        name="Churn", marker_color=COLORS["coral"],
        opacity=0.7, nbinsx=30,
    ))
    fig_hist.update_layout(
        barmode="overlay", height=340,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20),
        xaxis_title="Churn Score", yaxis_title="Donantes",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Fila 4: Top Doantes en riesgo ────────────────────────────────────────────
st.divider()
st.subheader("🚨 Top 20 Donantes en máxima prioridad")

top_cols = [c for c in [
    "cliente_id", "segmento", "region",
    "churn_score", "ltv_estimado", "mrr",
    "cuadrante", "dias_renovacion", "delta_nps", "friccion_score",
] if c in df_filtered.columns]

top20 = (
    df_filtered[df_filtered["cuadrante"] == "Máxima Prioridad"]
    [top_cols]
    .sort_values("churn_score", ascending=False)
    .head(20)
)

if len(top20) > 0:
    st.dataframe(
        top20.style.background_gradient(
            subset=["churn_score"], cmap="RdYlGn_r"
        ).format({
            "churn_score":  "{:.1%}",
            "ltv_estimado": "${:,.0f}",
            "mrr":          "${:,.0f}",
        }),
        use_container_width=True, height=380,
    )
    excel_top = to_excel_bytes(top20, "Top_Riesgo")
    st.download_button(
        "⬇️ Exportar tabla (Excel)",
        data=excel_top,
        file_name="top20_maxima_prioridad.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("No hay donantes en 'Máxima Prioridad' con los filtros actuales.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Churn Intelligence Dashboard · "
    "Modelos: GradientBoosting · RandomForest · LogisticRegression · "
    "EnsembleCalibrado · TwoStage · SurvivalAnalysis · UpliftModel"
)




# =================================================
# EJECUTARLO EN BASH:
# =================================================
# OJO, EL PYTHON 3.11.11 INSTALADO EN ESTE SPYDER, ESTÁ UBICADO AQUÍ:
# "C:\Users\cesar\AppData\Local\spyder-6\envs\spyder-runtime\python.exe"
    
# ENTONCES LA EJECUCIÓN DE ESTE STREAMLIT, SE DEBE HACER DE LA SIGUIENTE MANERA:
    
# cd C:\Users\cesar\OneDrive\Escritorio\UNICEF\CHURN STREAMLIT\SEGUNDA PRUEBA
# "C:\Users\cesar\AppData\Local\spyder-6\envs\spyder-runtime\python.exe" -m streamlit run Home.py



