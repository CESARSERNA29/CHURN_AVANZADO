# pages/6_Monitoreo.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from my_utils.helpers import COLORS, calculate_psi, psi_label, to_excel_bytes
from my_utils.models  import champion_challenger

st.set_page_config(page_title="Monitoreo", page_icon="📡", layout="wide")
st.title("📡 Monitoreo en Producción")
st.caption("PSI, Feature Drift, Champion-Challenger y alertas de degradación")

if "results" not in st.session_state:
    st.warning("Vuelve a la página **Home** para cargar los datos primero.")
    st.stop()

results = st.session_state["results"]
meta    = st.session_state["meta"]
df_fe   = st.session_state["df_fe"]

X_train    = meta["X_train"]
X_test     = meta["X_test"]
y_test     = meta["y_test"]
feat_names = meta["feature_names"]
gb_model   = results["GradientBoosting"]["model"]
rf_model   = results["RandomForest"]["model"]

# Simular 3 periodos de producción con drift progresivo
@st.cache_data
def simulate_prod_data(seed=42):
    rng = np.random.default_rng(seed)
    def make_prod(drift):
        idx = rng.integers(0, len(X_test), 400)
        Xp  = X_test[idx].copy().astype(float)
        if drift > 0:
            for j in range(5):
                Xp[:, j] += drift * X_test[:, j].std() * rng.normal(0, 1, 400)
        return Xp
    return {
        "Mes 1 (sin drift)":   make_prod(0.0),
        "Mes 2 (drift leve)":  make_prod(0.3),
        "Mes 3 (drift severo)": make_prod(0.8),
    }

prod_data = simulate_prod_data()
score_train = gb_model.predict_proba(X_train)[:, 1]

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 PSI", "🌊 Feature Drift",
    "🏆 Champion-Challenger", "🔔 Dashboard de alertas"
])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Population Stability Index (PSI)")
    st.caption("Detecta si la distribución del score del modelo se aleja del entrenamiento. PSI < 0.10 = estable · 0.10-0.20 = investigar · > 0.20 = reentrenar")

    psi_rows = []
    for periodo, X_prod in prod_data.items():
        score_prod = gb_model.predict_proba(X_prod)[:, 1]
        psi_val    = calculate_psi(score_train, score_prod)
        emoji, txt, color = psi_label(psi_val)
        psi_rows.append({
            "Periodo": periodo, "PSI": psi_val,
            "Estado": f"{emoji} {txt}", "color": color
        })

    col1, col2 = st.columns([1, 2])
    with col1:
        for row in psi_rows:
            st.markdown(
                f"""<div style="padding:12px 16px;margin-bottom:8px;border-radius:8px;
                    border-left:4px solid {row['color']};background:#FAFAF8">
                    <b>{row['Periodo']}</b><br>
                    PSI: <b>{row['PSI']:.4f}</b> · {row['Estado']}
                </div>""", unsafe_allow_html=True
            )

    with col2:
        fig_psi = go.Figure()
        fig_psi.add_trace(go.Bar(
            x=[r["Periodo"].split("(")[0].strip() for r in psi_rows],
            y=[r["PSI"] for r in psi_rows],
            marker_color=[r["color"] for r in psi_rows],
            text=[f"{r['PSI']:.4f}" for r in psi_rows],
            textposition="outside",
        ))
        fig_psi.add_hline(y=0.10, line_dash="dash", line_color=COLORS["amber"],
                           annotation_text="Umbral leve (0.10)", opacity=0.7)
        fig_psi.add_hline(y=0.20, line_dash="dash", line_color=COLORS["red"],
                           annotation_text="Umbral crítico (0.20)", opacity=0.7)
        fig_psi.update_layout(
            height=340, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20), yaxis_title="PSI", xaxis_title="",
        )
        st.plotly_chart(fig_psi, use_container_width=True)

    # Distribuciones comparativas
    st.subheader("Distribución del score: entrenamiento vs producción")
    periodo_sel = st.selectbox("Comparar contra", list(prod_data.keys()))
    X_sel   = prod_data[periodo_sel]
    sc_sel  = gb_model.predict_proba(X_sel)[:, 1]

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=score_train, name="Entrenamiento (ref.)",
        marker_color=COLORS["teal"], opacity=0.65, nbinsx=30,
        histnorm="probability density",
    ))
    fig_dist.add_trace(go.Histogram(
        x=sc_sel, name=periodo_sel,
        marker_color=COLORS["coral"], opacity=0.65, nbinsx=30,
        histnorm="probability density",
    ))
    fig_dist.update_layout(
        barmode="overlay", height=340,
        xaxis_title="Churn Score", yaxis_title="Densidad",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Feature Drift — KS Test")
    st.caption("Detecta qué variables cambiaron su distribución entre entrenamiento y producción")

    from scipy.stats import ks_2samp
    periodo_drift = st.selectbox("Periodo a analizar", list(prod_data.keys()), key="drift_sel")
    X_drift = prod_data[periodo_drift]

    drift_rows = []
    for j, fname in enumerate(feat_names):
        ks_stat, ks_p = ks_2samp(X_train[:, j], X_drift[:, j])
        drift_rows.append({
            "Feature":  fname,
            "KS Stat":  round(ks_stat, 4),
            "p-value":  round(ks_p, 4),
            "Drift":    "⚠️ Sí" if ks_p < 0.05 else "✅ No",
        })

    drift_df = pd.DataFrame(drift_rows).sort_values("KS Stat", ascending=False)
    n_drifted = (drift_df["p-value"] < 0.05).sum()
    st.metric("Features con drift significativo", f"{n_drifted} / {len(feat_names)}")

    fig_drift = px.bar(
        drift_df.head(15), x="KS Stat", y="Feature", orientation="h",
        color="p-value",
        color_continuous_scale=["#D85A30", "#FAEEDA", "#E1F5EE"],
        color_continuous_midpoint=0.05,
        title=f"Top 15 Features por KS Statistic — {periodo_drift}",
    )
    fig_drift.add_vline(x=0.05, line_dash="dash", line_color=COLORS["amber"],
                         annotation_text="Umbral p=0.05")
    fig_drift.update_layout(
        height=400, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=20, l=180),
        yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    # Detalle de distribución por feature seleccionada
    feat_drill = st.selectbox(
        "Ver distribución de feature específica",
        options=drift_df[drift_df["p-value"] < 0.05]["Feature"].tolist() or feat_names,
    )
    fidx = feat_names.index(feat_drill)
    fig_fd = go.Figure()
    fig_fd.add_trace(go.Histogram(
        x=X_train[:, fidx], name="Entrenamiento",
        marker_color=COLORS["teal"], opacity=0.65,
        histnorm="probability density", nbinsx=25,
    ))
    fig_fd.add_trace(go.Histogram(
        x=X_drift[:, fidx], name=periodo_drift,
        marker_color=COLORS["coral"], opacity=0.65,
        histnorm="probability density", nbinsx=25,
    ))
    fig_fd.update_layout(
        barmode="overlay", height=300, xaxis_title=feat_drill,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_fd, use_container_width=True)

    excel_drift = to_excel_bytes(drift_df, "Feature_Drift")
    st.download_button(
        "⬇️ Exportar tabla drift (Excel)", excel_drift,
        "feature_drift.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Champion-Challenger")
    st.caption("Prueba estadística de si el Challenger supera al Champion actual")

    gb_probs = results["GradientBoosting"]["probs"]
    rf_probs = results["RandomForest"]["probs"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Champion: GradientBoosting**
        - AUC-ROC: `{results['GradientBoosting']['auc_roc']}`
        - AUC-PR:  `{results['GradientBoosting']['auc_pr']}`
        """)
    with col2:
        st.markdown(f"""
        **Challenger: RandomForest**
        - AUC-ROC: `{results['RandomForest']['auc_roc']}`
        - AUC-PR:  `{results['RandomForest']['auc_pr']}`
        """)

    n_boot = st.slider("Iteraciones bootstrap", 200, 2000, 1000, 100)
    with st.spinner("Ejecutando bootstrap..."):
        cc = champion_challenger(y_test, gb_probs, rf_probs, n_boot=n_boot)

    c1, c2, c3 = st.columns(3)
    c1.metric("Diferencia media (Champ-Chall)", f"{cc['mean_diff']:+.4f}")
    c2.metric("IC 95%", f"[{cc['ci_low']:.4f}, {cc['ci_high']:.4f}]")
    c3.metric(
        "p-value",
        f"{cc['p_value']:.4f}",
        delta="✅ No significativo" if not cc["significant"] else "⚠️ Significativo",
        delta_color="off",
    )

    if cc["significant"]:
        winner = "Champion" if cc["mean_diff"] > 0 else "Challenger"
        st.warning(f"⚠️ Diferencia significativa detectada. **{winner}** es superior. "
                    f"{'Mantener Champion.' if winner == 'Champion' else 'Considera promover el Challenger.'}")
    else:
        st.success("✅ No hay diferencia estadísticamente significativa. El Champion sigue siendo válido.")

    # Distribución bootstrap
    fig_boot = go.Figure()
    fig_boot.add_trace(go.Histogram(
        x=cc["diffs"], nbinsx=50,
        marker_color=COLORS["purple"], opacity=0.8, name="Distribución bootstrap",
    ))
    fig_boot.add_vline(x=0, line_dash="dash", line_color=COLORS["red"],
                        annotation_text="H₀: Sin diferencia")
    fig_boot.add_vline(x=cc["mean_diff"], line_color=COLORS["teal"], line_width=2,
                        annotation_text=f"Media: {cc['mean_diff']:+.4f}")
    fig_boot.add_vrect(x0=cc["ci_low"], x1=cc["ci_high"],
                        fillcolor=COLORS["purple"], opacity=0.1, line_width=0,
                        annotation_text="IC 95%")
    fig_boot.update_layout(
        height=360, xaxis_title="Diferencia AUC (Champion - Challenger)",
        yaxis_title="Frecuencia bootstrap",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20), showlegend=False,
    )
    st.plotly_chart(fig_boot, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔔 Dashboard de alertas de monitoreo")

    # Calcular todos los PSI
    alerts = []
    for periodo, X_prod in prod_data.items():
        sc_prod = gb_model.predict_proba(X_prod)[:, 1]
        psi_val = calculate_psi(score_train, sc_prod)
        emoji, txt, color = psi_label(psi_val)
        alerts.append({
            "Tipo":     "PSI",
            "Métrica":  f"Score — {periodo.split('(')[0].strip()}",
            "Valor":    f"{psi_val:.4f}",
            "Estado":   f"{emoji} {txt}",
            "Acción":   "Ninguna" if psi_val < 0.10 else
                        "Investigar origen del drift" if psi_val < 0.20 else
                        "REENTRENAR EL MODELO INMEDIATAMENTE",
            "color":    color,
        })

    # Feature drift (último mes)
    X_last = list(prod_data.values())[-1]
    from scipy.stats import ks_2samp
    drifted_feats = []
    for j, fname in enumerate(feat_names):
        _, ks_p = ks_2samp(X_train[:, j], X_last[:, j])
        if ks_p < 0.05:
            drifted_feats.append(fname)
    alerts.append({
        "Tipo":    "Feature Drift",
        "Métrica": f"{len(drifted_feats)} features con drift (último mes)",
        "Valor":   str(len(drifted_feats)),
        "Estado":  "⚠️ Drift detectado" if drifted_feats else "✅ Estable",
        "Acción":  f"Revisar: {', '.join(drifted_feats[:3])}{'...' if len(drifted_feats)>3 else ''}"
                    if drifted_feats else "Sin acción requerida",
        "color": COLORS["amber"] if drifted_feats else COLORS["teal"],
    })

    # Champion-Challenger
    alerts.append({
        "Tipo":    "Champion-Challenger",
        "Métrica": "GradientBoosting vs RandomForest",
        "Valor":   f"p={cc['p_value']:.4f}",
        "Estado":  "⚠️ Significativo" if cc["significant"] else "✅ No significativo",
        "Acción":  "Evaluar promoción del Challenger" if cc["significant"] else "Mantener Champion",
        "color":   COLORS["amber"] if cc["significant"] else COLORS["teal"],
    })

    for alert in alerts:
        st.markdown(f"""
        <div style="border-left:4px solid {alert['color']};padding:12px 16px;
                    margin-bottom:10px;border-radius:0 10px 10px 0;background:#FAFAF8;
                    border:0.5px solid #D3D1C7;border-left:4px solid {alert['color']}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
              <b style="font-size:13px">{alert['Tipo']} — {alert['Métrica']}</b><br>
              <span style="font-size:12px;color:#5F5E5A">Valor: {alert['Valor']} · {alert['Estado']}</span>
            </div>
          </div>
          <div style="margin-top:6px;font-size:11px;color:#5F5E5A;font-style:italic">
            📌 Acción recomendada: {alert['Acción']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    alerts_df = pd.DataFrame([{k: v for k, v in a.items() if k != "color"} for a in alerts])
    excel_alerts = to_excel_bytes(alerts_df, "Alertas_Monitoreo")
    st.download_button(
        "⬇️ Exportar reporte de alertas (Excel)", excel_alerts,
        "alertas_monitoreo.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
