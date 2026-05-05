# pages/2_Modelos.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from my_utils.helpers import COLORS, to_excel_bytes
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve

st.set_page_config(page_title="Modelos", page_icon="🤖", layout="wide")
st.title("🤖 Modelos Predictivos")
st.caption("Comparativa de todos los modelos entrenados")

if "results" not in st.session_state:
    st.warning("Vuelve a la página **Home** para cargar los datos primero.")
    st.stop()

results = st.session_state["results"]
meta    = st.session_state["meta"]
df      = st.session_state.get("df_filtered", st.session_state["df_fe"])

MODEL_COLORS = {
    "GradientBoosting":   COLORS["teal"],
    "RandomForest":       COLORS["blue"],
    "LogisticRegression": COLORS["gray"],
    "TwoStage":           COLORS["purple"],
    "EnsembleCalibrado":  COLORS["coral"],
    "SurvivalAnalysis":   COLORS["amber"],
}

SCORED_MODELS = [k for k in results if k not in ("shap_df", "meta", "UpliftModel")
                  and "probs" in results.get(k, {})]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Comparativa", "📈 Curvas ROC / PR",
    "🎯 Lift & Deciles", "🔢 Calibración", "📋 Detalle por modelo"
])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Métricas comparativas")
    rows = []
    for mname in SCORED_MODELS:
        m = results[mname]
        rows.append({
            "Modelo":     mname,
            "AUC-ROC":   m["auc_roc"],
            "AUC-PR":    m["auc_pr"],
            "Brier":     m.get("brier", "-"),  # 👈 aquí nace el problema
        })

    df_metrics = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)

    # 👇 🔥 AGREGA ESTO AQUÍ
    cols = ["AUC-ROC", "AUC-PR", "Brier"]
    df_metrics[cols] = df_metrics[cols].apply(pd.to_numeric, errors="coerce")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            df_metrics.style.background_gradient(
                subset=["AUC-ROC", "AUC-PR"], cmap="Greens"
            ).format({
                "AUC-ROC": "{:.4f}",
                "AUC-PR": "{:.4f}",
                "Brier": "{:.4f}"
            }),
            width="stretch", height=280,   # 👈 actualizado
        )

    with col2:
        fig = go.Figure()
        for _, row in df_metrics.iterrows():
            fig.add_trace(go.Bar(
                name=row["Modelo"],
                x=["AUC-ROC", "AUC-PR"],
                y=[row["AUC-ROC"], row["AUC-PR"]],
                marker_color=MODEL_COLORS.get(row["Modelo"], COLORS["gray"]),
            ))
        fig.update_layout(
            barmode="group", height=300,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20), yaxis_range=[0.4, 1.0],
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Mejor modelo
    best = df_metrics.iloc[0]
    st.success(
        f"✅ Mejor modelo: **{best['Modelo']}** · "
        f"AUC-ROC: {best['AUC-ROC']:.4f} · "
        f"AUC-PR: {best['AUC-PR']:.4f}"
    )
st.caption("AUC-PR: Área bajo la curva Precision–Recall, usado para evaluar qué tan bien un modelo identifica la clase positiva - Churn")

st.caption("Lift: Métrica de negocio que responde a una pregunta muy práctica:  ¿Cuántas veces mejor se está identificando casos positivos (churn, fraude, etc.), que si eligiera al azar?")

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)
    y_test = meta["y_test"]

    with col1:
        st.subheader("Curvas ROC")
        fig_roc = go.Figure()
        for mname in SCORED_MODELS:
            probs = results[mname]["probs"]
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = results[mname]["auc_roc"]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{mname} ({auc:.3f})",
                line=dict(color=MODEL_COLORS.get(mname, COLORS["gray"]), width=2),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color=COLORS["gray"], width=1),
            showlegend=False,
        ))
        fig_roc.update_layout(
            xaxis_title="FPR", yaxis_title="TPR", height=380,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.45, font_size=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.subheader("Curvas Precision-Recall")
        fig_pr = go.Figure()
        for mname in SCORED_MODELS:
            probs = results[mname]["probs"]
            prec, rec, _ = precision_recall_curve(y_test, probs)
            apr = results[mname]["auc_pr"]
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                name=f"{mname} ({apr:.3f})",
                line=dict(color=MODEL_COLORS.get(mname, COLORS["gray"]), width=2),
            ))
        baseline = y_test.mean()
        fig_pr.add_hline(y=baseline, line_dash="dash",
                          line_color=COLORS["gray"], opacity=0.6,
                          annotation_text=f"Baseline ({baseline:.2f})")
        fig_pr.update_layout(
            xaxis_title="Recall", yaxis_title="Precision", height=380,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.45, font_size=10),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Lift por decil")
    fig_lift = go.Figure()
    for mname in SCORED_MODELS:
        probs   = results[mname]["probs"]
        dec_idx = np.argsort(probs)[::-1]
        lifts   = []
        for d in range(1, 11):
            top_n = int(len(probs) * d / 10)
            lift  = y_test[dec_idx[:top_n]].mean() / (y_test.mean() + 1e-9)
            lifts.append(lift)
        fig_lift.add_trace(go.Scatter(
            x=list(range(1, 11)), y=lifts, mode="lines+markers",
            name=mname,
            line=dict(color=MODEL_COLORS.get(mname, COLORS["gray"]), width=2),
            marker_size=6,
        ))
    fig_lift.add_hline(y=1, line_dash="dash", line_color=COLORS["gray"],
                        annotation_text="Baseline (lift=1)")
    fig_lift.update_layout(
        xaxis_title="Decil", yaxis_title="Lift",
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, font_size=10),
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    # Tabla lift
    st.subheader("Churn real vs predicho por decil (mejor modelo)")
    best_model_name = df_metrics.iloc[0]["Modelo"]
    best_probs = results[best_model_name]["probs"]
    df_test_decil = pd.DataFrame({
        "probs": best_probs, "churn": y_test
    })
    df_test_decil["decil"] = pd.qcut(
        df_test_decil["probs"], q=10, labels=False, duplicates="drop"
    ) + 1
    decil_table = (
        df_test_decil.groupby("decil")
        .agg(n=("churn", "count"),
             churn_real=("churn", "mean"),
             score_medio=("probs", "mean"))
        .reset_index()
    )
    decil_table["lift"] = decil_table["churn_real"] / (y_test.mean() + 1e-9)
    st.dataframe(
        decil_table.style.format({
            "churn_real":   "{:.1%}",
            "score_medio":  "{:.3f}",
            "lift":         "{:.2f}",
        }).background_gradient(subset=["lift"], cmap="RdYlGn"),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Calibración de probabilidades")
    fig_cal = go.Figure()
    for mname in ["GradientBoosting", "EnsembleCalibrado"]:
        if mname in results:
            probs = results[mname]["probs"]
            prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
            fig_cal.add_trace(go.Scatter(
                x=prob_pred, y=prob_true, mode="lines+markers",
                name=mname,
                line=dict(color=MODEL_COLORS.get(mname), width=2),
                marker_size=7,
            ))
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color=COLORS["gray"]),
        name="Perfectamente calibrado",
    ))
    fig_cal.update_layout(
        xaxis_title="Prob. predicha", yaxis_title="Prob. real observada",
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_cal, use_container_width=True)
    st.info("Un modelo bien calibrado sigue la línea diagonal. La calibración isotónica corrige la sobreconfianza del GradientBoosting.")

# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Matriz de confusión por modelo")
    model_sel = st.selectbox("Seleccionar modelo", options=SCORED_MODELS)
    umbral    = st.slider("Umbral de clasificación", 0.0, 1.0, 0.5, 0.05)

    probs_sel = results[model_sel]["probs"]
    y_pred    = (probs_sel >= umbral).astype(int)
    cm        = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm, text_auto=True,
        labels=dict(x="Predicho", y="Real"),
        x=["No Churn", "Churn"], y=["No Churn", "Churn"],
        color_continuous_scale=["#E1F5EE", "#1D9E75"],
        title=f"Matriz de Confusión — {model_sel} (umbral={umbral})",
    )
    fig_cm.update_layout(height=380, margin=dict(t=40, b=20),
                          coloraxis_showscale=False)
    st.plotly_chart(fig_cm, use_container_width=True)

    tn, fp, fn, tp = cm.ravel()
    c1, c2, c3, c4 = st.columns(4)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    c1.metric("Precisión",  f"{prec:.1%}")
    c2.metric("Recall",     f"{rec:.1%}")
    c3.metric("F1-Score",   f"{f1:.3f}")
    c4.metric("Especificidad", f"{tn/(tn+fp+1e-9):.1%}")
