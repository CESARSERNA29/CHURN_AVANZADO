# pages/5_Survival_Uplift.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from my_utils.helpers import COLORS, kaplan_meier, to_excel_bytes

st.set_page_config(page_title="Survival & Uplift", page_icon="📈", layout="wide")
st.title("📈 Survival Analysis & Uplift Model")
st.caption("Kaplan-Meier, Cox PH aproximado y modelo de intervención T-Learner")

st.caption("Uplift Model (Modelo de incremento o modelo de tratamiento causal): Modelo de ML que responder: ¿Cuál es el efecto real de una acción (tratamiento) sobre un individuo?")      
st.caption("Uplift no solo predice qué va a pasar, sino qué cambiaría si haces algo vs si no lo haces.")


if "df_fe" not in st.session_state:
    st.warning("Vuelve a la página **Home** para cargar los datos primero.")
    st.stop()

df      = st.session_state.get("df_filtered", st.session_state["df_fe"])
results = st.session_state["results"]
meta    = st.session_state["meta"]

tab1, tab2, tab3 = st.tabs([
    "⏱️ Kaplan-Meier", "🎯 Uplift Model", "📋 Tablas"
])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Curvas de supervivencia Kaplan-Meier")
    st.caption("Probabilidad de que un cliente siga activo en función del tiempo")

    col1, col2 = st.columns([2, 1])
    with col1:
        split_by = st.selectbox(
            "Segmentar por",
            options=["Global"] + [c for c in ["segmento", "region", "cuadrante"]
                                   if c in df.columns],
        )

        fig_km = go.Figure()
        seg_colors = [COLORS["teal"], COLORS["blue"], COLORS["amber"],
                      COLORS["coral"], COLORS["purple"], COLORS["gray"]]

        if split_by == "Global":
            t, s = kaplan_meier(
                df["tiempo_al_evento"].values, df["churn"].values
            )
            fig_km.add_trace(go.Scatter(
                x=t, y=s, mode="lines", name="Global",
                line=dict(color=COLORS["teal"], width=2.5),
            ))
        else:
            groups = df[split_by].dropna().unique()
            for i, grp in enumerate(sorted(groups)):
                sub = df[df[split_by] == grp]
                if len(sub) < 10:
                    continue
                t, s = kaplan_meier(
                    sub["tiempo_al_evento"].values, sub["churn"].values
                )
                fig_km.add_trace(go.Scatter(
                    x=t, y=s, mode="lines", name=str(grp),
                    line=dict(color=seg_colors[i % len(seg_colors)], width=2),
                ))

        fig_km.add_hline(y=0.5, line_dash="dot", line_color=COLORS["gray"],
                          opacity=0.7, annotation_text="S(t)=0.5")
        fig_km.update_layout(
            xaxis_title="Días desde inicio", yaxis_title="Probabilidad de supervivencia",
            yaxis_range=[0, 1.05], height=420,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_km, use_container_width=True)

    with col2:
        st.subheader("Supervivencia a horizontes clave")
        t_global, s_global = kaplan_meier(
            df["tiempo_al_evento"].values, df["churn"].values
        )
        for days in [30, 60, 90, 180, 365]:
            idx_d = np.searchsorted(t_global, days)
            if idx_d < len(s_global):
                surv = s_global[idx_d]
                st.metric(f"Días {days}", f"{surv:.1%}")

        st.divider()
        st.caption(
            "La curva KM estima la probabilidad de que un cliente siga activo "
            "a cada punto del tiempo, incorporando correctamente la censura "
            "(clientes activos al final del periodo de observación)."
        )

    # Log-rank test aproximado entre segmentos
    if "segmento" in df.columns:
        st.subheader("Supervivencia media por segmento a 90 días")
        seg_surv = []
        for seg in df["segmento"].dropna().unique():
            sub = df[df["segmento"] == seg]
            t_s, s_s = kaplan_meier(sub["tiempo_al_evento"].values, sub["churn"].values)
            idx_90 = np.searchsorted(t_s, 90)
            s_90   = s_s[idx_90] if idx_90 < len(s_s) else s_s[-1]
            seg_surv.append({"Segmento": seg, "Supervivencia 90d": s_90, "n": len(sub)})
        st.dataframe(
            pd.DataFrame(seg_surv).style.format({"Supervivencia 90d": "{:.1%}"}),
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Uplift Model — Efecto incremental de la intervención")
    st.caption(
        "El uplift score mide cuánto reduce la probabilidad de churn "
        "una intervención de retención. Positivo = la intervención ayuda."
    )

    uplift_data = results.get("UpliftModel", {})
    if not uplift_data:
        st.info("Modelo de uplift no disponible.")
    else:
        uplift_score = uplift_data["uplift_score"]
        p_ctrl       = uplift_data["p_control"]
        p_treat      = uplift_data["p_treated"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Uplift promedio",         f"{uplift_score.mean():.4f}")
        col2.metric("Clientes persuadibles (>5%)",
                    f"{(uplift_score > 0.05).sum():,}")
        col3.metric("Clientes donde no intervenir (≤0)",
                    f"{(uplift_score <= 0).sum():,}")

        # Histograma uplift
        fig_up = go.Figure()
        fig_up.add_trace(go.Histogram(
            x=uplift_score[uplift_score > 0],
            name="Positivo (intervención ayuda)",
            marker_color=COLORS["teal"], opacity=0.75, nbinsx=25,
        ))
        fig_up.add_trace(go.Histogram(
            x=uplift_score[uplift_score <= 0],
            name="Nulo/negativo",
            marker_color=COLORS["coral"], opacity=0.75, nbinsx=25,
        ))
        fig_up.add_vline(x=0.05, line_dash="dot", line_color=COLORS["amber"],
                          annotation_text="Umbral acción (0.05)")
        fig_up.update_layout(
            barmode="overlay", height=360,
            xaxis_title="Uplift Score", yaxis_title="Clientes",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_up, use_container_width=True)

        # Curva de ganancia de uplift
        st.subheader("Curva de ganancia acumulada del uplift")
        sorted_idx = np.argsort(uplift_score)[::-1]
        cum_uplift  = np.cumsum(uplift_score[sorted_idx])
        cum_random  = np.linspace(0, uplift_score.sum(), len(uplift_score))
        pct_pop     = np.linspace(0, 100, len(uplift_score))

        fig_gain = go.Figure()
        fig_gain.add_trace(go.Scatter(
            x=pct_pop, y=cum_uplift, mode="lines",
            name="Modelo", line=dict(color=COLORS["teal"], width=2.5),
        ))
        fig_gain.add_trace(go.Scatter(
            x=pct_pop, y=cum_random, mode="lines",
            name="Aleatorio", line=dict(dash="dash", color=COLORS["gray"], width=1.5),
        ))
        fig_gain.update_layout(
            xaxis_title="% Clientes contactados",
            yaxis_title="Uplift acumulado",
            height=360, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_gain, use_container_width=True)

        st.info(
            "**Interpretación:** Si contactas al 30% de los clientes ordenados por "
            "uplift score descendente, capturarás la mayor parte del efecto incremental "
            "de la campaña de retención, optimizando el ROI de la intervención."
        )

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Tablas de clientes: persuadibles vs no persuadibles")

    if "UpliftModel" in results:
        uplift_score_all = results["UpliftModel"]["uplift_score"]
        idx_test         = meta["idx_test"]
        df_test          = st.session_state["df_fe"].iloc[idx_test].copy()
        df_test["uplift_score"]   = uplift_score_all
        df_test["p_churn_control"] = results["UpliftModel"]["p_control"]
        df_test["p_churn_tratado"] = results["UpliftModel"]["p_treated"]
        df_test["tipo_cliente"]    = np.where(
            df_test["uplift_score"] > 0.05, "Persuadible",
            np.where(df_test["uplift_score"] > 0, "Poco persuadible", "No intervenir")
        )

        tipo_filter = st.selectbox(
            "Tipo de cliente",
            ["Todos", "Persuadible", "Poco persuadible", "No intervenir"],
        )
        if tipo_filter != "Todos":
            df_test = df_test[df_test["tipo_cliente"] == tipo_filter]

        show_cols = [c for c in [
            "cliente_id", "segmento", "region",
            "uplift_score", "p_churn_control", "p_churn_tratado",
            "tipo_cliente", "ltv_estimado", "mrr",
        ] if c in df_test.columns]

        st.dataframe(
            df_test[show_cols].sort_values("uplift_score", ascending=False)
            .reset_index(drop=True)
            .style.format({
                "uplift_score":      "{:.4f}",
                "p_churn_control":   "{:.1%}",
                "p_churn_tratado":   "{:.1%}",
                "ltv_estimado":      "${:,.0f}",
                "mrr":               "${:,.0f}",
            }).background_gradient(subset=["uplift_score"], cmap="RdYlGn"),
            use_container_width=True, height=420,
        )

        excel_up = to_excel_bytes(df_test[show_cols], "Uplift_Clientes")
        st.download_button(
            "⬇️ Exportar tabla uplift (Excel)", excel_up,
            "uplift_clientes.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
