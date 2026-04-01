# pages/3_Explicabilidad.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from my_utils.helpers import COLORS, to_excel_bytes
from my_utils.models  import counterfactual

st.set_page_config(page_title="Explicabilidad", page_icon="🔍", layout="wide")
st.title("🔍 Explicabilidad del Modelo")
st.caption("SHAP, importancias, dependency plots y contrafactuales")

if "results" not in st.session_state:
    st.warning("Vuelve a la página **Home** para cargar los datos primero.")
    st.stop()

results = st.session_state["results"]
meta    = st.session_state["meta"]
df      = st.session_state.get("df_filtered", st.session_state["df_fe"])

gb_model      = results["GradientBoosting"]["model"]
shap_df       = results["shap_df"]
feature_names = meta["feature_names"]
X_train       = meta["X_train"]
X_test        = meta["X_test"]
y_test        = meta["y_test"]

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Importancia global", "🔎 Explicación local",
    "📉 Dependency plots", "🔄 Contrafactuales"
])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Importancia global de features (Permutation Importance)")
    top_n = st.slider("Número de features a mostrar", 5, len(shap_df), 15)
    top_shap = shap_df.head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_shap["importance"],
        y=top_shap["feature"],
        orientation="h",
        error_x=dict(type="data", array=top_shap["std"]),
        marker_color=[COLORS["coral"] if v > 0 else COLORS["teal"]
                      for v in top_shap["importance"]],
    ))
    fig.update_layout(
        xaxis_title="Impacto en AUC-PR", yaxis_title="",
        height=max(300, top_n * 28),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20, l=180),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature importance nativa (GradientBoosting)")
        gb_imp = pd.DataFrame({
            "feature":    feature_names,
            "importance": gb_model.feature_importances_,
        }).nlargest(top_n, "importance")
        fig2 = px.bar(
            gb_imp, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=["#E1F5EE", "#1D9E75"],
        )
        fig2.update_layout(
            height=max(280, top_n * 25), plot_bgcolor="white",
            paper_bgcolor="white", margin=dict(t=10, b=10, l=160),
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.dataframe(shap_df.head(top_n).style.format({
            "importance": "{:.4f}", "std": "{:.4f}"
        }).background_gradient(subset=["importance"], cmap="Greens"),
        use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Explicación local — contribución por cliente")

    gb_probs_test = meta["gb_probs_test"]
    idx_alto  = np.where(gb_probs_test > 0.65)[0]
    idx_medio = np.where((gb_probs_test > 0.35) & (gb_probs_test <= 0.65))[0]
    idx_bajo  = np.where(gb_probs_test < 0.20)[0]

    perfil_sel = st.radio(
        "Perfil de cliente",
        ["Alto riesgo (>65%)", "Riesgo medio (35-65%)", "Bajo riesgo (<20%)"],
        horizontal=True,
    )
    if "Alto" in perfil_sel:
        pool = idx_alto
    elif "medio" in perfil_sel:
        pool = idx_medio
    else:
        pool = idx_bajo

    if len(pool) == 0:
        st.warning("No hay clientes en este rango con los filtros actuales.")
    else:
        idx_c = st.selectbox(
            f"Seleccionar cliente del grupo ({len(pool)} disponibles)",
            options=range(min(len(pool), 50)),
            format_func=lambda i: f"Cliente #{pool[i]} · Score: {gb_probs_test[pool[i]]:.1%}",
        )
        i_sel = pool[idx_c]
        X_i   = X_test[i_sel]

        # Contribuciones por perturbación individual
        pred_base = gb_model.predict_proba(X_i.reshape(1, -1))[0, 1]
        contribs  = []
        for j in range(len(feature_names)):
            X_p = X_i.copy()
            X_p[j] = X_train[:, j].mean()
            pred_p = gb_model.predict_proba(X_p.reshape(1, -1))[0, 1]
            contribs.append(pred_base - pred_p)

        contrib_df = pd.DataFrame({
            "feature":      feature_names,
            "contribution": contribs,
            "value":        X_i,
        }).reindex(shap_df.index).dropna()
        contrib_df = contrib_df.sort_values("contribution", key=abs, ascending=False).head(12)

        st.metric("Probabilidad de churn del cliente seleccionado", f"{pred_base:.1%}")

        fig_local = go.Figure(go.Bar(
            x=contrib_df["contribution"],
            y=contrib_df["feature"],
            orientation="h",
            marker_color=[COLORS["coral"] if v > 0 else COLORS["teal"]
                          for v in contrib_df["contribution"]],
            text=[f"val={v:.2f}" for v in contrib_df["value"]],
            textposition="outside",
        ))
        fig_local.add_vline(x=0, line_color=COLORS["gray"], line_width=1)
        fig_local.update_layout(
            xaxis_title="Contribución al score", height=380,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20, l=180),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_local, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Dependency Plot")
    feat_dep = st.selectbox(
        "Seleccionar feature",
        options=shap_df["feature"].head(12).tolist(),
        index=0,
    )

    if feat_dep in feature_names:
        fidx  = feature_names.index(feat_dep)
        fvals = X_test[:, fidx]
        preds = gb_probs_test

        # Binning por percentiles
        pct_bins = np.percentile(fvals, np.linspace(0, 100, 21))
        centers, means, stds = [], [], []
        for i in range(len(pct_bins) - 1):
            mask = (fvals >= pct_bins[i]) & (fvals < pct_bins[i + 1])
            if mask.sum() > 2:
                centers.append((pct_bins[i] + pct_bins[i + 1]) / 2)
                means.append(preds[mask].mean())
                stds.append(preds[mask].std())

        fig_dep = go.Figure()
        fig_dep.add_trace(go.Scatter(
            x=centers, y=[m + s for m, s in zip(means, stds)],
            fill=None, mode="lines", line_color=COLORS["coral"],
            line_width=0, showlegend=False,
        ))
        fig_dep.add_trace(go.Scatter(
            x=centers, y=[m - s for m, s in zip(means, stds)],
            fill="tonexty", mode="lines", line_color=COLORS["coral"],
            line_width=0, fillcolor=COLORS["coral_lt"],
            name="±1 std",
        ))
        fig_dep.add_trace(go.Scatter(
            x=centers, y=means, mode="lines+markers",
            line=dict(color=COLORS["coral"], width=2.5),
            marker_size=7, name=f"P(churn) medio",
        ))
        fig_dep.update_layout(
            xaxis_title=feat_dep.replace("_", " "),
            yaxis_title="P(Churn) promedio",
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_dep, use_container_width=True)
        st.caption(f"El gráfico muestra cómo cambia la probabilidad media de churn al variar **{feat_dep}**, promediando el efecto del resto de features.")

# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Contrafactual — ¿Qué debe cambiar para reducir el riesgo?")

    gb_probs_test = meta["gb_probs_test"]
    idx_high = np.where(gb_probs_test > 0.55)[0]

    if len(idx_high) == 0:
        st.info("No hay clientes de alto riesgo en el conjunto de test.")
    else:
        target_prob = st.slider(
            "Probabilidad objetivo (reducir hasta)",
            0.10, 0.50, 0.30, 0.05
        )

        idx_cf = st.selectbox(
            "Seleccionar cliente de alto riesgo",
            options=range(min(len(idx_high), 50)),
            format_func=lambda i: f"Cliente #{idx_high[i]} · Score actual: {gb_probs_test[idx_high[i]]:.1%}",
        )

        i_cf = idx_high[idx_cf]

        with st.spinner("Calculando contrafactual..."):
            _, changes = counterfactual(
                gb_model, X_test[i_cf], X_train,
                feature_names, target_prob=target_prob,
            )

        prob_orig = gb_probs_test[i_cf]

        st.metric(
            "Probabilidad actual",
            f"{prob_orig:.1%}",
            delta=f"Objetivo: {target_prob:.0%}",
            delta_color="inverse"
        )

        if changes:
            df_cf = pd.DataFrame(changes)

            # 🔥 LIMPIEZA CRÍTICA (evita errores)
            for col in ["valor_actual", "valor_recomendado", "cambio"]:
                if col in df_cf.columns:
                    df_cf[col] = pd.to_numeric(df_cf[col], errors="coerce")

            # =========================
            # GRÁFICO
            # =========================
            fig_cf = go.Figure()

            fig_cf.add_trace(go.Bar(
                name="Valor actual",
                x=df_cf["feature"],
                y=df_cf["valor_actual"],
                marker_color=COLORS["coral"],
                opacity=0.85,
            ))

            fig_cf.add_trace(go.Bar(
                name="Valor recomendado",
                x=df_cf["feature"],
                y=df_cf["valor_recomendado"],
                marker_color=COLORS["teal"],
                opacity=0.85,
            ))

            fig_cf.update_layout(
                barmode="group",
                height=360,
                xaxis_title="",
                yaxis_title="Valor de la feature",
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(t=20, b=60),
            )

            st.plotly_chart(fig_cf, use_container_width=True)

            # =========================
            # TABLA PRO (SIN STYLER BUGS)
            # =========================
            st.subheader("Cambios recomendados")

            # 👉 versión estable sin .style
            df_cf_display = df_cf.copy()

            df_cf_display["cambio_fmt"] = df_cf_display["cambio"].apply(
                lambda x: f"🔻 {x:.2f}" if pd.notna(x) and x < 0
                else f"🔺 {x:.2f}" if pd.notna(x)
                else "-"
            )

            df_cf_display["valor_actual"] = df_cf_display["valor_actual"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "-"
            )

            df_cf_display["valor_recomendado"] = df_cf_display["valor_recomendado"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "-"
            )

            st.dataframe(
                df_cf_display[[
                    "feature",
                    "valor_actual",
                    "valor_recomendado",
                    "cambio_fmt"
                ]],
                use_container_width=True,
            )

        else:
            st.success(
                "El cliente ya está por debajo del umbral objetivo o no se encontraron cambios necesarios."
            )