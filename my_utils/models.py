# utils/models.py
# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento, predicción y evaluación de todos los modelos.
# Sin Streamlit. Devuelve diccionarios con resultados listos para visualizar.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    brier_score_loss, classification_report,
)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp

from my_utils.feature_engineering import FEATURE_COLS
from my_utils.helpers import calculate_psi


# ── Entrenamiento principal ────────────────────────────────────────────────────
def train_all_models(df: pd.DataFrame, models_dir: str = "models_saved") -> dict:
    """
    Entrena todos los modelos sobre el DataFrame con features ya calculadas.
    Guarda artefactos en models_dir.
    Devuelve un dict con modelos, métricas y predicciones.
    """
    os.makedirs(models_dir, exist_ok=True)

    X = df[FEATURE_COLS].fillna(0).values
    y = df["churn"].values
    feature_names = FEATURE_COLS

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)), test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Gradient Boosting ──────────────────────────────────────────────────────
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    results["GradientBoosting"] = _metrics(y_test, gb_probs, "GradientBoosting")
    results["GradientBoosting"]["model"] = gb
    joblib.dump(gb, os.path.join(models_dir, "gb_model.pkl"))

    # ── Random Forest ──────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    results["RandomForest"] = _metrics(y_test, rf_probs, "RandomForest")
    results["RandomForest"]["model"] = rf
    joblib.dump(rf, os.path.join(models_dir, "rf_model.pkl"))

    # ── Logistic Regression ────────────────────────────────────────────────────
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_probs = lr.predict_proba(X_test_sc)[:, 1]
    results["LogisticRegression"] = _metrics(y_test, lr_probs, "LogisticRegression")
    results["LogisticRegression"]["model"] = lr
    joblib.dump({"model": lr, "scaler": scaler},
                os.path.join(models_dir, "lr_model.pkl"))

    # ── Ensemble Calibrado ─────────────────────────────────────────────────────
    stack = 0.45 * gb_probs + 0.40 * rf_probs + 0.15 * lr_probs
    gb_cv = cross_val_predict(
        GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                    learning_rate=0.1, random_state=42),
        X_train, y_train, cv=5, method="predict_proba"
    )[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(gb_cv, y_train)
    ens_probs = iso.predict(stack)
    results["EnsembleCalibrado"] = _metrics(y_test, ens_probs, "EnsembleCalibrado")
    results["EnsembleCalibrado"]["model"] = iso
    joblib.dump(iso, os.path.join(models_dir, "ensemble_model.pkl"))

    # ── Two-Stage ─────────────────────────────────────────────────────────────
    hr_mask = rf_probs > 0.25
    ts_probs = rf_probs.copy()
    if hr_mask.sum() > 20:
        gb2 = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42
        )
        gb2.fit(X_train, y_train)
        ts_probs[hr_mask] = (
            0.4 * rf_probs[hr_mask] +
            0.6 * gb2.predict_proba(X_test[hr_mask])[:, 1]
        )
    results["TwoStage"] = _metrics(y_test, ts_probs, "TwoStage")

    # ── Survival (Cox aproximado) ──────────────────────────────────────────────
    time_train = df["tiempo_al_evento"].values[idx_train]
    cox = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    cox.fit(X_train, np.log1p(time_train))
    log_surv = cox.predict(X_test)
    surv_score = log_surv
    surv_score_norm = (surv_score - surv_score.min()) / (surv_score.max() - surv_score.min() + 1e-9)
    results["SurvivalAnalysis"] = {
        "model": cox, "log_survival": log_surv,
        "probs": 1 - surv_score_norm,
        "auc_roc": roc_auc_score(y_test, 1 - surv_score_norm),
        "auc_pr":  average_precision_score(y_test, 1 - surv_score_norm),
    }
    joblib.dump(cox, os.path.join(models_dir, "cox_model.pkl"))

    # ── Uplift T-Learner ───────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    treatment = rng.binomial(1, 0.5, len(df))
    X_t = X[treatment == 1];  y_t = y[treatment == 1]
    X_c = X[treatment == 0];  y_c = y[treatment == 0]
    mt = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    mc = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    mt.fit(X_t, y_t)
    mc.fit(X_c, y_c)
    p_ctrl    = mc.predict_proba(X_test)[:, 1]
    p_treat   = mt.predict_proba(X_test)[:, 1]
    uplift_sc = p_ctrl - p_treat
    results["UpliftModel"] = {
        "model_treated": mt, "model_control": mc,
        "uplift_score": uplift_sc,
        "p_control": p_ctrl, "p_treated": p_treat,
    }

    # ── Permutation Importance ─────────────────────────────────────────────────
    perm = permutation_importance(
        gb, X_test, y_test, n_repeats=20,
        random_state=42, scoring="average_precision"
    )
    shap_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    results["shap_df"] = shap_df

    # ── Score ensemble para TODOS los clientes ─────────────────────────────────
    gb_all  = gb.predict_proba(X)[:, 1]
    rf_all  = rf.predict_proba(X)[:, 1]
    lr_all  = lr.predict_proba(scaler.transform(X))[:, 1]
    ens_all = iso.predict(0.45 * gb_all + 0.40 * rf_all + 0.15 * lr_all)

    # ── Segmentación en cuadrantes ─────────────────────────────────────────────
    ltv_med = df["ltv_estimado"].median()
    risk_th  = 0.40

    cuadrante = np.where(
        (ens_all >= risk_th) & (df["ltv_estimado"].values >= ltv_med),
        "Máxima Prioridad",
        np.where(
            (ens_all >= risk_th) & (df["ltv_estimado"].values < ltv_med),
            "Intervención Ligera",
            np.where(
                (ens_all < risk_th) & (df["ltv_estimado"].values >= ltv_med),
                "Proteger Relación",
                "Monitoreo Pasivo"
            )
        )
    )

    # ── Decil de riesgo ────────────────────────────────────────────────────────
    decil = pd.qcut(ens_all, q=10, labels=False, duplicates="drop") + 1

    # ── Microsegmentos K-Means (alto riesgo) ───────────────────────────────────
    hr_idx = np.where(ens_all >= risk_th)[0]
    microseg = np.full(len(df), -1, dtype=int)
    if len(hr_idx) >= 4:
        km_feats = np.column_stack([
            ens_all[hr_idx],
            df["ltv_estimado"].values[hr_idx],
            df["engagement_score"].values[hr_idx],
            df["friccion_score"].values[hr_idx],
            df["delta_nps"].values[hr_idx],
        ])
        km_feats_sc = StandardScaler().fit_transform(km_feats)
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        microseg[hr_idx] = km.fit_predict(km_feats_sc)

    # ── Metadata del entrenamiento ─────────────────────────────────────────────
    results["meta"] = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "idx_train": idx_train, "idx_test": idx_test,
        "scaler": scaler, "feature_names": feature_names,
        "gb_probs_test": gb_probs, "rf_probs_test": rf_probs,
        "ens_all": ens_all, "cuadrante": cuadrante,
        "decil": decil, "microseg": microseg,
        "ltv_median": ltv_med, "risk_threshold": risk_th,
        "uplift_score_test": uplift_sc,
    }

    return results


# ── Contrafactual ──────────────────────────────────────────────────────────────
def counterfactual(model, X_instance: np.ndarray, X_train: np.ndarray,
                   feature_names: list, target_prob: float = 0.30,
                   n_steps: int = 800) -> tuple:
    """
    Genera el contrafactual mínimo para reducir la probabilidad
    de churn de un cliente al target_prob dado.
    Devuelve (X_cf, lista de cambios).
    """
    X_cf = X_instance.copy().astype(float)
    current_prob = model.predict_proba(X_cf.reshape(1, -1))[0, 1]
    if current_prob <= target_prob:
        return X_cf, []

    feat_std = np.array(X_train, dtype=float).std(axis=0) + 1e-6
    rng = np.random.default_rng(99)

    for _ in range(n_steps):
        i = rng.integers(len(feature_names))
        direction = X_train[:, i].mean() - X_cf[i]
        X_trial = X_cf.copy()
        X_trial[i] += 0.1 * direction
        prob = model.predict_proba(X_trial.reshape(1, -1))[0, 1]
        if prob < current_prob:
            X_cf = X_trial
            current_prob = prob
        if current_prob <= target_prob:
            break

    changes = []
    for i, fname in enumerate(feature_names):
        delta = X_cf[i] - X_instance[i]
        if abs(delta) > 0.01 * (abs(X_instance[i]) + 1e-6):
            changes.append({
                "feature": fname,
                "valor_actual":      round(float(X_instance[i]), 3),
                "valor_recomendado": round(float(X_cf[i]), 3),
                "cambio":            round(float(delta), 3),
            })
    changes = sorted(changes, key=lambda x: abs(x["cambio"]), reverse=True)[:6]
    return X_cf, changes


# ── Monitoreo ──────────────────────────────────────────────────────────────────
def monitor_production(model, X_train: np.ndarray,
                       X_prod_list: list, feature_names: list) -> dict:
    """
    Calcula PSI y KS drift para una lista de datasets de producción.
    X_prod_list: lista de (nombre, array) con datos de cada periodo.
    """
    score_train = model.predict_proba(X_train)[:, 1]
    psi_results, drift_results = [], {}

    for nombre, X_prod in X_prod_list:
        score_prod = model.predict_proba(X_prod)[:, 1]
        psi_val = calculate_psi(score_train, score_prod)
        psi_results.append({"periodo": nombre, "psi": psi_val})

    # Feature drift en el último periodo
    if X_prod_list:
        _, X_last = X_prod_list[-1]
        for j, fname in enumerate(feature_names):
            ks_stat, ks_p = ks_2samp(X_train[:, j], X_last[:, j])
            drift_results[fname] = {
                "ks_stat": round(ks_stat, 4),
                "ks_p":    round(ks_p, 4),
                "drifted": bool(ks_p < 0.05),
            }

    return {
        "psi":   pd.DataFrame(psi_results),
        "drift": drift_results,
    }


# ── Champion-Challenger bootstrap ──────────────────────────────────────────────
def champion_challenger(y_true: np.ndarray, scores_champ: np.ndarray,
                         scores_chall: np.ndarray, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            a1 = roc_auc_score(y_true[idx], scores_champ[idx])
            a2 = roc_auc_score(y_true[idx], scores_chall[idx])
            diffs.append(a1 - a2)
        except Exception:
            pass
    diffs = np.array(diffs)
    p_val = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diffs": diffs, "p_value": round(p_val, 4),
        "ci_low": round(ci_lo, 4), "ci_high": round(ci_hi, 4),
        "mean_diff": round(diffs.mean(), 4),
        "significant": bool(p_val < 0.05),
    }


# ── Helper interno ─────────────────────────────────────────────────────────────
def _metrics(y_true, y_score, name):
    return {
        "name":    name,
        "probs":   y_score,
        "y_test":  y_true,
        "auc_roc": round(roc_auc_score(y_true, y_score), 4),
        "auc_pr":  round(average_precision_score(y_true, y_score), 4),
        "brier":   round(brier_score_loss(y_true, y_score), 4),
    }
