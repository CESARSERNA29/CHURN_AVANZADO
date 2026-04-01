# Churn Intelligence Dashboard

Dashboard de análisis predictivo de churn construido con Streamlit.

## Estructura del proyecto  (CAS)

```
CHURN_STREAMLIT/
├── Home.py                          # Punto de entrada principal
├── data/
│   └── 01_Datos_Churn_Streamlit.xlsx   # Tu dataset
├── pages/
│   ├── 1_Feature_Engineering.py
│   ├── 2_Modelos.py
│   ├── 3_Explicabilidad.py
│   ├── 4_Segmentacion.py
│   ├── 5_Survival_Uplift.py
│   └── 6_Monitoreo.py
├── utils/
│   ├── helpers.py
│   ├── feature_engineering.py
│   └── models.py
├── models_saved/                    # Se crea automáticamente
├── exports/                         # Se crea automáticamente
└── requirements.txt
```

## Instalación

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar el dataset
# Copiar 01_Datos_Churn_Streamlit.xlsx en la carpeta data/

# 4. Ejecutar el dashboard
streamlit run Home.py
```

## Columnas requeridas en el dataset

| Columna | Tipo | Descripción |
|---|---|---|
| cliente_id | str | Identificador único |
| segmento | str | Enterprise / SMB / Startup / Individual |
| region | str | Región geográfica |
| tenure_meses | int | Meses como cliente |
| plan_precio | float | Precio por usuario/mes |
| num_usuarios | int | Usuarios activos |
| logins_30d | int | Logins últimos 30 días |
| logins_30d_previos | int | Logins 30 días anteriores |
| features_usados | int | Features del producto usadas |
| tickets_soporte | int | Tickets abiertos |
| tickets_resueltos | int | Tickets resueltos |
| nps_actual | int | NPS actual (0-10) |
| nps_30d_atras | int | NPS hace 30 días (0-10) |
| paginas_visitadas | int | Páginas visitadas |
| sesiones_30d | int | Sesiones del mes |
| sesiones_completadas | int | Sesiones completadas |
| api_calls_30d | int | Llamadas API del mes |
| dias_sin_login | int | Días desde último login |
| dias_renovacion | int | Días para renovación (neg = vencido) |
| num_integraciones | int | Integraciones activas |
| fallos_pago | int | Fallos de pago recientes |
| degradaciones | int | Downgrades de plan |
| referidos_dados | int | Referidos generados |
| tiempo_al_evento | float | Días hasta churn (para survival) |
| churn | int | 1 = churneó, 0 = activo |

## Páginas del dashboard

| Página | Contenido |
|---|---|
| 🏠 Home | KPIs, matriz LTV×Riesgo, Top 20 en riesgo |
| ⚙️ Feature Engineering | RFM, señales de cambio, fricción, profundidad |
| 🤖 Modelos | ROC, PR, lift, calibración, matriz de confusión |
| 🔍 Explicabilidad | SHAP, local, dependency plots, contrafactuales |
| 🗺️ Segmentación | Matriz por región, segmento, microsegmentos |
| 📈 Survival & Uplift | Kaplan-Meier, T-Learner, curva de ganancia |
| 📡 Monitoreo | PSI, feature drift, Champion-Challenger, alertas |
