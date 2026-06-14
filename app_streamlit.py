"""
App de demo - TFM: Recomendador de margen de tierra vegetal
Carga el pipeline entrenado (modelo_margen_tierra.pkl) y, a partir de los
datos de una obra introducidos por el usuario, predice el margen adicional
recomendado y la cantidad de tierra a pedir.
"""

import streamlit as st
import pandas as pd
import joblib

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Recomendador de margen de tierra vegetal",
    page_icon="🌱",
    layout="centered",
)

st.title("🌱 Recomendador de margen de tierra vegetal")
st.markdown(
    "Introduce los datos de la obra para obtener el **margen adicional "
    "recomendado** sobre el cálculo teórico y la **cantidad de tierra a "
    "pedir**."
)

# =============================================================================
# CARGA DEL MODELO
# =============================================================================
@st.cache_resource
def load_model():
    return joblib.load("modelo_margen_tierra.pkl")

model = load_model()

# =============================================================================
# FORMULARIO DE ENTRADA
# =============================================================================
st.header("Datos de la obra")

col1, col2 = st.columns(2)

with col1:
    superficie_m2 = st.number_input(
        "Superficie (m²)", min_value=1.0, max_value=5000.0, value=200.0, step=1.0,
        help="Superficie a cubrir, en metros cuadrados."
    )
    espesor_cm = st.number_input(
        "Espesor de tierra (cm)", min_value=1.0, max_value=20.0, value=6.5, step=0.5,
        help="Espesor de la capa de tierra vegetal a aplicar."
    )
    densidad_suelo = st.number_input(
        "Densidad del suelo (kg/m³)", min_value=1000.0, max_value=1700.0, value=1345.0, step=1.0,
        help="Densidad aparente del suelo/tierra a utilizar."
    )
    tipo_trabajo = st.selectbox(
        "Tipo de trabajo",
        options=["instalacion", "renovacion", "ampliacion"],
        index=0,
    )
    tipo_suelo = st.selectbox(
        "Tipo de suelo",
        options=["arenoso", "arcilloso", "mixto"],
        index=2,
    )
    estacion = st.selectbox(
        "Estación del año",
        options=["primavera", "verano", "otoño", "invierno"],
        index=0,
    )

with col2:
    pendiente_pct = st.slider(
        "Pendiente del terreno (%)", min_value=0.0, max_value=25.0, value=5.0, step=0.5
    )
    nivelacion = st.slider(
        "Nivelación previa del terreno (1=mala, 5=excelente)",
        min_value=1, max_value=5, value=3,
    )
    accesibilidad = st.slider(
        "Accesibilidad al terreno (1=difícil, 5=fácil)",
        min_value=1, max_value=5, value=3,
    )
    operarios = st.slider(
        "Número de operarios", min_value=1, max_value=6, value=3,
    )
    experiencia_operarios = st.number_input(
        "Experiencia media de los operarios (años)",
        min_value=0.0, max_value=30.0, value=6.0, step=0.5,
    )
    lluvia_previa_mm = st.number_input(
        "Lluvia acumulada en los días previos (mm)",
        min_value=0.0, max_value=200.0, value=10.0, step=1.0,
    )

# =============================================================================
# PREDICCIÓN
# =============================================================================
if st.button("Calcular recomendación", type="primary"):

    # tierra_teorica_kg se calcula internamente (no es input del modelo,
    # pero sí es necesaria para traducir el margen a kg de tierra a pedir)
    tierra_teorica_kg = superficie_m2 * (espesor_cm / 100) * densidad_suelo

    # DataFrame con las columnas EXACTAS que espera el pipeline
    input_df = pd.DataFrame([{
        "superficie_m2": superficie_m2,
        "espesor_cm": espesor_cm,
        "densidad_suelo": densidad_suelo,
        "tipo_trabajo": tipo_trabajo,
        "tipo_suelo": tipo_suelo,
        "pendiente_pct": pendiente_pct,
        "nivelacion": nivelacion,
        "accesibilidad": accesibilidad,
        "operarios": operarios,
        "experiencia_operarios": experiencia_operarios,
        "lluvia_previa_mm": lluvia_previa_mm,
        "estacion": estacion,
    }])

    margen_pred = model.predict(input_df)[0]
    tierra_a_pedir_kg = tierra_teorica_kg * (1 + margen_pred / 100)

    st.header("Resultado")

    c1, c2, c3 = st.columns(3)
    c1.metric("Tierra teórica (kg)", f"{tierra_teorica_kg:,.0f}")
    c2.metric("Margen recomendado", f"{margen_pred:+.1f} %")
    c3.metric("Tierra a pedir (kg)", f"{tierra_a_pedir_kg:,.0f}")

    diferencia_kg = tierra_a_pedir_kg - tierra_teorica_kg
    if margen_pred >= 0:
        st.success(
            f"Se recomienda pedir **{diferencia_kg:,.0f} kg adicionales** "
            f"sobre el cálculo teórico ({margen_pred:.1f}% extra)."
        )
    else:
        st.info(
            f"El modelo recomienda pedir **{abs(diferencia_kg):,.0f} kg "
            f"menos** que el cálculo teórico ({abs(margen_pred):.1f}% menos), "
            f"dadas las condiciones favorables de la obra."
        )

    with st.expander("Ver detalle de los datos introducidos"):
        st.dataframe(input_df, hide_index=True)

# =============================================================================
# NOTA SOBRE EL MODELO
# =============================================================================
st.divider()
st.caption(
    "Modelo: Regresión Lineal entrenada sobre un dataset sintético de obras "
    "de jardinería (MAE≈1.96 puntos de margen, R²≈0.65 en test). "
    "Demo desarrollada para un Trabajo Fin de Máster en Data Science."
)