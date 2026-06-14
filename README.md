# Recomendador de margen de tierra vegetal — TFM Data Science

Modelo de Machine Learning que predice, antes de iniciar una obra de
jardinería, el **margen adicional óptimo** que conviene pedir sobre el
cálculo teórico de tierra vegetal, evitando tanto roturas de obra por
falta de material como excesos de pedido.

## Índice

- [Problema de negocio](#problema-de-negocio)
- [Archivos del repositorio](#archivos-del-repositorio)
- [Dataset](#dataset)
- [Control de Data Leakage](#control-de-data-leakage)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Demo (app Streamlit)](#demo-app-streamlit)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)

## Problema de negocio

Actualmente, la cantidad de tierra vegetal necesaria para una obra se
calcula con una fórmula teórica fija:

\`\`\`
tierra_teorica_kg = superficie_m2 × (espesor_cm / 100) × densidad_suelo
\`\`\`

Esta fórmula no tiene en cuenta factores reales de la obra (pendiente,
accesibilidad, experiencia del equipo, lluvia previa, tipo de suelo, etc.),
por lo que en la práctica casi nunca se ajusta: o falta material (hay que
parar la obra y hacer un segundo pedido) o sobra (coste extra de
transporte y devolución).

El objetivo del proyecto es predecir, **antes de ejecutar la obra**, el
\`margen_necesario_pct\`:

\`\`\`
margen_necesario_pct = ((tierra_consumida_real_kg - tierra_teorica_kg)
                          / tierra_teorica_kg) × 100
\`\`\`

## Archivos del repositorio

| Archivo | Descripción |
|---|---|
| \`01_eda_limpieza.ipynb\` | Análisis exploratorio de datos y limpieza de datos |
| \`02_modelado.ipynb\` | Selección de variables, modelado y evaluación |
| \`dataset_jardineria_tierra_sucio.csv\` | Dataset sintético "sucio" |
| \`dataset_jardineria_tierra_limpio.csv\` | Dataset tras limpieza |
| \`modelo_margen_tierra.pkl\` | Pipeline final entrenado (joblib) |
| \`app_streamlit.py\` | App de demo (Streamlit) |
| \`requirements.txt\` | Dependencias de la app |
| \`presentacion_tfm.pptx\` | Presentación del proyecto |

Los notebooks están en formato \`# %%\` (celdas), compatibles con
Jupyter, VSCode y PyCharm.

## Dataset

Dataset sintético de **656 obras de jardinería** (tras limpieza), generado
con \`generar_dataset.py\`. Cada fila representa una obra terminada, con 12
variables de entrada conocidas en fase de presupuesto:

| Variable | Descripción |
|---|---|
| \`superficie_m2\` | Superficie a cubrir (m²) |
| \`espesor_cm\` | Espesor de la capa de tierra (cm) |
| \`densidad_suelo\` | Densidad aparente del suelo (kg/m³) |
| \`tipo_trabajo\` | instalación / renovación / ampliación |
| \`tipo_suelo\` | arenoso / arcilloso / mixto |
| \`pendiente_pct\` | Pendiente del terreno (%) |
| \`nivelacion\` | Nivelación previa (1=mala, 5=excelente) |
| \`accesibilidad\` | Accesibilidad al terreno (1=difícil, 5=fácil) |
| \`operarios\` | Número de operarios |
| \`experiencia_operarios\` | Experiencia media del equipo (años) |
| \`lluvia_previa_mm\` | Lluvia acumulada en los días previos (mm) |
| \`estacion\` | primavera / verano / otoño / invierno |

La variable objetivo (\`margen_necesario_pct\`) se genera mediante un proceso
no lineal con efectos de saturación, interacciones entre variables (p. ej.
lluvia × tipo de suelo) y ruido estadístico, para simular un escenario
realista.

El dataset incluye además "suciedad" intencional (duplicados, nulos,
errores tipográficos, valores fuera de rango) para practicar limpieza de
datos.

## Control de Data Leakage

Tres variables solo existen **después** de finalizar la obra y se han
**excluido por completo** del entrenamiento:

- \`tierra_pedida_kg\`
- \`tierra_sobrante_kg\`
- \`tierra_consumida_real_kg\`

Esto garantiza que el modelo predice únicamente con información disponible
en el momento de hacer el presupuesto, antes de ejecutar la obra.

## Metodología

1. **EDA y limpieza** (\`01_eda_limpieza.ipynb\`): distribuciones, nulos, duplicados, outliers,
   correlaciones y validación visual de las relaciones de negocio
   esperadas (pendiente, nivelación, lluvia × suelo, experiencia).
   Eliminación de duplicados,
   normalización de categóricas (errores tipográficos), corrección de
   imposibles físicos (lluvia negativa, nivelación/accesibilidad fuera de
   rango [1,5]). Los valores nulos (~3-5% por columna) y los outliers
   altos pero físicamente posibles se **conservan**, para que el modelo
   los gestione de forma nativa.
2. **Modelado** (\`03_modelado.py\`): selección de variables (sin
   \`tierra_teorica_kg\`, para evitar multicolinealidad con sus componentes),
   codificación one-hot de categóricas, split train/test 80/20, y
   comparativa de 3 modelos mediante validación cruzada de 5 particiones:
   - Regresión Lineal (baseline)
   - XGBoost
   - LightGBM
3. **Selección del modelo final** y guardado del pipeline completo
   (preprocesado + imputación + modelo) con \`joblib\`.

## Resultados

### Comparativa de modelos (validación cruzada, 5-fold)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| **Regresión Lineal** | **1.79** | **2.26** | **0.61** |
| XGBoost | 1.84 | 2.31 | 0.60 |
| LightGBM | 1.87 | 2.35 | 0.59 |

La Regresión Lineal obtiene el mejor resultado en las tres métricas. Con
~520 obras de entrenamiento y el ruido introducido por diseño, un modelo
lineal generaliza mejor que modelos no lineales más complejos.

### Modelo final — evaluación en test (132 obras)

| Métrica | Valor |
|---|---|
| MAE | 1.96 |
| RMSE | 2.48 |
| R² | 0.65 |

### Interpretación (coeficientes del modelo)

Los signos de los coeficientes son coherentes con la lógica de negocio
diseñada:

- \`nivelacion\`, \`accesibilidad\`, \`experiencia_operarios\` → coeficiente
  **negativo** (mejoran estas variables → menor margen necesario).
- \`pendiente_pct\`, \`espesor_cm\`, \`lluvia_previa_mm\` → coeficiente
  **positivo**.
- \`tipo_suelo_arcilloso\` → coeficiente positivo (mayor margen base);
  \`tipo_suelo_arenoso\` → coeficiente negativo.

## Demo (app Streamlit)

\`app_streamlit.py\` carga el modelo entrenado y, a partir de los datos
de una obra introducidos en un formulario, calcula:

- Tierra teórica (kg)
- Margen recomendado (%)
- Tierra a pedir (kg)

### Ejecución local

\`\`\`bash
pip install -r requirements.txt
python -m streamlit run app_streamlit.py
\`\`\`

> En Windows, si \`streamlit run\` falla por restricciones de seguridad del
> sistema (Application Control), usa \`python -m streamlit run
> app_streamlit.py\` en su lugar.

## Limitaciones y trabajo futuro

- El dataset es **sintético**: las relaciones reales podrían diferir de
  las simuladas.
- El tamaño moderado (656 obras) limita la capacidad de los modelos no
  lineales para superar a un modelo lineal simple.
- R² ≈ 0.65: queda varianza sin explicar, parte de ella ruido introducido
  por diseño.
- Próximos pasos: validar con datos reales de obras, ampliar la muestra, y
  explorar nuevas variables (tipo de planta, maquinaria utilizada, etc.).
