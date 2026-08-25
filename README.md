# DLxGait-EDSS: Clasificación y Caracterización Biomecánica de la Marcha en Esclerosis Múltiple

## Análisis biomecánico mediante calcetines inteligentes (SCKS)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![PEP8](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)

---

## Descripción

Pipeline de extracción, análisis biomecánico temporal y clasificación de marcha a partir de datos de calcetines inteligentes (SCKS) en pacientes con Esclerosis Múltiple (EM). El objetivo es clasificar y caracterizar patrones de marcha mediante biomarcadores digitales objetivos, sentando una base metodológica que complemente el seguimiento clínico de la enfermedad.

El sistema se organiza en cuatro bloques:

1. **Biomecánica temporal** (`A06_ANALISIS_CINEMATICO`) — detección de eventos de marcha (Heel Strike/Toe Off) y métricas temporales bilaterales (duración de zancada, apoyo, vuelo, asimetría). No incluye reconstrucción espacial, descartada por fragilidad metodológica.
2. **Clasificación mediante Deep Learning** (`A04_TRANSFORMER`) — seis esquemas comparables (Transformer, FFT, Híbrido, y tres variantes de Late Fusion), validados mediante Leave-One-Patient-Out (LOPO) real sobre 27 pacientes.
3. **Clasificación enriquecida con GPS** (`A07_TRAYECTORIA_GPS`) — GPS como rama discreta de entrada, fusionada con IMU/presión mediante atención cruzada, más un cálculo exploratorio de velocidad de sesión.
4. **Informe consolidado** (`generar_informe_conjunto.py`) — combina los tres pipelines para un mismo paciente en un PDF corto.

---

## Instalación

\`\`\`bash
git clone https://github.com/MultipleSclerosisMonitoring/2026_DLxGait-EDSS.git
cd 2026_DLxGait-EDSS
conda create -n tfm python=3.10 && conda activate tfm
pip install -e .
\`\`\`

Antes de ejecutar cualquier pipeline, edite `A01_EXTRACCION_DATOS/config.yaml` con sus credenciales de InfluxDB (el archivo no contiene credenciales reales).

---

## Bloque 1 — Biomecánica Temporal (`A06_ANALISIS_CINEMATICO`)

Detecta eventos de marcha por presión plantar y calcula métricas temporales bilaterales.

\`\`\`bash
python A06_ANALISIS_CINEMATICO/Orquestador_temporal.py \\
    --paciente CODIGO_PACIENTE --inicio "yyyy-mm-dd hh:mm:ss" --fin "yyyy-mm-dd hh:mm:ss" \\
    --config-yaml A01_EXTRACCION_DATOS/config.yaml
\`\`\`

**Estructura:**
\`\`\`
A06_ANALISIS_CINEMATICO/
├── event_detector.py
├── kinematic_engine_temporal.py
├── Orquestador_temporal.py
└── tools/   (scripts de diagnóstico, ver docstrings)
\`\`\`

---

## Bloque 2 — Deep Learning (`A04_TRANSFORMER`)

Clasificación binaria marcha/reposo mediante seis esquemas: Transformer, FFT, Híbrido (Early Fusion), y tres variantes de Late Fusion (Media Geométrica, Voto Mayoritario, Meta-Clasificador), validados con LOPO real sobre 27 pacientes.

### Entrenar desde cero
\`\`\`bash
python A04_TRANSFORMER/AA_TRANSFORMER_V1.py --dataset "DATASET_HDF5/dataset_jerarquico.hdf5"
\`\`\`

### Inferencia con modelos preentrenados (evaluador agnóstico)
\`\`\`bash
python -m A04_TRANSFORMER.Agnostic_evaluator \\
    -c A01_EXTRACCION_DATOS/config.yaml -m A05_MODELOS_ENTRENADOS \\
    -r CODIGO_PACIENTE --start "yyyy-mm-dd hh:mm:ss" --end "yyyy-mm-dd hh:mm:ss" --modelo 6
\`\`\`

| Opción | Esquema |
|---|---|
| 1 / fft | FFT solo |
| 2 / transformer | Transformer solo |
| 3 / hibrido_early | Híbrido (Early Fusion) |
| 4 / hibrido_late | Late Fusion — Media Geométrica |
| 5 / hibrido_late_voto | Late Fusion — Voto Mayoritario |
| 6 / hibrido_late_meta | Late Fusion — Meta-Clasificador |

**Umbral de decisión fijo (0.5)**, sin calibración dinámica por diseño (ver memoria, Discusión 5.3).

### Resultados LOPO (27 pacientes, umbral 0.5)

| Esquema | AUC global | AUC promedio/paciente | STD |
|---|---|---|---|
| FFT (solo) | 0.9813 | 0.9761 | 0.0780 |
| Media Geométrica | 0.9781 | 0.9876 | 0.0303 |
| Meta-Clasificador | 0.9774 | 0.9879 | 0.0317 |
| Voto Mayoritario | 0.9529 | 0.9457 | 0.1171 |
| Híbrido (Early Fusion) | 0.9374 | — | — |
| Transformer (solo) | 0.9386 | — | — |

FFT domina en AUC global; Media Geométrica y Meta-Clasificador son más consistentes entre pacientes (ver memoria, Sección 4.2 y 5.1 para el análisis completo del trade-off).

\`\`\`bash
python A04_TRANSFORMER/EVALUACION_MODELOS.py --dataset "DATASET_HDF5/dataset_jerarquico.hdf5" --models "A05_MODELOS_ENTRENADOS" --run-lopo
\`\`\`
*Advertencia: reentrena 5 esquemas × 27 folds, puede tardar varias horas.*

---

## Bloque 3 — Clasificación con GPS (`A07_TRAYECTORIA_GPS`)

- **IMU/presión**: rama densa (PSD, 348 dimensiones), igual que A04.
- **GPS**: rama discreta con `delta_t`, posición UTM forward-filled, y máscara de observación (1.0 si es lectura real, 0.0 si es relleno).
- **Fusión**: `nn.MultiheadAttention`, IMU como query, GPS como key/value.

\`\`\`bash
python A07_TRAYECTORIA_GPS/modelo_gps_clasificacion.py \\
    --config-yaml A01_EXTRACCION_DATOS/config.yaml \\
    --excel A07_TRAYECTORIA_GPS/segmentos_A07_v2.xlsx \\
    --models-dir A05_MODELOS_ENTRENADOS
\`\`\`

**Resultado LOPO real:** AUC promedio 0.8221, con 40% de pacientes monoclase (menor volumen de datos que A04). Desempeño inferior a las arquitecturas de A04 — no se considera una alternativa viable en su estado actual (ver memoria, Sección 4.2.3).

También incluye cálculo exploratorio de velocidad de sesión:
\`\`\`bash
python A07_TRAYECTORIA_GPS/analizar_velocidad_gps_paciente.py \\
    --paciente CODIGO_PACIENTE --inicio "yyyy-mm-dd hh:mm:ss" --fin "yyyy-mm-dd hh:mm:ss" \\
    --config-yaml A01_EXTRACCION_DATOS/config.yaml
\`\`\`

---

## Bloque 4 — Informe Consolidado

Combina los 3 pipelines (Agnostic, A06, A07) para un mismo paciente/ventana en un PDF único. Requiere correr los tres primero, en orden, con la misma referencia temporal:

\`\`\`bash
python -m A04_TRANSFORMER.Agnostic_evaluator -c ... -r PACIENTE --start ... --end ... --modelo 6
python A06_ANALISIS_CINEMATICO/Orquestador_temporal.py --paciente PACIENTE --inicio ... --fin ...
python A07_TRAYECTORIA_GPS/analizar_velocidad_gps_paciente.py --paciente PACIENTE --inicio ... --fin ...
python generar_informe_conjunto.py --paciente PACIENTE --modelo-agnostic "hibrido_late_meta" ...
\`\`\`

---

## Tecnologías

| Categoría | Herramientas |
|---|---|
| Deep Learning | PyTorch, NumPy, SciPy |
| Ingeniería | Pydantic, Pytest, Pandas, Matplotlib, Reportlab |
| Biomecánica | IMU, FFT, Scikit-learn |
| Geoespacial | pyproj (UTM) |

Sigue PEP8, tipado estricto, y validación reproducible tras clonado limpio.

---

## Roadmap

**Completado:** extracción de datos · biomecánica temporal (A06) · seis esquemas de clasificación con LOPO real (27 pacientes) · evaluador agnóstico unificado (6 modos) · A07 rediseñado con GPS discreto · informe consolidado · compatibilidad multiplataforma validada.

**Líneas futuras:** ground truth validado clínicamente · ampliación de la muestra · investigación del caso atípico (paciente 20266247G-97) · incorporación de biomarcadores complementarios (pruebas cognitivas, marcadores oculares) hacia un sistema de caracterización del deterioro en EM.

---

## Autor

**Jairo Eduardo Paez Leal** — Máster en Ingeniería de Organización, ETSII, UPM
**Tutor:** Joaquín Bienvenido Ordieres Meré — ETSII, UPM
