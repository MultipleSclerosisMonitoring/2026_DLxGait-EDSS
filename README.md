# ANÁLISIS DE LA MARCHA PARA ESTIMACIÓN DE DETERIORO EN ESCLEROSIS MÚLTIPLE

## Análisis biomecánico mediante calcetines inteligentes (SCKS)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![PEP8](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)

---

## Descripción

Este repositorio contiene un pipeline de extracción, análisis biomecánico temporal
y clasificación de marcha para datos obtenidos mediante calcetines inteligentes
(SCKS) en pacientes con Esclerosis Múltiple (EM). El objetivo clínico final es
estimar automáticamente el **EDSS** (Expanded Disability Status Scale) a partir
de biomarcadores de marcha.

El sistema se organiza en cuatro bloques:

1. **Pipeline biomecánico temporal** (`A06_ANALISIS_CINEMATICO`) — extracción,
   detección de eventos de marcha (Heel Strike/Toe Off) y cálculo de métricas
   **temporales** bilaterales (duración de zancada, apoyo, vuelo, asimetría). No
   incluye reconstrucción espacial (trayectoria, velocidad, longitud de zancada,
   MTC): esa parte se descartó explícitamente por fragilidad metodológica (ver
   sección del bloque). No depende del bloque de Deep Learning.
2. **Pipeline de Deep Learning** (`A04_TRANSFORMER`) — clasificación de marcha
   mediante Transformer + FFT sobre espectrogramas híbridos generados en
   `A01_EXTRACCION_DATOS` / `A03_PREPROCESAMIENTO`, con evaluación LOPO real y
   comparación de estrategias de fusión (Early Fusion vs. Late Fusion).
3. **Clasificación enriquecida con GPS** (`A07_TRAYECTORIA_GPS`) — GPS tratado
   como rama discreta de entrada (no como target de trayectoria densa),
   fusionada con la rama IMU/presión mediante cross-attention, para mejorar la
   clasificación marcha/reposo. Incluye además un cálculo exploratorio de
   velocidad promedio de sesión vía GPS.
4. **Informe consolidado** (`generar_informe_conjunto.py`) — combina los
   resultados de los 3 pipelines anteriores (Agnostic, A06, A07) para un mismo
   paciente/sesión en un único PDF corto con métricas y gráficas.

---

## Instalación

```bash
git clone https://github.com/MultipleSclerosisMonitoring/2026_DLxGait-EDSS.git
cd 2026_DLxGait-EDSS

conda create -n tfm python=3.10
conda activate tfm

pip install -e .          # instalación editable (recomendada)
# o alternativamente:
pip install -r requirements.txt
```

### Configuración de InfluxDB

El archivo `config.yaml` no contiene credenciales reales. Antes de ejecutar
cualquier pipeline, edite `A01_EXTRACCION_DATOS/config.yaml`:

```yaml
influxdb:
  url: "https://YOUR_SERVER:8086/"
  org: "YOUR_ORG"
  bucket: "YOUR_BUCKET"
  token: "YOUR_TOKEN"
  tzval: "Europe/Madrid"
```

---

## Bloque 1 — Pipeline Biomecánico Temporal (`A06_ANALISIS_CINEMATICO`)

Módulo de detección de eventos de marcha y cálculo de métricas **temporales**
bilaterales, a partir de presión plantar (eventos) y giroscopio (Mid-Stance,
usado solo para diagnóstico de huecos). Se ejecuta y valida de forma
independiente del bloque de DL.

### Nota de migración importante

Una versión anterior de este bloque (`Orquestador_biomecanico.py` +
`kinematic_engine.py` + `fatigue_analysis.py`) incluía reconstrucción espacial
completa (fusión sensorial Madgwick, integración ZUPT, trayectoria 3D,
velocidad de marcha, longitud de zancada, MTC, y análisis de fatiga sobre
velocidad). **Esa parte espacial fue descartada** tras evaluación conjunta con
el director del TFM: la cadena de heurísticas necesaria (escala de aceleración
por "primera ventana segura", integración doble propensa a deriva) no producía
resultados clínicamente creíbles (velocidades y longitudes de zancada muy por
debajo de rangos fisiológicos plausibles en pruebas reales).

Las **métricas temporales** (duración de zancada, tiempo de apoyo, tiempo de
vuelo), que dependen únicamente de la detección de eventos por presión plantar,
sí se evaluaron como internamente consistentes y se mantienen.

Toda referencia a `Orquestador_biomecanico.py`, `kinematic_engine.py` o
`fatigue_analysis.py` en documentación anterior queda reemplazada por
`Orquestador_temporal.py` y `kinematic_engine_temporal.py`. `fatigue_analysis.py`
se eliminó sin reemplazo (dependía de una variable de velocidad que ya no se
calcula).

### Entrypoint oficial

```
A06_ANALISIS_CINEMATICO/Orquestador_temporal.py
```

### Ejecución

```bash
python A06_ANALISIS_CINEMATICO/Orquestador_temporal.py \
    --paciente CODIGO_PACIENTE \
    --inicio "yyyy-mm-dd hh:mm:ss" \
    --fin "yyyy-mm-dd hh:mm:ss" \
    --config-yaml A01_EXTRACCION_DATOS/config.yaml
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--fs` | 100 | Frecuencia de muestreo (Hz) |
| `--max-time-diff` | 0.20 | Tolerancia (s) para emparejar zancadas bilaterales |
| `--config-yaml` | `A01_EXTRACCION_DATOS/config.yaml` | Ruta al archivo de configuración de InfluxDB |
| `--output-dir` | `None` | Carpeta de salida. Si se omite, se calcula automáticamente como `A06_ANALISIS_CINEMATICO/RESULTADOS_TEMPORALES`, relativa a la ubicación real del script (portable entre clones del repositorio) |

### Flujo del pipeline

1. **Extracción** (`extract_data_plus.py`, en `A01_EXTRACCION_DATOS`): consulta
   InfluxDB (Flux) filtrando por `CodeID`, `Foot`, `type=SCKS` y rango temporal.
2. **Detección de eventos** (`event_detector.py`): Heel Strike / Toe Off a
   partir de presión plantar, con umbral adaptativo e histéresis temporal.
3. **Segmentación por tramos continuos**: si hay huecos > 2s entre eventos
   consecutivos (pausas, giros, pérdida de detección), el registro se corta en
   tramos y cada uno se procesa de forma aislada.
4. **Cálculo de métricas temporales** (`kinematic_engine_temporal.py`, por
   tramo): duración de zancada, tiempo de apoyo, tiempo de vuelo, usando
   Mid-Stance (derivado del giroscopio) únicamente como referencia interna de
   segmentación, no para ninguna integración de posición.
5. **Emparejamiento bilateral** (corregido, no voraz): cruza zancadas de ambos
   pies por proximidad temporal, marcando cada evento del pie izquierdo como
   "usado" tras emparejarlo, evitando reutilización implícita del mismo evento
   en más de un par.
6. **Cálculo de asimetría**: porcentaje de asimetría interpodal en duración de
   zancada, apoyo y vuelo.

### Herramientas de diagnóstico (`tools/`)

Desarrolladas durante la validación del motor cinemático anterior (con
reconstrucción espacial). Se conservan en el repositorio como referencia
histórica; `verificar_saturacion_acc.py`, `plot_deriva.py` y
`diagnostico_imu.py` en particular fueron diseñadas para diagnosticar la
cadena de reconstrucción espacial ya descartada, por lo que su utilidad para
el pipeline temporal actual es limitada.

| Script | Función |
|---|---|
| `TESTMAGNETO.py` | Audita measurements, field keys y tag keys reales del bucket InfluxDB (sintaxis Flux `schema.*`) |
| `verificar_saturacion_acc.py` | Cuantifica el % de muestras del acelerómetro saturadas en el límite de su rango dinámico, por eje y pie (relevante solo si se reactivara reconstrucción espacial) |
| `diagnosticar_eventos.py` | Imprime una "radiografía" de los huecos de detección de eventos > 2s por pie, clasificando la causa probable — sigue siendo útil con el pipeline temporal actual |
| `plot_deriva.py` | Grafica la presión plantar contra el umbral de detección en una ventana temporal específica |
| `diagnostico_imu.py` | Compara calidad de señal IMU cruda entre pies (diseñada originalmente para barrido de `madgwick_beta`, ya no aplicable) |

### Estructura de archivos

```
A06_ANALISIS_CINEMATICO/
├── __init__.py
├── event_detector.py
├── kinematic_engine_temporal.py
├── Orquestador_temporal.py
└── tools/
    ├── TESTMAGNETO.py
    ├── verificar_saturacion_acc.py
    ├── diagnosticar_eventos.py
    ├── plot_deriva.py
    └── diagnostico_imu.py
```

---

## Bloque 2 — Pipeline de Datos y Deep Learning (`A04_TRANSFORMER`)

Clasificación binaria de marcha/reposo mediante tres arquitecturas comparables
— Transformer temporal, FFT frecuencial, e Híbrido — evaluadas con test
independiente y, de forma más rigurosa, con validación **Leave-One-Patient-Out
(LOPO)** real (reentrenamiento completo por fold).

### Opción A — Entrenar modelos desde cero

```bash
python A04_TRANSFORMER/AA_TRANSFORMER_V1.py \
    --dataset "DATASET_HDF5/dataset_jerarquico.hdf5"
```

**Requisitos de hardware**: el entrenamiento usa GPU (CUDA) de forma automática
si está disponible (`torch.cuda.is_available()`). Si no hay GPU, el código cae
automáticamente a CPU sin necesidad de modificar nada, pero el entrenamiento de
los tres modelos será considerablemente más lento.

Entrena automáticamente: Transformer Temporal, Clasificador FFT, Modelo
Híbrido Early Fusion (Tiempo + Frecuencia); guarda el escalador
(`scaler_gait.joblib`), exporta pesos (`.pth`), guarda `particion_pacientes.json`
(identifica qué paciente quedó en test/val de esta corrida) y genera
métricas/gráficas de validación en `A05_MODELOS_ENTRENADOS/`.

**Nota sobre el umbral de clasificación**: el proyecto usa un umbral de
decisión fijo de 0.5 (no un umbral óptimo tipo Youden's J) en todo el pipeline
de producción, por decisión de diseño. El umbral está hardcodeado en el propio
código, no se carga desde ningún archivo `.joblib`.

**Archivos necesarios en `A05_MODELOS_ENTRENADOS/`** para el evaluador
agnóstico y la evaluación LOPO:

```
modelo_transformer.pth
modelo_fft.pth
modelo_hibrido.pth
scaler_gait.joblib
transformer_config.joblib
train_idx.npy / val_idx.npy / test_idx.npy
particion_pacientes.json
LATE_FUSION/meta_clasificador_late_fusion.joblib   (ver late_fusion.py más abajo)
```

### Opción B — Inferencia con modelos preentrenados (evaluador agnóstico unificado)

```bash
python A04_TRANSFORMER/Agnostic_evaluator.py \
    -c A01_EXTRACCION_DATOS/config.yaml \
    -m A05_MODELOS_ENTRENADOS \
    -r CODIGO_PACIENTE \
    --start "yyyy-mm-dd hh:mm:ss" \
    --end "yyyy-mm-dd hh:mm:ss" \
    --modelo 4
```

Si se omite `--modelo`, el script pregunta interactivamente por consola (no
usar sin `--modelo` en contextos automatizados).

| Opción | Modelo | Requiere |
|---|---|---|
| 1 / fft | FFTModel solo | `modelo_fft.pth` |
| 2 / transformer | GaitTransformer solo | `modelo_transformer.pth` |
| 3 / hibrido_early | GaitHybridModel (Early Fusion) | `modelo_hibrido.pth` |
| 4 / hibrido_late | Late Fusion (Media Geométrica) | `modelo_transformer.pth` + `modelo_fft.pth` (fórmula, sin `.joblib` adicional) |

**Nota sobre fechas**: si se pasa una fecha en formato ISO con sufijo `Z`
(ej. `2026-05-22T12:40:50Z`), verificar el comportamiento real de parseo antes
de asumir que se interpreta como UTC — en pruebas recientes, el evaluador trató
ese formato como hora local. Se recomienda pasar la hora ya convertida a hora
local (`tzval` del config, por defecto Europe/Madrid) en formato
`"yyyy-mm-dd hh:mm:ss"` para evitar ambigüedad.

Salida: `agnostico_{paciente}_{modelo}.csv` y
`{paciente}_{modelo}_agnostico_timeline.png` en `A04_TRANSFORMER/RESULTADO_AGNOSTIC/`.

### Comparación de modelos — LOPO real (Leave-One-Patient-Out)

Resultado más reciente, con el dataset ampliado a **27 pacientes** (frente a
los 14 originales), reentrenando cada esquema completo por fold:

| Método | AUC LOPO (global) |
|---|---|
| FFT (solo) | 0.9813 |
| Media Geométrica (Late Fusion) | 0.9781 |
| Meta-Clasificador (Late Fusion) | 0.9774 |
| Voto Mayoritario (Late Fusion) | 0.9529 |
| Híbrido (Early Fusion) | 0.9374 |

Ampliar el dataset de 14 a 27 pacientes mejoró el AUC LOPO en los 5 esquemas
de forma consistente (entre 4 y 6 puntos porcentuales respecto a la corrida
anterior). Con este dataset, **FFT solo** es el mejor en AUC global, aunque la
diferencia frente a Media Geométrica y Meta-Clasificador es marginal; en la
métrica de AUC promedio por fold (que pondera igual a cada paciente, en vez de
favorecer a los que aportan más muestras), Meta-Clasificador toma la delantera
por un margen igualmente estrecho. El evaluador agnóstico (`Agnostic_evaluator.py`,
opción 4) usa Media Geométrica por simplicidad (fórmula sin artefacto adicional
que sincronizar), no por ser el único candidato válido — ver
`A05_MODELOS_ENTRENADOS/LOPO/comparativa/resumen_general.txt` para el detalle
completo, incluyendo balanced accuracy, MCC y consistencia entre pacientes.

Ejecutar la evaluación LOPO completa (Híbrido + FFT + las 3 variantes de Late Fusion):

```bash
python A04_TRANSFORMER/EVALUACION_MODELOS.py \
    --dataset "DATASET_HDF5/dataset_jerarquico.hdf5" \
    --models "A05_MODELOS_ENTRENADOS" \
    --run-lopo
```

**Advertencia de tiempo**: esta corrida reentrena múltiples redes por cada
paciente dejado fuera (Híbrido, FFT, y Transformer+FFT por separado para Late
Fusion) — con 27 pacientes, puede tardar varias horas incluso con GPU.

Genera, dentro de `A05_MODELOS_ENTRENADOS/`:

```
evaluacion_superficial/     ← test independiente (1 paciente), metrica secundaria
    graficas/
    calibracion_eval.csv
    metricas_basicas.txt
    reporte_final_evaluacion.txt
LOPO/                       ← metrica principal de referencia
    hibrido/{graficas/, lopo_fold_aucs.csv, lopo_metricas_por_paciente.csv, metricas_lopo.csv, calibracion_lopo.csv}
    fft/{...}
    latefusion_media_geometrica/{...}
    latefusion_voto_mayoritario/{...}
    latefusion_meta_clasificador/{...}
    comparativa/
        resumen_general.txt
        detalles_por_modelo.txt
        graficas/comparativa_efectividad.png
        graficas/comparativa_efectividad_vs_costo.png
```

### Calcular solo Late Fusion (sin recalcular Híbrido/FFT vía LOPO)

Si los modelos ya están entrenados y solo se quiere obtener las 3 variantes de
Late Fusion sobre el test independiente (sin reentrenar, solo inferencia):

```bash
python A04_TRANSFORMER/late_fusion.py --dataset "DATASET_HDF5/dataset_jerarquico.hdf5" --models-dir A05_MODELOS_ENTRENADOS
```

Guarda el meta-clasificador final entrenado
(`meta_clasificador_late_fusion.joblib`) en
`A05_MODELOS_ENTRENADOS/LATE_FUSION/`.

**Nota metodológica**: el meta-clasificador final guardado se entrena sobre el
test independiente (único conjunto etiquetado disponible fuera de train/val).
El meta-clasificador reportado en el LOPO (`latefusion_meta_clasificador/`) es
uno distinto, honesto out-of-fold, calculado internamente en cada corrida de
`EVALUACION_MODELOS.py --run-lopo`; no es el mismo artefacto que el `.joblib`
de producción.

### Estructura de archivos

```
A04_TRANSFORMER/
├── __init__.py
├── AA_TRANSFORMER_V1.py
├── Agnostic_evaluator.py
├── EVALUACION_MODELOS.py
├── late_fusion.py
├── test_pipeline.py
└── RESULTADO_AGNOSTIC/
```

---

## Bloque 3 — Clasificación enriquecida con GPS (`A07_TRAYECTORIA_GPS`)

### Objetivo actual (rediseñado)

Este bloque **fue rediseñado por completo** respecto a su versión original.
El objetivo ya no es predecir trayectoria (posición X, Y) del cuerpo, sino
**clasificación** marcha/reposo — la misma tarea de A04 — usando el GPS de los
calcetines como **rama de entrada auxiliar**, fusionada con la rama IMU/presión
mediante cross-attention (`nn.MultiheadAttention`), para intentar mejorar el
AUC de clasificación respecto a usar solo IMU/presión.

Adicionalmente, se implementó un cálculo exploratorio de **velocidad promedio
de sesión** vía GPS (distancia recorrida real, sumando tramos entre lecturas
reales consecutivas — no desplazamiento neto, que subestimaría trayectos de
ida y vuelta) y un análisis de fatiga sobre esa velocidad, ambos documentados
como indicadores gruesos, no como sustituto de las métricas biomecánicas finas
de A06.

### Historial: por qué se descartó el enfoque original (trayectoria)

La primera versión de este bloque investigaba una propuesta del director del
TFM: usar presión + GPS, junto con fine-tuning de la arquitectura
`GaitTransformer` de A04, para predecir la **trayectoria** (posición X, Y)
como alternativa al motor Madgwick+ZUPT de A06.

**Conclusión de esa investigación: no viable con los datos actuales.**

- **Causa raíz**: el GPS de los calcetines es la única fuente de posición
  disponible (confirmado mediante inventario exhaustivo de todos los `_field`
  y tags de InfluxDB — no existe UWB, RTK ni ningún otro sensor de
  posicionamiento), con lecturas reales espaciadas entre **7 y 80 segundos**
  entre sí. Esto es varios órdenes de magnitud más disperso que la frecuencia
  (~10-25 Hz) necesaria para generar una trayectoria fina fiable.
- **Resultado**: el error final de predicción de trayectoria se mantuvo en el
  orden de **20-60 metros** según la variante probada (normalización por
  sesión, predicción de desplazamiento relativo, distintos learning rates),
  muy lejos de ser utilizable para calcular parámetros clínicos de marcha.
- **Problema metodológico adicional identificado**: el enfoque original
  generaba un "ground truth" denso interpolando (PCHIP) entre las lecturas GPS
  reales dispersas, y entrenaba el modelo contra esa curva interpolada — es
  decir, el modelo aprendía a imitar una **suposición matemática**, no el
  movimiento real del paciente entre lecturas.

**Decisión**: se descartó la reconstrucción de trayectoria vía GPS+Transformer
por completo. Los scripts de esa versión
(`preparar_dataset_trayectoria.py`, `conector_trayectoria.py`,
`trajectory_model.py`, `entrenar_trajectory_model.py`,
`entrenar_trajectory_model_multi.py`, `analizar_segmento_trayectoria.py`) **ya
no existen en el repositorio**. Se mantiene A06 (detección de eventos +
métricas temporales) como fuente principal de biomarcadores de marcha del
proyecto.

### Rediseño: GPS como rama discreta (no como target denso)

Siguiendo la recomendación del director del TFM tras evaluar la investigación
anterior, el GPS se trata ahora como lo que realmente es: una fuente de
observación **discreta, irregular y de baja frecuencia**, no una señal densa
comparable a IMU/presión, y **nunca** se usa simultáneamente como entrada y
como variable a predecir del mismo problema (evitar fuga de información).

Principios de diseño:

- **IMU/presión**: rama densa, tensor PSD combinado (mismo esquema que A04,
  348 dimensiones), sin cambios respecto al pipeline de clasificación.
- **GPS**: rama discreta — para cada frame de la rama densa, se registra
  `delta_t` (tiempo transcurrido), `x`/`y` (posición proyectada a metros UTM,
  forward-filled desde la última lectura real) y una **máscara de
  observación** (1.0 si ese frame coincide con una lectura GPS real, 0.0 si es
  relleno). No se interpola ninguna curva entre lecturas reales: el "hueco"
  entre lecturas queda representado por la máscara en 0, dejando que el modelo
  decida cuánto confiar en la posición, no una suposición matemática externa.
- **Fusión**: cross-attention (`nn.MultiheadAttention`) entre los embeddings
  por paso de la rama IMU/presión (query) y la rama GPS (key, value).
- **Objetivo**: clasificación marcha/reposo (etiqueta tomada de la columna
  `mov_type` del Excel de segmentos), nunca predicción de posición.

**Advertencia sobre desplazamiento incremental**: si se calcula distancia
recorrida a partir de las coordenadas GPS, debe sumarse el desplazamiento
entre **cada par de lecturas reales consecutivas**, no la distancia neta entre
el primer y el último punto. Un paciente que recorre un pasillo de ida y
vuelta puede terminar en (casi) la misma posición de inicio (distancia neta ≈
0) pese a haber recorrido el doble de la longitud del pasillo. Esta corrección
está implementada en `calcular_distancia_recorrida_real`.

### Contenido de la carpeta

| Archivo | Rol |
|---|---|
| `preparar_dataset_clasificacion_gps.py` | Extrae presión+IMU+GPS de InfluxDB (`extraer_datos_crudos`); construye la rama GPS discreta (`construir_rama_gps_discreta`); lee la tabla de segmentos desde Excel (`cargar_segmentos_desde_excel`); calcula velocidad de sesión (`calcular_velocidad_promedio_sesion`, `calcular_distancia_recorrida_real`) y fatiga exploratoria (`calcular_fatiga_por_tramos`) |
| `modelo_gps_clasificacion.py` | Conector (arma el tensor PSD + secuencia GPS alineados), arquitectura de 2 ramas con cross-attention (`ModeloClasificacionGPS`, `EncoderGPS`), y entrenamiento/evaluación LOPO real, todo en un único script |
| `analizar_velocidad_gps_paciente.py` | Calcula velocidad promedio de sesión para un paciente/segmento específico, guarda resultado individual (el LOPO de `modelo_gps_clasificacion.py` no reporta por paciente) |
| `segmentos_A07_v2.xlsx` | Tabla de segmentos (`Reference`, `datefrom`, `dateuntil`, `mov_type`, `es_utc`) usada para construir el dataset de entrenamiento. Editable directamente sin tocar código |
| `verificar_gps_por_pie.py` | Herramienta de diagnóstico: confirma que el GPS es compartido entre pies (un solo GPS por dispositivo, no independiente por pie) |
| `inventario_influxdb.py` | Herramienta de diagnóstico: lista todos los `_field`/tags de InfluxDB, usada para confirmar que no existe otra fuente de posicionamiento (UWB/RTK) |

### Cómo ejecutar

```bash
# Entrenar y evaluar (LOPO real) el modelo de clasificacion GPS+cross-attention
python A07_TRAYECTORIA_GPS/modelo_gps_clasificacion.py \
    --config-yaml A01_EXTRACCION_DATOS/config.yaml \
    --excel A07_TRAYECTORIA_GPS/segmentos_A07_v2.xlsx \
    --models-dir A05_MODELOS_ENTRENADOS
```

Reentrena el modelo completo (rama IMU/presión inicializada con los pesos de
`modelo_transformer.pth` de A04, rama GPS desde cero) en cada fold
Leave-One-Patient-Out. **Advertencia de tiempo**: con ~24-30 pacientes y varios
segmentos por paciente, esto puede tardar varias horas. Resultado:
`RESULTADOS_CLASIFICACION_GPS/lopo_clasificacion_gps.csv`.

```bash
# Calcular velocidad de sesion para un paciente/segmento especifico
python A07_TRAYECTORIA_GPS/analizar_velocidad_gps_paciente.py \
    --paciente CODIGO_PACIENTE \
    --inicio "yyyy-mm-dd hh:mm:ss" \
    --fin "yyyy-mm-dd hh:mm:ss" \
    --config-yaml A01_EXTRACCION_DATOS/config.yaml
```

Resultado: `RESULTADOS_VELOCIDAD_GPS/{paciente}_velocidad_gps.csv`, con
`distancia_m`, `duracion_s`, `velocidad_ms` (NaN si hay menos de 2 lecturas GPS
reales en el segmento) y `n_lecturas_gps_reales`.

**Advertencia metodológica**: esta velocidad es un promedio grueso de
desplazamiento de toda la sesión, no velocidad de marcha instantánea ni
sustituto de longitud de zancada/MTC — esas métricas requieren resolución de
centímetros, incompatible con la frecuencia real del GPS disponible.

### Bug conocido y corregido — lat/lng como texto desde InfluxDB

Durante la validación de este bloque se detectó que InfluxDB puede devolver
`lat`/`lng` como **texto** (`'0'`) en vez de numérico (`0.0`). Muchos
receptores GPS/GNSS reportan `lat=0, lng=0` antes de obtener su primer "fix"
satelital real; sin conversión explícita a numérico, una comparación directa
(`df["lat"] == 0`) nunca es `True` (compara string contra entero) y ese punto
espurio se trataba como una lectura real, introduciendo saltos de posición de
cientos de miles de kilómetros en los cálculos de distancia/velocidad.

**Corregido** en `construir_rama_gps_discreta` (conversión explícita con
`pd.to_numeric(..., errors="coerce")` antes de cualquier comparación, y
exclusión explícita de `(0, 0)` del conjunto de lecturas reales). Auditado
sobre los segmentos de entrenamiento reales: el bug no afectó ningún resultado
de LOPO ya reportado (0 de 78 segmentos auditados tenían el origen
contaminado por este problema).

### Estructura de archivos

```
A07_TRAYECTORIA_GPS/
├── __init__.py
├── preparar_dataset_clasificacion_gps.py
├── modelo_gps_clasificacion.py
├── analizar_velocidad_gps_paciente.py
├── segmentos_A07_v2.xlsx
├── verificar_gps_por_pie.py
├── inventario_influxdb.py
├── RESULTADOS_CLASIFICACION_GPS/
└── RESULTADOS_VELOCIDAD_GPS/
```

---

## Bloque 4 — Informe consolidado (`generar_informe_conjunto.py`)

Genera un informe PDF corto que combina los resultados de los **3 pipelines**
(clasificación vía Agnostic, biomecánica temporal vía A06, velocidad GPS vía
A07) para **un mismo paciente y ventana temporal**, con sus métricas más
relevantes y 2 gráficas (línea de probabilidad de marcha en el tiempo, y
barras de asimetría bilateral).

**El informe solo se genera si los 3 pipelines ya corrieron exitosamente para
ese paciente** — el script verifica la existencia de los 3 archivos de salida
esperados; si falta alguno, informa cuál y no genera ningún PDF parcial.

### Orden de ejecución

Antes de generar el informe, hay que correr los 3 pipelines **para el mismo
paciente y la misma ventana temporal**, en este orden:

```bash
# 1. Clasificacion (Agnostic)
python A04_TRANSFORMER/Agnostic_evaluator.py \
    -c A01_EXTRACCION_DATOS/config.yaml -m A05_MODELOS_ENTRENADOS \
    -r CODIGO_PACIENTE --start "yyyy-mm-dd hh:mm:ss" --end "yyyy-mm-dd hh:mm:ss" --modelo 4

# 2. Biomecanica temporal (A06)
python A06_ANALISIS_CINEMATICO/Orquestador_temporal.py \
    --paciente CODIGO_PACIENTE --inicio "yyyy-mm-dd hh:mm:ss" --fin "yyyy-mm-dd hh:mm:ss" \
    --config-yaml A01_EXTRACCION_DATOS/config.yaml

# 3. Velocidad GPS (A07)
python A07_TRAYECTORIA_GPS/analizar_velocidad_gps_paciente.py \
    --paciente CODIGO_PACIENTE --inicio "yyyy-mm-dd hh:mm:ss" --fin "yyyy-mm-dd hh:mm:ss" \
    --config-yaml A01_EXTRACCION_DATOS/config.yaml

# 4. Informe conjunto (resultado final: un PDF)
python generar_informe_conjunto.py \
    --paciente CODIGO_PACIENTE --modelo-agnostic "hibrido_late" \
    --dir-agnostic "A04_TRANSFORMER/RESULTADO_AGNOSTIC" \
    --dir-a06 "A06_ANALISIS_CINEMATICO/RESULTADOS_TEMPORALES" \
    --dir-a07 "A07_TRAYECTORIA_GPS/RESULTADOS_VELOCIDAD_GPS" \
    --output-pdf "informe_CODIGO_PACIENTE.pdf"
```

El paso 4 es el que produce el **resultado final**: un único PDF con el
resumen consolidado de los 3 pipelines para ese paciente.

---

## Compatibilidad y reproducibilidad

- Empaquetado editable moderno (`pyproject.toml` / `setup.py`)
- Resolución agnóstica de rutas (`Path(__file__)`), sin rutas absolutas locales
  hardcodeadas — verificado explícitamente en `Orquestador_temporal.py` y
  `analizar_velocidad_gps_paciente.py`, cuya carpeta de salida por defecto se
  calcula relativa a la ubicación del propio script, no a una ruta fija
- Compatibilidad equivalente entre `python -m ...` y `python script.py`
- Ejecución reproducible tras clonado limpio, multiplataforma (validado
  clonando el repositorio en una carpeta nueva y corriendo los 3 pipelines +
  informe conjunto de punta a punta)

---

## Tecnologías utilizadas

| Categoría | Herramientas |
|---|---|
| Deep Learning | PyTorch, NumPy, SciPy |
| Ingeniería | Pydantic, Pytest, Logging, Type Hinting, Pandas, Matplotlib, Openpyxl, Pyarrow, Reportlab |
| Biomecánica | IMUs, FFT, Sensor Fusion, Scikit-learn |
| Geoespacial | pyproj (proyección UTM) |

El proyecto sigue PEP8, tipado estricto, programación orientada a objetos,
modularidad y testing unitario.

---

## Roadmap

### Completado

- [x] Pipeline de extracción de datos
- [x] Detección de eventos y métricas biomecánicas temporales (A06)
- [x] Arquitectura Tiempo, Frecuencia e Híbrida (Early Fusion)
- [x] Late Fusion (Media Geométrica, Voto Mayoritario, Meta-Clasificador)
- [x] Validación LOPO real (5 esquemas, dataset ampliado a 27 pacientes)
- [x] Evaluación agnóstica continua (unificada, 4 modos de modelo)
- [x] Rediseño de A07: GPS como rama discreta de clasificación (cross-attention),
      reemplazando el enfoque de trayectoria interpolada (descartado)
- [x] Cálculo exploratorio de velocidad de sesión vía GPS
- [x] Informe consolidado (PDF) de los 3 pipelines por paciente
- [x] Auditorías de dataset (redundancia de ventanas, distribución de clases,
      fuga de datos)
- [x] Compatibilidad multiplataforma mediante empaquetado editable, validada
      con clonado limpio

### Próximas fases

- [ ] Regresor clínico EDSS
- [ ] Integración TabTransformer
- [ ] MLP-Mixer clínico
- [ ] Fusión con variables cognitivas

---

## Autor

**Jairo Eduardo Paez Leal**
Máster en Ingeniería de Organización
Escuela Técnica Superior de Ingenieros Industriales
Universidad Politécnica de Madrid (UPM)

**Joaquín Bienvenido Ordieres Meré**
Tutor
Escuela Técnica Superior de Ingenieros Industriales
Universidad Politécnica de Madrid (UPM)
