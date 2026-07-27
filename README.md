# ESTIMACIÓN DE DETERIORO EN ESCLEROSIS MÚLTIPLE

## Análisis biomecánico mediante calcetines inteligentes (SCKS)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![PEP8](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)

---

## Descripción

Este repositorio contiene un pipeline de extracción, reconstrucción biomecánica
y clasificación de marcha para datos obtenidos mediante calcetines inteligentes
(SCKS) en pacientes con Esclerosis Múltiple (EM). El objetivo clínico final es
estimar automáticamente el **EDSS** (Expanded Disability Status Scale) a partir
de biomarcadores de marcha.

El sistema se organiza en tres bloques:

1. **Pipeline biomecánico (`A06_ANALISIS_CINEMATICO`)** — extracción, detección
   de eventos de marcha, reconstrucción cinemática 3D (Madgwick+ZUPT) y
   biomarcadores espaciotemporales/fatiga. No depende del bloque de Deep
   Learning.
2. **Pipeline de Deep Learning (`A04_TRANSFORMER`)** — clasificación de marcha
   mediante Transformer + FFT sobre espectrogramas híbridos generados en
   `A01_EXTRACCION_DATOS` / `A03_PREPROCESAMIENTO`, con evaluación LOPO real y
   comparación de estrategias de fusión (Early Fusion vs. Late Fusion).
3. **Exploración de trayectoria GPS+Transformer (`A07_TRAYECTORIA_GPS`)** —
   investigación de una alternativa de trayectoria basada en GPS+presión con
   fine-tuning del Transformer. **Concluida como no viable** con los datos
   actuales (ver README propio de esa carpeta); se conserva como evidencia
   documentada del proceso.

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

## Configuración de InfluxDB

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

# Bloque 1 — Pipeline Biomecánico (`A06_ANALISIS_CINEMATICO`)

Módulo de reconstrucción cinemática 3D y cálculo de métricas espaciotemporales
de marcha bilateral, a partir de señales inerciales (IMU) y presión plantar.
Genera biomarcadores que pueden integrarse posteriormente en modelos clínicos
predictivos, pero se ejecuta y valida de forma independiente del bloque de DL.

## Entrypoint oficial

```
A06_ANALISIS_CINEMATICO/Orquestador_biomecanico.py
```

> **Nota de migración:** una versión previa y más simple (`run_gait_pipeline.py`
> / `run_pipeline.py`) existió en este directorio pero fue descartada. No
> implementaba auto-calibración de umbral, fusión con magnetómetro,
> segmentación por tramos continuos ni validaciones de calidad de señal, y solo
> procesaba el pie derecho. Toda referencia a un script `run_pipeline.py` en
> versiones anteriores de este README queda reemplazada por
> `Orquestador_biomecanico.py`.

## Ejecución

```bash
python A06_ANALISIS_CINEMATICO/Orquestador_biomecanico.py \
    --paciente CODIGO_PACIENTE \
    --inicio "yyyy-mm-dd hh:mm:ss" \
    --fin "yyyy-mm-dd hh:mm:ss" \
    --config-yaml A01_EXTRACCION_DATOS/config.yaml
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--fs` | 100 | Frecuencia de muestreo (Hz) |
| `--fatigue-target` | `Gait_Speed_ms` | Variable sobre la que se calcula la pendiente de fatiga |
| `--max-time-diff` | 0.20 | Tolerancia (s) para emparejar zancadas bilaterales |
| `--th-right` / `--th-left` | None | Umbral manual de detección de eventos por pie (si se omite, se auto-calibra) |
| `--sin-auto-calibrar` | False | Desactiva la auto-calibración de umbral |
| `--max-stride` | 1.9 | Longitud de zancada máxima plausible (m), para filtrar outliers |

## Flujo del pipeline

1. **Extracción** (`extract_data_plus.py`, en `A01_EXTRACCION_DATOS`): consulta
   InfluxDB (Flux) filtrando por `CodeID`, `Foot`, `type=SCKS` y rango temporal.
2. **Detección de eventos** (`event_detector.py`): Heel Strike / Toe Off a
   partir de presión plantar, con umbral adaptativo e histéresis temporal.
3. **Auto-calibración de umbral**: barrido de fracciones de umbral por pie,
   seleccionando la que produce un %stance fisiológico (45-70%).
4. **Segmentación por tramos continuos**: si hay huecos > 2s entre eventos
   consecutivos (pausas, giros, pérdida de detección), el registro se corta en
   tramos y cada uno se procesa de forma aislada, evitando que la integración
   arrastre deriva de un tramo a otro.
5. **Reconstrucción cinemática** (`kinematic_engine.py`, por tramo): fusión
   sensorial (Madgwick, con magnetómetro si está disponible), compensación
   gravitacional, ZUPT (Zero Velocity Update) e integración de posición 3D.
6. **Emparejamiento bilateral y asimetría**: cruza zancadas de ambos pies por
   proximidad temporal y calcula asimetría de velocidad, zancada y MTC.
7. **Análisis de fatiga** (`fatigue_analysis.py`): regresión lineal
   longitudinal sobre la variable objetivo, extrayendo la pendiente como
   biomarcador de degradación motora.

## Herramientas de diagnóstico (`tools/`)

Desarrolladas durante la validación del pipeline con datos reales. Útiles
para depurar sesiones nuevas antes de confiar en sus métricas.

| Script | Función |
|---|---|
| `TESTMAGNETO.py` | Audita measurements, field keys y tag keys reales del bucket InfluxDB (sintaxis Flux `schema.*`) |
| `verificar_saturacion_acc.py` | Cuantifica el % de muestras del acelerómetro saturadas en el límite de su rango dinámico, por eje y pie |
| `diagnosticar_eventos.py` | Imprime una "radiografía" de los huecos de detección de eventos > 2s por pie, clasificando la causa probable |
| `plot_deriva.py` | Grafica la presión plantar contra el umbral de detección en una ventana temporal específica |
| `diagnostico_imu.py` | Compara calidad de señal IMU cruda entre pies y realiza barrido de `madgwick_beta` |

Ejemplo de uso (todas aceptan la misma firma de argumentos):

```bash
python A06_ANALISIS_CINEMATICO/tools/verificar_saturacion_acc.py \
    --paciente CODIGO_PACIENTE \
    --inicio "yyyy-mm-dd hh:mm:ss" \
    --fin "yyyy-mm-dd hh:mm:ss"
```

Las herramientas que generan gráficas (`plot_deriva.py`, `diagnostico_imu.py`)
guardan su salida en `RESULTADOS_BIOMECANICO/<paciente>/`.

## Limitación conocida: saturación del acelerómetro (±2g)

La validación con tres pacientes reales confirmó que el acelerómetro de los
sensores SCKS satura sistemáticamente en ±2g (más frecuentemente en el eje de
progresión, `Ay`), introduciendo una subestimación estructural de la longitud
de zancada y la asimetría MTC calculadas por doble integración. Esta
limitación es independiente de la calidad de detección de eventos o de la
presencia de huecos temporales.

Se recomienda ejecutar `verificar_saturacion_acc.py` sobre cualquier sesión
nueva antes de reportar métricas espaciales (zancada, velocidad, MTC) como
definitivas. Las métricas temporales (`stride_times`, `stance_times`,
`swing_times`), al depender solo de la detección de eventos por presión
plantar, no están afectadas por esta limitación.

## Resultados de ejemplo (`RESULTADOS_BIOMECANICO/`)

El repositorio incluye, a modo de evidencia de validación, los resultados
generados durante las pruebas del pipeline sobre 4 sesiones reales (métricas,
diagnósticos de saturación y de eventos). Estos archivos no son necesarios
para ejecutar el pipeline — se regeneran automáticamente al correr
`Orquestador_biomecanico.py` sobre cualquier paciente/rango de fechas nuevo,
sobrescribiendo los existentes para ese mismo `CodeID`.

## Estructura de archivos

```
A06_ANALISIS_CINEMATICO/
├── __init__.py
├── event_detector.py
├── kinematic_engine.py
├── Orquestador_biomecanico.py
├── fatigue_analysis.py
└── tools/
    ├── TESTMAGNETO.py
    ├── verificar_saturacion_acc.py
    ├── diagnosticar_eventos.py
    ├── plot_deriva.py
    └── diagnostico_imu.py
```

---

# Bloque 2 — Pipeline de Datos y Deep Learning (`A04_TRANSFORMER`)

Clasificación binaria de marcha/reposo mediante tres arquitecturas
comparables — Transformer temporal, FFT frecuencial, e Híbrido — evaluadas
con test independiente y, de forma más rigurosa, con validación
Leave-One-Patient-Out (LOPO) real (reentrenamiento completo por fold).

## Opción A — Entrenar modelos desde cero

```bash
python A04_TRANSFORMER/AA_TRANSFORMER_V1.py \
    --dataset "DATASET_LISTO/dataset_jerarquico.hdf5" \
    --out_dir "A05_MODELOS_ENTRENADOS"
```

> **Requisitos de hardware:** el entrenamiento usa GPU (CUDA) de forma
> automática si está disponible (`torch.cuda.is_available()`). Si no hay GPU,
> el código cae automáticamente a CPU sin necesidad de modificar nada, pero el
> entrenamiento de los tres modelos (Temporal, FFT, Híbrido) será
> considerablemente más lento.

Entrena automáticamente: Transformer Temporal, Clasificador FFT, Modelo
Híbrido Early Fusion (Tiempo + Frecuencia); guarda el escalador
(`scaler_gait.joblib`), exporta pesos (`.pth`) y genera métricas/gráficas de
validación.

> **Nota sobre el umbral de clasificación:** el proyecto usa un umbral de
> decisión **fijo de 0.5** (no un umbral óptimo tipo Youden's J), por decisión
> de diseño. El umbral está **hardcodeado en el propio código**
> (`self.threshold = 0.5` en `Agnostic_evaluator.py`), no se carga desde
> ningún archivo `.joblib` — esto evita que un archivo de umbral corrupto o
> sobrescrito accidentalmente altere la clasificación sin que se note.

Archivos necesarios en `A05_MODELOS_ENTRENADOS/` para el evaluador agnóstico
y la evaluación LOPO:
```
modelo_transformer.pth
modelo_fft.pth
modelo_hibrido.pth
meta_clasificador_late_fusion.joblib   (meta-clasificador de Late Fusion; ver más abajo)
scaler_gait.joblib
transformer_config.joblib
train_idx.npy
val_idx.npy
test_idx.npy
y_test.npy
prob_fft.npy
prob_transformer.npy
```

## Opción B — Inferencia con modelos preentrenados (evaluador agnóstico unificado)

Con los modelos ya en `A05_MODELOS_ENTRENADOS`, se puede saltar el
entrenamiento y ejecutar el evaluador agnóstico directamente. **El evaluador
es único y unificado**: permite elegir, en tiempo de ejecución, cuál de los 4
modos de clasificación usar, sin mantener scripts separados por modelo.

```bash
python A04_TRANSFORMER/Agnostic_evaluator.py \
    -c A01_EXTRACCION_DATOS/config.yaml \
    -m A05_MODELOS_ENTRENADOS \
    -r CODIGO_PACIENTE \
    --start "yyyy-mm-ddThh:mm:ssZ" \
    --end "yyyy-mm-ddThh:mm:ssZ"
```

Si se omite `--modelo`, el script pregunta interactivamente por consola:

```
Seleccione el modelo a usar:
  1. FFT
  2. Transformer
  3. Hibrido Early Fusion
  4. Hibrido Late Fusion (Media Geometrica)
Opcion (1-4):
```

También puede pasarse directamente para evitar la pregunta interactiva:

```bash
python A04_TRANSFORMER/Agnostic_evaluator.py \
    -c A01_EXTRACCION_DATOS/config.yaml \
    -m A05_MODELOS_ENTRENADOS \
    -r CODIGO_PACIENTE \
    --start "yyyy-mm-ddThh:mm:ssZ" \
    --end "yyyy-mm-ddThh:mm:ssZ" \
    --modelo 4
```

| Opción | Modelo | Requiere |
|---|---|---|
| `1` / `fft` | FFTModel solo | `modelo_fft.pth` |
| `2` / `transformer` | GaitTransformer solo | `modelo_transformer.pth` |
| `3` / `hibrido_early` | GaitHybridModel (Early Fusion) | `modelo_hibrido.pth` |
| `4` / `hibrido_late` | Late Fusion (Media Geométrica) | `modelo_transformer.pth` + `modelo_fft.pth` (sin `.joblib` adicional; la combinación es una fórmula, no un modelo entrenado) |

El script carga únicamente los artefactos que requiere el modo elegido (no
carga los 3 `.pth` siempre).

> **Nota importante — uso interactivo únicamente:** el modo por defecto
> (sin `--modelo`) usa `input()` para preguntar el modelo por teclado. Esto
> es intencional para uso manual desde consola; **no usar sin `--modelo` en
> contextos automatizados/programados**, ya que se quedaría esperando una
> respuesta que nunca llega.

### Formatos de fecha admitidos

El evaluador detecta automáticamente el formato de `--start`/`--end`:

- **ISO 8601 / UTC** (recomendado): `2025-12-24T10:53:30Z`
- **Tradicional / hora local** (según `tzval` del config): `2025-12-24 10:53:30`

Si recibe errores de formato, verifique que no haya espacios extra y que las
fechas ISO incluyan `T` y `Z`.

### Parámetros del evaluador

| Parámetro | Descripción |
|---|---|
| `-c` | Ruta al archivo `config.yaml` |
| `-m` | Directorio de modelos entrenados |
| `-r` | Identificador del sujeto de estudio |
| `--start` | Inicio del intervalo temporal |
| `--end` | Fin del intervalo temporal |
| `--modelo` | `1`/`fft`, `2`/`transformer`, `3`/`hibrido_early`, `4`/`hibrido_late`. Si se omite, se pregunta interactivamente |
| `-o` | Carpeta de salida (CSV + gráfica); por defecto `A04_TRANSFORMER/RESULTADO_AGNOSTIC/` |

Ejemplo de salida (columnas genéricas, no dependen del modelo elegido):
```csv
timestamp,prob,pred,prob_smoothed,pred_final_smoothed
2025-12-24 10:53:35.866,0.9999752044677734,1,0.9999757289886475,1
```

Los archivos de salida incluyen el nombre del modelo usado
(`agnostico_{paciente}_{modelo}.csv`,
`{paciente}_{modelo}_agnostico_timeline.png`), evitando que corridas con
distintos modelos se sobrescriban entre sí.

## Comparación de modelos — LOPO real (Leave-One-Patient-Out)

A diferencia del test independiente (que en este proyecto satura en
AUC=1.0000 para los tres modelos base, señal de que no discrimina bien entre
alternativas), la validación LOPO real —reentrenando desde cero en cada
fold, dejando un paciente distinto fuera— sí diferencia el desempeño:

| Método | AUC LOPO |
|---|---|
| **FFT (solo)** | **0.9061** |
| Media Geométrica (Late Fusion) | 0.9002 |
| Meta-Clasificador (Late Fusion) | 0.8918 |
| Híbrido (Early Fusion) | 0.8771 |
| Voto Mayoritario (Late Fusion) | 0.8651 |

**Conclusión:** ninguna variante de Late Fusion supera al FFT solo, aunque
Media Geométrica y Meta-Clasificador sí superan al Híbrido (Early Fusion).
Es decir, Late Fusion mejora sobre Early Fusion, pero el mejor modelo
individual (FFT) sigue ganando a cualquier combinación probada. Por eso el
evaluador agnóstico (`Agnostic_evaluator.py`, opción `4`) usa **Media
Geométrica** como variante de Late Fusion (la mejor de las tres), no Voto
Mayoritario ni Meta-Clasificador.

Ejecutar la evaluación LOPO completa (Híbrido + FFT + las 3 variantes de
Late Fusion):

```bash
python A04_TRANSFORMER/EVALUACION_MODELOS.py \
    --dataset "DATASET_LISTO/dataset_jerarquico.hdf5" \
    --models "A05_MODELOS_ENTRENADOS" \
    --run-lopo
```

> **Advertencia de tiempo:** esta corrida reentrena múltiples redes por cada
> paciente dejado fuera (Híbrido, FFT, y Transformer+FFT por separado para
> Late Fusion), por lo que es considerablemente más lenta que un
> entrenamiento único.

### Calcular solo Late Fusion (sin recalcular Híbrido/FFT test independiente)

Si los modelos ya están entrenados y solo se quiere obtener las 3 variantes
de Late Fusion sobre el test independiente (sin reentrenar nada, solo
inferencia):

```bash
python A04_TRANSFORMER/late_fusion.py --models-dir A05_MODELOS_ENTRENADOS
```

Reconstruye `x_test`/`y_test` desde el HDF5 + `test_idx.npy`, corre
inferencia con los modelos ya entrenados, calcula Media Geométrica, Voto
Mayoritario y Meta-Clasificador, y guarda el meta-clasificador final
entrenado (`meta_clasificador_late_fusion.joblib`) en
`A05_MODELOS_ENTRENADOS/LATE_FUSION/`.

> **Nota metodológica:** el meta-clasificador final guardado se entrena
> sobre el mismo test set usado para reportar sus métricas (único conjunto
> etiquetado disponible fuera de train/val). Las métricas de
> "Meta-Clasificador" en el reporte LOPO sí son honestas (out-of-fold), pero
> el `.joblib` de producción no tiene la misma garantía de generalización.

## Estructura de archivos

```
A04_TRANSFORMER/
├── __init__.py
├── AA_TRANSFORMER_V1.py
├── Agnostic_evaluator.py
├── EVALUACION_MODELOS.py
├── late_fusion.py
└── test_pipeline.py
```

| Script | Función |
|---|---|
| `AA_TRANSFORMER_V1.py` | Entrenamiento de los tres modelos (Temporal, FFT, Híbrido Early Fusion) a partir del dataset HDF5; guarda pesos, escalador y config |
| `Agnostic_evaluator.py` | Evaluador agnóstico **unificado**: inferencia continua sobre un paciente/rango temporal, con selección de modelo (FFT / Transformer / Híbrido Early / Híbrido Late) |
| `EVALUACION_MODELOS.py` | Evaluación de test independiente + LOPO real (`run_lopo_evaluation`, `run_lopo_evaluation_fft`, `run_lopo_late_fusion`) para los 3 modelos base y las 3 variantes de Late Fusion |
| `late_fusion.py` | Calcula las 3 variantes de Late Fusion (Media Geométrica, Voto Mayoritario, Meta-Clasificador) sobre el test independiente, sin reentrenar; guarda el meta-clasificador de producción |
| `test_pipeline.py` | Tests unitarios del pipeline de Deep Learning |

## Arquitectura del modelo

- **Rama Temporal (Transformer Encoder)**: captura dependencias secuenciales
  de largo alcance en señales de acelerómetro y giroscopio, mediante
  Multi-Head Attention y Positional Encoding.
- **Rama Frecuencial (FFT + MLP)**: transforma ventanas temporales mediante
  RFFT (Real Fast Fourier Transform) para aislar micro-frecuencias
  biomecánicas generadas por impactos instantáneos de la marcha (basado en
  Müller et al., 2021).
- **Modelo Híbrido — Early Fusion**: concatena los espacios latentes de ambas
  ramas antes del cabezal de clasificación final.
- **Late Fusion**: entrena Transformer y FFT de forma completamente
  independiente, y combina sus probabilidades finales (no sus features
  intermedias) mediante media geométrica, voto mayoritario, o un
  meta-clasificador (regresión logística). Ver comparación LOPO arriba.

Validación LOPO mediante `LeaveOneGroupOut`, garantizando separación
estricta por paciente y evaluación clínica realista.

---

# Bloque 3 — Exploración de trayectoria GPS+Transformer (`A07_TRAYECTORIA_GPS`)

Investigación de una propuesta del director del TFM: usar los sensores de
presión y GPS, junto con la misma arquitectura Transformer de A04 (mediante
fine-tuning), para calcular trayectoria (posición X,Y) del cuerpo, como
alternativa al motor Madgwick+ZUPT de A06.

> **Conclusión: no viable con los datos actuales.** El GPS de los calcetines
> es la única fuente de posición disponible (confirmado con un inventario
> exhaustivo de todos los campos/tags de InfluxDB — no existe UWB, RTK, ni
> ningún otro sensor de posicionamiento), con lecturas reales espaciadas
> entre 7 y 80 segundos entre sí. El error final de trayectoria se mantuvo
> en el orden de 20-60 metros según la variante probada, muy lejos de ser
> utilizable. **Se mantiene A06 (Madgwick+ZUPT) como única fuente de
> trayectoria/velocidad/zancada/MTC del proyecto.** Este bloque se conserva
> como evidencia documentada del proceso de investigación, no como pipeline
> de producción. Ver `A07_TRAYECTORIA_GPS/README_A07.md` para el detalle
> completo de la metodología, decisiones y resultados.

## Estructura de archivos

```
A07_TRAYECTORIA_GPS/
├── __init__.py
├── README_A07.md
├── preparar_dataset_trayectoria.py
├── conector_trayectoria.py
├── trajectory_model.py
├── entrenar_trajectory_model.py
├── entrenar_trajectory_model_multi.py
├── analizar_segmento_trayectoria.py
├── verificar_gps_por_pie.py
├── inventario_influxdb.py
└── RESULTADOS_TRAYECTORIA/
```

---

## Compatibilidad y reproducibilidad

- Empaquetado editable moderno (`pyproject.toml` / `setup.py`)
- Resolución agnóstica de rutas (`Path(__file__)`), sin rutas absolutas locales
- Compatibilidad equivalente entre `python -m ...` y `python script.py`
- Ejecución reproducible tras clonado limpio, multiplataforma

## Tecnologías utilizadas

| Categoría | Herramientas |
|---|---|
| Deep Learning | PyTorch, NumPy, SciPy |
| Ingeniería | Pydantic, Pytest, Logging, Type Hinting, Pandas, Matplotlib, Openpyxl, Pyarrow |
| Biomecánica | IMUs, FFT, Sensor Fusion, ZUPT, AHRS, Scikit-learn |
| Geoespacial | pyproj (proyección UTM), scipy.interpolate (PCHIP) — solo en A07 |

El proyecto sigue PEP8, tipado estricto, programación orientada a objetos,
modularidad y testing unitario.

---

## Roadmap

**Completado**
- [x] Pipeline de extracción de datos
- [x] Pipeline de reconstrucción biomecánica y clasificación de marcha
- [x] Arquitectura Tiempo, Frecuencia e Híbrida (Early Fusion)
- [x] Reconstrucción cinemática 3D (IMU + presión plantar)
- [x] Corrección de deriva inercial (ZUPT 3D, compensación gravitacional, orientación global)
- [x] Extracción de métricas biomecánicas espaciales y temporales
- [x] Evaluación agnóstica continua (unificada, 4 modos de modelo)
- [x] Validación LOPO real (Híbrido, FFT, y 3 variantes de Late Fusion)
- [x] Exploración de trayectoria GPS+Transformer (concluida, no viable)
- [x] Tests unitarios (pytest)
- [x] Compatibilidad multiplataforma mediante empaquetado editable
- [x] Refactorización completa PEP8

**Próximas fases**
- [ ] Regresor clínico EDSS
- [ ] Integración TabTransformer
- [ ] MLP-Mixer clínico
- [ ] Fusión con variables cognitivas
- [ ] Aumento del volumen de muestras de entrenamiento

---

## Autor

```
Jairo Eduardo Paez Leal
Máster en Ingeniería de Organización
Escuela Técnica Superior de Ingenieros Industriales
Universidad Politécnica de Madrid (UPM)
```

```
Joaquín Bienvenido Ordieres Meré
Tutor
Escuela Técnica Superior de Ingenieros Industriales
Universidad Politécnica de Madrid (UPM)
```
