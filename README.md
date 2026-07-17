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

El sistema se organiza en dos bloques independientes:

1. **Pipeline biomecánico (`A06_ANALISIS_CINEMATICO`)** — extracción, detección
   de eventos de marcha, reconstrucción cinemática 3D y biomarcadores
   espaciotemporales/fatiga. No depende del bloque de Deep Learning.
2. **Pipeline de Deep Learning (`A04_TRANSFORMER`)** — clasificación de marcha
   mediante Transformer + FFT sobre espectrogramas híbridos generados en
   `A01_EXTRACCION_DATOS` / `A03_PREPROCESAMIENTO`.

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

Si se prefiere clonar el repositorio sin descargar estos archivos al disco
(por ejemplo, para no traer datos de pacientes de ejemplo), puede usarse un
sparse checkout:

```bash
git clone --filter=blob:none --no-checkout https://github.com/MultipleSclerosisMonitoring/2026_DLxGait-EDSS.git
cd 2026_DLxGait-EDSS
git sparse-checkout init --cone
git sparse-checkout set --skip-checks '/*' '!A06_ANALISIS_CINEMATICO/RESULTADOS_BIOMECANICO'
git checkout
```

Esto trae el historial completo del repositorio (los archivos siguen
disponibles para consulta en GitHub y recuperables localmente con
`git sparse-checkout disable` en cualquier momento), pero no materializa en
disco el contenido de `RESULTADOS_BIOMECANICO/` durante el checkout.

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

# Bloque 2 — Pipeline de Datos y Deep Learning

## Extracción y preprocesamiento

```bash
# 1. Extraccion de datos (genera tensores .parquet)
python A01_EXTRACCION_DATOS/extract_data_plus.py \
    --backend influxdbms \
    -c A01_EXTRACCION_DATOS/config.yaml \
    -e A01_EXTRACCION_DATOS/solicitud.xlsx

# 2. Limpieza y empaquetado (genera dataset_jerarquico.hdf5)
python A03_PREPROCESAMIENTO/LIMPIEZA.py \
    --input resultados \
    --output DATASET_LISTO \
    --excel A01_EXTRACCION_DATOS/solicitud.xlsx
```

## Opción A — Entrenar modelos desde cero

```bash
python A04_TRANSFORMER/AA_TRANSFORMER_V1.py \
    --dataset "DATASET_LISTO/dataset_jerarquico.hdf5" \
    --out_dir "A05_MODELOS_ENTRENADOS"
```

> **Requisitos de hardware:** el entrenamiento usa GPU (CUDA) de forma
> automática si está disponible (`torch.cuda.is_available()`), incluyendo
> mixed precision (AMP) para acelerar. Si no hay GPU, el código cae
> automáticamente a CPU sin necesidad de modificar nada — pero el
> entrenamiento de los tres modelos (Temporal, FFT, Híbrido) será
> considerablemente más lento. No se ha determinado un requisito mínimo de
> VRAM ni un tiempo de entrenamiento de referencia; quedan pendientes de
> medición empírica.

Entrena automáticamente: Transformer Temporal, Clasificador FFT, Modelo
Híbrido Tiempo + Frecuencia; guarda el escalador (`scaler_gait.joblib`),
exporta pesos (`.pth`) y genera métricas/gráficas de validación.

> **Nota sobre el umbral de clasificación:** `Agnostic_evaluator.py` no
> carga un umbral óptimo desde disco — usa un valor fijo definido en el
> propio script. Por eso `optimal_threshold_hibrido.joblib` no forma parte
> de los artefactos necesarios para la inferencia agnóstica, aunque
> versiones anteriores de este documento lo mencionaran como tal.

Archivos generados en `A05_MODELOS_ENTRENADOS/` (confirmado contra una
ejecución real):
```
modelo_transformer.pth
modelo_fft.pth
modelo_hibrido.pth
scaler_gait.joblib
transformer_config.joblib
train_idx.npy
val_idx.npy
test_idx.npy
ANALISIS_MODELOS/
LOPO/
```

> Las subcarpetas `ANALISIS_MODELOS/` y `LOPO/` corresponden a la evaluación
> y a la prueba de estrés LOPO (`run_lopo_stress_test`), generadas por
> `EVALUACION_MODELOS.py`, no por `AA_TRANSFORMER_V1.py` directamente.

## Opción B — Inferencia con modelos preentrenados

Con los modelos y escaladores ya en `A05_MODELOS_ENTRENADOS`, se puede saltar
el entrenamiento y ejecutar el evaluador agnóstico directamente:

```bash
python A04_TRANSFORMER/Agnostic_evaluator.py \
    -c A01_EXTRACCION_DATOS/config.yaml \
    -m A05_MODELOS_ENTRENADOS \
    -r CODIGO_PACIENTE \
    --start "yyyy-mm-ddThh:mm:ssZ" \
    --end "yyyy-mm-ddThh:mm:ssZ"
```

### Formatos de fecha admitidos

El evaluador detecta automáticamente el formato de `--start`/`--end`:

- **ISO 8601** (recomendado): `2025-12-24T10:53:30Z`
- **Tradicional**: `2025-12-24 10:53:30`

Si recibe errores de formato, verifique que no haya espacios extra y que las
fechas ISO incluyan `T` y `Z` (ej. `2025-12-24T10:53:30Z`). El sistema intenta
parseo ISO 8601 primero, con fallback automático al formato tradicional.

### Parámetros del evaluador

| Parámetro | Descripción |
|---|---|
| `-c` | Ruta al archivo `config.yaml` |
| `-m` | Directorio de modelos entrenados |
| `-r` | Identificador del sujeto de estudio (solo el código, sin ruta a Excel) |
| `--start` | Inicio del intervalo temporal |
| `--end` | Fin del intervalo temporal |
| `-o` | Archivo CSV de salida |

Ejemplo de salida:
```csv
timestamp,prob_hybrid,pred_hybrid,prob_smoothed,pred_final_smoothed
2025-12-24 10:53:35.866,0.9999752044677734,1,0.9999757289886475,1
```

## Ejecución como módulo

```bash
python -m A04_TRANSFORMER.Agnostic_evaluator
# equivalente a:
python A04_TRANSFORMER/Agnostic_evaluator.py
```

## Estructura de archivos

```
A04_TRANSFORMER/
├── __init__.py
├── AA_TRANSFORMER_V1.py
├── Agnostic_evaluator.py
├── EVALUACION_MODELOS.py
└── test_pipeline.py
```

| Script | Función |
|---|---|
| `AA_TRANSFORMER_V1.py` | Entrenamiento de los tres modelos (Temporal, FFT, Híbrido) a partir del dataset HDF5; guarda pesos, escalador y config |
| `Agnostic_evaluator.py` | Inferencia continua sobre un paciente/rango temporal usando los modelos ya entrenados |
| `EVALUACION_MODELOS.py` | Evaluación y validación de los modelos entrenados, incluyendo la prueba de estrés LOPO (`run_lopo_stress_test`) y generación de métricas/gráficas de validación |
| `test_pipeline.py` | Tests unitarios del pipeline de Deep Learning |

## Arquitectura del modelo

- **Rama Temporal (Transformer Encoder)**: captura dependencias secuenciales
  de largo alcance en señales de acelerómetro y giroscopio, mediante
  Multi-Head Attention y Positional Encoding.
- **Rama Frecuencial (FFT + MLP)**: transforma ventanas temporales mediante
  RFFT (Real Fast Fourier Transform) para aislar micro-frecuencias
  biomecánicas generadas por impactos instantáneos de la marcha (basado en
  Müller et al., 2021).
- **Modelo Híbrido (Early Fusion)**: concatena los espacios latentes de ambas
  ramas antes del cabezal de clasificación final, mejorando robustez,
  generalización y reducción de ruido biomecánico.

Validación inter-sujeto mediante `StratifiedGroupKFold`, garantizando
separación estricta por paciente y evaluación clínica realista.

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

El proyecto sigue PEP8, tipado estricto, programación orientada a objetos,
modularidad y testing unitario.

---

## Roadmap

**Completado**
- [x] Pipeline de extracción de datos
- [x] Pipeline de reconstrucción biomecánica y clasificación de marcha
- [x] Arquitectura Tiempo, Frecuencia e Híbrida
- [x] Reconstrucción cinemática 3D (IMU + presión plantar)
- [x] Corrección de deriva inercial (ZUPT 3D, compensación gravitacional, orientación global)
- [x] Extracción de métricas biomecánicas espaciales y temporales
- [x] Evaluación agnóstica continua
- [x] Tests unitarios (pytest)
- [x] Compatibilidad multiplataforma mediante empaquetado editable
- [x] Refactorización completa PEP8

**Próximas fases**
- [ ] Regresor clínico EDSS
- [ ] Integración TabTransformer
- [ ] MLP-Mixer clínico
- [ ] Fusión con variables cognitivas

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
