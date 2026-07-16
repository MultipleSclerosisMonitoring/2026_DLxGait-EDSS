# ESTIMACIÓN DE DETERIORO EN ESCLEROSIS MÚLTIPLE

## Análisis biomecánico mediante calcetines inteligentes (SCKS)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![PEP8](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)

---

# Descripción

Este repositorio contiene un pipeline robusto de extracción, reconstrucción biomecánica y clasificación de marcha para datos obtenidos mediante calcetines inteligentes (SCKS) en pacientes con Esclerosis Múltiple (EM).

El sistema implementa:

- Extracción de señales desde InfluxDB
- Preprocesamiento biomecánico
- Reconstrucción cinemática basada en IMUs
- Detección de eventos de marcha
- Modelado profundo mediante PyTorch
- Inferencia híbrida Tiempo + Frecuencia
- Inferencia continua sobre flujos biomecánicos
- Generación automática de espectrogramas híbridos
- Clasificación multimodal basada en Transformer + FFT

---

# Instalación

## 1. Clonar repositorio

```bash
git clone https://github.com/MultipleSclerosisMonitoring/2026_DLxGait-EDSS.git
cd 2026_DLxGait-EDSS
```

---

## 2. Crear entorno virtual

```bash
conda create -n tfm python=3.10
conda activate tfm
```

---

## 3. Instalar dependencias

### Instalación editable (recomendada)

```bash
pip install -e .
```

### Alternativamente

```bash
pip install -r requirements.txt
```

---

# Configuración de InfluxDB

El archivo `config.yaml` no contiene credenciales reales.

Antes de ejecutar cualquier pipeline, edite el archivo:

```text
A01_EXTRACCION_DATOS/config.yaml
```

Ejemplo:

```yaml
influxdb:
  url: "https://YOUR_SERVER:8086/"
  org: "YOUR_ORG"
  bucket: "YOUR_BUCKET"
  token: "YOUR_TOKEN"
  tzval: "Europe/Madrid"
```
---
# Pipeline de Datos (Extracción y Preprocesamiento)

Antes de realizar la inferencia o entrenar nuevos modelos, es necesario generar el dataset estructurado a partir de los registros de la base de datos.

## 1. Extracción de Datos

Este comando lee los intervalos temporales del archivo Excel de metadatos, extrae las señales desde InfluxDB y genera tensores en formato `.parquet` en el directorio de resultados.

```bash
python A01_EXTRACCION_DATOS/extract_data_plus.py \
    --backend influxdbms \
    -c A01_EXTRACCION_DATOS/config.yaml \
    -e A01_EXTRACCION_DATOS/solicitud.xlsx
```

## 2. Limpieza y Empaquetado (HDF5)

Este comando toma los archivos `.parquet` generados, aplica las ventanas deslizantes, cruza los identificadores y empaqueta todo en un dataset jerárquico balanceado (`dataset_jerarquico.hdf5`).

```bash
python A03_PREPROCESAMIENTO/LIMPIEZA.py \
    --input resultados \
    --output DATASET_LISTO \
    --excel A01_EXTRACCION_DATOS/solicitud.xlsx
```

---

# Flujo de Ejecución 

El pipeline permite dos opciones de flujos de trabajo principales, dependiendo de si se desea probar el entrenamiento de las arquitecturas o solo evaluar pacientes con el sistema ya consolidado.

## Opción A — Entrenar modelos desde cero
Genera nuevos artefactos (`.pth`, escaladores, umbrales) a partir del dataset procesado:

```bash
python A04_TRANSFORMER\AA_TRANSFORMER_V1.py ^
    --dataset "DATASET_LISTO\dataset_jerarquico.hdf5" ^
    --out_dir "A05_MODELOS_ENTRENADOS"
```

El proceso entrena de forma automática:

- Transformer Temporal
- Clasificador FFT
- Modelo Híbrido Tiempo + Frecuencia
- Cálculo automático de umbral óptimo
- Guardado de escalador (`scaler_gait.joblib`)
- Exportación de pesos (`.pth`)
- Generación de métricas y gráficas de validación

## Archivos generados

Tras finalizar el entrenamiento se crearán automáticamente:

```text
A05_MODELOS_ENTRENADOS/
│
├── modelo_transformer.pth
├── modelo_fft.pth
├── modelo_hibrido.pth
├── scaler_gait.joblib
├── optimal_threshold_hibrido.joblib
├── transformer_config.joblib
│
└── graficas/
    ├── MODELO_TRANSFORMER_Matriz_Confusion.png
    ├── MODELO_FFT_Matriz_Confusion.png
    └── MODELO_HIBRIDO_FINAL_Matriz_Confusion.png
```

## Opción B — Inferencia con modelos preentrenados
En la carpeta A05_MODELOS_ENTRENADOS están los modelos y escaladores, se puede saltar el entrenamiento y ejecutar el evaluador agnóstico directamente sobre el flujo continuo:

```bash
python A04_TRANSFORMER\Agnostic_evaluator.py ^
    -c A01_EXTRACCION_DATOS\config.yaml ^
    -m A05_MODELOS_ENTRENADOS ^
    -r CODIGO_PACIENTE ^
    --start "yyyy-mm-ddThh:mm:ssZ" ^
    --end "yyyy-mm-ddThh:mm:ssZ"
```
# A06_ANALISIS_CINEMATICO — Pipeline Biomecanico Bilateral

Modulo de reconstruccion cinematica 3D y calculo de metricas espaciotemporales
de marcha bilateral, a partir de senales inerciales (IMU) y presion plantar
adquiridas mediante calcetines inteligentes (SCKS). Este pipeline es previo
e independiente del modelo de Deep Learning (`A04_TRANSFORMER`); genera
biomarcadores biomecanicos que pueden integrarse posteriormente en modelos
clinicos predictivos.

## Entrypoint oficial

El unico orquestador soportado es:

```
Orquestador_biomecanico.py
```

> Nota de migracion: una version previa y mas simple (`run_gait_pipeline.py`)
> existio en este directorio pero fue descartada. No implementaba
> auto-calibracion de umbral, fusion con magnetometro, segmentacion por
> tramos continuos ni validaciones de calidad de senal, y solo procesaba
> el pie derecho. Toda la documentacion y comandos de este README se
> refieren exclusivamente a `Orquestador_biomecanico.py`.

## Ejecucion

```bash
python A06_ANALISIS_CINEMATICO/Orquestador_biomecanico.py \
    --paciente CODIGO_PACIENTE \
    --inicio "yyyy-mm-dd hh:mm:ss" \
    --fin "yyyy-mm-dd hh:mm:ss" \
    --config-yaml A01_EXTRACCION_DATOS/config.yaml
```

### Parametros opcionales

| Parametro | Default | Descripcion |
|---|---|---|
| `--fs` | 100 | Frecuencia de muestreo (Hz) |
| `--fatigue-target` | `Gait_Speed_ms` | Variable sobre la que se calcula la pendiente de fatiga |
| `--max-time-diff` | 0.20 | Tolerancia (s) para emparejar zancadas bilaterales |
| `--th-right` / `--th-left` | None | Umbral manual de deteccion de eventos por pie (si se omite, se auto-calibra) |
| `--sin-auto-calibrar` | False | Desactiva la auto-calibracion de umbral |
| `--max-stride` | 1.9 | Longitud de zancada maxima plausible (m), para filtrar outliers |

## Flujo del pipeline

1. **Extraccion** (`extract_data_plus.py`, en `A01_EXTRACCION_DATOS`): consulta
   InfluxDB (Flux) filtrando por `CodeID`, `Foot`, `type=SCKS` y rango temporal.
2. **Deteccion de eventos** (`event_detector.py`): Heel Strike / Toe Off a
   partir de presion plantar con umbral adaptativo e histeresis temporal.
3. **Auto-calibracion de umbral**: barrido de fracciones de umbral por pie,
   seleccionando la que produce un %stance fisiologico (45-70%).
4. **Segmentacion por tramos continuos**: si hay huecos > 2s entre eventos
   consecutivos (pausas, giros, perdida de deteccion), el registro se corta
   en tramos y cada uno se procesa de forma aislada, evitando que la
   integracion arrastre deriva de un tramo a otro.
5. **Reconstruccion cinematica** (`kinematic_engine.py`, por tramo): fusion
   sensorial (Madgwick, con magnetometro si esta disponible en los datos),
   compensacion gravitacional, ZUPT (Zero Velocity Update) e integracion de
   posicion 3D.
6. **Emparejamiento bilateral y asimetria**: cruza zancadas de ambos pies por
   proximidad temporal y calcula asimetria de velocidad, zancada y MTC.
7. **Analisis de fatiga** (`fatigue_analysis.py`): regresion lineal
   longitudinal sobre la variable objetivo, extrayendo la pendiente como
   biomarcador de degradacion motora.

## Herramientas de diagnostico (`tools/`)

Estas herramientas se desarrollaron durante la validacion del pipeline con
datos reales (ver `docs/MEMORIA_EXPLORACION_PIPELINE_BIOMECANICO.md` para el
detalle completo del proceso y los hallazgos). Son utiles para depurar
sesiones nuevas antes de confiar en sus metricas:

- **`TESTMAGNETO.py`**: audita measurements, field keys y tag keys reales
  del bucket InfluxDB (usa sintaxis Flux `schema.*`, no InfluxQL `SHOW`).
- **`verificar_saturacion_acc.py`**: cuantifica el % de muestras del
  acelerometro saturadas en el limite de su rango dinamico, por eje y pie.
  Uso:
  ```bash
  python A06_ANALISIS_CINEMATICO/tools/verificar_saturacion_acc.py \
      --paciente CODIGO_PACIENTE \
      --inicio "yyyy-mm-dd hh:mm:ss" \
      --fin "yyyy-mm-dd hh:mm:ss"
  ```
- **`diagnosticar_eventos.py`**: subclase del orquestador que imprime una
  "radiografia" de los huecos de deteccion de eventos > 2s por pie,
  clasificando la causa probable (umbral insuficiente vs filtro/refractario).
- **`plot_deriva.py`**: grafica la senal de presion plantar contra el umbral
  de deteccion calculado, en una ventana temporal especifica, para
  inspeccion visual. Guarda el resultado en
  `RESULTADOS_BIOMECANICO/<paciente>/`.
- **`diagnostico_imu.py`**: compara la calidad de la senal IMU cruda entre
  pie derecho e izquierdo (norma de aceleracion, dispersion en stance,
  divergencia de Madgwick a lo largo del tiempo) y realiza un barrido de
  `madgwick_beta` buscando el valor que minimiza la dispersion de la
  componente Z global durante los tramos de apoyo. Uso:
  ```bash
  python A06_ANALISIS_CINEMATICO/tools/diagnostico_imu.py \
      --paciente CODIGO_PACIENTE \
      --inicio "yyyy-mm-dd hh:mm:ss" \
      --fin "yyyy-mm-dd hh:mm:ss"
  ```
  Guarda sus graficas en `RESULTADOS_BIOMECANICO/<paciente>/`.

## Limitacion conocida: saturacion del acelerometro (+-2g)

La validacion con tres pacientes reales confirmo que el acelerometro de los
sensores SCKS satura sistematicamente en +-2g (mas frecuentemente en el eje
de progresion, `Ay`), lo cual introduce una subestimacion estructural de la
longitud de zancada y la asimetria MTC calculadas por doble integracion.
Esta limitacion es independiente de la calidad de deteccion de eventos o de
la presencia de huecos temporales (confirmado con un paciente que no
presento ninguno de esos dos problemas). Ver la memoria en `docs/` para el
detalle completo, incluyendo tabla comparativa de saturacion entre
pacientes.

Se recomienda ejecutar `verificar_saturacion_acc.py` sobre cualquier sesion
nueva antes de reportar metricas espaciales (zancada, velocidad, MTC) como
definitivas. Las metricas temporales (`stride_times`, `stance_times`,
`swing_times`), al depender solo de la deteccion de eventos por presion
plantar, no estan afectadas por esta limitacion.

## Estructura de archivos

```
A06_ANALISIS_CINEMATICO/
├── __init__.py
├── README.md
├── event_detector.py
├── kinematic_engine.py
├── Orquestador_biomecanico.py
├── fatigue_analysis.py
├── tools/
│   ├── TESTMAGNETO.py
│   ├── verificar_saturacion_acc.py
│   ├── diagnosticar_eventos.py
│   ├── plot_deriva.py
│   └── diagnostico_imu.py
└── docs/
    └── MEMORIA_EXPLORACION_PIPELINE_BIOMECANICO.md
```

---

# Ejecución

Gracias al empaquetado mediante `setup.py`, el proyecto puede ejecutarse desde cualquier directorio.

## Opción 1 — Ejecución como módulo

```bash
python -m A04_TRANSFORMER.Agnostic_evaluator
```

## Opción 2 — Ejecución mediante ruta relativa

```bash
python A04_TRANSFORMER/Agnostic_evaluator.py
```

---

# Compatibilidad y reproducibilidad

Para garantizar compatibilidad entre distintos entornos de ejecución y evitar dependencias implícitas de rutas locales se incorporaron las siguientes medidas:

- Empaquetado editable moderno mediante pyproject.toml y setup.py
- Resolución agnóstica de rutas (`Path(__file__)`)
- Compatibilidad equivalente entre:

```bash
python -m ...
```

y

```bash
python script.py
```

- Eliminación de dependencias de rutas absolutas locales
- Mejora de portabilidad para clonación limpia del repositorio
- Compatibilidad multiplataforma
- Ejecución reproducible tras clonado limpio

Estas modificaciones permiten ejecutar el pipeline correctamente en entornos nuevos tras un clonado limpio del repositorio.

---

# Manual de Operaciones — Inferencia Continua (Agnostic Evaluator)

## Descripción

Módulo encargado de realizar inferencias continuas sobre flujos de datos biomecánicos provenientes de InfluxDB.

El sistema:

1. Recupera señales biomecánicas desde InfluxDB
2. Alinea ambas extremidades
3. Genera espectrogramas híbridos
4. Ejecuta inferencia continua
5. Exporta probabilidades y predicciones en CSV

---

## Cambios recientes

### Flexibilidad en formatos de fecha

El evaluador admite de forma nativa dos formatos para los argumentos:

```text
--start
--end
```

### Formato ISO 8601 (recomendado para InfluxDB)

```text
YYYY-MM-DDTHH:MM:SSZ
```

Ejemplo:

```text
2025-12-24T10:53:30Z
```

### Formato tradicional

```text
YYYY-MM-DD HH:MM:SS
```

Ejemplo:

```text
2025-12-24 10:53:30
```

El sistema detecta automáticamente el formato proporcionado.

---

### Simplificación del argumento `-r`

El parámetro:

```bash
-r / --reference
```

ahora recibe únicamente el identificador del paciente/segmento:

```text
CODIGO_PACIENTE
```

eliminando la necesidad de especificar rutas completas hacia archivos Excel.

---

# Ejemplo de inferencia agnóstica

## Ejemplo usando formato ISO 8601

```bash
python A04_TRANSFORMER\Agnostic_evaluator.py ^
-c A01_EXTRACCION_DATOS\config.yaml ^
-m A05_MODELOS_ENTRENADOS ^
-r CODIGO_PACIENTE ^
--start "2025-12-24T10:53:30Z" ^
--end "2025-12-24T10:58:18Z"
```

## Ejemplo usando formato tradicional

```bash
python A04_TRANSFORMER\Agnostic_evaluator.py ^
-c A01_EXTRACCION_DATOS\config.yaml ^
-m A05_MODELOS_ENTRENADOS ^
-r CODIGO_PACIENTE ^
--start "2025-12-24 10:53:30" ^
--end "2025-12-24 10:58:18"
```

---

## Parámetros

| Parámetro | Descripción |
|---|---|
| `-c` | Ruta al archivo `config.yaml` |
| `-m` | Directorio de modelos entrenados |
| `-r` | Identificador del sujeto de estudio |
| `--start` | Inicio del intervalo temporal |
| `--end` | Fin del intervalo temporal |
| `-o` | Archivo CSV de salida |

---

## Ejemplo de salida

```csv
timestamp,prob_hybrid,pred_hybrid,prob_smoothed,pred_final_smoothed
2025-12-24 10:53:35.866,0.9999752044677734,1,0.9999757289886475,1
```

---

# Pipeline de Reconstrucción Cinemática

El pipeline biomecánico puede ejecutarse mediante formato de fecha tradicional o ISO 8601:

```bash TRADICIONAL
python run_pipeline.py ^
--paciente CODIGO_PACIENTE ^
--inicio "yyyy-mm-dd hh:mm:ss" ^
--fin "yyyy-mm-dd hh:mm:ss" ^
--out resultados.csv
```

```bash ISO
python run_pipeline.py ^
--paciente CODIGO_PACIENTE ^
--inicio "yyyy-mm-ddThh:mm:ssZ" ^
--fin "yyyy-mm-ddThh:mm:ssZ" ^
--out resultados.csv
```

El sistema:

1. Extrae señales IMU y presión plantar desde InfluxDB
2. Detecta eventos Heel Strike y Toe Off
3. Reconstruye trayectoria espacial 3D
4. Calcula métricas biomecánicas
5. Exporta resultados en CSV

---

# Arquitectura

## Pipeline Biomecánico Inercial

El repositorio incorpora un pipeline biomecánico tridimensional basado en IMUs para reconstrucción espacial de la marcha.

El sistema implementa:

- Sensor fusion mediante Madgwick Filter
- Compensación gravitacional
- Rotación al marco global
- Integración inercial XYZ
- Zero Velocity Update (ZUPT)
- Extracción de métricas espaciotemporales

Las métricas obtenidas incluyen:

- Stride Length
- Gait Speed
- Stride Time
- Stance Time
- Swing Time
- Minimum Toe Clearance (MTC)

### Bloque de Procesamiento Inercial

Previo al modelo Deep Learning, el sistema realiza una reconstrucción biomecánica completa utilizando señales inerciales provenientes de los calcetines inteligentes.

Este bloque genera biomarcadores espaciales y temporales robustos que posteriormente pueden integrarse en modelos clínicos predictivos.

---

## 1. Rama Temporal — Transformer Encoder

Captura dependencias secuenciales de largo alcance en señales IMU:

- Acelerómetro
- Giroscopio

### Características

- Multi-Head Attention
- Positional Encoding
- Aprendizaje temporal profundo
- Detección de patrones anómalos de marcha

---

## 2. Rama Frecuencial — FFT + MLP

Transforma ventanas temporales mediante:

```text
RFFT (Real Fast Fourier Transform)
```

### Objetivo

Aislar micro-frecuencias biomecánicas generadas por impactos instantáneos de la marcha.

### Fundamentación biomecánica

Basado en:

```text
Müller et al., 2021
```

---

## 3. Modelo Híbrido — Early Fusion

Concatena espacios latentes de:

- Rama Temporal
- Rama Frecuencial

antes del cabezal de clasificación final.

### Ventajas

- Mayor robustez
- Mejor generalización
- Inferencia multimodal
- Reducción del ruido biomecánico

---

# Validación

La validación inter-sujeto se implementa mediante:

```python
StratifiedGroupKFold
```

Esto garantiza:

- Separación estricta por paciente
- Prevención de leakage entre sujetos
- Evaluación clínica realista

---

# Pipeline Biomecánico (A06)

## Detección de Eventos

- Heel Strike (HS)
- Toe Off (TO)

mediante:

- Presión plantar
- Giroscopio sagital
- Umbral adaptativo
- Histéresis temporal

---

## Reconstrucción Cinemática

Incluye:

- Sensor Fusion (Madgwick)
- Rotación al marco global
- Compensación gravitacional
- ZUPT tridimensional
- Integración XYZ
- Estimación de trayectoria espacial

---

# Tecnologías Utilizadas

## Deep Learning

- PyTorch
- NumPy
- SciPy

## Ingeniería

- Pydantic
- Pytest
- Logging
- Type Hinting
- Pandas
- Matplotlib
- Openpyxl
- Pyarrow

## Biomecánica

- IMUs
- FFT
- Sensor Fusion
- ZUPT
- AHRS
- Scikit-learn

---

# Calidad del Código

El proyecto sigue:

- PEP8
- Tipado estricto
- Programación orientada a objetos
- Modularidad
- Testing unitario

---

# Troubleshooting

## Error de formato de fecha

Si recibe errores relacionados con fechas:

- Verifique que no existan espacios extra
- Compruebe que las fechas ISO incluyan:

```text
T
```

y

```text
Z
```

Ejemplo correcto:

```text
2025-12-24T10:53:30Z
```

El sistema intentará:

1. Parseo ISO 8601
2. Fallback automático al formato tradicional

---

# Roadmap

## Completado

- [x] Pipeline extracción de datos
- [x] Pipeline robusto de extracción, reconstrucción biomecánica y clasificación de marcha
- [x] Arquitectura Tiempo, Frecuencia e Híbrida
- [x] Pipeline de reconstrucción cinemática 3D mediante integración de señales IMU y presión plantar
- [x] Corrección de deriva inercial mediante ZUPT tridimensional, compensación gravitacional y orientación global
- [x] Extracción de métricas biomecánicas espaciales y temporales
- [x] Evaluación Agnóstica
- [x] Tests unitarios (pytest)
- [x] Compatibilidad multiplataforma mediante empaquetado editable
- [x] Ejecución reproducible tras clonación limpia del repositorio
- [x] Refactorización completa PEP8

---

## Próximas Fases

- [ ] Regresor Clínico EDSS
- [ ] Integración TabTransformer
- [ ] MLP-Mixer clínico
- [ ] Fusión con variables cognitivas

---

# Objetivo Clínico Final

Utilizar biomarcadores biomecánicos derivados de la marcha para estimar automáticamente:

```text
EDSS (Expanded Disability Status Scale)
```

en pacientes con **Esclerosis Múltiple**, integrando:

- Señales IMU
- Presión plantar
- Variables demográficas
- Variables cognitivas

---

# Autor

```text
Jairo Eduardo Paez Leal
Máster en Ingeniería de Organización
Escuela Técnica Superior de Ingenieros Industriales
Universidad Politécnica de Madrid (UPM)
```

```text
Joaquín Bienvenido Ordieres Meré
Tutor
Escuela Técnica Superior de Ingenieros Industriales
Universidad Politécnica de Madrid (UPM)
```
---
