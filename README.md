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

### Dependencia adicional

Para la estimación de orientación basada en cuaterniones:

```bash
pip install ahrs
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

El proyecto fue refactorizado para garantizar compatibilidad entre distintos entornos de ejecución y evitar dependencias implícitas de rutas locales.

Se incorporaron las siguientes mejoras:

- Empaquetado editable mediante `setup.py`
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

El pipeline biomecánico puede ejecutarse mediante:

```bash
python run_pipeline.py ^
--paciente CODIGO_PACIENTE ^
--inicio "2025-01-01 10:00:00" ^
--fin "2025-01-01 10:10:00" ^
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

## Warning de Scikit-learn

Mensajes como:

```text
InconsistentVersionWarning
```

indican diferencias entre versiones de `scikit-learn` utilizadas al entrenar y cargar modelos.

Se recomienda utilizar la misma versión del entorno de entrenamiento para garantizar reproducibilidad completa.

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

---
