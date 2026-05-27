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

El `config.yaml` no contiene credenciales reales.

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

Estas modificaciones permiten ejecutar el pipeline correctamente en entornos nuevos tras un clonado limpio del repositorio.

---

# Ejemplo de inferencia agnóstica

```bash
python A04_TRANSFORMER\Agnostic_evaluator.py ^
-c A01_EXTRACCION_DATOS\config.yaml ^
-m A05_MODELOS_ENTRENADOS ^
-r PACIENTE_001 ^
--start "2025-01-01 10:00:00" ^
--end "2025-01-01 10:05:00" ^
-o resultado_agnostic.csv
```

El sistema:

1. Recupera señales desde InfluxDB
2. Alinea ambas extremidades
3. Genera espectrogramas híbridos
4. Ejecuta inferencia continua
5. Exporta probabilidades y predicciones en CSV

Ejemplo de salida:

```csv
timestamp,prob_hybrid,pred_hybrid,prob_smoothed,pred_final_smoothed
2025-12-24 10:53:35.866,0.9999752044677734,1,0.9999757289886475,1
```

---

# Pipeline de Reconstrucción Cinemática

El pipeline biomecánico puede ejecutarse mediante:

```bash
python run_pipeline.py ^
--paciente PACIENTE_001 ^
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

# Roadmap

## Completado

- [x] Pipeline extracción de datos
- [x] Pipeline robusto de extracción, reconstrucción biomecánica y clasificación de marcha
- [x] Arquitectura híbrida Tiempo/Frecuencia
- [x] Refactorización completa PEP8
- [x] Tests unitarios (`pytest`)
- [x] Pipeline cinemático 3D
- [x] Integración IMU + presión plantar
- [x] Implementación de reconstrucción cinemática 3D basada en IMUs
- [x] Implementación de compensación gravitacional y orientación global
- [x] Implementación de Zero Velocity Update (ZUPT) tridimensional
- [x] Extracción de métricas biomecánicas espaciales y temporales
- [x] Compatibilidad multiplataforma mediante empaquetado editable
- [x] Ejecución reproducible tras clonación limpia del repositorio

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
