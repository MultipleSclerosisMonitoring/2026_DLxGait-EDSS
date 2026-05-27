# ESTIMACIÓN DE DETERIORO EN ESCLEROSIS MÚLTIPLE

## Análisis biomecánico mediante calcetines inteligentes (SCKS)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![PEP8](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)

---

# Descripción

Este repositorio contiene una arquitectura end-to-end para la extracción, preprocesamiento y modelado de datos biomecánicos obtenidos mediante calcetines inteligentes (SCKS) en pacientes con Esclerosis Múltiple (EM).

El sistema implementa:

- Extracción de señales desde InfluxDB
- Preprocesamiento biomecánico
- Reconstrucción cinemática basada en IMUs
- Detección de eventos de marcha
- Modelado profundo mediante PyTorch
- Inferencia híbrida Tiempo + Frecuencia

---

# Instalación

## 1. Crear entorno virtual

```bash
conda create -n tfm python=3.10
conda activate tfm
```

---

## 2. Instalar dependencias

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

El config.yaml no contiene credenciales reales.

Antes de ejecutar cualquier pipeline, edite el archivo:

```text
A01_EXTRACCION_DATOS/.config.yaml
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

## Biomecánica

- IMUs
- FFT
- Sensor Fusion
- ZUPT

---

# Calidad del Código

El proyecto sigue:

- PEP8
- Tipado estricto
- Programación orientada a objetos
- Modularidad corporativa
- Testing unitario

---

# Roadmap

## Completado

- [x] Pipeline extracción de datos
- [x] Clasificación Marcha vs Reposo
- [x] Arquitectura híbrida Tiempo/Frecuencia
- [x] Refactorización completa PEP8
- [x] Tests unitarios (`pytest`)
- [x] Pipeline cinemático 3D
- [x] Integración IMU + presión plantar

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
Universidad Politécnica de Madrid (UPM)
```

---


