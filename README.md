# ESTIMACIÓN DE DETERIORO EN ESCLEROSIS MÚLTIPLE FUNCIÓN DEL COMPORTAMIENTO EN TESTS PREESTABLECIDOS

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-success.svg)](#)
[![Code style: PEP8](https://img.shields.io/badge/Code%20Style-PEP8-yellow.svg)](https://pep8.org/)

Este repositorio contiene la arquitectura de extremo a extremo (*end-to-end*) para la extracción, preprocesamiento y modelado de datos biomecánicos obtenidos mediante calcetines inteligentes (SCKS) en pacientes con Esclerosis Múltiple.

---

## Instalación y Configuración

Cree y active un entorno virtual. Para garantizar la correcta resolución de paquetes e *imports* desde cualquier directorio, instale el proyecto en modo editable:

Aquí tienes el texto con el formato Markdown perfecto, con los bloques de código separados del texto para que lo copies y pegues directamente en tu README.md:

Markdown

```bash
pip install -e .
Alternativamente, las dependencias también pueden instalarse mediante:

Bash
pip install -r requirements.txt
Autenticación de Base de Datos (InfluxDB)
El proyecto utiliza InfluxDB v2, que requiere autenticación obligatoria mediante Token. Por motivos de seguridad, el repositorio incluye una plantilla.

Antes de ejecutar los procesos de extracción, edite el archivo A01_EXTRACCION_DATOS/.config.yaml y sustituya el campo token por sus credenciales locales reales:

YAML
influxdb:
  bucket: "Gait/autogen"
  org: "UPM"
  token: "SU_TOKEN_REAL_AQUI"  # <-- SUSTITUIR POR SU TOKEN
  url: "https://localhost:8086/"  # <-- AJUSTAR IP SI ES NECESARIO
Uso
Gracias al empaquetado del proyecto (setup.py), ambos modos de ejecución son totalmente equivalentes y válidos. Puede lanzar los scripts desde la raíz del repositorio de las siguientes formas:

Bash
# Opción 1: Ejecución como módulo
python -m A04_TRANSFORMER.Agnostic_evaluator

# Opción 2: Ejecución mediante ruta relativa
python A04_TRANSFORMER/Agnostic_evaluator.py
Arquitectura
La arquitectura ha sido migrada a PyTorch para garantizar un flujo continuo y evitar el Data Leakage. El sistema se divide en tres enfoques modulares:

Rama Temporal (Transformer Encoder):

Captura dependencias secuenciales a largo plazo en los datos crudos del giroscopio y acelerómetro.

Utiliza Multi-Head Attention para identificar patrones anómalos de la marcha en el dominio del tiempo.

Rama Frecuencial (FFT - Multi-Layer Perceptron):

Transforma las ventanas temporales mediante la Transformada Rápida de Fourier (RFFT).

Aísla las micro-frecuencias generadas por impactos instantáneos (apoyado en la justificación biomecánica de Müller et al., 2021).

Modelo Híbrido (Early Fusion):

Concatena los espacios latentes de ambas ramas (Tiempo + Frecuencia) antes del cabezal de clasificación, logrando una inferencia altamente robusta.

La validación inter-sujeto se garantiza mediante StratifiedGroupKFold.

Roadmap (Próximos Pasos)
[x] Fase 1-4: Pipeline de extracción y clasificación robusta Marcha vs. Reposo.

[x] Refactorización Core: Implementación de tests unitarios (pytest), empaquetado PEP 8, tipado estricto y POO corporativa.

[ ] --TENTATIVO-- Fase 5: Regresor Clínico (EDSS): Utilizar los embeddings latentes generados por el modelo híbrido como entrada para una red Tabular (TabTransformer/MLP-Mixer) capaz de predecir el grado de discapacidad EDSS, uniendo la marcha con las covariables demográficas y cognitivas de los pacientes.
