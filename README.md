# ESTIMACIÓN DE DETERIORO EN ESCLEROSIS MÚLTIPLE FUNCIÓN DEL COMPORTAMIENTO EN TESTS PREESTABLECIDOS

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-success.svg)](#)
[![Code style: PEP8](https://img.shields.io/badge/Code%20Style-PEP8-yellow.svg)](https://pep8.org/)

Este repositorio contiene la arquitectura de extremo a extremo (*end-to-end*) para la extracción, preprocesamiento y modelado de datos biomecánicos obtenidos mediante calcetines inteligentes (SCKS) en pacientes con Esclerosis Múltiple.

---

## Arquitectura 

La arquitectura ha sido migrada a **PyTorch** para garantizar un flujo continuo y evitar el *Data Leakage*. El sistema se divide en tres enfoques modulares:

1. **Rama Temporal (Transformer Encoder):** * Captura dependencias secuenciales a largo plazo en los datos crudos del giroscopio y acelerómetro.
   * Utiliza *Multi-Head Attention* para identificar patrones anómalos de la marcha en el dominio del tiempo.
2. **Rama Frecuencial (FFT - Multi-Layer Perceptron):**
   * Transforma las ventanas temporales mediante la Transformada Rápida de Fourier (RFFT).
   * Aísla las micro-frecuencias generadas por impactos instantáneos (apoyado en la justificación biomecánica de *Müller et al., 2021*).
3. **Modelo Híbrido (Early Fusion):**
   * Concatena los espacios latentes de ambas ramas (Tiempo + Frecuencia) antes del cabezal de clasificación, logrando una inferencia altamente robusta.

*La validación inter-sujeto se garantiza mediante `StratifiedGroupKFold`.*

---

## Instalación (Setup)

Para que el motor de Python reconozca la estructura de módulos locales y resuelva las dependencias correctamente, clona el repositorio y ejecuta el siguiente comando en la raíz (donde se encuentra el archivo `pyproject.toml`):

```bash
pip install -e .
Flujo de Ejecución (Quickstart)
El código ha sido refactorizado implementando tipado estricto, logging estructurado y validaciones CLI mediante argparse y Pydantic. Ejecutar siempre desde la raíz del proyecto.

1. Extracción de Datos (InfluxDB)
Bash
python -m a01_extraccion_datos.extract_data_plus --backend influxdbms -c ./config.yaml -e solicitud.xlsx -o ./resultados
2. Preprocesamiento y Balanceo (Generación HDF5)
Bash
python -m a03_preprocesamiento.LIMPIEZA --input ./resultados --output ./DATASET --excel solicitud.xlsx
3. Entrenamiento (Deep Learning)
Bash
python -m a04_transformer.AA_TRANSFORMER_V1 --dataset ./DATASET/dataset_jerarquico.hdf5 --output ./modelos_entrenados
4. Evaluación Agnóstica Continua (Inferencia InfluxDB)
Bash
python -m a04_transformer.Agnostic_evaluator -c "./config.yaml" -m "./modelos_entrenados" -r "JASAHUG010-85" --start "2025-12-24 09:51:00" --end "2025-12-24 10:38:00" -o "./evaluacion_continua.csv"
Troubleshooting (Solución de Problemas)
InfluxExtractionError: Configuración faltante:

Causa: El archivo config.yaml no se encuentra. Por políticas de seguridad, las credenciales no se suben al repositorio.

Solución: Solicita el archivo de configuración al administrador y colócalo en el directorio raíz.

extract_data_plus: error: the following arguments are required: -e/--excel:

Causa: Se está intentando extraer datos sin proveer el mapeo de identidades.

Solución: Añade la bandera -e solicitud.xlsx al ejecutar por terminal.

ModuleNotFoundError: No module named 'a01_extraccion_datos':

Causa: El proyecto no se ha instalado en el entorno virtual activo.

Solución: Ejecuta pip install -e . en la raíz del proyecto.

pydantic_core.ValidationError: Path does not point to a directory:

Causa: Las rutas pasadas por consola a los scripts de limpieza o inferencia no existen en el disco duro.

Solución: Verifica que la carpeta de entrada ha sido creada previamente.

Roadmap (Próximos Pasos)
[x] Fase 1-4: Pipeline de extracción y clasificación robusta Marcha vs. Reposo.

[x] Refactorización Core: Implementación de tests unitarios (pytest), empaquetado PEP 8, tipado estricto y POO corporativa.

[ ] --TENTATIVO-- Fase 5: Regresor Clínico (EDSS): Utilizar los embeddings latentes generados por el modelo híbrido como entrada para una red Tabular (TabTransformer/MLP-Mixer) capaz de predecir el grado de discapacidad EDSS, uniendo la marcha con las covariables demográficas y cognitivas de los pacientes.
