# ESTIMACIÓN DE DETERIORO EN ESCLEROSIS MÚLTIPLE FUNCIÓN DEL COMPORTAMIENTO EN TESTS PREESTABLECIDOS

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-success.svg)](#)
[![Code style: PEP8](https://img.shields.io/badge/Code%20Style-PEP8-yellow.svg)](https://pep8.org/)

Este repositorio contiene la arquitectura de extremo a extremo (*end-to-end*) para la extracción, preprocesamiento y modelado de datos biomecánicos obtenidos mediante calcetines inteligentes (SCKS) en pacientes con Esclerosis Múltiple.

---

## Arquitectura 

La arquitectura ha sido migrada a **PyTorch** para garantizar un flujo continuo y evitar el *Data Leakage*. El sistema se divide en tres enfoques modulares:

1. **Rama Temporal (Transformer Encoder):** 
   * Captura dependencias secuenciales a largo plazo en los datos crudos del giroscopio y acelerómetro.
   * Utiliza *Multi-Head Attention* para identificar patrones anómalos de la marcha en el dominio del tiempo.
2. **Rama Frecuencial (FFT - Multi-Layer Perceptron):**
   * Transforma las ventanas temporales mediante la Transformada Rápida de Fourier (RFFT).
   * Aísla las micro-frecuencias generadas por impactos instantáneos (apoyado en la justificación biomecánica de *Müller et al., 2021*).
3. **Modelo Híbrido (Early Fusion):**
   * Concatena los espacios latentes de ambas ramas (Tiempo + Frecuencia) antes del cabezal de clasificación, logrando una inferencia altamente robusta.

*La validación inter-sujeto se garantiza mediante `StratifiedGroupKFold`.*

---

## Flujo de Ejecución (Quickstart)

El código ha sido refactorizado implementando tipado estricto, logging estructurado y validaciones CLI mediante `argparse` y `Pydantic`.

### 1. Extracción de Datos (InfluxDB)
`python "01_EXTRACCION DE DATOS/extract_data_plus.py" --backend influxdbms -c .config.yaml -e solicitud.xlsx -o ./resultados`

### 2. Preprocesamiento y Balanceo (Generación HDF5)
`python "03_CODIGOS PREPROCESAMIENTO/LIMPIEZA.py" --input ./resultados --output ./DATASET --excel solicitud.xlsx`

### 3. Entrenamiento (Deep Learning)
`python "04_CODIGO TRANSFORMER/01_TRANSFORMER_V1.py" --dataset ./DATASET/dataset_jerarquico.hdf5 --output ./MODELOS_ENTRENADOS`

### 4. Inferencia y Pruebas Ciegas (Sliding Window)
`python "04_INFERENCIA/inferencia_sliding.py" --modelo ./MODELOS_ENTRENADOS/modelo_frecuencia.pth --scaler ./MODELOS_ENTRENADOS/scaler_gait.joblib --datos ./resultados/segment_000.parquet`

---

## Troubleshooting (Solución de Problemas)

* **`InfluxExtractionError: Configuración faltante`**: 
  * *Causa:* El archivo `.config.yaml` no se encuentra. Por políticas de seguridad, las credenciales no se suben al repositorio.
  * *Solución:* Solicita el archivo de configuración al administrador y colócalo en el directorio raíz de extracción.
* **`extract_data_plus.py: error: the following arguments are required: -e/--excel`**: 
  * *Causa:* Se está intentando extraer datos sin proveer el mapeo de identidades.
  * *Solución:* Añade la bandera `-e solicitud.xlsx` al ejecutar por terminal.
* **`pydantic_core.ValidationError: Path does not point to a directory`**: 
  * *Causa:* Las rutas pasadas por consola a los scripts de limpieza o inferencia no existen en el disco duro.
  * *Solución:* Verifica que la carpeta de entrada ha sido creada previamente.

---

## Roadmap (Próximos Pasos)

- [x] **Fase 1-4**: Pipeline de extracción y clasificación robusta Marcha vs. Reposo.
- [x] **Refactorización Core**: Implementación de tests unitarios (`pytest`), tipado estricto y POO corporativa.
- [ ] ** --TENTATIVO-- Fase 5: Regresor Clínico (EDSS)**: Utilizar los *embeddings* latentes generados por el modelo híbrido como entrada para una red Tabular (TabTransformer/MLP-Mixer) capaz de predecir el grado de discapacidad EDSS, uniendo la marcha con las covariables demográficas y cognitivas de los pacientes.
