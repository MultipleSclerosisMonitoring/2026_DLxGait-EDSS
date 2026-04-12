# Detección de Marcha en pacientes con Esclerosis Múltiple mediante Deep Learning

Este repositorio contiene el pipeline completo de ingeniería de datos y Deep Learning para la detección continua de marcha (Gait) en pacientes con Esclerosis Múltiple, utilizando datos biomecánicos extraídos en crudo.

---

## Mejoras 

1. **Data Leakage (Identidad Preservada):** El sistema de preprocesamiento mapea los segmentos con el ID del paciente. La validación cruzada utiliza `StratifiedGroupKFold` para aislar pacientes, asegurando que el modelo se evalúa frente a sujetos *nunca antes vistos* en entrenamiento.
2. **Eliminación del Sesgo de Muestreo:** Se ha implementado un balanceo aleatorio de clases (`random.shuffle`) antes de la selección, garantizando la representación de todos los pacientes en el dataset final.
3. **Inferencia Continua (Sliding Window):** Transición de inferencia estática a un motor dinámico (`inferencia_sliding.py`) capaz de leer secuencias ininterrumpidas, aplicar ventanas deslizantes con solapamiento, y detectar el milisegundo exacto de transición entre Reposo y Marcha.
4. **Portabilidad:** Rresolución dinámica con `pathlib`.
5. **Documentación Sphinx:** Implementación de docstrings estándar (PEP 257) en todas las clases y métodos principales.

---

## Estructura del Proyecto y Ejecución

El pipeline está diseñado para ejecutarse secuencialmente:

### 1. Extracción de Datos
* **Script:** `01_EXTRACCION DE DATOS/extract_data_plus.py`
* **Descripción:** Conecta con la base de datos (InfluxDB) usando los parámetros del `.config.yaml` (credenciales ocultas) y el excel de solicitudes, generando archivos `.parquet` en bruto.

### 2. Preprocesamiento, Resampling y Balanceo
* **Script:** `03_CODIGOS PREPROCESAMIENTO/LIMPIEZA.py`
* **Descripción:** Procesa los archivos `.parquet`, extrae la identidad del paciente, aplica interpolación para señales cortas y empaqueta todo en un archivo jerárquico `dataset_jerarquico.hdf5` libre de fugas de datos.

### 3. Entrenamiento Híbrido (Stress Test)
* **Script:** `04_CODIGO TRANSFORMER/01_TRANSFORMER_V1.py`
* **Descripción:** Entrena tres arquitecturas:
  - *Transformer (Dominio del Tiempo)*
  - *MLP sobre Transformada de Fourier (FFT)*
  - *Modelo Híbrido (Fusión Tiempo + Frecuencia)*
* Finaliza con un Stress Test (Validación Cruzada por Grupos de 5 Folds) para certificar la generalización del modelo FFT (>98% AUC).

### 4. Monitorización Temporal (Inferencia)
* **Script:** `04_INFERENCIA/inferencia_sliding.py`
* **Descripción:** Carga los pesos del modelo (`.pth`) y el escalador, procesando una secuencia ininterrumpida de movimientos y mostrando la transición temporal en la consola mediante ventanas deslizantes.

---
