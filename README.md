# ESTIMACIÓN DE DETERIORO EN ESCLEROSIS MÚLTIPLE FUNCIÓN DEL COMPORTAMIENTO EN TESTS PREESTABLECIDOS

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-success.svg)](#)
[![Code style: PEP8](https://img.shields.io/badge/Code%20Style-PEP8-yellow.svg)](https://pep8.org/)

Este repositorio contiene la arquitectura de extremo a extremo (*end-to-end*) para la extracción, preprocesamiento y modelado de datos biomecánicos obtenidos mediante calcetines inteligentes (SCKS) en pacientes con Esclerosis Múltiple.

---

# Instalación

Cree y active un entorno virtual y, posteriormente, instale el proyecto en modo editable:

```bash
pip install -e .
```

Alternativamente, las dependencias también pueden instalarse mediante:

```bash
pip install -r requirements.txt
```

# Uso

Los scripts deben ejecutarse como módulos Python desde la raíz del repositorio.

Ejemplo:

```bash
python -m A04_TRANSFORMER.Agnostic_evaluator
```

No ejecute los scripts directamente mediante rutas relativas como:

```bash
python A04_TRANSFORMER/Agnostic_evaluator.py
```

ya que esto puede provocar errores en la resolución de imports y paquetes Python.


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

Roadmap (Próximos Pasos)
[x] Fase 1-4: Pipeline de extracción y clasificación robusta Marcha vs. Reposo.

[x] Refactorización Core: Implementación de tests unitarios (pytest), empaquetado PEP 8, tipado estricto y POO corporativa.

[ ] --TENTATIVO-- Fase 5: Regresor Clínico (EDSS): Utilizar los embeddings latentes generados por el modelo híbrido como entrada para una red Tabular (TabTransformer/MLP-Mixer) capaz de predecir el grado de discapacidad EDSS, uniendo la marcha con las covariables demográficas y cognitivas de los pacientes.
