# A07_TRAYECTORIA_GPS — Estimación de trayectoria mediante fine-tuning de GaitTransformer con GPS+presión

## Objetivo

Explorar una propuesta del director del TFM: usar los sensores de presión y
GPS de los calcetines instrumentados, junto con la misma arquitectura
Transformer ya validada para clasificación (A04), para calcular la
trayectoria (posición X,Y) del cuerpo en cada instante, como alternativa o
complemento al motor Madgwick+ZUPT del pipeline biomecánico (A06).

## Conclusión (importante, léase antes de usar este código)

**Este enfoque se investigó a fondo y se concluyó que NO es viable con los
datos actualmente disponibles.** El error final de predicción de
trayectoria se mantuvo en el orden de 20-60 metros según la variante
probada (normalización por sesión, predicción de desplazamiento relativo,
distintos learning rates), muy lejos de ser utilizable para calcular
parámetros clínicos de marcha.

**Causa raíz identificada:** el GPS de los calcetines es la única fuente de
posición disponible (confirmado mediante inventario exhaustivo de todos los
`_field` y tags de InfluxDB — no existe UWB, RTK ni ningún otro sensor de
posicionamiento), y sus lecturas reales están espaciadas entre 7 y 80
segundos entre sí. Esto es varios órdenes de magnitud más disperso que la
frecuencia (~10-25Hz) que se necesitaría para generar un ground truth denso
y fiable. El resto de la "trayectoria" usada para entrenar es una
interpolación matemática (PCHIP) entre esos puntos dispersos, no una
medición real.

**Decisión:** se mantiene A06 (Madgwick+ZUPT) como única fuente de
trayectoria/velocidad/zancada/MTC del proyecto. Este código se conserva como
evidencia documentada del proceso de investigación (metodología correcta,
resultado negativo honesto), no como pipeline de producción.

## Contenido de la carpeta

| Archivo | Rol |
|---|---|
| `preparar_dataset_trayectoria.py` | Extrae presión+IMU+GPS de InfluxDB, proyecta GPS a metros locales (UTM), genera ground truth denso vía interpolación PCHIP, separa train/validación (validación = solo lecturas GPS reales, nunca interpoladas) |
| `conector_trayectoria.py` | Convierte el DataFrame combinado al formato que espera `GaitFeatureExtractor`, genera el tensor PSD combinado "Both" (348 dim, mismo esquema que el dataset de clasificación) |
| `trajectory_model.py` | Arquitectura `TrajectoryModel`: fine-tuning de una instancia de `GaitTransformer` + cabezal de regresión nuevo para predecir (X, Y). Incluye `VentanasSecuenciaDataset` |
| `entrenar_trajectory_model.py` | Entrena el modelo con un solo segmento/paciente |
| `entrenar_trajectory_model_multi.py` | Entrena el modelo GENÉRICO combinando múltiples segmentos de distintos pacientes (script principal usado para el resultado final) |
| `analizar_segmento_trayectoria.py` | Inferencia rápida sobre un segmento nuevo usando el modelo ya entrenado (sin reentrenar), calcula velocidad/zancada/stance/swing desde la trayectoria predicha (MTC queda fuera, requiere componente vertical Z que este modelo no predice) |
| `verificar_gps_por_pie.py` | Herramienta de diagnóstico: verifica si el GPS es independiente por pie o compartido (confirmado: compartido, un solo GPS por dispositivo) |
| `inventario_influxdb.py` | Herramienta de diagnóstico: lista todos los `_field`/tags de InfluxDB, usada para confirmar que no existe otra fuente de posicionamiento (UWB/RTK) |

## Cómo ejecutar (si se quiere reproducir la investigación)

Todos los scripts asumen que esta carpeta (`A07_TRAYECTORIA_GPS/`) está en
la raíz del proyecto, al mismo nivel que `A01_EXTRACCION_DATOS/`,
`A04_TRANSFORMER/`, `A05_MODELOS_ENTRENADOS/`, `A06_ANALISIS_CINEMATICO/`.

```bash
# Entrenar el modelo generico (multi-segmento)
python A07_TRAYECTORIA_GPS/entrenar_trajectory_model_multi.py --epochs 150

# Analizar un segmento nuevo con el modelo ya entrenado
python A07_TRAYECTORIA_GPS/analizar_segmento_trayectoria.py --paciente <ID> --inicio "YYYY-MM-DD HH:MM:SS" --fin "YYYY-MM-DD HH:MM:SS"
```

Los resultados (modelo entrenado, estadísticas de normalización) se guardan
en `A07_TRAYECTORIA_GPS/RESULTADOS_TRAYECTORIA/`.
