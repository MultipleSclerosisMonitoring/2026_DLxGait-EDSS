# Auditoría Del Dataset Actual

Fecha de auditoría: `2026-07-28`

Dataset auditado: `DATASET_LISTO/dataset_jerarquico.hdf5`

## Resumen global

- Muestras totales: `22,305`
- Pacientes totales: `14`
- Forma única de tensor: `(100, 290)`
- Clase `0` (`no_marcha`): `15,614` (`70.00%`)
- Clase `1` (`marcha`): `6,691` (`30.00%`)

## Recuento por paciente

| Paciente | Muestras | No marcha (0) | Marcha (1) | Ambas clases |
|---|---:|---:|---:|---|
| `01912299X-118` | 4065 | 3118 | 947 | Sí |
| `02548893X-118` | 524 | 0 | 524 | No |
| `2025-CJ1-SJ2-49` | 482 | 401 | 81 | Sí |
| `2025-CJ2-SJ2-50` | 334 | 326 | 8 | Sí |
| `20266247G-97` | 1633 | 260 | 1373 | Sí |
| `53471345W-118` | 4346 | 4142 | 204 | Sí |
| `EPGHUG006-25` | 2443 | 2060 | 383 | Sí |
| `JCGMHUG007-73` | 1027 | 879 | 148 | Sí |
| `MGM-202406-79` | 271 | 0 | 271 | No |
| `SBRHUG002-12` | 831 | 408 | 423 | Sí |
| `SFMHUG065-22` | 1710 | 826 | 884 | Sí |
| `SHGHUG021-18` | 687 | 290 | 397 | Sí |
| `SHSHUG037-1` | 3841 | 2895 | 946 | Sí |
| `TABUENCA01-45` | 111 | 9 | 102 | Sí |

## Split actual reproducido con la lógica de `AA_TRANSFORMER_V1.py`

La implementación actual define:

- `test_patient = unique_patients[-1]`
- `val_patient = unique_patients[-2]`
- `train = resto de pacientes`

Con el HDF5 actual eso produce:

### Train

- Pacientes: `12`
- Muestras: `18,353`
- Clase `0`: `12,710` (`69.25%`)
- Clase `1`: `5,643` (`30.75%`)

### Validación

- Paciente: `SHSHUG037-1`
- Muestras: `3,841`
- Clase `0`: `2,895` (`75.37%`)
- Clase `1`: `946` (`24.63%`)

### Test

- Paciente: `TABUENCA01-45`
- Muestras: `111`
- Clase `0`: `9` (`8.11%`)
- Clase `1`: `102` (`91.89%`)

## Auditoría de leakage

### Comprobación de solapamiento de pacientes entre splits

- `train ∩ val = ∅`
- `train ∩ test = ∅`
- `val ∩ test = ∅`

Conclusión:

- No se observa leakage por paciente en el split actual si se reconstruye con la lógica del código.
- La separación por grupos/paciente sí evita que ventanas del mismo paciente aparezcan a la vez en entrenamiento y evaluación.

## Riesgos de sesgo detectados

### 1. Test de un solo paciente

El conjunto de test actual está formado únicamente por `TABUENCA01-45`.

Impacto:

- La generalización queda muy condicionada por la distribución particular de ese paciente.
- Un test monópaciente no representa de forma robusta el comportamiento clínico global.

### 2. Desbalance severo en test

Distribución en test:

- `no_marcha`: `9`
- `marcha`: `102`

Impacto:

- La especificidad en test depende de solo `9` ejemplos negativos.
- Pequeños cambios en falsos positivos alteran mucho la métrica.
- El AUC y la matriz de confusión del test independiente son mucho menos estables de lo que sugieren sus cifras.

### 3. Pacientes monoclase

Pacientes con una sola clase presente:

- `02548893X-118`: solo `marcha`
- `MGM-202406-79`: solo `marcha`

Impacto:

- El dataset no está equilibrado a nivel de sujeto.
- Parte de la señal aprendida puede reflejar diferencias entre pacientes y no solo entre estados de marcha/reposo.

### 4. Inconsistencia entre el HDF5 actual y los artefactos históricos

Artefactos guardados en `A05_MODELOS_ENTRENADOS`:

- `train_idx.npy`: `21,313`
- `val_idx.npy`: `3,012`
- `test_idx.npy`: `254`
- Total histórico implícito: `24,579`

Dataset actual:

- Total actual: `22,305`

Conclusión:

- Los índices y reportes históricos no corresponden al HDF5 actual.
- Por tanto, las métricas históricas de test y los splits guardados no pueden asumirse como válidos para este archivo HDF5 sin reentrenar o, al menos, reconstruir la partición.

## Conclusión operativa

- `1)` El recuento por clase y por paciente del HDF5 actual está auditado arriba.
- `2)` No hay leakage por paciente en el split actual reconstruido desde el código.
- Sí hay riesgo claro de sesgo por composición del dataset y del split:
  - test monópaciente
  - test muy desbalanceado
  - pacientes monoclase
  - desacople entre dataset actual y artefactos históricos

## Recomendación

Para evaluación fiable del dataset actual, priorizar:

- `LeaveOneGroupOut` por paciente como criterio principal
- regenerar `train_idx.npy`, `val_idx.npy` y `test_idx.npy` desde el HDF5 actual
- no reutilizar directamente las métricas históricas guardadas para justificar rendimiento del dataset presente
