# -*- coding: utf-8 -*-
"""
Entrena TrajectoryModel (fine-tuning de GaitTransformer) usando MULTIPLES
segmentos de marcha ya validados manualmente (por inspeccion visual del
usuario) y verificados con GPS real suficiente (>=3 lecturas distintas).

A diferencia de entrenar_trajectory_model.py (un solo paciente/segmento),
este script concatena el dataset de entrenamiento (ground truth PCHIP) y
de validacion (GPS real) de TODOS los segmentos de la lista SEGMENTOS_ENTRENAMIENTO
antes de entrenar, produciendo un modelo GENERICO reutilizable para
inferencia rapida sobre cualquier segmento nuevo (ver analizar_segmento_trayectoria.py,
aun por construir), sin necesidad de reentrenar cada vez.
"""

from __future__ import annotations
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def fijar_semilla(seed: int = 13) -> None:
    """
    Fija todas las semillas aleatorias relevantes para reproducibilidad
    total entre corridas (numpy, torch, random, y determinismo de CUDA).

    :param seed: Valor de la semilla.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "A01_EXTRACCION_DATOS"))
sys.path.insert(0, str(PROJECT_ROOT / "A04_TRANSFORMER"))

from extract_data_plus import ExtractionParams
from AA_TRANSFORMER_V1 import TransformerConfig

import preparar_dataset_trayectoria as pdt
import conector_trayectoria as ct
from trajectory_model import TrajectoryModel, VentanasSecuenciaDataset


# =============================================================================
# LISTA DE SEGMENTOS DE ENTRENAMIENTO (22 segmentos validados manualmente
# por inspeccion visual del usuario, confirmados con GPS real >=3 lecturas)
#
# EXCLUIDOS (comentados, no descartados definitivamente): AMIR2026-54 y
# SHSHUG037-1 se excluyeron tras detectar que su normalizacion por sesion
# (media_x/std_x) queda en ordenes de magnitud absurdos comparado con las
# demas 18 sesiones (ej. SHSHUG037-1: std_x=16019 vs el resto en el rango
# 1-150), lo cual dispara el error de evaluacion desnormalizado. Pendiente
# investigar la causa raiz antes de reincorporarlas (sospecha: AMIR2026-54
# pudo tener un bug de resta de origen UTM al ser el unico segmento UTC;
# SHSHUG037-1, con 71 puntos GPS reales, pudo generar un salto de
# interpolacion PCHIP extremo entre dos lecturas).
# =============================================================================
SEGMENTOS_ENTRENAMIENTO = [
    ("MGM-202406-79", datetime(2024, 6, 16, 13, 26, 4), datetime(2024, 6, 16, 13, 31, 4), False),
    ("TABUENCA01-45", datetime(2024, 4, 25, 16, 38, 33), datetime(2024, 4, 25, 16, 49, 5), False),
    # ("AMIR2026-54", datetime(2026, 5, 22, 12, 40, 50), datetime(2026, 5, 22, 13, 10, 0), True),  # EXCLUIDO: ver nota arriba
    ("02548893X-118", datetime(2025, 2, 28, 22, 57, 14), datetime(2025, 2, 28, 23, 6, 44), False),
    ("04845288Q-121", datetime(2025, 3, 1, 11, 50, 57), datetime(2025, 3, 1, 11, 52, 6), False),
    ("2025-CJ1-SJ2-49", datetime(2025, 5, 15, 20, 5, 24), datetime(2025, 5, 15, 20, 7, 2), False),
    ("20266247G-97", datetime(2025, 3, 28, 18, 59, 0), datetime(2025, 3, 28, 19, 9, 0), False),
    ("EPGHUG006-25", datetime(2025, 12, 13, 10, 33, 0), datetime(2025, 12, 13, 10, 40, 0), False),
    ("JCGMHUG007-73", datetime(2025, 12, 17, 5, 26, 0), datetime(2025, 12, 17, 5, 27, 0), False),
    ("JCGMHUG007-73", datetime(2025, 12, 17, 6, 54, 0), datetime(2025, 12, 17, 6, 56, 0), False),
    ("RSBHUGOO5-11", datetime(2025, 12, 6, 13, 13, 0), datetime(2025, 12, 6, 13, 14, 0), False),
    ("RSBHUGOO5-11", datetime(2025, 12, 6, 13, 23, 0), datetime(2025, 12, 6, 13, 24, 0), False),
    ("SBRHUG002-12", datetime(2025, 12, 18, 9, 28, 38), datetime(2025, 12, 18, 9, 29, 23), False),
    ("SBRHUG002-12", datetime(2025, 12, 18, 9, 31, 2), datetime(2025, 12, 18, 9, 38, 10), False),
    ("SFMHUG065-22", datetime(2026, 5, 8, 12, 11, 45), datetime(2026, 5, 8, 12, 19, 21), False),
    ("SFMHUG065-22", datetime(2026, 5, 8, 12, 27, 23), datetime(2026, 5, 8, 12, 30, 18), False),
    ("SFMHUG065-22", datetime(2026, 5, 8, 12, 34, 34), datetime(2026, 5, 8, 12, 37, 18), False),
    ("SFMHUG065-22", datetime(2026, 5, 8, 13, 22, 45), datetime(2026, 5, 8, 13, 25, 9), False),
    ("SFMHUG065-22", datetime(2026, 5, 8, 13, 30, 15), datetime(2026, 5, 8, 13, 31, 14), False),
    ("SHGHUG021-18", datetime(2026, 1, 30, 9, 35, 50), datetime(2026, 1, 30, 9, 36, 44), False),
    ("SHGHUG021-18", datetime(2026, 1, 30, 9, 38, 53), datetime(2026, 1, 30, 9, 45, 25), False),
    # ("SHSHUG037-1", datetime(2026, 3, 12, 10, 41, 50), datetime(2026, 3, 12, 10, 58, 50), False),  # EXCLUIDO: ver nota arriba
]


def preparar_todos_los_segmentos(config_path: str) -> tuple:
    """
    Extrae y prepara el dataset (train + val) de cada segmento en
    SEGMENTOS_ENTRENAMIENTO, y concatena los tensores PSD + posicion de
    todos los segmentos exitosos. Los segmentos que fallen (por ejemplo,
    por caida de conexion) se omiten con una advertencia, sin detener el
    resto del proceso.

    NORMALIZACION POR SESION: la posicion objetivo (X_m, Y_m) de cada
    segmento se normaliza independientemente (resta su propia media,
    divide por su propia desviacion estandar) ANTES de concatenar con
    los demas segmentos. Esto es necesario porque cada sesion tiene su
    propio origen local (primer punto GPS de esa sesion) y su propio
    rango de desplazamiento real, que puede variar en un orden de
    magnitud entre sesiones (ej. 130 m en una sesion larga vs unos pocos
    metros en una sesion de 1 minuto) -- sin esto, el modelo intenta
    aprender una funcion que salta entre escalas muy dispares de un
    segmento a otro, dificultando la generalizacion con pocos datos.

    :param config_path: Ruta al config.yaml de InfluxDB.
    :return: Tupla (x_train_total, y_train_total, x_val_total, y_val_total,
        segmentos_exitosos, segmentos_fallidos, stats_normalizacion), donde
        stats_normalizacion es un dict {segmento_idx: (media_x, std_x, media_y, std_y)}
        para poder desnormalizar predicciones de vuelta a metros reales.
    """
    params = ExtractionParams(f_start_hz=0.25, f_stop_hz=29.0)

    x_train_list, y_train_list = [], []
    x_val_list, y_val_list, mascara_val_list = [], [], []
    etiquetas_exitosas = []
    segmentos_exitosos = []
    segmentos_fallidos = []
    stats_normalizacion = {}

    for idx_seg, (paciente, inicio, fin, es_utc) in enumerate(SEGMENTOS_ENTRENAMIENTO):
        print(f"\n{'='*70}")
        print(f"PROCESANDO: {paciente} | {inicio} -> {fin} | UTC={es_utc}")
        print(f"{'='*70}")
        try:
            df_train, df_val = pdt.preparar_dataset_trayectoria(
                config_path, paciente, inicio, fin, es_utc=es_utc
            )

            x_tr, y_tr = ct.generar_tensores_psd_y_posicion(
                df_train, params, columna_x="dX_m_gt", columna_y="dY_m_gt",
                devolver_mascara_gps=False
            )
            x_val, y_val, mascara_gps_val = ct.generar_tensores_psd_y_posicion(
                df_val, params, columna_x="dX_m", columna_y="dY_m",
                devolver_mascara_gps=True
            )

            # DESCARTAR FRAMES DE VAL DONDE dX_m/dY_m QUEDO NaN (frames de
            # contexto interpolado que no coinciden con un punto GPS real
            # consecutivo a otro punto GPS real; solo los frames marcados
            # en mascara_gps_val son evaluables, pero ademas necesitan
            # tener un delta real valido, no NaN)
            mascara_gps_val = mascara_gps_val & ~np.isnan(y_val[:, 0]) & ~np.isnan(y_val[:, 1])
            y_val = np.nan_to_num(y_val, nan=0.0)

            # CALCULAR ESTADISTICAS DE NORMALIZACION SOBRE EL TRAIN DE ESTA
            # SESION (nunca sobre val, para no filtrar informacion del GPS
            # real hacia los parametros de normalizacion)
            media_x, std_x = float(y_tr[:, 0].mean()), float(y_tr[:, 0].std())
            media_y, std_y = float(y_tr[:, 1].mean()), float(y_tr[:, 1].std())
            std_x = std_x if std_x > 1e-6 else 1.0
            std_y = std_y if std_y > 1e-6 else 1.0

            y_tr_norm = y_tr.copy()
            y_tr_norm[:, 0] = (y_tr[:, 0] - media_x) / std_x
            y_tr_norm[:, 1] = (y_tr[:, 1] - media_y) / std_y

            y_val_norm = y_val.copy()
            if len(y_val) > 0:
                y_val_norm[:, 0] = (y_val[:, 0] - media_x) / std_x
                y_val_norm[:, 1] = (y_val[:, 1] - media_y) / std_y

            stats_normalizacion[len(segmentos_exitosos)] = {
                "paciente": paciente, "media_x": media_x, "std_x": std_x,
                "media_y": media_y, "std_y": std_y
            }

            x_train_list.append(x_tr)
            y_train_list.append(y_tr_norm)
            x_val_list.append(x_val)
            y_val_list.append(y_val_norm)
            mascara_val_list.append(mascara_gps_val)
            etiquetas_exitosas.append(f"{paciente} ({inicio})")
            segmentos_exitosos.append((paciente, str(inicio), str(fin)))

            print(f"OK: {len(x_tr)} frames de train, {len(x_val)} frames de val "
                  f"({int(mascara_gps_val.sum())} con GPS real evaluable)")
            print(f"  Normalizacion: media_x={media_x:.2f}, std_x={std_x:.2f}, "
                  f"media_y={media_y:.2f}, std_y={std_y:.2f}")

        except Exception as e:
            print(f"FALLO: {e}")
            segmentos_fallidos.append((paciente, str(inicio), str(fin), str(e)))

    # CONCATENAR CON VALIDACION DE FORMA: si algun segmento produjo un
    # tensor con dimensiones inconsistentes respecto a los demas (se ha
    # observado variabilidad entre corridas para segmentos muy cortos,
    # posiblemente por datos que cambian levemente en InfluxDB entre
    # consultas), se descarta ese segmento especifico en vez de fallar
    # todo el proceso.
    def _concatenar_robusto(lista_arrays: list, nombre: str, etiquetas: list) -> np.ndarray:
        """Concatena una lista de arrays, descartando los que no coincidan
        en la segunda dimension con la mayoria (moda) de las formas vistas.

        :param etiquetas: Lista PARALELA (mismo orden y longitud) a
            lista_arrays, con un identificador legible por elemento, para
            reportar exactamente cual segmento se descarta.
        """
        if not lista_arrays:
            return np.zeros((0, 2 if "y_" in nombre else 348), dtype=np.float32)

        arrays_no_vacios = []
        etiquetas_no_vacias = []
        for a, etq in zip(lista_arrays, etiquetas):
            if len(a) > 0:
                arrays_no_vacios.append(a)
                etiquetas_no_vacias.append(etq)

        if not arrays_no_vacios:
            return np.zeros((0, 2 if "y_" in nombre else 348), dtype=np.float32)

        formas_dim1 = [a.shape[1] for a in arrays_no_vacios]
        forma_esperada = max(set(formas_dim1), key=formas_dim1.count)

        validos = []
        for a, etq in zip(arrays_no_vacios, etiquetas_no_vacias):
            if a.shape[1] == forma_esperada:
                validos.append(a)
            else:
                print(f"ADVERTENCIA: descartado de '{nombre}' -> {etq} "
                      f"(forma {a.shape[1]}, esperado {forma_esperada})")

        return np.concatenate(validos, axis=0)

    x_train_total = _concatenar_robusto(x_train_list, "x_train", etiquetas_exitosas)
    y_train_total = _concatenar_robusto(y_train_list, "y_train", etiquetas_exitosas)
    x_val_total = _concatenar_robusto(x_val_list, "x_val", etiquetas_exitosas)
    y_val_total = _concatenar_robusto(y_val_list, "y_val", etiquetas_exitosas)

    # LA MASCARA ES 1D (bool), se concatena directo sin el chequeo de
    # segunda dimension que usa _concatenar_robusto para los tensores PSD
    mascara_val_total = (
        np.concatenate([m for m in mascara_val_list if len(m) > 0])
        if any(len(m) > 0 for m in mascara_val_list) else np.zeros(0, dtype=bool)
    )

    print(f"\n\n{'='*70}")
    print("RESUMEN DE PREPARACION")
    print(f"{'='*70}")
    print(f"Segmentos exitosos: {len(segmentos_exitosos)} de {len(SEGMENTOS_ENTRENAMIENTO)}")
    print(f"Segmentos fallidos: {len(segmentos_fallidos)}")
    for p, i, f, err in segmentos_fallidos:
        print(f"  - {p} ({i} -> {f}): {err}")
    print(f"\nTotal frames de entrenamiento: {len(x_train_total)}")
    print(f"Total frames de validacion (incluye contexto): {len(x_val_total)}")
    print(f"De los cuales son GPS real evaluable: {int(mascara_val_total.sum())}")
    print("\nNOTA: y_train_total/y_val_total estan en unidades NORMALIZADAS "
          "(z-score por sesion), no en metros reales. Ver stats_normalizacion "
          "para desnormalizar.")

    return (x_train_total, y_train_total, x_val_list, y_val_list, mascara_val_list,
            segmentos_exitosos, segmentos_fallidos, stats_normalizacion)


def entrenar_modelo_generico(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val_list: list,
    y_val_list: list,
    mascara_val_list: list,
    t_cfg: TransformerConfig,
    ruta_pesos_preentrenados: Path,
    device: str,
    stats_normalizacion: dict,
    epochs: int = 150,
    lr: float = 0.0001,
    patience: int = 20,
    congelar_transformer: bool = False
) -> tuple:
    """
    Entrena TrajectoryModel sobre el tensor de entrenamiento ya concatenado
    y normalizado (z-score por sesion) de multiples segmentos, con
    fine-tuning de modelo_transformer.pth. Evalua el error final POR
    SEGMENTO (no sobre un unico tensor de validacion concatenado), para
    evitar mezclar el contexto de una sesion con el punto GPS real de otra.

    :param x_train: Tensor PSD combinado de entrenamiento, forma (N, 348).
    :param y_train: Posicion objetivo NORMALIZADA de entrenamiento, forma (N, 2).
    :param x_val_list: Lista (una por segmento) de tensores PSD de
        validacion, cada uno con su propio contexto + puntos GPS reales.
    :param y_val_list: Lista paralela a x_val_list, con la posicion
        NORMALIZADA (usando la normalizacion de ESE segmento especifico,
        no el promedio global) de cada frame de validacion.
    :param mascara_val_list: Lista paralela, cada elemento indica que
        frames de ese segmento de validacion corresponden a un punto GPS
        real (evaluable), no a contexto interpolado.
    :param t_cfg: TransformerConfig, debe coincidir con el checkpoint.
    :param ruta_pesos_preentrenados: Ruta a modelo_transformer.pth.
    :param device: "cuda" o "cpu".
    :param stats_normalizacion: Diccionario de medias/desviaciones por
        sesion (indexado igual que x_val_list), usado para desnormalizar
        el error de cada segmento a metros reales, con la escala correcta
        de ESA sesion especifica (no un promedio global).
    :param epochs: Numero maximo de epocas.
    :param lr: Tasa de aprendizaje.
    :param patience: Paciencia para early stopping.
    :param congelar_transformer: Si True, solo entrena el cabezal de regresion.
    :return: Tupla (modelo entrenado, historial de perdidas, error_medio_metros).
    """
    max_len = t_cfg.max_len
    dataset_train = VentanasSecuenciaDataset(x_train, y_train, max_len)
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    modelo = TrajectoryModel(t_cfg, congelar_transformer=congelar_transformer).to(device)
    modelo.cargar_pesos_preentrenados(ruta_pesos_preentrenados, device)

    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="min", factor=0.5, patience=7)

    mejor_loss = float("inf")
    sin_mejora = 0
    mejor_estado = None
    historial = []

    print(f"\nEntrenando con {len(dataset_train)} ventanas de secuencia (max_len={max_len})")
    print("NOTA: el Loss (MSE) esta en unidades normalizadas (z-score), no metros^2 directos.")

    for epoch in range(epochs):
        modelo.train()
        perdida_epoch = 0.0

        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizador.zero_grad()
            pred = modelo(xb)
            loss = criterio(pred, yb)
            loss.backward()
            optimizador.step()
            perdida_epoch += loss.item() * xb.size(0)

        perdida_epoch /= len(dataset_train)
        historial.append(perdida_epoch)
        scheduler.step(perdida_epoch)

        if perdida_epoch < mejor_loss:
            mejor_loss = perdida_epoch
            sin_mejora = 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1

        if epoch % 10 == 0 or sin_mejora >= patience:
            print(f"Epoch {epoch:03d} | Loss (MSE normalizado): {perdida_epoch:.4f} | Sin mejora: {sin_mejora}")

        if sin_mejora >= patience:
            print(f"Early stopping en epoch {epoch}")
            break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)

    # EVALUACION FINAL POR SEGMENTO: para cada sesion, se arman ventanas de
    # max_len frames terminando en cada frame marcado como GPS real, y se
    # mide el error SOLO ahi (nunca en el contexto interpolado). Se
    # desnormaliza con la escala PROPIA de esa sesion (no un promedio
    # global), ya que aqui si se conoce con certeza a que sesion pertenece
    # cada punto de validacion.
    modelo.eval()
    errores_metros_todos = []

    with torch.no_grad():
        for seg_i, (x_val, y_val, mascara) in enumerate(zip(x_val_list, y_val_list, mascara_val_list)):
            if len(x_val) < max_len or mascara.sum() == 0:
                continue

            if seg_i not in stats_normalizacion:
                continue
            stats_seg = stats_normalizacion[seg_i]
            std_x_seg, std_y_seg = stats_seg["std_x"], stats_seg["std_y"]

            indices_reales = np.where(mascara)[0]
            for idx_real in indices_reales:
                if idx_real < max_len - 1:
                    continue
                ventana = x_val[idx_real - max_len + 1: idx_real + 1]
                xb = torch.from_numpy(ventana).unsqueeze(0).to(device)
                pred = modelo(xb).cpu().numpy()[0]
                yb_np = y_val[idx_real]

                error_x_m = (pred[0] - yb_np[0]) * std_x_seg
                error_y_m = (pred[1] - yb_np[1]) * std_y_seg
                errores_metros_todos.append(float(np.sqrt(error_x_m**2 + error_y_m**2)))

    error_medio_metros = None
    if errores_metros_todos:
        error_medio_metros = float(np.mean(errores_metros_todos))
        print(f"\nError medio de DESPLAZAMIENTO (delta) vs GPS real: {error_medio_metros:.3f} m/frame "
              f"(sobre {len(errores_metros_todos)} puntos GPS reales evaluados, "
              f"desnormalizado con la escala propia de cada sesion)")
    else:
        print(
            "\nADVERTENCIA: no se pudo evaluar ningun punto GPS real "
            f"(se requieren >= {max_len} frames de contexto antes del punto real "
            "en al menos un segmento)."
        )

    return modelo, historial, error_medio_metros


if __name__ == "__main__":
    fijar_semilla(13)

    parser = argparse.ArgumentParser(description="Entrena el modelo GENERICO de trayectoria con multiples segmentos.")
    parser.add_argument("--config-yaml", type=str, default=str(PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml"))
    parser.add_argument("--models-dir", type=str, default=str(PROJECT_ROOT / "A05_MODELOS_ENTRENADOS"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--congelar-transformer", action="store_true")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "A07_TRAYECTORIA_GPS" / "RESULTADOS_TRAYECTORIA")
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)

    t_cfg_dict = joblib.load(models_dir / "transformer_config.joblib")
    t_cfg = TransformerConfig(**t_cfg_dict)

    x_train, y_train, x_val_list, y_val_list, mascara_val_list, exitosos, fallidos, stats_normalizacion = preparar_todos_los_segmentos(args.config_yaml)

    modelo, historial, error_medio_metros = entrenar_modelo_generico(
        x_train, y_train, x_val_list, y_val_list, mascara_val_list, t_cfg,
        ruta_pesos_preentrenados=models_dir / "modelo_transformer.pth",
        device=device, stats_normalizacion=stats_normalizacion,
        epochs=args.epochs, lr=args.lr,
        congelar_transformer=args.congelar_transformer
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_modelo = out_dir / "trajectory_model_generico.pth"
    torch.save(modelo.state_dict(), ruta_modelo)

    ruta_stats = out_dir / "trajectory_model_stats_normalizacion.joblib"
    joblib.dump(stats_normalizacion, ruta_stats)

    print(f"\nModelo GENERICO guardado en: {ruta_modelo}")
    print(f"Estadisticas de normalizacion guardadas en: {ruta_stats}")
    print(f"Entrenado con {len(exitosos)} segmentos de {len(SEGMENTOS_ENTRENAMIENTO)}")
    if error_medio_metros is not None:
        print(f"ERROR FINAL DE TRAYECTORIA vs GPS REAL: {error_medio_metros:.3f} metros")
