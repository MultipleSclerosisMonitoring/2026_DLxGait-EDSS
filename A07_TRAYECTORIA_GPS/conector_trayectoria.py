# -*- coding: utf-8 -*-
"""
Conecta la salida de preparar_dataset_trayectoria.py (DataFrame combinado,
columnas sufijadas por pie: Ax_Right, Ax_Left, etc.) con GaitFeatureExtractor
de extract_data_plus.py, que espera un diccionario {"Left": df, "Right": df}
con columnas SIN sufijo (Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz, S0, S1, S2),
y arma los tensores PSD (X_right, X_left) mas la posicion objetivo (X, Y)
por ventana, listos para entrenar TrajectoryModel.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from A01_EXTRACCION_DATOS.extract_data_plus import GaitFeatureExtractor, ExtractionParams

# CAMPOS CRUDOS QUE GaitFeatureExtractor ESPERA POR PIE (SIN SUFIJO)
CAMPOS_CRUDOS = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz", "S0", "S1", "S2"]


def dataframe_combinado_a_dict_por_pie(df: pd.DataFrame) -> dict:
    """
    Convierte el DataFrame combinado (columnas sufijadas, ej. 'Ax_Right')
    en el diccionario {"Left": df_left, "Right": df_right} con columnas
    sin sufijo, formato que espera GaitFeatureExtractor.process_interval.

    :param df: DataFrame de salida de extraer_datos_crudos/preparar_dataset_trayectoria,
        con columnas '<campo>_Right' y '<campo>_Left' para cada campo en CAMPOS_CRUDOS.
    :return: Diccionario {"Left": DataFrame, "Right": DataFrame}.
    """
    dict_por_pie = {}
    for pie in ["Left", "Right"]:
        columnas_pie = {f"{campo}_{pie}": campo for campo in CAMPOS_CRUDOS}
        faltantes = [c for c in columnas_pie if c not in df.columns]
        if faltantes:
            raise KeyError(
                f"Faltan columnas para el pie {pie}: {faltantes}. "
                f"Verifique que preparar_dataset_trayectoria.py extrajo Mx/My/Mz."
            )
        df_pie = df[list(columnas_pie.keys())].rename(columns=columnas_pie)
        dict_por_pie[pie] = df_pie

    return dict_por_pie


def generar_tensores_psd_y_posicion(
    df: pd.DataFrame,
    params: "ExtractionParams",
    columna_x: str = "X_m_gt",
    columna_y: str = "Y_m_gt",
    devolver_mascara_gps: bool = False
):
    """
    Genera el tensor PSD combinado (ambos pies apilados, 348 features) y
    la posicion objetivo asociada a cada frame temporal.

    IMPORTANTE (confirmado inspeccionando dataset_jerarquico.hdf5): el
    GaitTransformer pre-entrenado NO recibe un tensor de 174 features por
    pie por separado -- recibe UN SOLO tensor de 348 features por frame,
    resultado de apilar (vstack) las 174 features de un pie con las 174
    del otro (174 x 2 = 348, confirmado con freq_psd_hz=75, f_stop_hz=29.0
    dando 29 bins x 6 features = 174 por pie). El dataset real etiqueta
    esto como pie "Both", no "Right"/"Left" separados. Por eso aqui se
    genera un unico tensor combinado, y el modelo usa una sola instancia
    de GaitTransformer (no dos), a diferencia del diseno inicial.

    Los valores tambien se normalizan de vuelta a [0,1] (dividiendo por
    255), ya que el HDF5 original almacena los espectrogramas cuantizados
    a uint8 (0-255) segun se confirmo en compute_psd_spectrogram.

    :param df: DataFrame combinado, con columnas crudas sufijadas por pie
        y las columnas de posicion objetivo (columna_x, columna_y).
    :param params: Parametros de extraccion PSD (ExtractionParams), deben
        coincidir con los usados para entrenar el GaitTransformer original
        (freq_psd_hz=75, f_start_hz=0.25, f_stop_hz=29.0).
    :param columna_x: Nombre de la columna de posicion X a usar como objetivo.
    :param columna_y: Nombre de la columna de posicion Y a usar como objetivo.
    :param devolver_mascara_gps: Parametro EXPLICITO (no inferido de si
        existe la columna 'tiene_gps', para evitar cambios de firma
        implicitos y fragiles) que controla si se devuelve un tercer
        elemento con la mascara de GPS real. Usar True SOLO para df_val
        (que tiene tiene_gps=True en el punto real y False en el contexto
        interpolado circundante); False para df_train (donde no aplica
        distincion real/interpolado para efectos de evaluacion).
    :return: Si devolver_mascara_gps=True: tupla (x_both, y_pos, mascara_gps_real).
        Si False (default): tupla (x_both, y_pos).
    """
    if devolver_mascara_gps and "tiene_gps" not in df.columns:
        raise ValueError(
            "Se pidio devolver_mascara_gps=True pero el DataFrame no tiene "
            "columna 'tiene_gps'. Verifique que se este pasando df_val "
            "(salida de dividir_train_val), no un DataFrame arbitrario."
        )

    dict_por_pie = dataframe_combinado_a_dict_por_pie(df)

    lfeat = ["acc_mag", "gyro_mag", "mag_mag", "S0", "S1", "S2"]

    extractor = GaitFeatureExtractor(params=params, lfeat=lfeat)
    tensores = extractor.process_interval(dict_por_pie)

    if "Left" not in tensores or "Right" not in tensores:
        raise ValueError("GaitFeatureExtractor no genero tensores para ambos pies.")

    # process_interval devuelve (Sxx, t, f) por pie, con Sxx de forma
    # (n_features_apiladas, n_frames_temporales) -- confirmado empiricamente
    x_right_raw, t_frames_right, _f_right = tensores["Right"]
    x_left_raw, t_frames_left, _f_left = tensores["Left"]

    min_len = min(len(t_frames_right), len(t_frames_left))

    # APILAR AMBOS PIES: (174, n_frames) + (174, n_frames) -> (348, n_frames),
    # reproduciendo la forma "Both" del dataset original
    x_both_raw = np.vstack([x_right_raw[:, :min_len], x_left_raw[:, :min_len]])

    # TRANSPONER a (n_frames, 348) y NORMALIZAR de uint8 (0-255) a [0,1],
    # igual que se almaceno en el HDF5 original
    x_both = (x_both_raw.T.astype(np.float32)) / 255.0

    # ALINEAR POSICION OBJETIVO A LOS FRAMES DE SALIDA DEL PSD
    t_absolutos = (df.index - df.index[0]).total_seconds().values

    y_pos = np.zeros((min_len, 2), dtype=np.float32)
    mascara_gps_real = np.zeros(min_len, dtype=bool)

    for i, t_centro in enumerate(t_frames_right[:min_len]):
        idx_cercano = int(np.argmin(np.abs(t_absolutos - t_centro)))
        y_pos[i, 0] = df[columna_x].iloc[idx_cercano]
        y_pos[i, 1] = df[columna_y].iloc[idx_cercano]
        if devolver_mascara_gps:
            mascara_gps_real[i] = bool(df["tiene_gps"].iloc[idx_cercano])

    if devolver_mascara_gps:
        return x_both, y_pos, mascara_gps_real
    return x_both, y_pos
