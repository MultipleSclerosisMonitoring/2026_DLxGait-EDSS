# -*- coding: utf-8 -*-
"""
Analiza un segmento de marcha (paciente + rango de fechas, por consola,
igual que Agnostic_evaluator.py) usando el modelo GENERICO de trayectoria
GPS+Transformer ya entrenado (trajectory_model_generico.pth), SIN
reentrenar. Calcula los parametros clinicos de marcha especificados en
el TFM (Muller et al., 2021):

  - Velocidad de marcha, longitud de zancada, tiempo de zancada,
    stance/swing: calculados a partir de la trayectoria (X, Y) predicha
    por el modelo GPS+Transformer.
  - MTC (Minimum Toe Clearance): requiere la componente vertical (Z) de
    la posicion del pie, que el modelo GPS+Transformer NO predice (el
    GPS solo da posicion horizontal). Se calcula, de forma explicita y
    documentada, con el motor Madgwick+ZUPT del pipeline A06 original
    (Orquestador_biomecanico.py / kinematic_engine.py), presentado como
    fuente independiente y complementaria, no mezclada con la trayectoria
    del Transformer.
  - Pendiente de fatiga: regresion lineal (fatigue_analysis.py) sobre los
    parametros de ambas fuentes, permitiendo comparar.

Esto NO reentrena nada -- carga el modelo ya entrenado por
entrenar_trajectory_model_multi.py y corre solo inferencia.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "A01_EXTRACCION_DATOS"))
sys.path.insert(0, str(PROJECT_ROOT / "A04_TRANSFORMER"))
sys.path.insert(0, str(PROJECT_ROOT / "A06_ANALISIS_CINEMATICO"))

from extract_data_plus import ExtractionParams
from AA_TRANSFORMER_V1 import TransformerConfig

import preparar_dataset_trayectoria as pdt
import conector_trayectoria as ct
from trajectory_model import TrajectoryModel
from entrenar_trajectory_model_multi import VentanasSecuenciaDataset

from event_detector import EventDetector, EventDetectorConfig
from kinematic_engine import KinematicEngine, KinematicConfig
from fatigue_analysis import FatigueAnalyzer, FatigueConfig


def predecir_trayectoria(
    df: pd.DataFrame,
    params: ExtractionParams,
    t_cfg: TransformerConfig,
    modelo: TrajectoryModel,
    device: str,
    media_x: float,
    std_x: float,
    media_y: float,
    std_y: float
) -> np.ndarray:
    """
    Corre inferencia del modelo GPS+Transformer sobre un DataFrame ya
    preparado, y desnormaliza la salida a metros reales.

    :param df: DataFrame combinado (columnas crudas sufijadas por pie),
        sin necesidad de columnas de posicion objetivo.
    :param params: ExtractionParams usados para generar el PSD (deben
        coincidir con los usados al entrenar).
    :param t_cfg: TransformerConfig del checkpoint.
    :param modelo: TrajectoryModel ya cargado con pesos entrenados.
    :param device: "cuda" o "cpu".
    :param media_x, std_x, media_y, std_y: Parametros de normalizacion a
        usar para desnormalizar (por defecto, los promedios globales del
        entrenamiento multi-segmento; ver stats_normalizacion).
    :return: Array (n_frames, 2) con la trayectoria predicha en metros,
        alineada a los mismos frames que devuelve el extractor PSD.
    """
    df_temp = df.copy()
    df_temp["X_m_gt"] = 0.0
    df_temp["Y_m_gt"] = 0.0

    x_psd, _y_dummy = ct.generar_tensores_psd_y_posicion(
        df_temp, params, columna_x="X_m_gt", columna_y="Y_m_gt"
    )

    max_len = t_cfg.max_len
    n_frames = len(x_psd)

    if n_frames < max_len:
        raise ValueError(
            f"Segmento demasiado corto: se generaron {n_frames} frames de PSD, "
            f"se necesitan al menos {max_len} para una ventana del Transformer."
        )

    modelo.eval()
    trayectoria_norm = np.zeros((n_frames, 2), dtype=np.float32)
    conteo = np.zeros(n_frames, dtype=np.int32)

    with torch.no_grad():
        for i in range(n_frames - max_len + 1):
            ventana = x_psd[i:i + max_len]
            xb = torch.from_numpy(ventana).unsqueeze(0).to(device)
            pred = modelo(xb).cpu().numpy()[0]
            idx_destino = i + max_len - 1
            trayectoria_norm[idx_destino] += pred
            conteo[idx_destino] += 1

    primer_valido = int(np.argmax(conteo > 0))
    for i in range(primer_valido):
        trayectoria_norm[i] = trayectoria_norm[primer_valido]
        conteo[i] = 1

    conteo[conteo == 0] = 1
    trayectoria_norm /= conteo[:, np.newaxis]

    trayectoria_m = np.zeros_like(trayectoria_norm)
    trayectoria_m[:, 0] = trayectoria_norm[:, 0] * std_x + media_x
    trayectoria_m[:, 1] = trayectoria_norm[:, 1] * std_y + media_y

    return trayectoria_m


def calcular_parametros_desde_trayectoria(
    trayectoria_m: np.ndarray,
    t_frames_seg: np.ndarray,
    presion: np.ndarray,
    fs: int,
    umbral_presion: float
) -> dict:
    """
    Calcula velocidad, longitud/tiempo de zancada y stance/swing a partir
    de la trayectoria (X, Y) predicha por el modelo GPS+Transformer y la
    deteccion de eventos HS/TO por presion plantar.

    :param trayectoria_m: Trayectoria (n_frames_psd, 2) en metros, con
        muestreo temporal igual a t_frames_seg (NO a fs directamente).
    :param t_frames_seg: Tiempos (segundos desde inicio) de cada fila de
        trayectoria_m.
    :param presion: Señal de presion plantar combinada (S0+S1+S2), a
        frecuencia fs, usada solo para detectar eventos HS/TO.
    :param fs: Frecuencia de muestreo de la señal de presion cruda.
    :param umbral_presion: Umbral de deteccion (fraccion), auto-calibrado
        o fijo.
    :return: Diccionario con arrays de metricas por zancada.
    """
    detector = EventDetector(EventDetectorConfig(fs=fs))
    hs_idx, to_idx, stance_pct = detector.detect_from_pressure(
        presion, threshold_fraction_override=umbral_presion
    )

    if len(to_idx) > 0 and len(hs_idx) > 0 and to_idx[0] < hs_idx[0]:
        to_idx = to_idx[1:]
    min_len = min(len(hs_idx), len(to_idx))
    hs_idx, to_idx = hs_idx[:min_len], to_idx[:min_len]

    hs_tiempos_s = hs_idx / fs

    pos_en_hs_x = np.interp(hs_tiempos_s, t_frames_seg, trayectoria_m[:, 0])
    pos_en_hs_y = np.interp(hs_tiempos_s, t_frames_seg, trayectoria_m[:, 1])

    engine = KinematicEngine(KinematicConfig(fs=fs))

    temporal_metrics = engine.compute_temporal_metrics(hs_idx, to_idx)

    stride_lengths = []
    for i in range(len(hs_idx) - 1):
        dx = pos_en_hs_x[i + 1] - pos_en_hs_x[i]
        dy = pos_en_hs_y[i + 1] - pos_en_hs_y[i]
        stride_lengths.append(np.sqrt(dx**2 + dy**2))
    stride_lengths = np.array(stride_lengths)

    gait_speed = engine.compute_gait_speed(stride_lengths, temporal_metrics["stride_times"])

    return {
        "hs_idx": hs_idx,
        "to_idx": to_idx,
        "stance_pct": stance_pct,
        "stride_lengths": stride_lengths,
        "gait_speed": gait_speed,
        **temporal_metrics
    }


def main() -> None:
    """Punto de entrada CLI: analiza un segmento usando el modelo GPS+Transformer ya entrenado."""
    parser = argparse.ArgumentParser(description="Analiza un segmento con el modelo GPS+Transformer (sin reentrenar).")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fin", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--es-utc", action="store_true", help="Si inicio/fin ya estan en UTC")
    parser.add_argument("--config-yaml", type=str, default=str(PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml"))
    parser.add_argument("--models-dir", type=str, default=str(PROJECT_ROOT / "A05_MODELOS_ENTRENADOS"))
    parser.add_argument(
        "--trayectoria-dir", type=str,
        default=str(PROJECT_ROOT / "A07_TRAYECTORIA_GPS" / "RESULTADOS_TRAYECTORIA")
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "A07_TRAYECTORIA_GPS" / "RESULTADOS_TRAYECTORIA")
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)
    trayectoria_dir = Path(args.trayectoria_dir)

    t_cfg_dict = joblib.load(models_dir / "transformer_config.joblib")
    t_cfg = TransformerConfig(**t_cfg_dict)

    params = ExtractionParams(f_start_hz=0.25, f_stop_hz=29.0)

    inicio_dt = datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S")
    fin_dt = datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*70}\nANALIZANDO SEGMENTO: {args.paciente} | {inicio_dt} -> {fin_dt}\n{'='*70}")

    df_completo = pdt.extraer_datos_crudos(
        args.config_yaml, args.paciente, inicio_dt, fin_dt, es_utc=args.es_utc
    )

    modelo = TrajectoryModel(t_cfg, congelar_transformer=False).to(device)
    ruta_modelo = trayectoria_dir / "trajectory_model_generico.pth"
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=device))
    print(f"Modelo GPS+Transformer cargado desde: {ruta_modelo}")

    stats_normalizacion = joblib.load(trayectoria_dir / "trajectory_model_stats_normalizacion.joblib")
    media_x_prom = float(np.mean([s["media_x"] for s in stats_normalizacion.values()]))
    std_x_prom = float(np.mean([s["std_x"] for s in stats_normalizacion.values()]))
    media_y_prom = float(np.mean([s["media_y"] for s in stats_normalizacion.values()]))
    std_y_prom = float(np.mean([s["std_y"] for s in stats_normalizacion.values()]))
    print(
        "ADVERTENCIA: se usan medias/std PROMEDIO globales del entrenamiento "
        "para desnormalizar (no hay GPS real en este segmento nuevo para "
        "calibrar una normalizacion especifica); la posicion absoluta "
        "resultante debe interpretarse como orientativa, no exacta."
    )

    trayectoria_m = predecir_trayectoria(
        df_completo, params, t_cfg, modelo, device,
        media_x_prom, std_x_prom, media_y_prom, std_y_prom
    )

    dict_por_pie_tmp = ct.dataframe_combinado_a_dict_por_pie(df_completo)
    from extract_data_plus import GaitFeatureExtractor
    extractor_tmp = GaitFeatureExtractor(params=params, lfeat=["acc_mag", "gyro_mag", "mag_mag", "S0", "S1", "S2"])
    tensores_tmp = extractor_tmp.process_interval(dict_por_pie_tmp)
    _, t_frames_seg, _ = tensores_tmp["Right"]
    t_frames_seg = t_frames_seg[:len(trayectoria_m)]

    presion_right = (
        df_completo["S0_Right"].astype(float)
        + df_completo["S1_Right"].astype(float)
        + df_completo["S2_Right"].astype(float)
    ).values
    baseline = np.percentile(presion_right, 5)
    presion_right = np.clip(presion_right - baseline, 0, None)

    fs = 100
    parametros_transformer = calcular_parametros_desde_trayectoria(
        trayectoria_m, t_frames_seg, presion_right, fs=fs, umbral_presion=0.3
    )

    print("\n" + "=" * 70)
    print("PARAMETROS DESDE TRAYECTORIA GPS+TRANSFORMER (X, Y horizontal)")
    print("=" * 70)
    print(f"Pasos detectados: {len(parametros_transformer['hs_idx'])}")
    print(f"Velocidad media: {np.nanmean(parametros_transformer['gait_speed']):.3f} m/s")
    print(f"Zancada media:   {np.nanmean(parametros_transformer['stride_lengths']):.3f} m")
    print(f"Tiempo zancada:  {np.nanmean(parametros_transformer['stride_times']):.3f} s")
    print(f"Stance medio:    {np.nanmean(parametros_transformer['stance_times']):.3f} s")
    print(f"Swing medio:     {np.nanmean(parametros_transformer['swing_times']):.3f} s")

    print(
        "\nNOTA: MTC (Minimum Toe Clearance) requiere la componente vertical "
        "(Z) de la posicion del pie, que este modelo NO predice (el GPS solo "
        "da posicion horizontal). Para MTC, usar Orquestador_biomecanico.py "
        "(motor Madgwick+ZUPT de A06) sobre el mismo segmento, como fuente "
        "independiente y complementaria."
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_export = pd.DataFrame({
        "stride_length_m": parametros_transformer["stride_lengths"],
        "stride_time_s": parametros_transformer["stride_times"][:len(parametros_transformer["stride_lengths"])],
        "gait_speed_ms": parametros_transformer["gait_speed"],
    })
    ruta_csv = out_dir / f"{args.paciente}_parametros_trayectoria_transformer.csv"
    df_export.to_csv(ruta_csv, index=False)
    print(f"\nParametros guardados en: {ruta_csv}")


if __name__ == "__main__":
    main()
