# -*- coding: utf-8 -*-
"""
Diagnostico de calidad de senal IMU cruda entre pie derecho e izquierdo.

Este script NO modifica ni reemplaza el pipeline. Su unico objetivo es
responder una pregunta: el drift asimetrico observado en la trayectoria
3D, ya esta presente en el dato crudo del sensor (offset, ruido, escala
distinta entre pies), o aparece recien durante el procesamiento
(Madgwick, integracion)? Sin esta evidencia, cualquier cambio en
kinematic_engine.py seria una corazonada, no un diagnostico.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from A01_EXTRACCION_DATOS.extract_data_plus import cInfluxDB
from A06_ANALISIS_CINEMATICO.event_detector import EventDetector, EventDetectorConfig
from A06_ANALISIS_CINEMATICO.kinematic_engine import KinematicEngine, KinematicConfig


def extraer_pie(config_yaml: Path, paciente: str, inicio: datetime, fin: datetime, pie: str) -> pd.DataFrame:
    """Extrae telemetria cruda de un pie, igual que el orquestador."""
    extractor = cInfluxDB(config_path=str(config_yaml))
    try:
        df_pie = extractor.query_data(
            from_date=inicio, to_date=fin, qtok=paciente, pie=pie
        )
    finally:
        extractor.close()

    if df_pie is None or df_pie.empty:
        raise ValueError(f"DATOS NO ENCONTRADOS PARA PIE {pie}")

    return df_pie


def resolver_columnas_imu(df_pie: pd.DataFrame):
    """Detecta el esquema de columnas IMU presente, igual que el orquestador."""
    esquemas_acc = [["Acc_X", "Acc_Y", "Acc_Z"], ["Ax", "Ay", "Az"]]
    esquemas_gyro = [["Gyro_X", "Gyro_Y", "Gyro_Z"], ["Gx", "Gy", "Gz"]]

    columnas_acc = next(
        (e for e in esquemas_acc if all(c in df_pie.columns for c in e)), None
    )
    columnas_gyro = next(
        (e for e in esquemas_gyro if all(c in df_pie.columns for c in e)), None
    )

    if columnas_acc is None or columnas_gyro is None:
        raise KeyError(f"Columnas IMU no reconocidas: {list(df_pie.columns)}")

    return columnas_acc, columnas_gyro


def barrer_beta(
    acc_xyz: np.ndarray,
    gyro_xyz: np.ndarray,
    hs_idx: np.ndarray,
    to_idx: np.ndarray,
    mascara_stance: np.ndarray,
    pie: str,
    fs: int
) -> float:
    """
    Barre valores de madgwick_beta y mide estabilidad en stance.

    Para cada beta candidato, ejecuta Madgwick completo y mide el
    desvio estandar de la componente Z global durante las fases de
    stance detectadas. Un beta optimo deberia minimizar esa dispersion,
    ya que en stance el pie esta quieto y la aceleracion deberia estar
    dominada casi por completo por la gravedad estatica y estable.

    :param acc_xyz: Aceleracion cruda del sensor, forma (N, 3).
    :param gyro_xyz: Velocidad angular cruda en grados, forma (N, 3).
    :param hs_idx: Indices Heel Strike, requeridos por _convertir_aceleracion.
    :param to_idx: Indices Toe Off, requeridos por _convertir_aceleracion.
    :param mascara_stance: Mascara booleana de muestras en stance.
    :param pie: Lado analizado, solo para trazabilidad en log.
    :param fs: Frecuencia de muestreo (Hz).
    :return: Valor de beta que minimiza la dispersion en stance.
    """
    betas_candidatos = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    resultados = []

    gyro_rad = np.deg2rad(gyro_xyz)

    print(f"\n--- BARRIDO DE BETA (PIE {pie}) ---")

    for beta in betas_candidatos:
        engine_prueba = KinematicEngine(KinematicConfig(fs=fs, madgwick_beta=beta))
        acc_convertida = engine_prueba._convertir_aceleracion(acc_xyz, hs_idx, to_idx)
        quaternions = engine_prueba.estimate_orientation(gyro_rad, acc_convertida)
        global_acc = engine_prueba.rotate_to_global(acc_convertida, quaternions)

        if mascara_stance.sum() > 0:
            z_stance = global_acc[mascara_stance, 2]
            media = z_stance.mean()
            std = z_stance.std()
        else:
            media = np.nan
            std = np.nan

        resultados.append((beta, media, std))
        print(f"  beta={beta:.2f}: Z_stance media={media:.3f}  std={std:.3f}")

    # SELECCIONAR MENOR DISPERSION
    stds = [r[2] for r in resultados]
    indice_optimo = int(np.nanargmin(stds))
    beta_optimo = resultados[indice_optimo][0]

    print(f"BETA OPTIMO PIE {pie}: {beta_optimo:.2f} "
          f"(menor std = {resultados[indice_optimo][2]:.3f})")

    return beta_optimo


def diagnosticar_pie(
    config_yaml: Path, paciente: str, inicio: datetime, fin: datetime, pie: str, fs: int
) -> dict:
    """Calcula estadisticas crudas de calidad de senal para un pie."""
    df_pie = extraer_pie(config_yaml, paciente, inicio, fin, pie)
    columnas_acc, columnas_gyro = resolver_columnas_imu(df_pie)

    acc_xyz = df_pie[columnas_acc].astype(float).values
    gyro_xyz = df_pie[columnas_gyro].astype(float).values

    # NORMA ACELERACION CRUDA (deberia rondar 1.0 g o 9.81 m/s^2 segun escala)
    norma_acc = np.linalg.norm(acc_xyz, axis=1)

    # NORMA GIROSCOPIO CRUDO
    norma_gyro = np.linalg.norm(gyro_xyz, axis=1)

    # DETECTAR STANCE REAL
    presion = df_pie[["S0", "S1", "S2"]].astype(float).sum(axis=1).values
    baseline = np.percentile(presion, 5)
    presion = presion - baseline
    presion[presion < 0] = 0

    detector = EventDetector(EventDetectorConfig(fs=fs))
    hs_idx, to_idx, stance_pct = detector.detect_from_pressure(presion)

    # CONSTRUIR MASCARA STANCE
    mascara_stance = np.zeros(len(presion), dtype=bool)
    for hs in hs_idx:
        futuros_to = to_idx[to_idx > hs]
        if len(futuros_to) == 0:
            continue
        to = futuros_to[0]
        mascara_stance[hs:to] = True

    # BARRER BETA OPTIMO
    beta_optimo = barrer_beta(acc_xyz, gyro_xyz, hs_idx, to_idx, mascara_stance, pie, fs)

    # NORMA ACC SOLO DURANTE STANCE
    if mascara_stance.sum() > 0:
        norma_acc_stance = norma_acc[mascara_stance]
        norma_acc_stance_mean = norma_acc_stance.mean()
        norma_acc_stance_std = norma_acc_stance.std()
    else:
        norma_acc_stance_mean = np.nan
        norma_acc_stance_std = np.nan

    # ESTIMAR ORIENTACION CON MADGWICK REAL (usa hs_idx/to_idx, firma actual)
    engine = KinematicEngine(KinematicConfig(fs=fs))
    acc_convertida = engine._convertir_aceleracion(acc_xyz, hs_idx, to_idx)
    gyro_rad = np.deg2rad(gyro_xyz)
    quaternions = engine.estimate_orientation(gyro_rad, acc_convertida)
    global_acc = engine.rotate_to_global(acc_convertida, quaternions)

    # NORMA GLOBAL Z SOLO EN STANCE (deberia acercarse a 9.81 y ser ESTABLE)
    if mascara_stance.sum() > 0:
        global_z_stance = global_acc[mascara_stance, 2]
        global_z_stance_mean = global_z_stance.mean()
        global_z_stance_std = global_z_stance.std()
    else:
        global_z_stance_mean = np.nan
        global_z_stance_std = np.nan

    # COMPARAR PRIMERA VS ULTIMA FASE STANCE (divergencia con el tiempo)
    indices_stance = np.where(mascara_stance)[0]
    if len(indices_stance) > 200:
        primera_decima = indices_stance[:len(indices_stance) // 10]
        ultima_decima = indices_stance[-len(indices_stance) // 10:]
        z_inicio = global_acc[primera_decima, 2].mean()
        z_final = global_acc[ultima_decima, 2].mean()
    else:
        z_inicio = np.nan
        z_final = np.nan

    resumen = {
        "pie": pie,
        "n_muestras": len(df_pie),
        "columnas_acc": columnas_acc,
        "columnas_gyro": columnas_gyro,
        "acc_mean_por_eje": acc_xyz.mean(axis=0),
        "acc_std_por_eje": acc_xyz.std(axis=0),
        "norma_acc_mean": norma_acc.mean(),
        "norma_acc_std": norma_acc.std(),
        "norma_acc_min": norma_acc.min(),
        "norma_acc_max": norma_acc.max(),
        "gyro_mean_por_eje": gyro_xyz.mean(axis=0),
        "gyro_std_por_eje": gyro_xyz.std(axis=0),
        "norma_gyro_mean": norma_gyro.mean(),
        "norma_gyro_std": norma_gyro.std(),
        "acc_xyz": acc_xyz,
        "gyro_xyz": gyro_xyz,
        "norma_acc": norma_acc,
        "stance_pct": stance_pct,
        "norma_acc_stance_mean": norma_acc_stance_mean,
        "norma_acc_stance_std": norma_acc_stance_std,
        "global_z_stance_mean": global_z_stance_mean,
        "global_z_stance_std": global_z_stance_std,
        "global_acc": global_acc,
        "mascara_stance": mascara_stance,
        "z_inicio_sesion": z_inicio,
        "z_final_sesion": z_final,
        "beta_optimo": beta_optimo,
    }

    return resumen


def imprimir_resumen(resumen: dict) -> None:
    """Imprime el resumen de un pie en consola."""
    print(f"\n{'=' * 60}")
    print(f"PIE {resumen['pie']}")
    print(f"{'=' * 60}")
    print(f"Muestras:              {resumen['n_muestras']}")
    print(f"Columnas Acc usadas:   {resumen['columnas_acc']}")
    print(f"Columnas Gyro usadas:  {resumen['columnas_gyro']}")
    print()
    print(f"Acc media por eje (X,Y,Z): {resumen['acc_mean_por_eje']}")
    print(f"Acc std por eje (X,Y,Z):   {resumen['acc_std_por_eje']}")
    print(f"Norma Acc: media={resumen['norma_acc_mean']:.3f}  "
          f"std={resumen['norma_acc_std']:.3f}  "
          f"min={resumen['norma_acc_min']:.3f}  "
          f"max={resumen['norma_acc_max']:.3f}")
    print(f"  (Referencia: en reposo o marcha normal, la norma de Acc")
    print(f"   deberia oscilar cerca de 9.81 m/s^2 si las unidades son m/s^2,")
    print(f"   o cerca de 1.0 'g' si las unidades estan normalizadas por g)")
    print()
    print(f"Gyro media por eje (X,Y,Z): {resumen['gyro_mean_por_eje']}")
    print(f"Gyro std por eje (X,Y,Z):   {resumen['gyro_std_por_eje']}")
    print(f"Norma Gyro: media={resumen['norma_gyro_mean']:.3f}  "
          f"std={resumen['norma_gyro_std']:.3f}")
    print()
    print(f"Stance detectado: {resumen['stance_pct']:.1f}%")
    print(f"Norma Acc SOLO EN STANCE (pie quieto en el suelo): "
          f"media={resumen['norma_acc_stance_mean']:.3f}  "
          f"std={resumen['norma_acc_stance_std']:.3f}")
    print(f"  (En stance, la aceleracion deberia estar dominada casi por")
    print(f"   completo por la gravedad estatica. Si este valor converge")
    print(f"   cerca de 1.0, las unidades son 'g'. Si converge cerca de")
    print(f"   9.81, las unidades ya son m/s^2 y el problema es otro.)")
    print()
    print("--- ORIENTACION MADGWICK (marco global, tras conversion y rotacion) ---")
    print(f"Z global en stance: media={resumen['global_z_stance_mean']:.3f}  "
          f"std={resumen['global_z_stance_std']:.3f}")
    print("  (Si Madgwick converge bien, esto deberia acercarse a 9.81 con")
    print("   poca dispersion. Alta std indica orientacion inestable/ruidosa.)")
    print(f"Z global PRIMER 10% de stance: {resumen['z_inicio_sesion']:.3f}")
    print(f"Z global ULTIMO 10% de stance: {resumen['z_final_sesion']:.3f}")
    print("  (Si estos dos valores difieren mucho entre si, Madgwick esta")
    print("   DIVERGIENDO con el tiempo, no solo teniendo un offset fijo.)")


def graficar_comparativo(resumen_right: dict, resumen_left: dict, output_dir: Path, paciente: str) -> Path:
    """Genera grafica comparativa de norma de aceleracion entre pies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(resumen_right["norma_acc"], label="Pie Derecho", color="blue", linewidth=0.8)
    axes[0].axhline(9.81, color="black", linestyle="--", linewidth=1, label="9.81 m/s^2 (referencia)")
    axes[0].set_title(f"Norma Aceleracion Cruda - Pie Derecho - {paciente}")
    axes[0].set_ylabel("m/s^2")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(resumen_left["norma_acc"], label="Pie Izquierdo", color="red", linewidth=0.8)
    axes[1].axhline(9.81, color="black", linestyle="--", linewidth=1, label="9.81 m/s^2 (referencia)")
    axes[1].set_title(f"Norma Aceleracion Cruda - Pie Izquierdo - {paciente}")
    axes[1].set_xlabel("Muestras")
    axes[1].set_ylabel("m/s^2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(paciente, fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / f"{paciente}_diagnostico_imu.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def graficar_orientacion(resumen_right: dict, resumen_left: dict, output_dir: Path, paciente: str) -> Path:
    """Genera grafica de Z global durante stance a lo largo del tiempo."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    z_right = resumen_right["global_acc"][:, 2].copy()
    z_right[~resumen_right["mascara_stance"]] = np.nan
    axes[0].plot(z_right, label="Pie Derecho (Z global en stance)", color="blue", linewidth=0.8)
    axes[0].axhline(9.81, color="black", linestyle="--", linewidth=1, label="9.81 m/s^2 (referencia)")
    axes[0].set_title(f"Z Global en Stance a lo largo de la sesion - Pie Derecho - {paciente}")
    axes[0].set_ylabel("m/s^2")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    z_left = resumen_left["global_acc"][:, 2].copy()
    z_left[~resumen_left["mascara_stance"]] = np.nan
    axes[1].plot(z_left, label="Pie Izquierdo (Z global en stance)", color="red", linewidth=0.8)
    axes[1].axhline(9.81, color="black", linestyle="--", linewidth=1, label="9.81 m/s^2 (referencia)")
    axes[1].set_title(f"Z Global en Stance a lo largo de la sesion - Pie Izquierdo - {paciente}")
    axes[1].set_xlabel("Muestras")
    axes[1].set_ylabel("m/s^2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(paciente, fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / f"{paciente}_diagnostico_orientacion.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Diagnostico de calidad de senal IMU cruda.")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fin", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument(
        "--config-yaml", type=str,
        default=str(PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml")
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).resolve().parent.parent / "RESULTADOS_BIOMECANICO")
    )
    args = parser.parse_args()

    paciente = args.paciente
    inicio = datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S")
    fin = datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S")
    config_yaml = Path(args.config_yaml)

    # CARPETA DE SALIDA POR PACIENTE
    output_dir = Path(args.output_dir) / paciente
    output_dir.mkdir(parents=True, exist_ok=True)

    resumen_right = diagnosticar_pie(config_yaml, paciente, inicio, fin, "Right", args.fs)
    resumen_left = diagnosticar_pie(config_yaml, paciente, inicio, fin, "Left", args.fs)

    imprimir_resumen(resumen_right)
    imprimir_resumen(resumen_left)

    print(f"\n{'=' * 60}")
    print("COMPARATIVA DIRECTA")
    print(f"{'=' * 60}")
    print(f"Diferencia norma Acc media (Right - Left): "
          f"{resumen_right['norma_acc_mean'] - resumen_left['norma_acc_mean']:.3f}")
    print(f"Diferencia norma Acc std  (Right - Left): "
          f"{resumen_right['norma_acc_std'] - resumen_left['norma_acc_std']:.3f}")
    print(f"Diferencia norma Gyro std (Right - Left): "
          f"{resumen_right['norma_gyro_std'] - resumen_left['norma_gyro_std']:.3f}")

    ruta_grafica = graficar_comparativo(resumen_right, resumen_left, output_dir, paciente)
    print(f"\nGrafica guardada en: {ruta_grafica}")

    ruta_orientacion = graficar_orientacion(resumen_right, resumen_left, output_dir, paciente)
    print(f"Grafica de orientacion guardada en: {ruta_orientacion}")

    # DETERMINAR UNIDADES REALES
    print(f"\n{'=' * 60}")
    print("CONCLUSION SOBRE UNIDADES")
    print(f"{'=' * 60}")

    referencia_stance = np.nanmean([
        resumen_right["norma_acc_stance_mean"],
        resumen_left["norma_acc_stance_mean"]
    ])

    distancia_a_g = abs(referencia_stance - 1.0)
    distancia_a_ms2 = abs(referencia_stance - 9.81)

    print(f"Norma Acc promedio en stance (ambos pies): {referencia_stance:.3f}")

    if distancia_a_g < distancia_a_ms2:
        print(
            f"CONCLUSION: el valor esta mucho mas cerca de 1.0 'g' que de "
            f"9.81 m/s^2 (distancias: {distancia_a_g:.3f} vs {distancia_a_ms2:.3f}). "
            f"Los datos crudos de aceleracion probablemente estan en 'g'."
        )
    else:
        print(
            f"CONCLUSION: el valor esta mas cerca de 9.81 m/s^2 que de 1.0 'g' "
            f"(distancias: {distancia_a_g:.3f} vs {distancia_a_ms2:.3f}). "
            f"Las unidades parecen correctas; el drift observado tendria "
            f"otra causa distinta a las unidades de aceleracion."
        )


if __name__ == "__main__":
    main()