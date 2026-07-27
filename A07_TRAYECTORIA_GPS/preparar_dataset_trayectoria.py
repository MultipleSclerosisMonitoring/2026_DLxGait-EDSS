# -*- coding: utf-8 -*-
"""
Extraccion y preparacion del dataset de trayectoria corporal, combinando
presion plantar + IMU de ambos pies como entrada, y GPS (compartido,
proyectado a metros locales via UTM) como ground truth supervisado solo
en los instantes donde hay lectura GPS real.

A diferencia del pipeline biomecanico A06 (que usa Madgwick+ZUPT, motor
fisico deterministico), este dataset se genera para entrenar un modelo
de aprendizaje que prediga la trayectoria del cuerpo directamente desde
presion+IMU, usando el GPS real como supervision dispersa.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Tuple

import yaml
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from pyproj import Transformer
from scipy.interpolate import PchipInterpolator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _utm_epsg_para(lat: float, lng: float) -> int:
    """
    Determina el codigo EPSG de la zona UTM correcta para una coordenada
    dada (hemisferio norte/sur segun signo de latitud).

    :param lat: Latitud en grados decimales.
    :param lng: Longitud en grados decimales.
    :return: Codigo EPSG de la zona UTM correspondiente.
    """
    zona = int((lng + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zona  # UTM Norte
    return 32700 + zona  # UTM Sur


def extraer_datos_crudos(
    config_path: str, paciente: str, inicio: datetime, fin: datetime, es_utc: bool = False
) -> pd.DataFrame:
    """
    Consulta InfluxDB por presion plantar, IMU (Acc/Gyro) y GPS (lat/lng)
    de ambos pies, en el rango de fechas dado.

    :param config_path: Ruta al config.yaml de InfluxDB.
    :param paciente: CodeID del paciente.
    :param inicio: Fecha de inicio. Hora local (segun tzval del config)
        salvo que es_utc=True.
    :param fin: Fecha de fin. Misma regla que inicio.
    :param es_utc: Si True, inicio/fin ya estan en UTC y no se aplica
        conversion de zona horaria (util para segmentos dados en formato
        ISO/UTC con sufijo 'Z', evitando el desfase horario que ocurre si
        se interpretan por error como hora local).
    :return: DataFrame con columnas de ambos pies, indexado por tiempo.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["influxdb"]

    if es_utc:
        inicio_str = inicio.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fin_str = fin.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        tzval = cfg.get("tzval", "Europe/Madrid")
        inicio_str = inicio.replace(tzinfo=ZoneInfo(tzval)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fin_str = fin.replace(tzinfo=ZoneInfo(tzval)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    client = InfluxDBClient(url=cfg["url"], token=cfg["token"], org=cfg["org"], verify_ssl=False)


    metrics = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz", "S0", "S1", "S2"]
    metrics_str = " or ".join([f'r._field == "{m}"' for m in metrics])
    columns_str = ", ".join([f'"{m}"' for m in metrics] + ['"lat"', '"lng"'])

    dfs_por_pie = {}
    try:
        for pie in ["Left", "Right"]:
            query = f'''
            from(bucket: "{cfg['bucket']}")
            |> range(start: {inicio_str}, stop: {fin_str})
            |> filter(fn: (r) => r._measurement == "Gait")
            |> filter(fn: (r) => {metrics_str})
            |> filter(fn: (r) => r["CodeID"] == "{paciente}" and r["type"] == "SCKS" and r["Foot"] == "{pie}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> keep(columns: ["_time", {columns_str}])
            '''
            tables = client.query_api().query(query, org=cfg["org"])
            data = [rec.values for t in tables for rec in t.records]
            df_pie = pd.DataFrame(data).drop(columns=["result", "table"], errors="ignore")

            if df_pie.empty:
                raise ValueError(f"DATOS NO ENCONTRADOS PARA PIE {pie}")

            df_pie["_time"] = pd.to_datetime(df_pie["_time"], unit="ns", utc=True)
            df_pie = df_pie.set_index("_time").sort_index()

            # RENOMBRAR COLUMNAS CON SUFIJO POR PIE (excepto lat/lng, que son compartidas)
            renombre = {c: f"{c}_{pie}" for c in metrics}
            df_pie = df_pie.rename(columns=renombre)

            dfs_por_pie[pie] = df_pie
    finally:
        client.close()

    # UNIR AMBOS PIES POR TIMESTAMP MAS CERCANO (merge_asof, tolerancia 20ms)
    df_left = dfs_por_pie["Left"].reset_index()
    df_right = dfs_por_pie["Right"].reset_index()

    df_left = df_left.sort_values("_time")
    df_right = df_right.sort_values("_time")

    columnas_right_sin_gps = [c for c in df_right.columns if c not in ("_time", "lat", "lng")]

    df_combinado = pd.merge_asof(
        df_left, df_right[["_time"] + columnas_right_sin_gps],
        on="_time", direction="nearest", tolerance=pd.Timedelta("20ms")
    )

    df_combinado = df_combinado.dropna(
        subset=[c for c in df_combinado.columns if c not in ("lat", "lng")]
    )

    return df_combinado.set_index("_time")


def proyectar_gps_a_metros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proyecta las columnas lat/lng a coordenadas locales en metros (X, Y),
    usando la zona UTM correspondiente al primer punto GPS valido de la
    sesion como referencia.

    IMPORTANTE: el GPS en origen viene "forward-filled" (el mismo valor se
    repite en decenas/cientos de muestras de IMU consecutivas hasta que el
    GPS real se actualiza), no llega uno nuevo por cada muestra de IMU. Por
    eso 'tiene_gps' NO marca simplemente valores no nulos (eso daria 100%),
    sino unicamente las filas donde lat/lng CAMBIARON respecto a la fila
    anterior -- es decir, el primer instante de cada lectura GPS real nueva.
    El resto de filas se tratan como sin supervision directa para evitar
    ensenar al modelo que la posicion se mantuvo fija durante ese intervalo,
    cuando en realidad el GPS simplemente no se habia refrescado todavia.

    :param df: DataFrame con columnas 'lat' y 'lng' (forward-filled en origen).
    :return: Mismo DataFrame con columnas nuevas 'X_m', 'Y_m' y 'tiene_gps'.
    """
    gps_validos = df.dropna(subset=["lat", "lng"])
    if gps_validos.empty:
        raise ValueError("NO HAY NINGUNA LECTURA GPS VALIDA EN EL RANGO SOLICITADO.")

    lat0 = float(gps_validos["lat"].iloc[0])
    lng0 = float(gps_validos["lng"].iloc[0])
    epsg_utm = _utm_epsg_para(lat0, lng0)

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)

    # ORIGEN LOCAL: PRIMER PUNTO GPS DE LA SESION
    x0, y0 = transformer.transform(lng0, lat0)

    x_m = np.full(len(df), np.nan)
    y_m = np.full(len(df), np.nan)

    mask_no_nulos = df["lat"].notna() & df["lng"].notna()
    lats = df.loc[mask_no_nulos, "lat"].astype(float).values
    lngs = df.loc[mask_no_nulos, "lng"].astype(float).values

    xs, ys = transformer.transform(lngs, lats)
    x_m[mask_no_nulos.values] = np.array(xs) - x0
    y_m[mask_no_nulos.values] = np.array(ys) - y0

    df = df.copy()
    df["X_m"] = x_m
    df["Y_m"] = y_m

    # DETECTAR CAMBIOS REALES: primera fila de cada "meseta" de lat/lng constante
    lat_cambio = df["lat"].ne(df["lat"].shift(1))
    lng_cambio = df["lng"].ne(df["lng"].shift(1))
    df["tiene_gps"] = (mask_no_nulos & (lat_cambio | lng_cambio)).values

    n_lecturas_reales = df["tiene_gps"].sum()
    print(f"Zona UTM utilizada: EPSG:{epsg_utm}")
    print(f"Origen local (lat, lng): ({lat0:.7f}, {lng0:.7f})")
    print(f"Filas con lat/lng no nulo (incluye forward-fill): {mask_no_nulos.sum()} de {len(df)} "
          f"({100 * mask_no_nulos.sum() / len(df):.2f}%)")
    print(f"Lecturas GPS REALMENTE NUEVAS (cambios de valor): {n_lecturas_reales} "
          f"({100 * n_lecturas_reales / len(df):.2f}%)")

    return df


def interpolar_ground_truth_pchip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un ground truth denso de trayectoria (X_m_gt, Y_m_gt) para TODAS
    las filas, interpolando entre los puntos GPS REALES (tiene_gps=True)
    mediante PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).

    Se eligio PCHIP sobre un cubic spline estandar porque, con solo ~18
    puntos reales espaciados varios segundos entre si, un cubic spline
    puede oscilar ("overshoot") entre puntos y generar trayectorias que se
    alejan de forma poco plausible del rango real observado. PCHIP preserva
    monotonia local y evita ese sobre-ajuste, a costa de una curva algo
    menos suave -- es la opcion mas conservadora disponible en scipy para
    este caso.

    IMPORTANTE (limitacion metodologica, debe documentarse en la memoria):
    este ground truth denso es una SUPOSICION MATEMATICA de como se movio
    el cuerpo entre dos lecturas GPS reales, no una medicion. Con ~35
    segundos de espaciado promedio entre puntos, la interpolacion puede no
    reflejar giros, paradas o cambios de direccion reales ocurridos en ese
    intervalo. El modelo entrenado contra este ground truth aprende, en el
    mejor caso, a imitar la interpolacion a partir de presion+IMU -- no
    necesariamente la trayectoria real. Por eso los 18 puntos reales se
    reservan EXCLUSIVAMENTE para validacion final (ver dividir_train_val),
    nunca se usan en el entrenamiextra, para que el error reportado al
    final sea honesto y no circular.

    :param df: DataFrame con columnas 'X_m', 'Y_m' (solo validas donde
        tiene_gps=True) y 'tiene_gps'.
    :return: Mismo DataFrame con columnas nuevas 'X_m_gt', 'Y_m_gt'
        (ground truth denso, interpolado, para TODAS las filas).
    """
    puntos_reales = df[df["tiene_gps"]]
    n_puntos = len(puntos_reales)

    if n_puntos < 3:
        raise ValueError(
            f"Se requieren al menos 3 puntos GPS reales para interpolar con PCHIP, "
            f"se encontraron {n_puntos}."
        )

    # PCHIP REQUIERE ABSCISAS ESTRICTAMENTE CRECIENTES: usar segundos desde
    # el inicio de la sesion como variable independiente
    t_todos = (df.index - df.index[0]).total_seconds().values
    t_reales = (puntos_reales.index - df.index[0]).total_seconds().values

    interp_x = PchipInterpolator(t_reales, puntos_reales["X_m"].values)
    interp_y = PchipInterpolator(t_reales, puntos_reales["Y_m"].values)

    df = df.copy()
    df["X_m_gt"] = interp_x(t_todos)
    df["Y_m_gt"] = interp_y(t_todos)

    # DESPLAZAMIENTO (DELTA) ENTRE FRAMES CONSECUTIVOS: se predice esto en
    # vez de la posicion absoluta, porque la velocidad de marcha es
    # fisiologicamente similar entre sesiones (independiente de cuanto
    # dure o cuanto se desplace en total la sesion), mientras que la
    # posicion absoluta puede variar en ordenes de magnitud entre
    # sesiones cortas (~pocos metros) y largas (~cientos de metros).
    # Esto deberia dar una escala mucho mas consistente para normalizar
    # y entrenar. El primer frame no tiene delta anterior, se rellena con 0.
    df["dX_m_gt"] = df["X_m_gt"].diff().fillna(0.0)
    df["dY_m_gt"] = df["Y_m_gt"].diff().fillna(0.0)

    # VALIDACION DE PLAUSIBILIDAD FISIOLOGICA: un desplazamiento de mas de
    # MAX_DELTA_PLAUSIBLE_M por FRAME de IMU (frecuencia alta, ~100Hz) es
    # fisiologicamente imposible para marcha humana. Si aparece, es
    # evidencia de que PCHIP genero un salto irreal entre dos puntos GPS
    # reales demasiado separados/mal distribuidos para esta sesion -- se
    # rechaza la sesion COMPLETA en vez de solo recortar (clip) el valor,
    # ya que un salto asi sugiere que la interpolacion completa de esa
    # sesion no es confiable, no solo ese punto puntual.
    MAX_DELTA_PLAUSIBLE_M = 2.0
    delta_max_abs = max(df["dX_m_gt"].abs().max(), df["dY_m_gt"].abs().max())
    if delta_max_abs > MAX_DELTA_PLAUSIBLE_M:
        raise ValueError(
            f"Desplazamiento interpolado implausible: {delta_max_abs:.2f} m en un "
            f"solo frame (umbral: {MAX_DELTA_PLAUSIBLE_M} m). Esto indica que PCHIP "
            f"genero un salto irreal para esta sesion (puntos GPS reales demasiado "
            f"separados/mal distribuidos); se descarta la sesion completa."
        )

    # TAMBIEN SE CALCULA EL DELTA DEL GPS REAL (X_m/Y_m), SOLO VALIDO EN
    # LOS FRAMES CON tiene_gps=True, para poder evaluar contra el
    # desplazamiento real entre dos lecturas GPS consecutivas (no
    # interpoladas) durante la validacion
    df["dX_m"] = np.nan
    df["dY_m"] = np.nan
    if n_puntos > 1:
        x_reales = puntos_reales["X_m"].values
        y_reales = puntos_reales["Y_m"].values
        dx_reales = np.diff(x_reales, prepend=x_reales[0])
        dy_reales = np.diff(y_reales, prepend=y_reales[0])
        # USAR .values EN LA MASCARA BOOLEANA (no el Index) PARA EVITAR
        # "Must have equal len keys and value" cuando el indice tiene
        # timestamps duplicados (ya visto en SBRHUG002-12)
        mask_reales = df["tiene_gps"].values
        df.loc[mask_reales, "dX_m"] = dx_reales
        df.loc[mask_reales, "dY_m"] = dy_reales

    print(f"Ground truth PCHIP generado a partir de {n_puntos} puntos GPS reales, "
          f"espaciado promedio: {np.diff(t_reales).mean():.1f} s")
    print(f"Desplazamiento (delta) por frame -- rango dX_m_gt: "
          f"[{df['dX_m_gt'].min():.4f}, {df['dX_m_gt'].max():.4f}] m, "
          f"rango dY_m_gt: [{df['dY_m_gt'].min():.4f}, {df['dY_m_gt'].max():.4f}] m")

    return df


def dividir_train_val(
    df: pd.DataFrame, margen_val_s: float = 0.5, contexto_val_s: float = 16.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el dataset en entrenamiento (todas las filas fuera del area de
    validacion, usando el ground truth interpolado X_m_gt/Y_m_gt) y
    validacion (una ventana de CONTEXTO continuo alrededor de cada punto
    GPS real, no solo el instante exacto).

    IMPORTANTE (correccion respecto a una version anterior): un punto GPS
    real es un unico instante disperso (los puntos reales tipicamente
    estan a >30s de distancia entre si). Si df_val solo contuviera esos
    instantes exactos, nunca podria formarse ni una sola ventana de
    secuencia de max_len frames (ej. 20) para el Transformer, porque esos
    instantes nunca son consecutivos entre si. Por eso ahora se incluye
    una ventana de contexto de +/- contexto_val_s alrededor de cada punto
    real: los frames de contexto proveen la secuencia de entrada necesaria
    para el Transformer, pero la METRICA DE ERROR solo se calcula sobre el
    frame central que coincide exactamente con el punto GPS real (ver
    analizar_segmento_trayectoria.py / entrenar_trajectory_model_multi.py),
    nunca sobre los frames de contexto interpolados.

    EL VALOR DE contexto_val_s (16.0s) SE CALCULO explicitamente, no es
    arbitrario: con freq_psd_hz=75, nperseg=75 (fact_hz=1.0), noverlap=37
    (window_overlap=0.5), el avance real entre frames de PSD consecutivos
    es step=38 muestras. Para generar max_len=20 frames de PSD se
    requieren nperseg + (max_len-1)*step = 75+19*38 = 797 muestras =
    10.63s de señal cruda a 75Hz. Se uso 16s (~1.5x de margen) para
    asegurar que, incluso con algo de perdida de muestras en la
    alineacion entre pies (merge_asof), siempre haya al menos max_len
    frames de contexto disponibles.

    :param df: DataFrame con columnas 'tiene_gps', 'X_m_gt', 'Y_m_gt', 'X_m', 'Y_m'.
    :param margen_val_s: Segundos a excluir del entrenamiento alrededor de
        cada punto GPS real (evita fuga: el train no ve directamente el
        entorno inmediato de un punto que se usara para medir el error).
    :param contexto_val_s: Segundos de contexto (antes del punto real) a
        incluir en df_val. Ver calculo explicito arriba.
    :return: Tupla (df_train, df_val). df_val incluye tanto los frames de
        contexto como el frame exacto con GPS real (marcado en 'tiene_gps').
    """
    tiempos_reales = df.index[df["tiene_gps"]]

    mask_val = pd.Series(False, index=df.index)
    mask_excluir_train = pd.Series(False, index=df.index)

    for t_real in tiempos_reales:
        mask_val |= (df.index >= t_real - pd.Timedelta(seconds=contexto_val_s)) & (df.index <= t_real)
        mask_excluir_train |= (df.index >= t_real - pd.Timedelta(seconds=margen_val_s)) & \
                               (df.index <= t_real + pd.Timedelta(seconds=margen_val_s))

    df_val = df[mask_val].copy()
    df_train = df[~mask_excluir_train].copy()

    print(f"Division train/val: {len(df_train)} filas de entrenamiento, "
          f"{len(df_val)} filas de val (incluye contexto de {contexto_val_s}s "
          f"antes de cada uno de los {len(tiempos_reales)} puntos GPS reales), "
          f"{mask_excluir_train.sum()} filas excluidas del train por margen de +/-{margen_val_s}s")

    return df_train, df_val

    return df_train, df_val


def preparar_dataset_trayectoria(
    config_path: str, paciente: str, inicio: datetime, fin: datetime, es_utc: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo: extrae datos crudos de ambos pies + GPS, proyecta el
    GPS a metros locales, genera el ground truth denso via PCHIP, y separa
    train/validacion.

    :param config_path: Ruta al config.yaml de InfluxDB.
    :param paciente: CodeID del paciente.
    :param inicio: Fecha de inicio. Hora local salvo que es_utc=True.
    :param fin: Fecha de fin. Misma regla que inicio.
    :param es_utc: Si True, inicio/fin ya estan en UTC.
    :return: Tupla (df_train, df_val).
    """
    df = extraer_datos_crudos(config_path, paciente, inicio, fin, es_utc=es_utc)
    df = proyectar_gps_a_metros(df)
    df = interpolar_ground_truth_pchip(df)
    df_train, df_val = dividir_train_val(df)
    return df_train, df_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset de trayectoria (presion+IMU -> GPS proyectado a metros).")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fin", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument(
        "--config-yaml", type=str,
        default=str(PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml")
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Si se indica, guarda train.csv y val.csv en esta carpeta."
    )
    args = parser.parse_args()

    inicio_dt = datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S")
    fin_dt = datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S")

    df_train, df_val = preparar_dataset_trayectoria(args.config_yaml, args.paciente, inicio_dt, fin_dt)

    print(f"\nTrain: {df_train.shape[0]} filas, {df_train.shape[1]} columnas")
    print(f"Val:   {df_val.shape[0]} filas, {df_val.shape[1]} columnas")
    print("\nGround truth (train, interpolado):")
    print(df_train[["X_m_gt", "Y_m_gt"]].describe())
    print("\nGPS real (val, sin interpolar):")
    print(df_val[["X_m", "Y_m"]].describe())

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(out_dir / f"{args.paciente}_train.csv")
        df_val.to_csv(out_dir / f"{args.paciente}_val.csv")
        print(f"\nGuardado en: {out_dir}")
