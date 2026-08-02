# -*- coding: utf-8 -*-
"""
A07 REDISEÑADO: extraccion de datos para clasificacion marcha/reposo
enriquecida con GPS como rama discreta de entrada (NO como ground truth
de trayectoria interpolada).

Cambio de enfoque respecto a la version anterior de A07, siguiendo la
recomendacion del director del TFM: el GPS es una fuente de observacion
discreta, irregular y de baja frecuencia -- no debe tratarse como una
señal densa comparable a IMU/presion, ni usarse simultaneamente como
entrada Y como target del mismo problema (fuga de informacion). Aqui el
GPS entra UNICAMENTE como rama de entrada auxiliar para clasificacion
(objetivo = marcha/reposo, tomado de la columna mov_type del Excel de
segmentos), nunca como variable a predecir.

Para cada segmento, se generan dos representaciones paralelas:
  - Rama IMU/presion: tensor PSD combinado (mismo esquema que A04, 348
    dim), reutilizando GaitFeatureExtractor sin cambios.
  - Rama GPS: secuencia IRREGULAR de observaciones (delta_t desde el
    inicio del segmento, x, y proyectados a metros UTM, mascara de
    observacion 1/0). No se interpola nada entre lecturas reales.
"""

from __future__ import annotations
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
from sklearn.linear_model import LinearRegression

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
        return 32600 + zona
    return 32700 + zona


def cargar_segmentos_desde_excel(ruta_excel: Path) -> pd.DataFrame:
    """
    Lee la tabla de segmentos (Reference, datefrom, dateuntil, mov_type,
    es_utc) desde un Excel, en vez de tenerla hardcodeada en Python. Esto
    permite anadir/quitar segmentos editando el Excel directamente, sin
    tocar codigo.

    :param ruta_excel: Ruta al archivo .xlsx con las columnas esperadas.
    :return: DataFrame con los segmentos, tipos ya convertidos
        (datefrom/dateuntil como datetime, mov_type como int, es_utc como bool).
    """
    df = pd.read_excel(ruta_excel, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]

    columnas_esperadas = {"Reference", "datefrom", "dateuntil", "mov_type"}
    faltantes = columnas_esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el Excel: {faltantes}")

    df["datefrom"] = pd.to_datetime(df["datefrom"], dayfirst=True, format="mixed")
    df["dateuntil"] = pd.to_datetime(df["dateuntil"], dayfirst=True, format="mixed")
    df["mov_type"] = df["mov_type"].astype(int)

    if "es_utc" not in df.columns:
        df["es_utc"] = False
    df["es_utc"] = df["es_utc"].fillna(False).astype(bool)

    return df


def extraer_datos_crudos(
    config_path: str, paciente: str, inicio: datetime, fin: datetime, es_utc: bool = False
) -> pd.DataFrame:
    """
    Consulta InfluxDB por presion plantar, IMU (Acc/Gyro/Mag) y GPS
    (lat/lng) de ambos pies, en el rango de fechas dado. Identico en
    logica a la version usada en el A07 original (reutilizable sin
    cambios), ya que la extraccion cruda en si era correcta -- lo que
    cambia es como se USA el GPS despues de extraerlo.

    :param config_path: Ruta al config.yaml de InfluxDB.
    :param paciente: CodeID del paciente.
    :param inicio: Fecha de inicio. Hora local salvo que es_utc=True.
    :param fin: Fecha de fin. Misma regla que inicio.
    :param es_utc: Si True, inicio/fin ya estan en UTC.
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

    client = InfluxDBClient(url=cfg["url"], token=cfg["token"], org=cfg["org"], verify_ssl=False, timeout=30_000)

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

            renombre = {c: f"{c}_{pie}" for c in metrics}
            df_pie = df_pie.rename(columns=renombre)

            dfs_por_pie[pie] = df_pie
    finally:
        client.close()

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


def construir_rama_gps_discreta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la representacion de la rama GPS como secuencia IRREGULAR
    de observaciones discretas, sin interpolar nada entre lecturas
    reales. Para cada fila del DataFrame (frecuencia de IMU), se agregan
    4 columnas nuevas:

      - gps_delta_t: segundos transcurridos desde el INICIO DEL SEGMENTO
        hasta esta fila (tiempo continuo, siempre disponible).
      - gps_x_m, gps_y_m: posicion en metros locales (proyeccion UTM),
        SOLO rellenada en los frames donde hay una lectura GPS real
        (forward-fill del ultimo valor conocido en el resto -- esto es
        aceptable aqui porque la mascara de observacion, no el valor en
        si, es lo que el modelo debe aprender a usar; el valor "stale"
        entre lecturas NO se presenta como ground truth continuo, solo
        como contexto de la ultima ancla conocida).
      - gps_mascara: 1.0 si esta fila coincide con una lectura GPS REAL
        (no forward-filled), 0.0 en caso contrario. Esta es la señal
        clave que distingue "observacion real" de "relleno", y es la
        que un encoder de secuencia irregular (o cross-attention) deberia
        usar para ponderar cuanto confiar en gps_x_m/gps_y_m en cada paso.

    A diferencia de la version anterior de A07 (interpolar_ground_truth_pchip),
    NO se genera ninguna curva suave entre puntos reales: el "hueco" entre
    lecturas GPS quedaria representado por gps_mascara=0, dejando que el
    modelo (no una suposicion matematica externa) decida que hacer con esa
    incertidumbre.

    :param df: DataFrame con columnas 'lat', 'lng' (crudas, con
        forward-fill de origen, tal como llegan de InfluxDB).
    :return: Mismo DataFrame con las 4 columnas gps_* añadidas.
    """
    df = df.copy()

    t_absolutos = (df.index - df.index[0]).total_seconds().values
    df["gps_delta_t"] = t_absolutos

    lat_num = pd.to_numeric(df["lat"], errors="coerce")
    lng_num = pd.to_numeric(df["lng"], errors="coerce")

    lat_cambio = lat_num.ne(lat_num.shift(1))
    lng_cambio = lng_num.ne(lng_num.shift(1))

    # FILTRAR (0, 0) COMO LECTURA INVALIDA: muchos receptores GPS/GNSS
    # reportan lat=0, lng=0 antes de obtener su primer "fix" satelital
    # real (o durante una perdida temporal de senal). (0,0) cae en el
    # Golfo de Guinea, a miles de km de cualquier sesion real de este
    # proyecto -- tratarlo como lectura real introduce un salto de
    # posicion absurdo (varios cientos de miles de km) que arruina
    # cualquier calculo de distancia/velocidad. Se excluye explicitamente
    # de mask_no_nulos, ademas del filtro de NaN ya existente.
    #
    # IMPORTANTE: lat/lng pueden llegar como texto (str) desde InfluxDB
    # (confirmado empiricamente: '0' en vez de 0.0), en cuyo caso una
    # comparacion directa df["lat"] == 0 NUNCA es True (compara str
    # contra int) y el filtro queda inerte en silencio. Se convierte
    # explicitamente a numerico con pd.to_numeric ANTES de comparar.
    mask_no_nulos = (
        lat_num.notna() & lng_num.notna()
        & ~((lat_num == 0) & (lng_num == 0))
    )
    mascara_real = (mask_no_nulos & (lat_cambio | lng_cambio)).values

    df["gps_mascara"] = mascara_real.astype(np.float32)

    x_m = np.full(len(df), np.nan)
    y_m = np.full(len(df), np.nan)

    puntos_reales = df.loc[mascara_real]
    if len(puntos_reales) > 0:
        lat0 = float(puntos_reales["lat"].iloc[0])
        lng0 = float(puntos_reales["lng"].iloc[0])
        epsg_utm = _utm_epsg_para(lat0, lng0)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
        x0, y0 = transformer.transform(lng0, lat0)

        lats = df.loc[mask_no_nulos, "lat"].astype(float).values
        lngs = df.loc[mask_no_nulos, "lng"].astype(float).values
        xs, ys = transformer.transform(lngs, lats)
        x_m[mask_no_nulos.values] = np.array(xs) - x0
        y_m[mask_no_nulos.values] = np.array(ys) - y0

    df["gps_x_m"] = x_m
    df["gps_y_m"] = y_m

    df["gps_x_m"] = df["gps_x_m"].ffill().bfill().fillna(0.0)
    df["gps_y_m"] = df["gps_y_m"].ffill().bfill().fillna(0.0)

    n_reales = int(mascara_real.sum())
    print(f"Rama GPS: {n_reales} lecturas reales de {len(df)} frames totales "
          f"({100 * n_reales / len(df):.3f}%)")

    return df


def preparar_segmento_clasificacion(
    config_path: str, paciente: str, inicio: datetime, fin: datetime,
    mov_type: int, es_utc: bool = False
) -> Tuple[pd.DataFrame, int]:
    """
    Pipeline completo para UN segmento: extrae datos crudos + construye
    la rama GPS discreta. El label de clasificacion (marcha/reposo) viene
    dado externamente (columna mov_type del Excel), no se infiere de los
    datos.

    :param config_path: Ruta al config.yaml de InfluxDB.
    :param paciente: CodeID del paciente.
    :param inicio: Fecha de inicio del segmento.
    :param fin: Fecha de fin del segmento.
    :param mov_type: Label de clasificacion (1=marcha, 0=reposo), tomado
        del Excel de segmentos.
    :param es_utc: Si True, inicio/fin ya estan en UTC.
    :return: Tupla (df_segmento_con_gps, mov_type).
    """
    df = extraer_datos_crudos(config_path, paciente, inicio, fin, es_utc=es_utc)
    df = construir_rama_gps_discreta(df)
    return df, mov_type


# =============================================================================
# VELOCIDAD Y FATIGA CON GPS (agregado, no frame-a-frame)
#
# ADVERTENCIA METODOLOGICA IMPORTANTE: estas funciones NO intentan
# reconstruir velocidad instantanea de marcha ni longitud de zancada
# (esas metricas requieren resolucion de centimetros, incompatible con
# GPS civil de baja frecuencia). Calculan unicamente una VELOCIDAD
# PROMEDIO DE SESION (distancia recorrida real / tiempo total), util
# como indicador grueso de desplazamiento, no como sustituto de las
# metricas biomecanicas finas que se calculaban en A06.
# =============================================================================

def calcular_distancia_recorrida_real(df: pd.DataFrame) -> float:
    """
    Calcula la distancia TOTAL recorrida sumando el desplazamiento entre
    cada par de lecturas GPS REALES consecutivas (gps_mascara == 1), NO
    la distancia neta entre el primer y el ultimo punto.

    Esto corrige explicitamente la advertencia del director del TFM:
    si el paciente recorre un pasillo de ida y vuelta, la posicion neta
    final puede ser ~0 (mismo punto de inicio y fin) aunque haya
    recorrido el doble de la longitud del pasillo. Sumar los tramos
    reales entre lecturas consecutivas SI captura ese recorrido, aunque
    con baja frecuencia de GPS puede subestimar movimientos no lineales
    ocurridos ENTRE dos lecturas reales (ej. un giro completo que el GPS
    no alcanzo a capturar).

    :param df: DataFrame de un segmento, con columnas gps_x_m, gps_y_m,
        gps_mascara ya calculadas por construir_rama_gps_discreta.
    :return: Distancia total recorrida, en metros. 0.0 si hay menos de
        2 lecturas GPS reales.
    """
    puntos_reales = df.loc[df["gps_mascara"] == 1, ["gps_x_m", "gps_y_m"]].values

    if len(puntos_reales) < 2:
        return 0.0

    deltas = np.diff(puntos_reales, axis=0)
    distancias_tramo = np.linalg.norm(deltas, axis=1)
    return float(distancias_tramo.sum())


def calcular_velocidad_promedio_sesion(df: pd.DataFrame) -> dict:
    """
    Calcula la velocidad promedio de UNA sesion/segmento, como distancia
    recorrida real (ver calcular_distancia_recorrida_real) dividida entre
    la duracion total del segmento -- NO velocidad instantanea de marcha.

    :param df: DataFrame de un segmento, con columnas gps_delta_t,
        gps_x_m, gps_y_m, gps_mascara ya calculadas.
    :return: Diccionario con distancia_m, duracion_s, velocidad_ms,
        n_lecturas_gps_reales. velocidad_ms es NaN si no hay suficientes
        lecturas GPS reales (< 2) para calcular un desplazamiento.
    """
    distancia_m = calcular_distancia_recorrida_real(df)
    duracion_s = float(df["gps_delta_t"].iloc[-1] - df["gps_delta_t"].iloc[0])
    n_lecturas = int((df["gps_mascara"] == 1).sum())

    velocidad_ms = distancia_m / duracion_s if (duracion_s > 0 and n_lecturas >= 2) else float("nan")

    return {
        "distancia_m": distancia_m,
        "duracion_s": duracion_s,
        "velocidad_ms": velocidad_ms,
        "n_lecturas_gps_reales": n_lecturas,
    }


def calcular_fatiga_por_tramos(
    lista_velocidades: list, lista_tiempos_inicio: list
) -> dict:
    """
    Ajusta una regresion lineal de velocidad promedio (una por
    segmento/sesion) contra el tiempo, como biomarcador exploratorio de
    fatiga -- mismo criterio que fatigue_analysis.py (A06), pero aqui la
    entrada es velocidad promedio de sesion via GPS, no velocidad de
    marcha instantanea via IMU integrada.

    ADVERTENCIA: la fiabilidad de esta fatiga hereda TODAS las
    limitaciones de calcular_velocidad_promedio_sesion (velocidad
    promedio gruesa, sensible a la escasez de lecturas GPS por sesion).
    Con pocos segmentos por paciente, la pendiente resultante tiene alta
    incertidumbre y debe tratarse como exploratoria, no como conclusion
    clinica fuerte -- exactamente la misma salvedad que ya aplicaba
    fatigue_analysis.py en A06.

    :param lista_velocidades: Velocidad promedio (m/s) de cada segmento,
        en el mismo orden que lista_tiempos_inicio. Los NaN (segmentos
        sin suficientes lecturas GPS) se excluyen automaticamente.
    :param lista_tiempos_inicio: Marca de tiempo de inicio de cada
        segmento (ej. timestamp unix, o indice de sesion), mismo orden
        y longitud que lista_velocidades.
    :return: Diccionario con slope, intercept, y n_puntos_validos. Si
        hay menos de 2 puntos validos, slope=intercept=0.0 (mismo
        "escudo anti-crash" que ya usaba FatigueAnalyzer en A06).
    """
    velocidades = np.array(lista_velocidades, dtype=float)
    tiempos = np.array(lista_tiempos_inicio, dtype=float)

    mask_validos = ~np.isnan(velocidades)
    velocidades_validas = velocidades[mask_validos]
    tiempos_validos = tiempos[mask_validos]

    if len(velocidades_validas) < 2:
        print("ADVERTENCIA FATIGA: datos insuficientes (< 2 segmentos con "
              "velocidad GPS valida) para calcular degradacion.")
        return {"slope": 0.0, "intercept": 0.0, "n_puntos_validos": len(velocidades_validas)}

    x = tiempos_validos.reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(x, velocidades_validas)

    return {
        "slope": float(modelo.coef_[0]),
        "intercept": float(modelo.intercept_),
        "n_puntos_validos": len(velocidades_validas),
    }



    import argparse
    parser = argparse.ArgumentParser(description="Prepara UN segmento (presion+IMU+GPS discreto) para clasificacion.")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fin", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--mov-type", type=int, required=True, choices=[0, 1])
    parser.add_argument("--es-utc", action="store_true")
    parser.add_argument("--config-yaml", type=str, default=str(PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml"))
    args = parser.parse_args()

    inicio_dt = datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S")
    fin_dt = datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S")

    df_resultado, label = preparar_segmento_clasificacion(
        args.config_yaml, args.paciente, inicio_dt, fin_dt, args.mov_type, args.es_utc
    )

    print(f"\nSegmento: {args.paciente} | label={label}")
    print(f"Filas: {len(df_resultado)}")
    print(df_resultado[["gps_delta_t", "gps_x_m", "gps_y_m", "gps_mascara"]].describe())
