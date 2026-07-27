# -*- coding: utf-8 -*-
"""
Verifica si las coordenadas GPS (lat/lng) son independientes entre pie
izquierdo y derecho (dos sensores GPS distintos) o si es un unico GPS
compartido replicado en ambos registros de InfluxDB. Esto es necesario
para decidir si el modelo de trayectoria GPS+presion debe predecir
posiciones independientes por pie o una posicion compartida del cuerpo.
"""

import argparse
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import yaml
import pandas as pd
from influxdb_client import InfluxDBClient


def verificar_gps_por_pie(
    config_path: str, paciente: str, inicio: datetime, fin: datetime, es_utc: bool = False
) -> None:
    """Compara lat/lng de pie Left vs Right para el mismo paciente/rango.

    :param config_path: Ruta al config.yaml de InfluxDB.
    :param paciente: CodeID del paciente.
    :param inicio: Fecha de inicio. Se interpreta como hora local
        (segun tzval del config) salvo que es_utc=True.
    :param fin: Fecha de fin. Misma regla que inicio.
    :param es_utc: Si True, 'inicio'/'fin' ya estan en UTC y no se
        aplica conversion de zona horaria (util para timestamps con
        sufijo 'Z' como los que usa Agnostic_evaluator.py, evitando el
        desfase que ocurre si se interpretan por error como hora local).
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

    try:
        resultados = {}
        for pie in ["Left", "Right"]:
            query = f'''
            from(bucket: "{cfg['bucket']}")
            |> range(start: {inicio_str}, stop: {fin_str})
            |> filter(fn: (r) => r["CodeID"] == "{paciente}" and r["Foot"] == "{pie}")
            |> filter(fn: (r) => r._field == "Ax")
            |> keep(columns: ["_time", "lat", "lng"])
            |> limit(n: 20)
            '''
            tables = client.query_api().query(query, org=cfg["org"])
            filas = []
            for t in tables:
                for rec in t.records:
                    filas.append({
                        "time": rec.get_time(),
                        "lat": rec.values.get("lat"),
                        "lng": rec.values.get("lng"),
                    })
            resultados[pie] = pd.DataFrame(filas)
            print(f"\n--- Primeras lecturas GPS, PIE {pie} ---")
            print(resultados[pie].to_string(index=False))

        # COMPARAR SI LAT/LNG COINCIDEN ENTRE PIES EN TIMESTAMPS CERCANOS
        if not resultados["Left"].empty and not resultados["Right"].empty:
            lats_left = set(resultados["Left"]["lat"].dropna())
            lats_right = set(resultados["Right"]["lat"].dropna())
            interseccion = lats_left & lats_right
            print(f"\n--- COMPARACION ---")
            print(f"Valores de 'lat' unicos en Left:  {len(lats_left)}")
            print(f"Valores de 'lat' unicos en Right: {len(lats_right)}")
            print(f"Valores de 'lat' EN COMUN entre ambos pies: {len(interseccion)}")
            if interseccion:
                print("Los pies COMPARTEN valores identicos de lat -> probablemente 1 solo GPS compartido.")
            else:
                print("Los pies NO comparten valores de lat -> probablemente 2 sensores GPS independientes.")

    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica si el GPS es independiente por pie o compartido.")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fin", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument(
        "--config-yaml", type=str,
        default=r"C:\Users\jairi\OneDrive\Escritorio\TFM_CLONADO_FINALFINAL\A01_EXTRACCION_DATOS\config.yaml"
    )
    args = parser.parse_args()

    inicio_dt = datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S")
    fin_dt = datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S")

    verificar_gps_por_pie(args.config_yaml, args.paciente, inicio_dt, fin_dt)
