# -*- coding: utf-8 -*-
"""
Verifica si el acelerometro presenta saturacion (clipping) real en el
limite de su rango dinamico, y compara la distribucion de valores
entre pie derecho e izquierdo para detectar asimetrias de hardware
o de escala entre sensores.
"""

import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from influxdb_client import InfluxDBClient


def verificar_saturacion(
    config_path: str,
    paciente: str,
    pie: str,
    inicio: datetime,
    fin: datetime
) -> None:
    """
    Descarga Ax, Ay, Az y cuenta cuantas muestras caen exactamente en
    el valor maximo/minimo observado (indicio de clipping de hardware),
    ademas de mostrar percentiles de la distribucion completa.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["influxdb"]

    tzval = cfg.get("tzval", "Europe/Madrid")
    inicio_str = inicio.replace(tzinfo=ZoneInfo(tzval)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fin_str = fin.replace(tzinfo=ZoneInfo(tzval)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    client = InfluxDBClient(url=cfg["url"], token=cfg["token"], org=cfg["org"], verify_ssl=False)

    metrics = ["Ax", "Ay", "Az"]
    metrics_str = " or ".join([f'r._field == "{m}"' for m in metrics])
    columns_str = ", ".join([f'"{m}"' for m in metrics])

    query = f'''
    from(bucket: "{cfg['bucket']}")
    |> range(start: {inicio_str}, stop: {fin_str})
    |> filter(fn: (r) => r._measurement == "Gait")
    |> filter(fn: (r) => {metrics_str})
    |> filter(fn: (r) => r["CodeID"] == "{paciente}" and r["type"] == "SCKS" and r["Foot"] == "{pie}")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", {columns_str}])
    '''

    try:
        tables = client.query_api().query(query, org=cfg["org"])
        data = [rec.values for t in tables for rec in t.records]
        df = pd.DataFrame(data)

        if df.empty:
            print(f"\nSIN DATOS PARA {paciente}/{pie}.")
            return

        print(f"\n{'='*55}")
        print(f"PIE: {pie}  (n = {len(df)} muestras)")
        print(f"{'='*55}")

        for eje in metrics:
            valores = df[eje].astype(float).values
            v_min, v_max = valores.min(), valores.max()

            n_en_max = np.sum(np.isclose(valores, v_max, atol=1e-6))
            n_en_min = np.sum(np.isclose(valores, v_min, atol=1e-6))
            pct_saturado = 100.0 * (n_en_max + n_en_min) / len(valores)

            percentiles = np.percentile(valores, [1, 5, 25, 50, 75, 95, 99])

            print(f"\n  Eje {eje}:")
            print(f"    Rango: [{v_min:.3f}, {v_max:.3f}]")
            print(f"    Muestras en el limite max ({v_max:.3f}): {n_en_max} ({100*n_en_max/len(valores):.2f}%)")
            print(f"    Muestras en el limite min ({v_min:.3f}): {n_en_min} ({100*n_en_min/len(valores):.2f}%)")
            print(f"    Total saturado: {pct_saturado:.2f}%")
            print(f"    Percentiles [1,5,25,50,75,95,99]: {np.round(percentiles, 3)}")

    finally:
        client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verifica saturacion del acelerometro para un paciente.")
    parser.add_argument("--paciente", type=str, required=True, help="CodeID del paciente")
    parser.add_argument("--inicio", type=str, required=True, help="Fecha inicio 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fin", type=str, required=True, help="Fecha fin 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument(
        "--config-yaml", type=str,
        default=r"C:\Users\jairi\OneDrive\Escritorio\TFM_CLONADO_FINALFINAL\A01_EXTRACCION_DATOS\config.yaml",
        help="Ruta al config.yaml de InfluxDB"
    )
    args = parser.parse_args()

    inicio_dt = datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S")
    fin_dt = datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S")

    verificar_saturacion(args.config_yaml, args.paciente, "Right", inicio_dt, fin_dt)
    verificar_saturacion(args.config_yaml, args.paciente, "Left", inicio_dt, fin_dt)