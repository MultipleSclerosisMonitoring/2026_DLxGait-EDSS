# -*- coding: utf-8 -*-
"""
TESTMAGNETO

Diagnostico de escala del acelerometro.
Verifica si Ax, Ay, Az llegan en g, en m/s^2, o en unidades crudas de ADC,
comparando la norma de la senal durante un tramo de reposo/stance
contra los valores esperados para cada escala posible.
"""

import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from influxdb_client import InfluxDBClient


def diagnosticar_escala_acelerometro(
    config_path: str,
    paciente: str,
    pie: str,
    inicio: datetime,
    fin: datetime,
    n_muestras_reposo: int = 100
) -> None:
    """
    Descarga Ax, Ay, Az crudos y analiza su norma para inferir la escala real.

    :param config_path: Ruta al YAML de configuracion InfluxDB.
    :param paciente: CodeID del paciente.
    :param pie: Foot ("Right" o "Left").
    :param inicio: Fecha de inicio (hora local).
    :param fin: Fecha de fin (hora local).
    :param n_muestras_reposo: Numero de muestras iniciales asumidas en reposo/stance.
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

        acc = df[metrics].astype(float).values
        norma_completa = np.linalg.norm(acc, axis=1)
        norma_reposo = norma_completa[:n_muestras_reposo]

        print(f"\n--- DIAGNOSTICO ESCALA ACELEROMETRO: {paciente}/{pie} ---")
        print(f"Total muestras: {len(acc)}")
        print(f"Norma media (todas las muestras):  {norma_completa.mean():.3f}")
        print(f"Norma media (primeras {n_muestras_reposo}, asumido reposo): {norma_reposo.mean():.3f}")
        print(f"Norma std  (primeras {n_muestras_reposo}):                  {norma_reposo.std():.3f}")
        print(f"Valor min / max Ax: [{acc[:,0].min():.3f}, {acc[:,0].max():.3f}]")
        print(f"Valor min / max Ay: [{acc[:,1].min():.3f}, {acc[:,1].max():.3f}]")
        print(f"Valor min / max Az: [{acc[:,2].min():.3f}, {acc[:,2].max():.3f}]")

        print("\nCOMPARACION CONTRA ESCALAS CONOCIDAS (norma esperada en reposo = 1g):")
        candidatos = {
            "g (norma ~1.0)": 1.0,
            "m/s^2 (norma ~9.81)": 9.81,
            "ADC 16-bit +-2g (norma ~16384)": 16384.0,
            "ADC 16-bit +-4g (norma ~8192)": 8192.0,
            "ADC 16-bit +-8g (norma ~4096)": 4096.0,
            "ADC 16-bit +-16g (norma ~2048)": 2048.0,
        }
        for nombre, esperado in candidatos.items():
            ratio = norma_reposo.mean() / esperado
            print(f"  {nombre:35s} -> ratio observado/esperado = {ratio:.3f}")

        print(
            "\nInterpretacion: el candidato con ratio mas cercano a 1.0 indica la escala real "
            "en la que llega Ax/Ay/Az desde InfluxDB."
        )

    finally:
        client.close()


if __name__ == "__main__":
    CFG_PATH = r"C:\Users\jairi\OneDrive\Escritorio\TFM_CLONADO_FINALFINAL\A01_EXTRACCION_DATOS\config.yaml"

    diagnosticar_escala_acelerometro(
        config_path=CFG_PATH,
        paciente="TABUENCA01-45",
        pie="Right",
        inicio=datetime.strptime("2024-04-25 16:38:33", "%Y-%m-%d %H:%M:%S"),
        fin=datetime.strptime("2024-04-25 16:49:05", "%Y-%m-%d %H:%M:%S")
    )

    diagnosticar_escala_acelerometro(
        config_path=CFG_PATH,
        paciente="TABUENCA01-45",
        pie="Left",
        inicio=datetime.strptime("2024-04-25 16:38:33", "%Y-%m-%d %H:%M:%S"),
        fin=datetime.strptime("2024-04-25 16:49:05", "%Y-%m-%d %H:%M:%S")
    )