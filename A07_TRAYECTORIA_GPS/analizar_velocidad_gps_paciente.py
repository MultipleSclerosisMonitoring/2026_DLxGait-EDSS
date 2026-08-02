# -*- coding: utf-8 -*-
"""
Analiza velocidad y fatiga GPS de UN paciente/segmento especifico
(complementa el LOPO agregado de A07, que no reporta resultados por
paciente individual). Guarda el resultado en un CSV con nombre por
paciente, mismo patron que Agnostic_evaluator.py y
Orquestador_temporal.py (A06), para poder despues cruzar los 3
pipelines en un informe conjunto.

Reutiliza preparar_segmento_clasificacion (extraccion + rama GPS) y las
funciones de velocidad/fatiga ya construidas en
preparar_dataset_clasificacion_gps.py.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preparar_dataset_clasificacion_gps as pdc


def analizar_segmento_velocidad_fatiga(
    config_path: str, paciente: str, inicio: datetime, fin: datetime,
    es_utc: bool, output_dir: Path
) -> dict:
    """
    Extrae un segmento, calcula su rama GPS, y de ahi la velocidad
    promedio de esa sesion. Como un solo segmento no permite calcular
    fatiga (requiere >= 2 puntos en el tiempo), la fatiga se deja como
    NaN/0.0 aqui -- para fatiga real hace falta correr esta funcion
    sobre varios segmentos del mismo paciente y combinar los resultados
    con calcular_fatiga_por_tramos (ver bloque main).

    :param config_path: Ruta al config.yaml de InfluxDB.
    :param paciente: CodeID del paciente.
    :param inicio: Inicio del segmento.
    :param fin: Fin del segmento.
    :param es_utc: Si True, inicio/fin ya estan en UTC.
    :param output_dir: Carpeta donde guardar el CSV de resultado.
    :return: Diccionario con el resultado (mismo formato que
        calcular_velocidad_promedio_sesion, mas metadata de paciente/fecha).
    """
    df_seg, _label = pdc.preparar_segmento_clasificacion(
        config_path, paciente, inicio, fin, mov_type=1, es_utc=es_utc
    )

    resultado_velocidad = pdc.calcular_velocidad_promedio_sesion(df_seg)

    resultado = {
        "paciente": paciente,
        "inicio": inicio.isoformat(),
        "fin": fin.isoformat(),
        **resultado_velocidad,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ruta_csv = output_dir / f"{paciente}_velocidad_gps.csv"
    pd.DataFrame([resultado]).to_csv(ruta_csv, index=False)
    print(f"Resultado guardado en: {ruta_csv}")

    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza velocidad GPS de un segmento especifico (A07 por paciente).")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fin", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--es-utc", action="store_true")
    parser.add_argument("--config-yaml", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).resolve().parent / "RESULTADOS_VELOCIDAD_GPS")
    )
    args = parser.parse_args()

    inicio_dt = datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S")
    fin_dt = datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S")

    resultado = analizar_segmento_velocidad_fatiga(
        args.config_yaml, args.paciente, inicio_dt, fin_dt, args.es_utc, Path(args.output_dir)
    )

    print(f"\nVelocidad promedio: {resultado['velocidad_ms']}")
    print(f"Distancia recorrida: {resultado['distancia_m']:.2f} m")
    print(f"Lecturas GPS reales: {resultado['n_lecturas_gps_reales']}")
