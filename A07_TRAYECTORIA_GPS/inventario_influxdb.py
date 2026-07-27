# -*- coding: utf-8 -*-
"""
Inventario completo de campos (_field) y tags disponibles en el bucket
InfluxDB, para verificar si existe algun sistema de posicionamiento mas
denso que GPS (ej. UWB) que no se haya explorado hasta ahora.
"""

import argparse
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import yaml
from influxdb_client import InfluxDBClient


def inventario_completo(config_path: str, paciente: str, inicio: datetime, fin: datetime) -> None:
    """Lista TODOS los _field y tags disponibles para un paciente/rango, sin filtrar por nombre conocido."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["influxdb"]

    tzval = cfg.get("tzval", "Europe/Madrid")
    inicio_str = inicio.replace(tzinfo=ZoneInfo(tzval)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fin_str = fin.replace(tzinfo=ZoneInfo(tzval)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    client = InfluxDBClient(url=cfg["url"], token=cfg["token"], org=cfg["org"], verify_ssl=False, timeout=30_000)

    try:
        # 1. TODOS LOS _field DISPONIBLES (sin filtrar por nombre conocido)
        query_fields = f'''
        from(bucket: "{cfg['bucket']}")
        |> range(start: {inicio_str}, stop: {fin_str})
        |> filter(fn: (r) => r["CodeID"] == "{paciente}")
        |> keep(columns: ["_field"])
        |> distinct(column: "_field")
        '''
        tables = client.query_api().query(query_fields, org=cfg["org"])
        fields = sorted({rec.get_value() for t in tables for rec in t.records})
        print("=== TODOS LOS _field DISPONIBLES ===")
        for f in fields:
            print(f"  - {f}")

        # 2. TODOS LOS TAGS DISPONIBLES (nombres de columnas de tags, via schema)
        query_tagkeys = f'''
        import "influxdata/influxdb/schema"
        schema.tagKeys(bucket: "{cfg['bucket']}", predicate: (r) => r["CodeID"] == "{paciente}", start: {inicio_str}, stop: {fin_str})
        '''
        tables_tags = client.query_api().query(query_tagkeys, org=cfg["org"])
        tagkeys = sorted({rec.get_value() for t in tables_tags for rec in t.records})
        print("\n=== TODOS LOS TAG KEYS DISPONIBLES ===")
        for tk in tagkeys:
            print(f"  - {tk}")

        # 3. BUSCAR PALABRAS CLAVE RELACIONADAS CON POSICIONAMIENTO
        candidatos = [f for f in fields + tagkeys if any(
            kw in f.lower() for kw in ["uwb", "pos", "dist", "anchor", "beacon", "rtk", "loc", "trilat"]
        )]
        print("\n=== CANDIDATOS RELACIONADOS CON POSICIONAMIENTO (fields + tags) ===")
        if candidatos:
            for c in candidatos:
                print(f"  - {c}")
        else:
            print("  Ninguno encontrado ademas de lat/lng.")

    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventario completo de fields/tags en InfluxDB.")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fin", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument(
        "--config-yaml", type=str,
        default=r"C:\Users\jairi\OneDrive\Escritorio\TFMCLONFINAL\A01_EXTRACCION_DATOS\config.yaml"
    )
    args = parser.parse_args()

    inicio_dt = datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S")
    fin_dt = datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S")

    inventario_completo(args.config_yaml, args.paciente, inicio_dt, fin_dt)
