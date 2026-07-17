# -*- coding: utf-8 -*-
"""
Auditor de campos InfluxDB (API Flux, InfluxDB v2).
Lista measurements y field keys disponibles en el bucket configurado.
"""

import argparse
from pathlib import Path

import yaml
import pandas as pd
from influxdb_client import InfluxDBClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class InfluxAuditor:
    """Auditor de campos InfluxDB usando consultas Flux."""

    def __init__(self, config_path: str) -> None:
        """Inicializa conexion con parametros del YAML."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # ACCESO AL BLOQUE ANIDADO
        self.db_cfg = self.config["influxdb"]

        # CONEXION IGNORANDO SSL
        self.client = InfluxDBClient(
            url=self.db_cfg["url"],
            token=self.db_cfg["token"],
            org=self.db_cfg["org"],
            verify_ssl=False
        )
        self.query_api = self.client.query_api()

    def listar_measurements(self) -> None:
        """Muestra todas las mediciones disponibles en el bucket, via Flux schema.measurements."""
        bucket = self.db_cfg["bucket"]
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurements(bucket: "{bucket}")
        '''
        tables = self.query_api.query(query, org=self.db_cfg["org"])

        print("\n--- MEDICIONES DISPONIBLES ---")
        for table in tables:
            for record in table.records:
                print(f"Medicion encontrada: {record.get_value()}")

    def listar_field_keys(self, measurement: str) -> None:
        """Muestra los field keys de un measurement, via Flux schema.measurementFieldKeys."""
        bucket = self.db_cfg["bucket"]
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurementFieldKeys(bucket: "{bucket}", measurement: "{measurement}")
        '''
        tables = self.query_api.query(query, org=self.db_cfg["org"])

        print(f"\n--- FIELD KEYS DE '{measurement}' ---")
        for table in tables:
            for record in table.records:
                print(f"Field: {record.get_value()}")

    def listar_tag_keys(self, measurement: str) -> None:
        """Muestra los tag keys de un measurement (utiles para saber que identifica 'pie', 'paciente', etc.)."""
        bucket = self.db_cfg["bucket"]
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurementTagKeys(bucket: "{bucket}", measurement: "{measurement}")
        '''
        tables = self.query_api.query(query, org=self.db_cfg["org"])

        print(f"\n--- TAG KEYS DE '{measurement}' ---")
        for table in tables:
            for record in table.records:
                print(f"Tag: {record.get_value()}")

    def audit_and_save(self, measurement: str, output_file: str = "auditoria_campos_influx.csv") -> None:
        """Consulta field keys de un measurement y guarda CSV."""
        print(f"\nAuditando: {measurement}...")
        bucket = self.db_cfg["bucket"]
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurementFieldKeys(bucket: "{bucket}", measurement: "{measurement}")
        '''
        try:
            tables = self.query_api.query(query, org=self.db_cfg["org"])
            fields = [{"field": record.get_value()} for table in tables for record in table.records]

            df = pd.DataFrame(fields)
            df.to_csv(output_file, index=False)
            print(f"EXITO: CSV guardado en {output_file}")
            print(f"Total campos: {len(df)}")

        except Exception as e:
            print(f"ERROR en consulta: {e}")

    def close(self) -> None:
        """Cierra la conexion con InfluxDB."""
        self.client.close()


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Auditor de schema InfluxDB (Flux).")
    parser.add_argument(
        "--config-yaml", type=str,
        default=str(PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml"),
        help="Ruta al config.yaml de InfluxDB"
    )
    parser.add_argument(
        "--measurement", type=str, default=None,
        help="Nombre del measurement a auditar (si se omite, solo lista measurements disponibles)"
    )
    args = parser.parse_args()

    auditor = InfluxAuditor(args.config_yaml)

    try:
        # 1. LISTAR MEASUREMENTS PARA IDENTIFICAR TABLA
        auditor.listar_measurements()

        if args.measurement:
            # 2. LISTAR TODOS LOS FIELDS (columnas de valores: Acc_X, Gyro_Y, Mag_Z, S0, etc.)
            auditor.listar_field_keys(args.measurement)

            # 3. LISTAR TAGS (identificadores: paciente, pie, etc.)
            auditor.listar_tag_keys(args.measurement)

            # 4. GUARDAR AUDITORIA EN CSV
            auditor.audit_and_save(args.measurement)
        else:
            print(
                "\nUsa --measurement <nombre> (ej. 'Gait') para auditar field keys, "
                "tag keys y guardar CSV de ese measurement."
            )
    finally:
        auditor.close()


if __name__ == "__main__":
    main()
