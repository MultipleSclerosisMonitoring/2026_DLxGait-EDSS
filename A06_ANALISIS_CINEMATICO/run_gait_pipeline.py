# -*- coding: utf-8 -*-
"""
run_gait_pipeline.py
=============================================================
Orquestador principal del pipeline biomecánico.
Ejecuta la detección de eventos y el motor cinemático.
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt

# IMPORTAR MODULOS LOCALES
from A06_ANALISIS_CINEMATICO.event_detector import EventDetector, EventDetectorConfig
from A06_ANALISIS_CINEMATICO.kinematic_engine import KinematicEngine, KinematicConfig

# IMPORTAR EXTRACCION DATOS
try:
    from A01_EXTRACCION_DATOS.extract_data_plus import cInfluxDB
except ImportError as e:
    print(f"ERROR EXTRACCION: {e}")
    sys.exit(1)

# CONFIGURAR LOGGER PRINCIPAL
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PipelineConfig(BaseModel):
    """
    Configuración validada del orquestador.
    """
    paciente: str
    inicio: str
    fin: str
    fs: int = Field(default=100)
    out: Path = Field(default=Path("resultados.csv"))

class GaitOrchestrator:
    """
    Clase principal orquestación biomecánica.
    """
    def __init__(self, config: PipelineConfig) -> None:
        """
        Inicializa dependencias y motores.

        :param config: Parámetros de ejecución validados.
        :type config: PipelineConfig
        :return: Nada.
        :rtype: None
        """
        self.config = config
        self.detector = EventDetector(EventDetectorConfig(fs=self.config.fs))
        self.kinematic = KinematicEngine(KinematicConfig(fs=self.config.fs))
        
        # DEFINIR RUTA CONFIGURACION
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_yaml = self.project_root / "A01_EXTRACCION_DATOS" / ".config.yaml"

    def _fetch_data(self) -> pd.DataFrame:
        """
        Consulta datos inerciales remotos.

        :return: Matriz de datos extraídos.
        :rtype: pd.DataFrame
        """
        if not self.config_yaml.exists():
            logger.error("YAML NO ENCONTRADO")
            sys.exit(1)

        # EXTRAER DATOS INFLUXDB
        try:
            extractor = cInfluxDB(config_path=str(self.config_yaml))
            logger.info("CONSULTANDO BASE DATOS")
            
            start_dt = datetime.strptime(self.config.inicio, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(self.config.fin, "%Y-%m-%d %H:%M:%S")
            
            df = extractor.query_data(
                from_date=start_dt, 
                to_date=end_dt, 
                qtok=self.config.paciente, 
                pie="Right"
            )
            return df
        except Exception as e:
            logger.error(f"ERROR INFLUXDB: {e}")
            sys.exit(1)
        finally:
            if 'extractor' in locals():
                extractor.close()

    def _process_pressure(self, df: pd.DataFrame) -> np.ndarray:
        """
        Limpia señal presión plantar.

        :param df: Datos brutos extraídos.
        :type df: pd.DataFrame
        :return: Vector de presión filtrada.
        :rtype: np.ndarray
        """
        try:
            # PREPARAR PRESION PLANTAR
            presion = df[["S0", "S1", "S2"]].astype(float).sum(axis=1).values
            presion -= np.percentile(presion, 5)
            presion[presion < 0] = 0
            return presion
        except KeyError:
            logger.error("FALTAN COLUMNAS PRESION")
            sys.exit(1)

    def run(self) -> None:
        """
        Ejecuta flujo trabajo completo.

        :return: Nada.
        :rtype: None
        """
        # OBTENER DATOS CRUDOS
        df = self._fetch_data()

        if df is None or df.empty:
            logger.error("DATOS VACIOS")
            sys.exit(1)

        # PROCESAR SENSORES PRESION
        presion = self._process_pressure(df)

        # DETECTAR EVENTOS TEMPORALES
        hs_idx, to_idx = self.detector.detect_from_pressure(presion)

        if len(hs_idx) < 3:
            logger.error("INSUFICIENTES EVENTOS MARCHA")
            sys.exit(1)

        logger.info(f"EVENTOS DETECTADOS: {len(hs_idx)} HS, {len(to_idx)} TO")


        # EXTRAER DATOS INERCIALES
        try:
            acc = df[["Ax", "Ay", "Az"]].astype(float).values
            gyro = df[["Gx", "Gy", "Gz"]].astype(float).values
        except KeyError as e:
            logger.error(f"FALTAN COLUMNAS INERCIALES: {e}")
            sys.exit(1)

        # EJECUTAR MOTOR CINEMATICO
        logger.info("EJECUTANDO MOTOR CINEMATICO")
        try:
            res = self.kinematic.run_pipeline(acc, gyro, hs_idx, to_idx)
        except Exception as e:
            logger.error(f"ERROR CINEMATICO: {e}")
            sys.exit(1)

        # EXPORTAR RESULTADOS CSV
        n_strides = len(res["stride_lengths"])
        df_out = pd.DataFrame({
            "Stride_Length_m": res["stride_lengths"][:n_strides],
            "Gait_Speed_ms": res["gait_speed"][:n_strides],
            "Stride_Time_s": res["stride_times"][:n_strides]
        })
        
        n_mtc = min(n_strides, len(res["mtc"]))
        df_out.loc[:n_mtc-1, "MTC_m"] = res["mtc"][:n_mtc]

        df_out.to_csv(self.config.out, index=False)
        logger.info(f"EXPORTADO A: {self.config.out}")
        pos = res["position"]
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            linewidth=1
        )
        
        ax.set_title("Trayectoria 3D")
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        plt.show()

def main() -> None:
    """
    Punto entrada de aplicación.
    """
    # PARSEAR ARGUMENTOS TERMINAL
    parser = argparse.ArgumentParser(description="Pipeline Gait Analysis")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True)
    parser.add_argument("--fin", type=str, required=True)
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--out", type=str, default="resultados.csv")
    args = parser.parse_args()

    # INSTANCIAR CONFIGURACION PYDANTIC
    config = PipelineConfig(
        paciente=args.paciente,
        inicio=args.inicio,
        fin=args.fin,
        fs=args.fs,
        out=Path(args.out)
    )

    # INICIAR ORQUESTADOR PIPELINE
    orchestrator = GaitOrchestrator(config)
    orchestrator.run()

if __name__ == "__main__":
    main()