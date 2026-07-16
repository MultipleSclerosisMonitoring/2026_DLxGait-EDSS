# -*- coding: utf-8 -*-
"""
Script independiente para optimizar y validar empíricamente 
el umbral de detección de presión en ciclos de marcha.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from datetime import datetime
from pydantic import BaseModel, Field

class OptimizerConfig(BaseModel):
    """Configuracion del optimizador."""
    fs: int = Field(default=100, gt=0)
    hysteresis_ms: int = Field(default=50, ge=0)

class ThresholdOptimizer:
    """Optimizador empirico de umbrales."""

    def __init__(self, config: OptimizerConfig) -> None:
        """Inicializa parametros base."""
        self.config = config
        self.min_samples = int((self.config.hysteresis_ms / 1000.0) * self.config.fs)

    def _apply_hysteresis(self, mask: np.ndarray) -> np.ndarray:
        """Aplica histeresis temporal."""
        clean_mask = np.copy(mask)
        current_state = clean_mask[0]
        counter = 0
        
        for i in range(1, len(clean_mask)):
            if clean_mask[i] == current_state:
                counter = 0
            else:
                counter += 1
                
            if counter >= self.min_samples:
                current_state = clean_mask[i]
                counter = 0
                
            clean_mask[i] = current_state
            
        return clean_mask

    def detect_events(self, pressure: np.ndarray, fraction: float) -> Tuple[int, int]:
        """Detecta eventos dado umbral."""
        p_min = np.percentile(pressure, 5)
        p_max = np.percentile(pressure, 95)
        
        # EVITAR DIVISION CERO
        if p_max == p_min:
            return 0, 0
            
        umbral = p_min + fraction * (p_max - p_min)
        stance_mask = pressure > umbral
        stance_clean = self._apply_hysteresis(stance_mask)
        
        hs_count = np.sum((stance_clean[:-1] == False) & (stance_clean[1:] == True))
        to_count = np.sum((stance_clean[:-1] == True) & (stance_clean[1:] == False))
        
        return hs_count, to_count

    def sweep_thresholds(self, pressure: np.ndarray) -> pd.DataFrame:
        """Barre fracciones de umbral."""
        fractions = np.arange(0.10, 0.95, 0.05)
        results = []
        
        # ITERAR MULTIPLES UMBRALES
        for frac in fractions:
            hs, to = self.detect_events(pressure, frac)
            results.append({"Fraccion": frac, "Heel_Strikes": hs, "Toe_Offs": to})
            
        return pd.DataFrame(results)

    def plot_stability(self, df_results: pd.DataFrame) -> None:
        """Grafica estabilidad de deteccion."""
        plt.figure(figsize=(10, 5))
        plt.plot(df_results["Fraccion"], df_results["Heel_Strikes"], marker="o", label="Heel Strikes")
        plt.plot(df_results["Fraccion"], df_results["Toe_Offs"], marker="x", label="Toe Offs", linestyle="--")
        
        # CONFIGURAR FORMATO GRAFICA
        plt.title("Analisis de Sensibilidad del Umbral de Presion")
        plt.xlabel("Fraccion del Umbral (0.1 a 0.9)")
        plt.ylabel("Numero de Eventos Detectados")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # GUARDAR GRAFICA EN DISCO
        plt.tight_layout()
        plt.savefig("analisis_sensibilidad.png")

# ============================================================
# BLOQUE DE EJECUCION
# ============================================================
# INYECTAR RUTA PROYECTO
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTAR CLASE EXTRACCION
from A01_EXTRACCION_DATOS.extract_data_plus import cInfluxDB

# ============================================================
# BLOQUE DE EJECUCION
# ============================================================
if __name__ == "__main__":
    # CONFIGURAR INSTANCIA OPTIMIZADOR
    config = OptimizerConfig(fs=100)
    optimizer = ThresholdOptimizer(config)
    
    # CONFIGURAR CONEXION INFLUXDB
    config_path = r"C:\Users\jairi\OneDrive\Escritorio\TFM_CLONADO_FINALFINAL\A01_EXTRACCION_DATOS\config.yaml"
    extractor = cInfluxDB(config_path=config_path)
    
    try:
        # PARSEAR FECHAS DATETIME
        start_date = datetime.strptime("2026-02-09 22:37:00", "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime("2026-02-09 22:41:00", "%Y-%m-%d %H:%M:%S")

        # EXTRAER DATOS REALES
        print("Extrayendo datos InfluxDB...")
        df_pie = extractor.query_data(
            from_date=start_date, 
            to_date=end_date, 
            qtok="PRCHUG025-11", 
            pie="Right"
        )
        
        # VERIFICAR DATOS EXISTENTES
        if df_pie.empty:
            print("DATOS NO ENCONTRADOS.")
            sys.exit(0)
            
        # CALCULAR PRESION TOTAL
        presion_real = df_pie[["S0", "S1", "S2"]].astype(float).sum(axis=1).values
        
        # EJECUTAR BARRIDO UMBRALES
        print("Iniciando barrido umbrales...")
        df_analisis = optimizer.sweep_thresholds(presion_real)
        
        # MOSTRAR TABLA RESULTADOS
        print("\nRESULTADOS DEL BARRIDO:")
        print(df_analisis.to_string(index=False))
        
        # GRAFICAR ESTABILIDAD
        plt.figure(figsize=(10, 5))
        plt.plot(df_analisis["Fraccion"], df_analisis["Heel_Strikes"], marker="o", label="Heel Strikes")
        plt.plot(df_analisis["Fraccion"], df_analisis["Toe_Offs"], marker="x", label="Toe Offs", linestyle="--")
        plt.title("Sensibilidad Umbral Presion: PRCHUG025-11")
        plt.xlabel("Fraccion Umbral")
        plt.ylabel("Eventos Detectados")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # GUARDAR GRAFICA DISCO
        plt.savefig("analisis_sensibilidad_PRCHUG025-11.png")
        print("GRAFICA GUARDADA CORRECTAMENTE.")
        
    finally:
        # CERRAR CONEXION INFLUXDB
        extractor.close()