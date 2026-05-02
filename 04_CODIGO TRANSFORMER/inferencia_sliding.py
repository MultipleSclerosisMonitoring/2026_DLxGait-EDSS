# -*- coding: utf-8 -*-
"""
Script de inferencia para secuencias mixtas (REPOSO + MARCHA).
Simula una transicion real para validar la respuesta temporal del modelo
mediante el uso de un algoritmo de ventana deslizante continua.
"""

import torch
import joblib
import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
from typing import Tuple, List
import torch.nn as nn
from scipy.fft import rfft
from pydantic import BaseModel, FilePath, PositiveInt, Field

# CONFIGURACION LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# EXCEPCIONES PERSONALIZADAS
class InferenceError(Exception):
    """Excepción para errores durante el motor de inferencia."""
    pass

class EngineConfig(BaseModel):
    """Configuración para el motor de inferencia deslizante."""
    model_path: FilePath
    scaler_path: FilePath
    window_size: PositiveInt = Field(default=100)
    step_size: PositiveInt = Field(default=25)
    sampling_hz: PositiveInt = Field(default=75)

class FFTModel(nn.Module):
    """
    Arquitectura del modelo de red neuronal basada en la Transformada 
    Rápida de Fourier (FFT).
    """
    def __init__(self, input_dim: int = 51 * 290) -> None:
        """
        Inicializa las capas del clasificador lineal.

        :param input_dim: Dimensión del tensor aplanado.
        :type input_dim: int
        """
        super(FFTModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inferencia del modelo.

        :param x: Tensor de entrada frecuencial.
        :type x: torch.Tensor
        :return: Logits de predicción.
        :rtype: torch.Tensor
        """
        return self.classifier(x)

class MotorInferenciaMixta:
    """
    Motor de inferencia para procesar secuencias continuas y 
    detectar transiciones temporales.
    """
    def __init__(self, config: EngineConfig) -> None:
        """
        Inicializa el motor cargando configuración, modelo y escalador.

        :param config: Configuración con rutas y parámetros de ventana.
        :type config: EngineConfig
        """
        self.config = config
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.scaler = joblib.load(self.config.scaler_path)
            self.model = FFTModel().to(self.device)
            self.model.load_state_dict(
                torch.load(self.config.model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            logger.info(f"MOTOR CARGADO EN DISPOSITIVO: {self.device.upper()}")
        except Exception as e:
            logger.error(f"FALLO CARGANDO MODELOS: {e}")
            raise InferenceError("No se pudo inicializar el motor de inferencia.") from e

    def _get_fft(self, data: np.ndarray) -> np.ndarray:
        """
        Aplica FFT para extraer características.

        :param data: Array de datos escalados (batch, ventanas, características).
        :type data: np.ndarray
        :return: Magnitudes FFT normalizadas.
        :rtype: np.ndarray
        """
        data_fft = np.abs(rfft(data, axis=1))
        return (data_fft / data.shape[1]).astype(np.float32)

    def procesar_mixto(self, paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga, concatena y aplica ventana deslizante sobre múltiples archivos.

        :param paths: Lista de rutas Parquet.
        :type paths: List[Path]
        :return: Tupla con array de tiempos (segundos) y probabilidades.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        try:
            dfs: List[pd.DataFrame] = []
            for p in paths:
                if not p.exists():
                    raise FileNotFoundError(f"El archivo {p.name} no existe.")
                dfs.append(pd.read_parquet(p).T.dropna())
            
            datos_crudos: np.ndarray = pd.concat(dfs, axis=0).values
            num_muestras: int = len(datos_crudos)
            
            predicciones: List[float] = []
            tiempos: List[float] = []
            
            win: int = self.config.window_size
            step: int = self.config.step_size
            hz: int = self.config.sampling_hz

            logger.info("INICIANDO SLIDING WINDOW SOBRE SECUENCIA CONTINUA")

            for start in range(0, num_muestras - win + 1, step):
                ventana: np.ndarray = datos_crudos[start : start + win, :]
                
                v_flat: np.ndarray = ventana.reshape(-1, ventana.shape[1])
                v_scaled: np.ndarray = self.scaler.transform(v_flat).reshape(1, win, ventana.shape[1])
                
                v_fft: np.ndarray = self._get_fft(v_scaled)
                tensor_x: torch.Tensor = torch.from_numpy(v_fft).float().to(self.device)
                
                with torch.no_grad():
                    out: torch.Tensor = self.model(tensor_x)
                    prob: float = torch.softmax(out, dim=1)[0, 1].item()
                
                predicciones.append(prob)
                tiempos.append((start + (win / 2)) / hz)

            return np.array(tiempos), np.array(predicciones)

        except (OSError, pd.errors.EmptyDataError, ValueError) as e:
            logger.error(f"ERROR PROCESANDO ARCHIVOS MIXTOS: {e}")
            raise InferenceError("Fallo en la extracción y predicción continua.") from e

    def reporte_transicion(self, tiempos: np.ndarray, probabilidades: np.ndarray) -> None:
        """
        Imprime un reporte mostrando el estado detectado y alertando transiciones.

        :param tiempos: Array temporal en segundos.
        :type tiempos: np.ndarray
        :param probabilidades: Array con probabilidades de marcha.
        :type probabilidades: np.ndarray
        """
        print("\n" + "="*60)
        print("# ANALISIS DE TRANSICION TEMPORAL (MIXTO)")
        print("="*60)
        print(f"{'TIEMPO (s)':<15} | {'ESTADO':<15} | {'PROBABILIDAD'}")
        print("-" * 60)

        for i, (t, p) in enumerate(zip(tiempos, probabilidades)):
            estado: str = "MARCHA" if p >= 0.5 else "REPOSO"
            
            aviso: str = ""
            if i > 0:
                est_prev: str = "MARCHA" if probabilidades[i-1] >= 0.5 else "REPOSO"
                if estado != est_prev:
                    aviso = " <--- CAMBIO DETECTADO!"

            if i % 4 == 0 or aviso != "":
                print(f"{t:<15.2f} | {estado:<15} | {p:.4f} {aviso}")

        print("="*60 + "\n")

def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Inferencia Continua Biomecánica")
    parser.add_argument("--modelo", type=Path, required=True, help="Ruta al archivo .pth")
    parser.add_argument("--scaler", type=Path, required=True, help="Ruta al archivo .joblib")
    parser.add_argument("--datos", type=Path, nargs="+", required=True, help="Lista de archivos .parquet a evaluar")
    args = parser.parse_args()

    try:
        cfg = EngineConfig(
            model_path=args.modelo,
            scaler_path=args.scaler
        )
        motor = MotorInferenciaMixta(cfg)
        
        # PROCESAR LISTA DE PARQUETS PASADA POR TERMINAL
        t, p = motor.procesar_mixto(args.datos)
        motor.reporte_transicion(t, p)
        
    except InferenceError as e:
        logger.critical(f"EJECUCION ABORTADA: {e}")

if __name__ == "__main__":
    main()
