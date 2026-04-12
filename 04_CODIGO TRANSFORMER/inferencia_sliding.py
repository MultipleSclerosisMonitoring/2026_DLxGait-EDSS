# -*- coding: utf-8 -*-
"""
Script de inferencia para secuencias mixtas (REPOSO + MARCHA).
Simula una transicion real para validar la respuesta temporal del modelo.
"""

import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import torch.nn as nn
from scipy.fft import rfft

class FFTModel(nn.Module):
    """
    Arquitectura del modelo de red neuronal basada en la Transformada Rápida de Fourier (FFT).
    """
    def __init__(self, input_dim: int = 51 * 290):
        """
        Inicializa las capas del clasificador lineal.

        :param input_dim: Dimensión del tensor de entrada tras aplanarlo. Por defecto 51 * 290.
        :type input_dim: int
        """
        super(FFTModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante (forward pass) del modelo.

        :param x: Tensor de entrada con las características de frecuencia.
        :type x: torch.Tensor
        :return: Tensor con las predicciones sin normalizar (logits) para las clases.
        :rtype: torch.Tensor
        """
        return self.classifier(x)

class MotorInferenciaMixta:
    """
    Motor de inferencia para procesar secuencias continuas y detectar transiciones temporales.
    """
    def __init__(self, mod_path: Path, scl_path: Path):
        """
        Inicializa el motor cargando el modelo y el escalador entrenados.

        :param mod_path: Ruta al archivo del modelo de pesos (.pth).
        :type mod_path: Path
        :param scl_path: Ruta al archivo del escalador guardado (.joblib).
        :type scl_path: Path
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = joblib.load(scl_path)
        self.model = FFTModel().to(self.device)
        self.model.load_state_dict(torch.load(mod_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
    def _get_fft(self, data: np.ndarray) -> np.ndarray:
        """
        Aplica la Transformada Rápida de Fourier (FFT) para extraer características de frecuencia.

        :param data: Array de datos escalados con forma (batch, ventanas, características).
        :type data: np.ndarray
        :return: Array con las magnitudes de la FFT normalizadas.
        :rtype: np.ndarray
        """
        data_fft = np.abs(rfft(data, axis=1))
        return (data_fft / data.shape[1]).astype(np.float32)

    def procesar_mixto(self, paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga, concatena y aplica una ventana deslizante sobre múltiples archivos para simular una transición.

        :param paths: Lista de rutas a los archivos Parquet que se concatenarán.
        :type paths: List[Path]
        :return: Tupla conteniendo un array de tiempos (segundos) y un array de probabilidades de marcha.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        dfs = []
        for p in paths:
            dfs.append(pd.read_parquet(p).T.dropna())
        
        datos_crudos = pd.concat(dfs, axis=0).values
        num_muestras = len(datos_crudos)
        
        predicciones, tiempos = [], []
        win, step, hz = 100, 25, 75

        for start in range(0, num_muestras - win + 1, step):
            ventana = datos_crudos[start : start + win, :]
            
            v_flat = ventana.reshape(-1, ventana.shape[1])
            v_scaled = self.scaler.transform(v_flat).reshape(1, win, ventana.shape[1])
            
            v_fft = self._get_fft(v_scaled)
            tensor_x = torch.from_numpy(v_fft).float().to(self.device)
            
            with torch.no_grad():
                out = self.model(tensor_x)
                prob = torch.softmax(out, dim=1)[0, 1].item()
            
            predicciones.append(prob)
            tiempos.append((start + (win / 2)) / hz)

        return np.array(tiempos), np.array(predicciones)

    def reporte_transicion(self, tiempos: np.ndarray, probabilidades: np.ndarray):
        """
        Imprime un reporte en consola mostrando el estado detectado y alertando de las transiciones.

        :param tiempos: Array temporal en segundos.
        :type tiempos: np.ndarray
        :param probabilidades: Array con las probabilidades de detección de marcha.
        :type probabilidades: np.ndarray
        """
        print("\n" + "="*60)
        print("# ANALISIS DE TRANSICION TEMPORAL (MIXTO)")
        print("="*60)
        print(f"{'TIEMPO (s)':<15} | {'ESTADO':<15} | {'PROBABILIDAD'}")
        print("-" * 60)

        for i, (t, p) in enumerate(zip(tiempos, probabilidades)):
            estado = "MARCHA" if p >= 0.5 else "REPOSO"
            
            aviso = ""
            if i > 0:
                est_prev = "MARCHA" if probabilidades[i-1] >= 0.5 else "REPOSO"
                if estado != est_prev:
                    aviso = " <--- CAMBIO DETECTADO!"

            if i % 4 == 0 or aviso != "":
                print(f"{t:<15.2f} | {estado:<15} | {p:.4f} {aviso}")

        print("="*60 + "\n")

if __name__ == "__main__":
    BASE = Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM")
    MOD = BASE / "MODELOS_ENTRENADOS" / "modelo_frecuencia.pth"
    SCL = BASE / "MODELOS_ENTRENADOS" / "scaler_gait.joblib"
    RES = BASE / "01_EXTRACCION DE DATOS" / "resultados"
    
    FILE_REPOSO = RES / "segment_002_tensor-0.parquet"
    FILE_MARCHA = RES / "segment_000_tensor-1.parquet"

    try:
        motor = MotorInferenciaMixta(MOD, SCL)
        t, p = motor.procesar_mixto([FILE_REPOSO, FILE_MARCHA])
        motor.reporte_transicion(t, p)
    except Exception as e:
        print(f"# ERROR: {e}")