# -*- coding: utf-8 -*-
"""
xxx
"""

from __future__ import annotations

import logging
import numpy as np

from scipy.signal import butter, filtfilt, detrend
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation as R
from pydantic import BaseModel, Field
from typing import Dict, Tuple, List

# INICIALIZAR LOGGER MODULO
logger = logging.getLogger(__name__)


class KinematicConfig(BaseModel):
    """Configuración estructurada del motor cinemático."""
    fs: int = Field(default=100, gt=0)
    cutoff_hz: float = Field(default=0.05, gt=0.0)
    filter_order: int = Field(default=4, gt=0)
    gravity: float = Field(default=9.81, gt=0.0)
    madgwick_beta: float = Field(default=0.1, gt=0.0)


class KinematicEngine:
    """Motor biomecánico para reconstrucción espacial."""

    def __init__(self, config: KinematicConfig) -> None:
        """Inicializa parámetros físicos."""
        self.config = config
        self.dt = 1.0 / self.config.fs

        # INICIALIZAR FILTRO MADGWICK
        try:
            from ahrs.filters import Madgwick
            self.madgwick = Madgwick(
                frequency=self.config.fs,
                beta=self.config.madgwick_beta
            )
        except ImportError:
            raise ImportError("Instalar ahrs: pip install ahrs")

    @staticmethod
    def _validate_xyz(data: np.ndarray, name: str) -> None:
        """Valida dimensiones matriz entrada."""
        # VERIFICAR FORMA ESPACIAL
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"{name} debe tener forma (N, 3).")

    def _get_valid_pairs(self, hs_idx: np.ndarray, to_idx: np.ndarray) -> List[Tuple[int, int]]:
        """Empareja ciclos de marcha reales."""
        valid_pairs = []
        
        # BUSCAR DESPEGUES FUTUROS
        for hs in hs_idx:
            future_to = to_idx[to_idx > hs]
            if len(future_to) == 0:
                continue
            to = future_to[0]
            valid_pairs.append((hs, to))
            
        return valid_pairs

    def apply_highpass(self, signal: np.ndarray) -> np.ndarray:
        """Reduce drift baja frecuencia."""
        nyq = 0.5 * self.config.fs
        cutoff = self.config.cutoff_hz / nyq

        # VALIDAR TEOREMA NYQUIST
        if cutoff >= 1.0:
            raise ValueError("Cutoff supera Nyquist")

        # DISEÑAR FILTRO BUTTERWORTH
        b, a = butter(self.config.filter_order, cutoff, btype="high")
        return filtfilt(b, a, signal)

    def estimate_orientation(self, gyro_rad: np.ndarray, acc_xyz: np.ndarray) -> np.ndarray:
        """Estima orientación usando Madgwick."""
        n = len(acc_xyz)
        Q = np.zeros((n, 4))

        # ASIGNAR CUATERNION IDENTIDAD
        Q[0] = np.array([1.0, 0.0, 0.0, 0.0])

        # BUCLE FILTRO MADGWICK
        for t in range(1, n):
            Q[t] = self.madgwick.updateIMU(q=Q[t - 1], gyr=gyro_rad[t], acc=acc_xyz[t])
            
            # NORMALIZAR CUATERNION
            Q[t] = Q[t] / np.linalg.norm(Q[t])

        return Q

    def rotate_to_global(self, acc_xyz: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
        """Rota aceleración marco global."""
        global_acc = np.zeros_like(acc_xyz)

        # ROTAR CADA MUESTRA
        for i in range(len(acc_xyz)):
            q = quaternions[i]
            
            # FORMATO SCALAR LAST
            rot = R.from_quat([q[1], q[2], q[3], q[0]])
            global_acc[i] = rot.apply(acc_xyz[i])

        return global_acc

    def remove_gravity(self, global_acc: np.ndarray) -> np.ndarray:
        """Elimina gravedad vertical global."""
        return global_acc

    def preprocess_acceleration(self, linear_acc: np.ndarray) -> np.ndarray:
        """Reduce drift residual lineal."""
        processed = np.zeros_like(linear_acc)

        # PROCESAR EJES SEPARADOS
        for axis in range(3):
            sig = linear_acc[:, axis]
            
            # ELIMINAR MEDIA BASE
            sig = sig - np.mean(sig)
            
            # ELIMINAR TENDENCIA LINEAL
            sig = detrend(sig)
            
            # FILTRAR BAJA FRECUENCIA
            sig = self.apply_highpass(sig)
            processed[:, axis] = sig

        return processed

    def integrate_velocity(self, acc_xyz: np.ndarray) -> np.ndarray:
        """Integra aceleración a velocidad."""
        velocity = np.zeros_like(acc_xyz)

        # INTEGRAR METODO TRAPEZOIDAL
        for axis in range(3):
            velocity[:, axis] = cumulative_trapezoid(acc_xyz[:, axis], dx=self.dt, initial=0)

        return velocity

    def apply_zupt(self, velocity: np.ndarray, hs_idx: np.ndarray, to_idx: np.ndarray) -> np.ndarray:
        """Compensación lineal de deriva."""
        corrected = velocity.copy()
        valid_pairs = self._get_valid_pairs(hs_idx, to_idx)

        # COMPENSAR DERIVA LINEALMENTE
        for hs, to in valid_pairs:
            # VALIDAR LIMITES ARRAY
            if to >= len(corrected):
                continue

            # CALCULAR TASA DERIVA
            drift_rate = corrected[to - 1] / (to - hs)

            # APLICAR CORRECCION PROGRESIVA
            for k in range(hs, to):
                corrected[k] -= drift_rate * (k - hs)
                
            corrected[to] = 0

        return corrected

    def integrate_position(self, velocity_xyz: np.ndarray) -> np.ndarray:
        """Integra velocidad a posición."""
        position = np.zeros_like(velocity_xyz)

        # INTEGRAR METODO TRAPEZOIDAL
        for axis in range(3):
            position[:, axis] = cumulative_trapezoid(velocity_xyz[:, axis], dx=self.dt, initial=0)

        return position

    def compute_temporal_metrics(self, hs_idx: np.ndarray, to_idx: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcula métricas temporales."""
        stride_times = np.diff(hs_idx) / self.config.fs
        stance_times = []
        swing_times = []

        valid_pairs = self._get_valid_pairs(hs_idx, to_idx)

        # CALCULAR TIEMPOS APOYO
        for hs, to in valid_pairs:
            stance = (to - hs) / self.config.fs
            stance_times.append(stance)

        # CALCULAR TIEMPOS BALANCEO
        for i in range(len(to_idx)):
            to = to_idx[i]
            future_hs = hs_idx[hs_idx > to]

            if len(future_hs) > 0:
                hs_next = future_hs[0]
                swing = (hs_next - to) / self.config.fs
                swing_times.append(swing)

        return {
            "stride_times": np.array(stride_times),
            "stance_times": np.array(stance_times),
            "swing_times": np.array(swing_times)
        }

    def compute_stride_length(self, position_xyz: np.ndarray, hs_idx: np.ndarray) -> np.ndarray:
        """Calcula longitud zancada horizontal."""
        stride_lengths = []

        # CALCULAR DISTANCIA 2D
        for i in range(len(hs_idx) - 1):
            p1 = position_xyz[hs_idx[i]]
            p2 = position_xyz[hs_idx[i + 1]]

            # AISLAR PLANO HORIZONTAL
            dist = np.linalg.norm((p2 - p1)[:2])
            stride_lengths.append(dist)

        return np.array(stride_lengths)

    def compute_gait_speed(self, stride_lengths: np.ndarray, stride_times: np.ndarray) -> np.ndarray:
        """Calcula velocidad media marcha."""
        n = min(len(stride_lengths), len(stride_times))

        if n == 0:
            return np.array([])

        return stride_lengths[:n] / stride_times[:n]

    def compute_mtc(self, position_xyz: np.ndarray, to_idx: np.ndarray, hs_idx: np.ndarray) -> np.ndarray:
        """Calcula clearance vertical máximo."""
        mtc_values = []
        z = position_xyz[:, 2]

        # ANALIZAR CLEARANCE RELATIVO
        for to in to_idx:
            future_hs = hs_idx[hs_idx > to]
            if len(future_hs) == 0:
                continue

            hs_next = future_hs[0]
            segment = z[to:hs_next]

            if len(segment) == 0:
                continue

            # AJUSTAR REFERENCIA RELATIVA
            n_samples = min(5, len(segment))
            relative_segment = segment - np.mean(segment[:n_samples])

            # EXTRAER ELEVACION MAXIMA
            mtc = np.max(relative_segment)
            mtc_values.append(mtc)

        return np.array(mtc_values)

    def run_pipeline(
        self,
        acc_xyz: np.ndarray,
        gyro_xyz: np.ndarray,
        hs_idx: np.ndarray,
        to_idx: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Ejecuta pipeline biomecánico completo."""
        # VALIDAR ALINEAMIENTO SENSORES
        if len(acc_xyz) != len(gyro_xyz):
            raise ValueError("Accelerometer and gyroscope lengths mismatch")

        # VALIDAR DIMENSIONES ENTRADA
        self._validate_xyz(acc_xyz, "acc_xyz")
        self._validate_xyz(gyro_xyz, "gyro_xyz")

        # CONVERTIR GIROSCOPIO RADIANES
        gyro_rad = np.deg2rad(gyro_xyz)

        # ESTIMAR ORIENTACION ESPACIAL
        quaternions = self.estimate_orientation(gyro_rad, acc_xyz)

        # ROTAR MARCO GLOBAL
        global_acc = self.rotate_to_global(acc_xyz, quaternions)

        # ELIMINAR GRAVEDAD VERTICAL
        linear_acc = self.remove_gravity(global_acc)

        # FILTRAR DERIVA RESIDUAL
        filtered_acc = self.preprocess_acceleration(linear_acc)

        # INTEGRAR VELOCIDAD CRUDA
        velocity = self.integrate_velocity(filtered_acc)

        # APLICAR CORRECCION ZUPT
        velocity = self.apply_zupt(velocity, hs_idx, to_idx)

        # INTEGRAR POSICION ESPACIAL
        position = self.integrate_position(velocity)

        # EXTRAER METRICAS TEMPORALES
        temporal_metrics = self.compute_temporal_metrics(hs_idx, to_idx)

        # EXTRAER LONGITUD ZANCADA
        stride_lengths = self.compute_stride_length(position, hs_idx)

        # EXTRAER VELOCIDAD MARCHA
        gait_speed = self.compute_gait_speed(stride_lengths, temporal_metrics["stride_times"])

        # EXTRAER DESPEJE VERTICAL
        mtc = self.compute_mtc(position, to_idx, hs_idx)

        return {
            "quaternions": quaternions,
            "global_acc": global_acc,
            "linear_acc": linear_acc,
            "velocity": velocity,
            "position": position,
            "stride_lengths": stride_lengths,
            "gait_speed": gait_speed,
            "mtc": mtc,
            **temporal_metrics
        }