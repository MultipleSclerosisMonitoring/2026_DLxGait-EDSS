# -*- coding: utf-8 -*-
"""
Detector de eventos biomecánicos.
Detecta Heel Strike y Toe Off utilizando señales de presión plantar
o giroscopio, aplicando histéresis temporal.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.signal import find_peaks
from pydantic import BaseModel, Field
from scipy.ndimage import minimum_filter1d


class EventDetectorConfig(BaseModel):
    """
    Configuración para detección adaptativa.
    Umbrales relajados para pacientes patológicos.
    """
    fs: int = Field(default=100, gt=0)
    
    # UMBRAL PRESION RELAJADO
    threshold_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    
    # UMBRAL GIRO RELAJADO
    gyro_prominence: float = Field(default=15.0, gt=0.0)
    
    # HISTERESIS SIN CAMBIOS
    hysteresis_ms: int = Field(default=50, ge=0)


class EventDetector:
    """Clase para detectar eventos biomecánicos."""

    def __init__(self, config: EventDetectorConfig) -> None:
        """
        Inicializa el detector cinemático.

        :param config: Configuración estructurada.
        :type config: EventDetectorConfig
        """
        self.config = config
        self.min_samples = int(
            (self.config.hysteresis_ms / 1000.0) * self.config.fs
        )

    def detect_from_pressure(
        self,
        pressure_sig: np.ndarray,
        threshold_fraction_override: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Identifica eventos usando umbral dinámico sobre señal corregida.
        Incluye eliminación de deriva (baseline drift).
        """
        # 1. CORRECTOR DINAMICO DE LINEA BASE (Filtro morfológico)
        # Ventana de 2 segundos. Busca el mínimo local (pie en el aire) y lo resta.
        window_size = int(2.0 * self.config.fs)
        baseline = minimum_filter1d(pressure_sig, size=window_size)
        
        # Señal limpia donde el vuelo vuelve a cero
        pressure_clean = pressure_sig - baseline

        # 2. CALCULAR UMBRAL DINAMICO (Sobre la señal limpia)
        p_min = np.percentile(pressure_clean, 5)
        p_max = np.percentile(pressure_clean, 95)

        fraction = (
            threshold_fraction_override
            if threshold_fraction_override is not None
            else self.config.threshold_fraction
        )

        dyn_threshold = p_min + fraction * (p_max - p_min)

        # 3. MASCARA DE APOYO
        stance_mask = pressure_clean > dyn_threshold
        samples_above = np.sum(stance_mask)
        stance_pct = 100.0 * samples_above / len(pressure_clean) if len(pressure_clean) > 0 else 0.0

        # 4. APLICAR FILTRO HISTERESIS
        stance_clean = self._apply_hysteresis(stance_mask)

        # 5. DETECTAR FLANCOS EVENTOS
        hs_indices = np.where(
            (stance_clean[:-1] == False) & (stance_clean[1:] == True)
        )[0]

        to_indices = np.where(
            (stance_clean[:-1] == True) & (stance_clean[1:] == False)
        )[0]

        # 6. VALIDAR HUECOS ANOMALOS
        self._warn_anomalous_gaps(hs_indices, to_indices)

        return hs_indices, to_indices, stance_pct

    def detect_from_gyro(
        self,
        gyro_y_sig: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta eventos usando rotación sagital.
        """
        # PICOS POSITIVOS HS
        hs_idx, _ = find_peaks(
            gyro_y_sig,
            prominence=self.config.gyro_prominence,
            distance=self.min_samples
        )

        # PICOS NEGATIVOS TO
        to_idx, _ = find_peaks(
            -gyro_y_sig,
            prominence=self.config.gyro_prominence,
            distance=self.min_samples
        )

        return hs_idx, to_idx

    def _apply_hysteresis(self, mask: np.ndarray) -> np.ndarray:
        """
        Suprime transiciones espurias temporalmente.
        """
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

    def _warn_anomalous_gaps(self, hs_idx: np.ndarray, to_idx: np.ndarray) -> None:
        """
        Advierte sobre posibles fallos de detección.
        """
        if len(hs_idx) < 2:
            return

        # VALIDAR DISTANCIA PASOS
        for i in range(len(hs_idx) - 1):
            duracion = (hs_idx[i+1] - hs_idx[i]) / self.config.fs
            if duracion > 2.5:
                # SE OMITE PRINT PARA PRODUCCION
                pass