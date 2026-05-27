# -*- coding: utf-8 -*-
"""
Motor cinemático tridimensional para reconstrucción biomecánica
basada en IMUs utilizando fusión sensorial y ZUPT.
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from pydantic import BaseModel, Field

# ============================================================
# INYECTAR RUTA PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# IMPORTAR CLASE EXTRACCION
# ============================================================

try:
    from A01_EXTRACCION_DATOS.extract_data_plus import cInfluxDB
except ImportError as e:
    print(f"ERROR DE IMPORTACION: {e}")
    sys.exit(1)

# ============================================================
# CONFIGURACION DETECTOR
# ============================================================

class EventDetectorConfig(BaseModel):
    """
    Configuracion estructurada para deteccion adaptativa.
    """

    fs: int = Field(default=100, gt=0)

    # SUBIDO PARA EVITAR STANCE PERMANENTE
    threshold_fraction: float = Field(default=0.50, ge=0.0, le=1.0)

    gyro_prominence: float = Field(default=50.0, gt=0.0)

    hysteresis_ms: int = Field(default=50, ge=0)


# ============================================================
# DETECTOR EVENTOS
# ============================================================

class EventDetector:
    """
    Clase para detectar eventos biomecanicos adaptativos.
    """

    def __init__(self, config: EventDetectorConfig) -> None:
        """
        Inicializa el detector cinematico.
        """

        self.config = config

        # DEFINIR VENTANA REBOTE
        self.min_samples = int(
            (self.config.hysteresis_ms / 1000.0) * self.config.fs
        )

    def detect_from_pressure(
        self,
        pressure_sig: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identifica eventos usando umbral dinámico.
        """

        # ====================================================
        # CALCULAR UMBRAL DINAMICO
        # ====================================================

        p_min = np.min(pressure_sig)
        p_max = np.max(pressure_sig)

        dyn_threshold = (
            p_min
            + self.config.threshold_fraction * (p_max - p_min)
        )

        print("\n==============================")
        print("DIAGNOSTICO PRESION")
        print("==============================")
        print(f"MIN PRESION: {p_min:.2f}")
        print(f"MAX PRESION: {p_max:.2f}")
        print(f"THRESHOLD:   {dyn_threshold:.2f}")

        samples_above = np.sum(pressure_sig > dyn_threshold)

        print(f"MUESTRAS SOBRE UMBRAL: {samples_above}")
        print(f"TOTAL MUESTRAS:        {len(pressure_sig)}")

        # ====================================================
        # APLICAR UMBRAL DINAMICO
        # ====================================================

        stance_mask = pressure_sig > dyn_threshold

        # ====================================================
        # APLICAR HISTÉRESIS
        # ====================================================

        stance_clean = self._apply_hysteresis(stance_mask)

        # ====================================================
        # DETECTAR FLANCOS
        # ====================================================

        # HEEL STRIKE = FALSE -> TRUE
        hs_indices = np.where(
            (stance_clean[:-1] == False)
            & (stance_clean[1:] == True)
        )[0]

        # TOE OFF = TRUE -> FALSE
        to_indices = np.where(
            (stance_clean[:-1] == True)
            & (stance_clean[1:] == False)
        )[0]

        return hs_indices, to_indices

    def detect_from_gyro(
        self,
        gyro_y_sig: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta eventos usando rotacion sagital.
        """

        # PICOS POSITIVOS
        hs_idx, _ = find_peaks(
            gyro_y_sig,
            prominence=self.config.gyro_prominence,
            distance=self.min_samples
        )

        # PICOS NEGATIVOS
        to_idx, _ = find_peaks(
            -gyro_y_sig,
            prominence=self.config.gyro_prominence,
            distance=self.min_samples
        )

        return hs_idx, to_idx

    def _apply_hysteresis(
        self,
        mask: np.ndarray
    ) -> np.ndarray:
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

                # VALIDAR CAMBIO
                if counter >= self.min_samples:

                    current_state = clean_mask[i]

                    counter = 0

            clean_mask[i] = current_state

        return clean_mask


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from datetime import datetime

    # ========================================================
    # ARGUMENTOS
    # ========================================================

    parser = argparse.ArgumentParser(
        description="Extraccion eventos reales"
    )

    parser.add_argument(
        "--paciente",
        type=str,
        required=True,
        help="ID paciente"
    )

    parser.add_argument(
        "--inicio",
        type=str,
        required=True,
        help="Fecha inicio"
    )

    parser.add_argument(
        "--fin",
        type=str,
        required=True,
        help="Fecha fin"
    )

    parser.add_argument(
        "--fs",
        type=int,
        default=100,
        help="Frecuencia muestreo"
    )

    args = parser.parse_args()

    # ========================================================
    # CONFIGURAR DETECTOR
    # ========================================================

    config = EventDetectorConfig(fs=args.fs)

    detector = EventDetector(config)

    # ========================================================
    # CONFIG YAML
    # ========================================================

    config_yaml_path = (
        PROJECT_ROOT
        / "A01_EXTRACCION_DATOS"
        / ".config.yaml"
    )

    if not config_yaml_path.exists():

        print(
            f"ERROR: Archivo .config.yaml no encontrado en "
            f"{config_yaml_path}"
        )

        sys.exit(1)

    # ========================================================
    # EXTRAER DATOS
    # ========================================================

    try:

        extractor = cInfluxDB(
            config_path=str(config_yaml_path)
        )

        print("CONSULTANDO BASE DATOS...")

        start_dt = datetime.strptime(
            args.inicio,
            "%Y-%m-%d %H:%M:%S"
        )

        end_dt = datetime.strptime(
            args.fin,
            "%Y-%m-%d %H:%M:%S"
        )

        df_paciente = extractor.query_data(
            from_date=start_dt,
            to_date=end_dt,
            qtok=args.paciente,
            pie="Right"
        )

    except Exception as e:

        print(f"ERROR EXTRACCION INFLUXDB: {e}")

        sys.exit(1)

    finally:

        if 'extractor' in locals():
            extractor.close()

    # ========================================================
    # VALIDAR DATOS
    # ========================================================

    if df_paciente is None or df_paciente.empty:

        print(
            f"ERROR: DATOS NO ENCONTRADOS PARA "
            f"{args.paciente}"
        )

        sys.exit(1)

    # ========================================================
    # PREPARAR SEÑAL PRESION
    # ========================================================

    try:

        presion_real = (
            df_paciente[["S0", "S1", "S2"]]
            .astype(float)
            .sum(axis=1)
            .values
        )

        # ====================================================
        # ELIMINAR OFFSET BASAL
        # ====================================================

        baseline = np.percentile(presion_real, 5)

        presion_real = presion_real - baseline

        presion_real[presion_real < 0] = 0

    except KeyError:

        print("ERROR: FALTAN COLUMNAS PRESION")

        sys.exit(1)

    # ========================================================
    # ESTADISTICAS
    # ========================================================

    print("\n==============================")
    print("ESTADISTICAS PRESION")
    print("==============================")

    print(
        df_paciente[["S0", "S1", "S2"]]
        .describe()
    )

    # ========================================================
    # DETECCION EVENTOS
    # ========================================================

    hs_idx, to_idx = detector.detect_from_pressure(
        presion_real
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print("\n==============================")
    print("ANALISIS COMPLETADO")
    print("==============================")

    print(f"Paciente: {args.paciente}")

    print(f"Muestras procesadas: {len(df_paciente)}")

    print(f"Heel Strikes: {len(hs_idx)}")

    print(f"Toe Offs: {len(to_idx)}")

    # ========================================================
    # VISUALIZACION
    # ========================================================

    plt.figure(figsize=(16, 6))

    plt.plot(
        presion_real,
        linewidth=1
    )

    # HEEL STRIKES
    plt.scatter(
        hs_idx,
        presion_real[hs_idx],
        marker="o",
        s=30,
        label="Heel Strike"
    )

    # TOE OFFS
    plt.scatter(
        to_idx,
        presion_real[to_idx],
        marker="x",
        s=30,
        label="Toe Off"
    )

    plt.title(
        f"Deteccion Eventos - {args.paciente}"
    )

    plt.xlabel("Muestras")

    plt.ylabel("Presion")

    plt.legend()

    plt.grid(True)

    plt.show()