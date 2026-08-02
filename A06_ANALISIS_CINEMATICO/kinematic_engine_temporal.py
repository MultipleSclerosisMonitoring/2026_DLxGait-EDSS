# -*- coding: utf-8 -*-
"""
Motor de METRICAS TEMPORALES biomecanicas (version recortada de A06).

Esta version elimina TODA la reconstruccion espacial (orientacion via
Madgwick, integracion de aceleracion a velocidad/posicion, longitud de
zancada, velocidad de marcha, MTC, velocidad pico de vuelo), siguiendo
la evaluacion del director del TFM: "las metricas temporales parecen
internamente consistentes" pero "las metricas espaciales no son
clinicamente creibles para una prueba real de marcha continua" -- la
parte espacial se descarta, no se repara, porque su fragilidad viene de
una cadena de heuristicas (escala de acceleracion por "primera ventana
seguro", integracion doble propensa a deriva) que no se resuelve con un
ajuste puntual.

Lo que SI se mantiene (evaluado como solido):
  - Segmentacion por huecos (segmentar_por_huecos).
  - Deteccion de Mid-Stance (calcular_mid_stance), usada aqui solo como
    apoyo para diagnostico de huecos, no para integracion.
  - Metricas temporales: stride_times, stance_times, swing_times
    (compute_temporal_metrics).
  - Diagnostico de huecos temporales anomalos entre Mid-Stances.

Lo que se elimina (documentado como fragil, no reparado aqui):
  - Estimacion de orientacion (Madgwick), rotacion a marco global.
  - Integracion ZUPT de velocidad y posicion.
  - stride_length, gait_speed, mtc, peak_swing_velocity (todas dependen
    de posicion 3D integrada).
  - fatigue_analysis.py completo (dependia de Gait_Speed_ms, que ya no
    se calcula aqui).
"""

import numpy as np
from typing import Dict, Tuple, List
from pydantic import BaseModel, Field


class KinematicConfig(BaseModel):
    """Configuracion del motor de metricas temporales."""
    fs: int = Field(default=100, gt=0)


class KinematicEngine:
    """Motor de metricas TEMPORALES biomecanicas (sin reconstruccion espacial)."""

    def __init__(self, config: KinematicConfig) -> None:
        self.config = config
        self.dt = 1.0 / self.config.fs

    @staticmethod
    def _validate_xyz(data: np.ndarray, name: str) -> None:
        """Valida dimensiones matriz (N, 3), usada para gyro_xyz."""
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"{name} debe tener forma (N, 3).")

    def _get_valid_pairs(self, hs_idx: np.ndarray, to_idx: np.ndarray) -> List[Tuple[int, int]]:
        """Empareja cada Heel Strike con el Toe Off inmediatamente posterior."""
        valid_pairs = []
        for hs in hs_idx:
            future_to = to_idx[to_idx > hs]
            if len(future_to) > 0:
                valid_pairs.append((hs, future_to[0]))
        return valid_pairs

    def _get_swing_pairs(self, hs_idx: np.ndarray, to_idx: np.ndarray) -> List[Tuple[int, int]]:
        """Empareja cada Toe Off con el Heel Strike siguiente (fase de vuelo)."""
        swing_pairs = []
        for to in to_idx:
            future_hs = hs_idx[hs_idx > to]
            if len(future_hs) > 0:
                hs_next = future_hs[0]
                if ((hs_next - to) / self.config.fs) <= 2.0:
                    swing_pairs.append((to, hs_next))
        return swing_pairs

    def calcular_mid_stance(
        self, hs_idx: np.ndarray, to_idx: np.ndarray, gyro_xyz: np.ndarray
    ) -> np.ndarray:
        """
        Calcula indices de Mid-Stance (momento de menor rotacion dentro de
        cada apoyo), usado unicamente para diagnostico de huecos
        temporales -- ya NO alimenta ninguna integracion de posicion.
        """
        gyro_norm = np.linalg.norm(gyro_xyz, axis=1)
        mid_stance_idx = []

        for hs, to in self._get_valid_pairs(hs_idx, to_idx):
            ventana = gyro_norm[hs:to]
            if len(ventana) > 0:
                mid_stance_idx.append(hs + np.argmin(ventana))

        return np.array(mid_stance_idx, dtype=int)

    def diagnosticar_huecos_mid_stance(
        self, ms_idx: np.ndarray, umbral_seg: float = 2.0
    ) -> Dict[str, float]:
        """
        Cuenta y localiza huecos temporales entre Mid-Stances consecutivos
        que superan el umbral fisiologico de duracion de un paso.

        :param ms_idx: Indices de Mid-Stance.
        :param umbral_seg: Duracion (s) a partir de la cual un hueco se considera anomalo.
        :return: Diccionario con conteo, duracion total y lista de huecos (inicio_s, fin_s, duracion_s).
        """
        if len(ms_idx) < 2:
            return {"n_huecos": 0, "duracion_total_s": 0.0, "huecos": []}

        huecos = []
        for i in range(len(ms_idx) - 1):
            duracion = (ms_idx[i + 1] - ms_idx[i]) / self.config.fs
            if duracion > umbral_seg:
                huecos.append({
                    "inicio_s": float(ms_idx[i] / self.config.fs),
                    "fin_s": float(ms_idx[i + 1] / self.config.fs),
                    "duracion_s": float(duracion)
                })

        duracion_total = sum(h["duracion_s"] for h in huecos)
        return {
            "n_huecos": len(huecos),
            "duracion_total_s": duracion_total,
            "huecos": huecos
        }

    def segmentar_por_huecos(
        self, hs_idx: np.ndarray, to_idx: np.ndarray, umbral_seg: float = 2.0
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Divide los eventos HS/TO en tramos continuos de marcha, cortando
        en cada hueco temporal mayor a umbral_seg entre HS consecutivos.

        :param hs_idx: Indices Heel Strike.
        :param to_idx: Indices Toe Off.
        :param umbral_seg: Duracion (s) a partir de la cual se corta un tramo.
        :return: Lista de tuplas (hs_segmento, to_segmento), una por tramo continuo.
        """
        if len(hs_idx) < 2:
            return [(hs_idx, to_idx)]

        cortes = []
        for i in range(len(hs_idx) - 1):
            duracion = (hs_idx[i + 1] - hs_idx[i]) / self.config.fs
            if duracion > umbral_seg:
                cortes.append(i + 1)

        limites = [0] + cortes + [len(hs_idx)]
        segmentos = []

        for i in range(len(limites) - 1):
            inicio, fin = limites[i], limites[i + 1]
            hs_seg = hs_idx[inicio:fin]
            if len(hs_seg) == 0:
                continue

            mask_to = (to_idx >= hs_seg[0]) & (to_idx <= hs_seg[-1] + int(2.0 * self.config.fs))
            to_seg = to_idx[mask_to]

            segmentos.append((hs_seg, to_seg))

        return segmentos

    def compute_temporal_metrics(
        self, hs_idx: np.ndarray, to_idx: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calcula metricas temporales (las unicas que el director del TFM
        confirmo como internamente consistentes): duracion de zancada,
        tiempo de apoyo (stance) y tiempo de vuelo (swing).
        """
        stride_times = np.diff(hs_idx) / self.config.fs
        stance_times, swing_times = [], []

        for hs, to in self._get_valid_pairs(hs_idx, to_idx):
            stance_times.append((to - hs) / self.config.fs)

        for to in to_idx:
            future_hs = hs_idx[hs_idx > to]
            if len(future_hs) > 0:
                swing_times.append((future_hs[0] - to) / self.config.fs)

        return {
            "stride_times": np.array(stride_times),
            "stance_times": np.array(stance_times),
            "swing_times": np.array(swing_times)
        }

    def run_pipeline(
        self,
        gyro_xyz: np.ndarray,
        hs_idx: np.ndarray,
        to_idx: np.ndarray,
    ) -> dict:
        """
        Orquesta el calculo de metricas TEMPORALES para un tramo continuo
        (sin reconstruccion espacial). Requiere unicamente el giroscopio
        (para Mid-Stance/diagnostico de huecos) y los eventos HS/TO ya
        detectados por EventDetector.

        :param gyro_xyz: Giroscopio (N, 3), en las unidades originales
            del sensor (no requiere conversion a rad/s, solo se usa su
            norma relativa para localizar Mid-Stance).
        :param hs_idx: Indices Heel Strike.
        :param to_idx: Indices Toe Off.
        :return: Diccionario con metricas temporales y diagnostico de huecos.
        """
        self._validate_xyz(gyro_xyz, "gyro_xyz")

        ms_idx = self.calcular_mid_stance(hs_idx, to_idx, gyro_xyz)
        diagnostico_huecos = self.diagnosticar_huecos_mid_stance(ms_idx, umbral_seg=2.0)
        temporal_metrics = self.compute_temporal_metrics(hs_idx, to_idx)

        return {
            "mid_stance_idx": ms_idx,
            "huecos_diagnostico": diagnostico_huecos,
            **temporal_metrics
        }

    def run_pipeline_segmentado(
        self,
        gyro_xyz: np.ndarray,
        hs_idx: np.ndarray,
        to_idx: np.ndarray,
        umbral_hueco_seg: float = 2.0,
        min_pasos_segmento: int = 4
    ) -> dict:
        """
        Ejecuta el calculo de metricas temporales por tramos continuos de
        marcha, cortando en los huecos temporales entre HS consecutivos
        (mismo criterio defensivo del A06 original, mantenido porque el
        director del TFM lo evaluo como una "buena decision defensiva").

        :param gyro_xyz: Giroscopio (N, 3).
        :param hs_idx: Indices Heel Strike globales.
        :param to_idx: Indices Toe Off globales.
        :param umbral_hueco_seg: Duracion a partir de la cual se corta un tramo.
        :param min_pasos_segmento: Minimo de pasos para procesar un segmento
            (segmentos mas cortos se descartan por no ser representativos).
        :return: Diccionario con metricas temporales concatenadas y metadata
            de segmento (segmento_id) por cada zancada valida.
        """
        segmentos = self.segmentar_por_huecos(hs_idx, to_idx, umbral_seg=umbral_hueco_seg)

        acumulado: Dict[str, list] = {
            "hs_times_s": [], "stride_times": [], "stance_times": [],
            "swing_times": [], "segmento_id": []
        }
        n_descartados = 0

        for seg_id, (hs_seg, to_seg) in enumerate(segmentos):
            if len(hs_seg) < min_pasos_segmento:
                n_descartados += 1
                continue

            resultado_seg = self.run_pipeline(
                gyro_xyz=gyro_xyz,
                hs_idx=hs_seg,
                to_idx=to_seg,
            )

            n_pasos_seg = len(resultado_seg["stride_times"])
            acumulado["hs_times_s"].extend((hs_seg[:-1] / self.config.fs).tolist())
            acumulado["stride_times"].extend(resultado_seg["stride_times"])
            acumulado["stance_times"].extend(resultado_seg["stance_times"])
            acumulado["swing_times"].extend(resultado_seg["swing_times"])
            acumulado["segmento_id"].extend([seg_id] * n_pasos_seg)

        print(
            f"[SEGMENTACION] {len(segmentos)} tramos detectados, "
            f"{len(segmentos) - n_descartados} procesados, "
            f"{n_descartados} descartados (< {min_pasos_segmento} pasos)."
        )

        return {
            "hs_times_s": np.array(acumulado["hs_times_s"]),
            "stride_times": np.array(acumulado["stride_times"]),
            "stance_times": np.array(acumulado["stance_times"]),
            "swing_times": np.array(acumulado["swing_times"]),
            "segmento_id": np.array(acumulado["segmento_id"]),
            "n_segmentos_totales": len(segmentos),
            "n_segmentos_procesados": len(segmentos) - n_descartados
        }