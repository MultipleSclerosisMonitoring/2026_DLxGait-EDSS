# -*- coding: utf-8 -*-
"""
Orquestador biomecanico bilateral -- VERSION TEMPORAL (recortes de A06).

Conecta EventDetector y KinematicEngine (version recortada, sin
reconstruccion espacial) para calcular metricas de marcha TEMPORALES
bilaterales: duracion de zancada, tiempo de apoyo, tiempo de vuelo, y
asimetria interpodal en esas metricas temporales.

NO calcula trayectoria, velocidad de marcha, longitud de zancada, MTC ni
fatiga (dependiente de velocidad) -- esas metricas se descartaron por
evaluacion del director del TFM (fragilidad de la reconstruccion
espacial: escala de aceleracion heuristica, integracion doble propensa
a deriva). Ver kinematic_engine_temporal.py para el detalle de que se
elimino y por que.
"""

from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from event_detector import EventDetector, EventDetectorConfig
from kinematic_engine_temporal import KinematicEngine, KinematicConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from A01_EXTRACCION_DATOS.extract_data_plus import cInfluxDB
except ImportError as error:
    raise ImportError(f"ERROR DE IMPORTACION: {error}") from error


class PipelineConfig(BaseModel):
    """Configuracion global del orquestador temporal."""
    paciente: str = Field(..., description="ID paciente")
    inicio: datetime = Field(..., description="Fecha inicio")
    fin: datetime = Field(..., description="Fecha fin")
    config_yaml_path: Path = Field(..., description="Ruta config InfluxDB")
    fs: int = Field(default=100, gt=0)
    output_dir: Optional[Path] = Field(
        default=None,
        description="Carpeta de salida. Si no se especifica, se calcula "
                     "relativa a la ubicacion del propio script (PROJECT_ROOT/"
                     "A06_ANALISIS_CINEMATICO/RESULTADOS_TEMPORALES), no una "
                     "ruta absoluta fija -- para que el script sea portable "
                     "entre distintos clones del repositorio."
    )
    max_time_diff_s: float = Field(default=0.20, gt=0.0)
    umbral_hueco_seg: float = Field(default=2.0, gt=0.0)


class BilateralPipelineTemporal:
    """Orquestador de metricas biomecanicas TEMPORALES (sin reconstruccion espacial)."""

    def __init__(self, config: PipelineConfig) -> None:
        """Inicializa el orquestador y los motores de calculo."""
        self.config = config

        # RESOLVER output_dir DE FORMA PORTABLE: si no se especifico
        # explicitamente (None), se calcula relativo a PROJECT_ROOT (la
        # ubicacion real del propio script en el clon actual), en vez de
        # depender de una ruta absoluta fija que rompe en otros clones.
        if self.config.output_dir is None:
            self.config.output_dir = PROJECT_ROOT / "A06_ANALISIS_CINEMATICO" / "RESULTADOS_TEMPORALES"

        config_eventos = EventDetectorConfig(
            fs=self.config.fs,
            threshold_fraction=0.15,
            gyro_prominence=15.0
        )
        self.detector = EventDetector(config_eventos)
        self.kinematic_engine = KinematicEngine(KinematicConfig(fs=self.config.fs))
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _extraer_pie(self, pie: str) -> pd.DataFrame:
        """Extrae telemetria desde InfluxDB."""
        extractor = cInfluxDB(config_path=str(self.config.config_yaml_path))
        try:
            df_pie = extractor.query_data(
                from_date=self.config.inicio,
                to_date=self.config.fin,
                qtok=self.config.paciente,
                pie=pie
            )
        finally:
            extractor.close()

        if df_pie is None or df_pie.empty:
            raise ValueError(f"DATOS NO ENCONTRADOS PARA PIE {pie}")
        return df_pie

    @staticmethod
    def _preparar_presion(df_pie: pd.DataFrame) -> np.ndarray:
        """Normaliza senal de presion."""
        presion = df_pie[["S0", "S1", "S2"]].astype(float).sum(axis=1).values
        baseline = np.percentile(presion, 5)
        presion = presion - baseline
        presion[presion < 0] = 0
        return presion

    @staticmethod
    def _extraer_gyro(df_pie: pd.DataFrame) -> np.ndarray:
        """Extrae matriz de giroscopio (unica senal que requiere el motor temporal)."""
        esquemas_gyro = [["Gyro_X", "Gyro_Y", "Gyro_Z"], ["Gx", "Gy", "Gz"]]
        columnas_gyro = next((e for e in esquemas_gyro if all(c in df_pie.columns for c in e)), None)

        if not columnas_gyro:
            raise KeyError("Columnas de giroscopio no reconocidas.")

        return df_pie[columnas_gyro].astype(float).values

    def _procesar_pie(self, pie: str) -> Dict[str, np.ndarray]:
        """Procesa un pie: deteccion de eventos + metricas temporales segmentadas."""
        df_pie = self._extraer_pie(pie)
        presion = self._preparar_presion(df_pie)

        umbral = 0.15
        hs_idx, to_idx, _ = self.detector.detect_from_pressure(
            presion, threshold_fraction_override=umbral
        )

        if len(to_idx) > 0 and len(hs_idx) > 0 and to_idx[0] < hs_idx[0]:
            to_idx = to_idx[1:]

        min_len = min(len(hs_idx), len(to_idx))
        hs_idx = hs_idx[:min_len]
        to_idx = to_idx[:min_len]

        print(f"\n[AJUSTE {pie}] -> Alineado a: {len(hs_idx)} pares HS/TO.")

        gyro_xyz = self._extraer_gyro(df_pie)

        resultado = self.kinematic_engine.run_pipeline_segmentado(
            gyro_xyz=gyro_xyz,
            hs_idx=hs_idx,
            to_idx=to_idx,
            umbral_hueco_seg=self.config.umbral_hueco_seg
        )

        return resultado

    def _emparejar_zancadas(self, right: Dict[str, np.ndarray], left: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        """
        Empareja zancadas bilaterales por proximidad temporal, SIN
        reutilizacion (cada HS izquierdo se marca como usado una vez
        emparejado) -- corrige el emparejamiento voraz no biyectivo que
        el director del TFM senalo como fragil en la version original.
        """
        pares = []
        right_times = right["hs_times_s"]
        left_times = left["hs_times_s"]
        usados_left = np.zeros(len(left_times), dtype=bool)

        for i, t_right in enumerate(right_times):
            diffs = np.abs(left_times - t_right)
            diffs_disponibles = np.where(usados_left, np.inf, diffs)

            if len(diffs_disponibles) == 0 or np.all(np.isinf(diffs_disponibles)):
                continue

            j = int(np.argmin(diffs_disponibles))
            if diffs_disponibles[j] > self.config.max_time_diff_s:
                continue

            if i >= len(right["stride_times"]) or j >= len(left["stride_times"]):
                continue

            usados_left[j] = True

            pares.append({
                "t_right_s": float(t_right),
                "t_left_s": float(left_times[j]),
                "stride_time_right": float(right["stride_times"][i]),
                "stride_time_left": float(left["stride_times"][j]),
                "stance_time_right": float(right["stance_times"][i]) if i < len(right["stance_times"]) else np.nan,
                "stance_time_left": float(left["stance_times"][j]) if j < len(left["stance_times"]) else np.nan,
                "swing_time_right": float(right["swing_times"][i]) if i < len(right["swing_times"]) else np.nan,
                "swing_time_left": float(left["swing_times"][j]) if j < len(left["swing_times"]) else np.nan,
            })
        return pares

    def _calcular_asimetria(self, pares: List[Dict[str, float]]) -> pd.DataFrame:
        """Calcula metricas de asimetria interpodal TEMPORAL."""
        filas = []
        for par in pares:
            fila = dict(par)
            for metrica in ["stride_time", "stance_time", "swing_time"]:
                v_der = par[f"{metrica}_right"]
                v_izq = par[f"{metrica}_left"]
                denom = (v_der + v_izq) / 2.0
                fila[f"Asimetria_{metrica}_pct"] = (
                    100.0 * (v_der - v_izq) / denom if denom else np.nan
                )
            filas.append(fila)
        return pd.DataFrame(filas)

    def ejecutar(self) -> Dict[str, float]:
        """Ejecuta orquestacion global TEMPORAL."""
        right, left = self._procesar_pie("Right"), self._procesar_pie("Left")

        pasos_der = len(right.get("stride_times", []))
        pasos_izq = len(left.get("stride_times", []))

        print("\n" + "*" * 45)
        print("AUDITORIA DE PASOS VALIDOS (DER VS IZQ)")
        print(f"Total Pasos DER: {pasos_der}")
        print(f"Total Pasos IZQ: {pasos_izq}")
        print("*" * 45 + "\n")

        pares = self._emparejar_zancadas(right, left)
        if not pares:
            raise ValueError("SIN ZANCADAS EMPAREJADAS VALIDAS")

        df_metricas = self._calcular_asimetria(pares)

        csv_path = self.config.output_dir / f"{self.config.paciente}_metricas_temporales.csv"
        df_metricas.to_csv(csv_path, index=False)

        print("\n" + "=" * 45)
        print("RESUMEN BIOMECANICO TEMPORAL")
        print("=" * 45)
        print(f"Duracion zancada media: {df_metricas['stride_time_right'].mean():.3f}s (der) / "
              f"{df_metricas['stride_time_left'].mean():.3f}s (izq)")
        print(f"Tiempo apoyo medio:     {df_metricas['stance_time_right'].mean():.3f}s (der) / "
              f"{df_metricas['stance_time_left'].mean():.3f}s (izq)")
        print(f"Tiempo vuelo medio:     {df_metricas['swing_time_right'].mean():.3f}s (der) / "
              f"{df_metricas['swing_time_left'].mean():.3f}s (izq)")
        print(f"Asimetria zancada:      {df_metricas['Asimetria_stride_time_pct'].mean():.1f}%")
        print("=" * 45 + "\n")
        print("NOTA: metricas espaciales (velocidad, longitud de zancada, MTC, trayectoria)")
        print("y analisis de fatiga NO se calculan en esta version -- descartadas por")
        print("fragilidad de la reconstruccion espacial (ver documentacion del proyecto).")

        return {
            "csv_path": str(csv_path),
            "n_zancadas_emparejadas": len(pares),
            "asimetria_stride_time_pct_media": float(df_metricas["Asimetria_stride_time_pct"].mean()),
        }


def parse_args() -> argparse.Namespace:
    """Parsea CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True)
    parser.add_argument("--fin", type=str, required=True)
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--config-yaml", type=str, default=None)
    parser.add_argument("--max-time-diff", type=float, default=0.20)
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Carpeta de salida. Si se omite, se usa PROJECT_ROOT/A06_ANALISIS_CINEMATICO/"
             "RESULTADOS_TEMPORALES (relativa al clon actual, no una ruta fija)."
    )
    return parser.parse_args()


def construir_config(args: argparse.Namespace) -> PipelineConfig:
    """Valida argumentos y emite config."""
    cfg_path = Path(args.config_yaml) if args.config_yaml else PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml"
    out_dir = Path(args.output_dir) if args.output_dir else None
    return PipelineConfig(
        paciente=args.paciente,
        inicio=datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S"),
        fin=datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S"),
        config_yaml_path=cfg_path,
        fs=args.fs,
        max_time_diff_s=args.max_time_diff,
        output_dir=out_dir,
    )


def main() -> None:
    """Punto de entrada."""
    args = parse_args()
    try:
        config = construir_config(args)
        pipeline = BilateralPipelineTemporal(config)
        resultado = pipeline.ejecutar()
    except Exception as error:
        print(f"ERROR PIPELINE: {error}")
        sys.exit(1)

    print(f"Resultado: {resultado}")


if __name__ == "__main__":
    main()
