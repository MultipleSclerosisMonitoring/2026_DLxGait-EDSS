# -*- coding: utf-8 -*-
"""
Orquestador biomecanico bilateral.

Conecta secuencialmente los modulos EventDetector, KinematicEngine
y FatigueAnalyzer para calcular metricas de marcha bilaterales,
asimetria interpodal y diagnostico de fatiga motora.
"""

from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pydantic import BaseModel, Field

# ASEGURAR QUE EL PROPIO DIRECTORIO ESTE EN EL PATH
# (necesario cuando este archivo se importa como submodulo desde otro
# script, ej. tools/diagnosticar_eventos.py, en vez de ejecutarse
# directamente como script principal)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from event_detector import EventDetector, EventDetectorConfig
from kinematic_engine import KinematicEngine, KinematicConfig
from fatigue_analysis import FatigueAnalyzer, FatigueConfig

# RUTA RELATIVA REAL
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from A01_EXTRACCION_DATOS.extract_data_plus import cInfluxDB
except ImportError as error:
    raise ImportError(f"ERROR DE IMPORTACION: {error}") from error


class PipelineConfig(BaseModel):
    """Configuracion global del orquestador."""
    paciente: str = Field(..., description="ID paciente")
    inicio: datetime = Field(..., description="Fecha inicio")
    fin: datetime = Field(..., description="Fecha fin")
    config_yaml_path: Path = Field(..., description="Ruta config InfluxDB")
    fs: int = Field(default=100, gt=0)
    output_dir: Path = Field(
        default=Path(
            r"C:\Users\jairi\OneDrive\Escritorio\TFM_CLONADO_FINALFINAL"
            r"\A06_ANALISIS_CINEMATICO\RESULTADOS_BIOMECANICO"
        )
    )
    fatigue_target: str = Field(default="Gait_Speed_ms")
    max_time_diff_s: float = Field(default=0.20, gt=0.0)
    auto_calibrar_umbral: bool = Field(default=True)
    umbral_barrido_min: float = Field(default=0.15, ge=0.0, le=1.0)
    umbral_barrido_max: float = Field(default=0.70, ge=0.0, le=1.0)
    umbral_barrido_paso: float = Field(default=0.05, gt=0.0)
    th_right_manual: Optional[float] = Field(default=None)
    th_left_manual: Optional[float] = Field(default=None)
    max_stride_m: float = Field(default=1.9, gt=0.0)
    # TOLERANCIA MAXIMA ASIMETRIA CONTEO PASOS
    max_diff_pasos_pct: float = Field(default=10.0, gt=0.0)
    # COTA MAXIMA RANGO ESPACIAL PLAUSIBLE (m)
    max_rango_espacial_m: float = Field(default=50.0, gt=0.0)


class BilateralPipeline:
    """Orquestador de analisis biomecanico."""

    def __init__(self, config: PipelineConfig) -> None:
        """Inicializa el orquestador y los motores de calculo."""
        self.config = config

        # CONFIGURACION BASE DETECTOR (umbral real se calibra por pie)
        config_eventos = EventDetectorConfig(fs=self.config.fs)
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
    def _extraer_matrices(df_pie: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extrae matrices IMU, incluyendo magnetometro si esta disponible."""
        esquemas_acc = [["Acc_X", "Acc_Y", "Acc_Z"], ["Ax", "Ay", "Az"]]
        esquemas_gyro = [["Gyro_X", "Gyro_Y", "Gyro_Z"], ["Gx", "Gy", "Gz"]]
        esquemas_mag = [["Mag_X", "Mag_Y", "Mag_Z"], ["Mx", "My", "Mz"]]

        columnas_acc = next((e for e in esquemas_acc if all(c in df_pie.columns for c in e)), None)
        columnas_gyro = next((e for e in esquemas_gyro if all(c in df_pie.columns for c in e)), None)
        columnas_mag = next((e for e in esquemas_mag if all(c in df_pie.columns for c in e)), None)

        if not columnas_acc or not columnas_gyro:
            raise KeyError("Columnas IMU no reconocidas.")

        matrices = {
            "acc_xyz": df_pie[columnas_acc].astype(float).values,
            "gyro_xyz": df_pie[columnas_gyro].astype(float).values,
            "mag_xyz": df_pie[columnas_mag].astype(float).values if columnas_mag else None
        }
        return matrices

    def _auto_calibrar_umbral(self, presion: np.ndarray, pie: str) -> float:
        """Calibra fraccion de umbral optima."""
        fracciones = np.arange(
            self.config.umbral_barrido_min,
            self.config.umbral_barrido_max + self.config.umbral_barrido_paso,
            self.config.umbral_barrido_paso
        )

        conteos, stances = [], []
        for fraccion in fracciones:
            hs_idx, _, stance_pct = self.detector.detect_from_pressure(
                presion, threshold_fraction_override=float(fraccion)
            )
            conteos.append(len(hs_idx))
            stances.append(stance_pct)

        stances = np.array(stances)
        indices_validos = np.where((stances >= 45.0) & (stances <= 70.0))[0]

        if len(indices_validos) == 0:
            idx = int(np.argmin(np.abs(stances - 57.5)))
            print(f"[CALIBRACION {pie}] Sin stance en rango 45-70%; usando umbral por cercania: {fracciones[idx]:.2f}")
            return float(fracciones[idx])

        conteos_validos = np.array(conteos)[indices_validos]
        mejor_inicio, mejor_largo, inicio_actual = 0, 1, 0

        for i in range(1, len(conteos_validos)):
            if abs(int(conteos_validos[i]) - int(conteos_validos[inicio_actual])) <= 1:
                largo_actual = i - inicio_actual + 1
                if largo_actual > mejor_largo:
                    mejor_largo = largo_actual
                    mejor_inicio = inicio_actual
            else:
                inicio_actual = i

        idx_rel = mejor_inicio + (mejor_largo - 1) // 2
        umbral_final = float(fracciones[int(indices_validos[idx_rel])])
        print(f"[CALIBRACION {pie}] Umbral seleccionado: {umbral_final:.2f} (stance {stances[indices_validos[idx_rel]]:.1f}%)")
        return umbral_final

    def _procesar_pie(self, pie: str) -> Dict[str, np.ndarray]:
        """Procesa pie mediante deteccion y cinematica, segmentando en tramos continuos."""
        df_pie = self._extraer_pie(pie)
        presion = self._preparar_presion(df_pie)

        # SELECCIONAR UMBRAL: MANUAL > AUTO-CALIBRADO > DEFECTO
        umbral_manual = self.config.th_right_manual if pie == "Right" else self.config.th_left_manual

        if umbral_manual is not None:
            umbral = umbral_manual
        elif self.config.auto_calibrar_umbral:
            umbral = self._auto_calibrar_umbral(presion, pie)
        else:
            umbral = self.detector.config.threshold_fraction

        hs_idx, to_idx, stance_pct = self.detector.detect_from_pressure(
            presion,
            threshold_fraction_override=umbral
        )

        # LOGICA DE SINCRONIZACION: HS SIEMPRE DEBE IR ANTES QUE TO
        if len(to_idx) > 0 and len(hs_idx) > 0 and to_idx[0] < hs_idx[0]:
            to_idx = to_idx[1:]

        min_len = min(len(hs_idx), len(to_idx))
        hs_idx = hs_idx[:min_len]
        to_idx = to_idx[:min_len]

        print(f"\n[AJUSTE {pie}] -> Umbral: {umbral:.2f} | Stance: {stance_pct:.1f}% | Alineado a: {len(hs_idx)} pares HS/TO.")

        matrices = self._extraer_matrices(df_pie)
        print(f"[{pie}] Magnetometro usado: {'SI' if matrices['mag_xyz'] is not None else 'NO'}")

        # PROCESAR POR TRAMOS CONTINUOS (evita arrastrar deriva a traves de huecos)
        resultado = self.kinematic_engine.run_pipeline_segmentado(
            acc_xyz=matrices["acc_xyz"],
            gyro_xyz=matrices["gyro_xyz"],
            hs_idx=hs_idx,
            to_idx=to_idx,
            mag_xyz=matrices["mag_xyz"],
            umbral_hueco_seg=2.0,
            min_pasos_segmento=4
        )


        return resultado

    def _validar_coherencia_pasos(self, pasos_der: int, pasos_izq: int) -> None:
        """Advierte si el conteo de pasos entre pies difiere demasiado."""
        mayor = max(pasos_der, pasos_izq)
        if mayor == 0:
            return
        diff_pct = 100.0 * abs(pasos_der - pasos_izq) / mayor
        if diff_pct > self.config.max_diff_pasos_pct:
            print(
                f"\n[ADVERTENCIA] Diferencia de pasos entre pies: {diff_pct:.1f}% "
                f"(DER={pasos_der}, IZQ={pasos_izq}). Umbral tolerado: {self.config.max_diff_pasos_pct:.1f}%.\n"
                "Las metricas de asimetria bilateral pueden no ser comparables. "
                "Revisar calibracion de umbral o calidad de senal.\n"
            )

    def _validar_deriva_espacial(self, position_segmentos: List[np.ndarray], pie: str) -> None:
        """Advierte si el rango espacial de algun segmento excede una cota plausible."""
        if not position_segmentos:
            return
        for idx, position in enumerate(position_segmentos):
            if position is None or len(position) == 0:
                continue
            rango = position[:, :2].max(axis=0) - position[:, :2].min(axis=0)
            if np.any(rango > self.config.max_rango_espacial_m):
                print(
                    f"\n[ADVERTENCIA] Pie {pie} (segmento {idx}): rango espacial XY = "
                    f"{rango[0]:.1f} x {rango[1]:.1f} m supera la cota plausible de "
                    f"{self.config.max_rango_espacial_m:.1f} m. "
                    "Probable deriva de integracion; resultados de posicion no confiables.\n"
                )

    def _emparejar_zancadas(self, right: Dict[str, np.ndarray], left: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        """Empareja zancadas bilaterales."""
        pares = []
        right_times, left_times = right["hs_times_s"], left["hs_times_s"]

        for i, t_right in enumerate(right_times):
            diffs = np.abs(left_times - t_right)
            if len(diffs) == 0:
                continue

            j = int(np.argmin(diffs))
            if diffs[j] > self.config.max_time_diff_s:
                continue
            if i >= len(right["gait_speed"]) or j >= len(left["gait_speed"]):
                continue

            s_r = float(right["stride_lengths"][i])
            s_l = float(left["stride_lengths"][j])

            if s_r > self.config.max_stride_m or s_l > self.config.max_stride_m:
                continue

            pares.append({
                "t_right_s": float(t_right),
                "t_left_s": float(left_times[j]),
                "gait_speed_right": float(right["gait_speed"][i]),
                "gait_speed_left": float(left["gait_speed"][j]),
                "stride_length_right": s_r,
                "stride_length_left": s_l,
                "mtc_right": float(right["mtc"][i]) if i < len(right["mtc"]) else np.nan,
                "mtc_left": float(left["mtc"][j]) if j < len(left["mtc"]) else np.nan,
                "segmento_right": int(right["segmento_id"][i]) if i < len(right.get("segmento_id", [])) else -1,
                "segmento_left": int(left["segmento_id"][j]) if j < len(left.get("segmento_id", [])) else -1,
            })
        return pares

    def _calcular_asimetria(self, pares: List[Dict[str, float]]) -> pd.DataFrame:
        """Calcula metricas de asimetria interpodal."""
        filas = []
        for par in pares:
            fila = dict(par)
            denom_vel = (par["gait_speed_right"] + par["gait_speed_left"]) / 2.0
            denom_len = (par["stride_length_right"] + par["stride_length_left"]) / 2.0
            denom_mtc = (par["mtc_right"] + par["mtc_left"]) / 2.0

            fila["Asimetria_Velocidad_pct"] = 100.0 * (par["gait_speed_right"] - par["gait_speed_left"]) / denom_vel if denom_vel else 0.0
            fila["Asimetria_Zancada_pct"] = 100.0 * (par["stride_length_right"] - par["stride_length_left"]) / denom_len if denom_len else 0.0
            fila["Asimetria_MTC_pct"] = 100.0 * (par["mtc_right"] - par["mtc_left"]) / denom_mtc if denom_mtc else 0.0
            fila["Gait_Speed_ms"] = denom_vel
            filas.append(fila)
        return pd.DataFrame(filas)

    @staticmethod
    def _reparar_saltos_espurios(position: np.ndarray, max_delta_m: float = 0.08) -> np.ndarray:
        """Interpola picos aislados de ruido UWB/GPS."""
        pos = position.copy()
        if len(pos) < 5:
            return pos

        dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        saltos = np.zeros(len(pos), dtype=bool)

        for i in range(1, len(dist) - 1):
            if dist[i] > max_delta_m and dist[i - 1] < max_delta_m * 0.5 and dist[i + 1] < max_delta_m * 0.5:
                saltos[i + 1] = True

        for eje in range(3):
            serie = pos[:, eje].copy()
            serie[saltos] = np.nan
            buenos = ~np.isnan(serie)
            if buenos.sum() > 1:
                serie[np.isnan(serie)] = np.interp(np.flatnonzero(np.isnan(serie)), np.flatnonzero(buenos), serie[buenos])
            pos[:, eje] = serie
        return pos

    def _graficar_trayectoria_3d(
        self, position_segmentos_right: List[np.ndarray], position_segmentos_left: List[np.ndarray]
    ) -> Path:
        """
        VISUALIZACION POST PROCESADO:
        Grafica cada tramo continuo de marcha por separado (sin conectar
        segmentos discontinuos entre si) y aplica Savitzky-Golay dentro de
        cada tramo puramente para un render limpio; no altera las metricas.
        """
        fig = plt.figure(figsize=(10, 8))
        eje = fig.add_subplot(111, projection="3d")
        v, o, u = 51, 3, 0.08

        etiqueta_r, etiqueta_l = "Pie Derecho", "Pie Izquierdo"
        for pos_r in position_segmentos_right or []:
            if pos_r is not None and len(pos_r) > v:
                pos_r = self._reparar_saltos_espurios(pos_r, max_delta_m=u)
                eje.plot(
                    savgol_filter(pos_r[:, 0], v, o), savgol_filter(pos_r[:, 1], v, o), savgol_filter(pos_r[:, 2], v, o),
                    color="blue", label=etiqueta_r
                )
                etiqueta_r = None  # SOLO UNA ENTRADA EN LEYENDA

        for pos_l in position_segmentos_left or []:
            if pos_l is not None and len(pos_l) > v:
                pos_l = self._reparar_saltos_espurios(pos_l, max_delta_m=u)
                eje.plot(
                    savgol_filter(pos_l[:, 0], v, o), savgol_filter(pos_l[:, 1], v, o), savgol_filter(pos_l[:, 2], v, o),
                    color="red", label=etiqueta_l
                )
                etiqueta_l = None

        eje.set_xlabel("X (m)"), eje.set_ylabel("Y (m)"), eje.set_zlabel("Z (m)")
        eje.set_title(f"Trayectoria 3D por tramos continuos - {self.config.paciente}")
        eje.legend()

        out = self.config.output_dir / f"{self.config.paciente}_trayectoria_3d.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def _graficar_fatiga(self, csv_path: Path) -> Path:
        """Grafica degradacion motora."""
        df = pd.read_csv(csv_path)
        y = df[self.config.fatigue_target].dropna().values
        x = np.arange(len(y))
        p, i = np.polyfit(x, y, 1)

        fig = plt.figure()
        plt.scatter(x, y)
        plt.plot(x, i + p * x)
        plt.title(f"Fatiga: {self.config.fatigue_target}")
        plt.xlabel("Pasos")
        plt.ylabel(self.config.fatigue_target)

        out = self.config.output_dir / f"{self.config.paciente}_fatiga.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def ejecutar(self) -> Dict[str, float]:
        """Ejecuta orquestacion global."""
        # PROCESAR CADA PIE
        right, left = self._procesar_pie("Right"), self._procesar_pie("Left")

        # AUDITAR PASOS VALIDOS
        pasos_der = len(right.get("stride_lengths", []))
        pasos_izq = len(left.get("stride_lengths", []))

        print("\n" + "*" * 45)
        print("AUDITORIA DE PASOS VALIDOS (DER VS IZQ)")
        print(f"Total Pasos DER: {pasos_der}")
        print(f"Total Pasos IZQ: {pasos_izq}")
        print("*" * 45 + "\n")

        # VALIDAR COHERENCIA ENTRE PIES
        self._validar_coherencia_pasos(pasos_der, pasos_izq)

        # VALIDAR DERIVA ESPACIAL
        self._validar_deriva_espacial(right.get("position_segmentos", []), "Derecho")
        self._validar_deriva_espacial(left.get("position_segmentos", []), "Izquierdo")

        # EMPAREJAR ZANCADAS
        pares = self._emparejar_zancadas(right, left)
        if not pares:
            raise ValueError("SIN ZANCADAS EMPAREJADAS VALIDAS")

        # CALCULAR METRICAS CLINICAS
        df_metricas = self._calcular_asimetria(pares)

        # GRAFICAR TRAYECTORIA 3D
        self._graficar_trayectoria_3d(right.get("position_segmentos", []), left.get("position_segmentos", []))

        # GUARDAR RESULTADOS CSV
        csv_path = self.config.output_dir / f"{self.config.paciente}_metricas.csv"
        df_metricas.to_csv(csv_path, index=False)

        # EJECUTAR ANALISIS FATIGA
        fatigue_cfg = FatigueConfig(
            file_path=str(csv_path),
            target_feature=self.config.fatigue_target
        )
        analyzer = FatigueAnalyzer(fatigue_cfg)
        biomarcadores = analyzer.run_analysis()
        self._graficar_fatiga(csv_path)

        # OBTENER VELOCIDAD VUELO
        vel_pico_der = right.get("peak_swing_velocity", [])
        vel_pico_izq = left.get("peak_swing_velocity", [])

        # CALCULAR PROMEDIOS VELOCIDAD
        prom_vel_der = np.mean(vel_pico_der) if len(vel_pico_der) > 0 else 0.0
        prom_vel_izq = np.mean(vel_pico_izq) if len(vel_pico_izq) > 0 else 0.0

        # OBTENER RANGOS ESPACIALES POR SEGMENTO (max entre tramos individuales)
        def _rango_maximo(position_segmentos: List[np.ndarray]) -> np.ndarray:
            """Calcula el rango [min,max] por eje tomando el tramo de mayor extension."""
            rangos = []
            for pos in position_segmentos or []:
                if pos is not None and len(pos) > 0:
                    rangos.append((pos[:, 0].min(), pos[:, 0].max(),
                                    pos[:, 1].min(), pos[:, 1].max(),
                                    pos[:, 2].min(), pos[:, 2].max()))
            if not rangos:
                return np.zeros(6)
            rangos = np.array(rangos)
            return np.array([
                rangos[:, 0].min(), rangos[:, 1].max(),
                rangos[:, 2].min(), rangos[:, 3].max(),
                rangos[:, 4].min(), rangos[:, 5].max()
            ])

        r_der = _rango_maximo(right.get("position_segmentos", []))
        r_izq = _rango_maximo(left.get("position_segmentos", []))

        # IMPRIMIR RESUMEN CONSOLA
        print("\n" + "=" * 45)
        print("RESUMEN BIOMECANICO GLOBAL")
        print("=" * 45)

        print("METRICAS CLINICAS PROMEDIO:")
        print(f"Velocidad Media: {df_metricas['Gait_Speed_ms'].mean():.3f} m/s")
        print(f"Zancada Der:     {df_metricas['stride_length_right'].mean():.3f} m")
        print(f"Zancada Izq:     {df_metricas['stride_length_left'].mean():.3f} m")
        print(f"Vel Vuelo Der:   {prom_vel_der:.3f} m/s")
        print(f"Vel Vuelo Izq:   {prom_vel_izq:.3f} m/s")
        print(f"Asimetria MTC:   {df_metricas['Asimetria_MTC_pct'].mean():.1f} %")
        print(f"Pendiente Fatiga:{biomarcadores['slope']:.6f}")

        print("\nRANGOS ESPACIALES POR TRAMO (max entre segmentos continuos):")
        print(f"Pie DER -> X: [{r_der[0]:.1f}, {r_der[1]:.1f}] Y: [{r_der[2]:.1f}, {r_der[3]:.1f}] Z: [{r_der[4]:.1f}, {r_der[5]:.1f}]")
        print(f"Pie IZQ -> X: [{r_izq[0]:.1f}, {r_izq[1]:.1f}] Y: [{r_izq[2]:.1f}, {r_izq[3]:.1f}] Z: [{r_izq[4]:.1f}, {r_izq[5]:.1f}]")
        print("=" * 45 + "\n")

        return biomarcadores


def parse_args() -> argparse.Namespace:
    """Parsea CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True)
    parser.add_argument("--fin", type=str, required=True)
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--config-yaml", type=str, default=None)
    parser.add_argument("--fatigue-target", type=str, default="Gait_Speed_ms")
    parser.add_argument("--max-time-diff", type=float, default=0.20)
    parser.add_argument("--th-right", type=float, default=None)
    parser.add_argument("--th-left", type=float, default=None)
    parser.add_argument("--sin-auto-calibrar", action="store_true")
    parser.add_argument("--max-stride", type=float, default=1.9)
    return parser.parse_args()


def construir_config(args: argparse.Namespace) -> PipelineConfig:
    """Valida argumentos y emite config."""
    cfg_path = Path(args.config_yaml) if args.config_yaml else PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml"
    return PipelineConfig(
        paciente=args.paciente,
        inicio=datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S"),
        fin=datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S"),
        config_yaml_path=cfg_path,
        fs=args.fs,
        fatigue_target=args.fatigue_target,
        max_time_diff_s=args.max_time_diff,
        th_right_manual=args.th_right,
        th_left_manual=args.th_left,
        auto_calibrar_umbral=not args.sin_auto_calibrar,
        max_stride_m=args.max_stride
    )


def main() -> None:
    """Punto de entrada."""
    args = parse_args()
    try:
        config = construir_config(args)
        pipeline = BilateralPipeline(config)
        biomarcadores = pipeline.ejecutar()
    except Exception as error:
        print(f"ERROR PIPELINE: {error}")
        sys.exit(1)

    print(f"Pendiente fatiga: {biomarcadores['slope']:.6f}")


if __name__ == "__main__":
    main()