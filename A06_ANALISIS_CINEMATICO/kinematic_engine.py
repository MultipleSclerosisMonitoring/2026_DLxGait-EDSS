# -*- coding: utf-8 -*-
"""
Motor cinematico tridimensional para reconstruccion biomecanica.
Implementa compensacion gravitacional, fusion sensorial y correccion
de deriva mediante actualizaciones de velocidad cero (ZUPT).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from pydantic import BaseModel, Field
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation as R


class KinematicConfig(BaseModel):
    """Configuracion del motor cinematico."""
    fs: int = Field(default=100, gt=0)
    cutoff_hz: float = Field(default=0.05, gt=0.0)
    filter_order: int = Field(default=4, gt=0)
    gravity: float = Field(default=9.81, gt=0.0)
    # AJUSTE OPTIMO PARA SENSOR IMU (6-DOF) SIN MAGNETOMETRO
    madgwick_beta: float = Field(default=0.033, gt=0.0)
    mtc_window_fraction: float = Field(default=0.4, gt=0.0, le=1.0)
    acc_input_in_g: bool = Field(default=True)
    calibrar_bias_estatico: bool = Field(default=True)
    muestras_inicializacion: int = Field(default=20, gt=0)


class KinematicEngine:
    """Motor biomecanico para reconstruccion espacial."""

    def __init__(self, config: KinematicConfig) -> None:
        self.config = config
        self.dt = 1.0 / self.config.fs

    def _get_safe_stance_window(
        self, sig: np.ndarray, hs_idx: np.ndarray, to_idx: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extrae el 50% central del primer apoyo para evitar picos de impacto (reposo real)."""
        if len(hs_idx) == 0 or len(to_idx) == 0:
            return None

        primer_hs = hs_idx[0]
        futuros_to = to_idx[to_idx > primer_hs]

        if len(futuros_to) == 0:
            return None

        primer_to = futuros_to[0]
        largo = primer_to - primer_hs

        if largo < 6:
            return sig[primer_hs:primer_to]

        inicio = primer_hs + int(largo * 0.25)
        fin = primer_hs + int(largo * 0.75)
        return sig[inicio:fin]

    def _convertir_aceleracion(
        self, acc_xyz: np.ndarray, hs_idx: np.ndarray, to_idx: np.ndarray
    ) -> np.ndarray:
        """Convierte a m/s^2 aislando el impacto."""
        if not self.config.acc_input_in_g:
            return acc_xyz

        factor_escala = 1.0
        ventana_stance = self._get_safe_stance_window(acc_xyz, hs_idx, to_idx)

        if ventana_stance is not None:
            factor_escala = np.linalg.norm(ventana_stance, axis=1).mean()

        if factor_escala < 0.1:  # EVITAR DIVISION POR CERO
            factor_escala = 1.0

        return (acc_xyz / factor_escala) * self.config.gravity

    @staticmethod
    def _validate_xyz(data: np.ndarray, name: str) -> None:
        """Valida dimensiones matriz."""
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"{name} debe tener forma (N, 3).")

    def _get_flat_foot_periods(
        self, hs_idx: np.ndarray, to_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula ventanas completas de pie plano para bloqueo cinematico."""
        ff_starts, ff_ends = [], []

        for hs, to in self._get_valid_pairs(hs_idx, to_idx):
            largo = to - hs
            # MARGEN: Ignora impacto (HS) y despegue (TO) aislando el centro
            margen = min(int(largo * 0.25), int(0.3 * self.config.fs))
            ff_starts.append(hs + margen)
            ff_ends.append(to - margen)

        return np.array(ff_starts, dtype=int), np.array(ff_ends, dtype=int)

    def _get_valid_pairs(self, hs_idx: np.ndarray, to_idx: np.ndarray) -> List[Tuple[int, int]]:
        valid_pairs = []
        for hs in hs_idx:
            future_to = to_idx[to_idx > hs]
            if len(future_to) > 0:
                valid_pairs.append((hs, future_to[0]))
        return valid_pairs

    def _get_swing_pairs(self, hs_idx: np.ndarray, to_idx: np.ndarray) -> List[Tuple[int, int]]:
        swing_pairs = []
        for to in to_idx:
            future_hs = hs_idx[hs_idx > to]
            if len(future_hs) > 0:
                hs_next = future_hs[0]
                if ((hs_next - to) / self.config.fs) <= 2.0:
                    swing_pairs.append((to, hs_next))
        return swing_pairs

    def _estimar_cuaternion_inicial(self, acc_xyz: np.ndarray) -> np.ndarray:
        """Estima orientacion inicial usando la gravedad."""
        n_muestras = min(self.config.muestras_inicializacion, len(acc_xyz))
        vec_grav = acc_xyz[:n_muestras].mean(axis=0)
        norma = np.linalg.norm(vec_grav)

        if norma < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])

        vec_grav = vec_grav / norma
        eje_z = np.array([0.0, 0.0, 1.0])
        eje_rot = np.cross(vec_grav, eje_z)
        norma_eje = np.linalg.norm(eje_rot)
        cos_ang = np.clip(np.dot(vec_grav, eje_z), -1.0, 1.0)

        if norma_eje < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0]) if cos_ang > 0 else np.array([0.0, 1.0, 0.0, 0.0])

        eje_rot = eje_rot / norma_eje
        angulo = np.arccos(cos_ang)
        w = np.cos(angulo / 2.0)
        xyz = eje_rot * np.sin(angulo / 2.0)
        q = np.array([w, xyz[0], xyz[1], xyz[2]])

        return q / np.linalg.norm(q)

    def estimate_orientation(
        self, gyro_rad: np.ndarray, acc_xyz: np.ndarray, mag_xyz: np.ndarray = None
    ) -> np.ndarray:
        from ahrs.filters import Madgwick
        q0_est = self._estimar_cuaternion_inicial(acc_xyz)
        madgwick = Madgwick(
            gyr=gyro_rad,
            acc=acc_xyz,
            mag=mag_xyz,
            frequency=self.config.fs,
            gain=self.config.madgwick_beta,
            q0=q0_est
        )
        return madgwick.Q

    def rotate_to_global(self, acc_xyz: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
        global_acc = np.zeros_like(acc_xyz)
        for i in range(len(acc_xyz)):
            q = quaternions[i]
            rot = R.from_quat([q[1], q[2], q[3], q[0]])
            global_acc[i] = rot.apply(acc_xyz[i])
        return global_acc

    def remove_gravity(self, global_acc: np.ndarray) -> np.ndarray:
        return global_acc - np.array([0.0, 0.0, self.config.gravity])

    def calcular_mid_stance(
        self, hs_idx: np.ndarray, to_idx: np.ndarray, gyro_xyz: np.ndarray
    ) -> np.ndarray:
        """Calcula indices de Mid-Stance."""
        # CALCULAR NORMA GIROSCOPIO
        gyro_norm = np.linalg.norm(gyro_xyz, axis=1)
        mid_stance_idx = []

        # EMPAREJAR EVENTOS VALIDOS
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

            # ACOTAR TO AL RANGO TEMPORAL DEL SEGMENTO HS
            mask_to = (to_idx >= hs_seg[0]) & (to_idx <= hs_seg[-1] + int(2.0 * self.config.fs))
            to_seg = to_idx[mask_to]

            segmentos.append((hs_seg, to_seg))

        return segmentos

    def integrate_and_zupt(
        self, linear_acc: np.ndarray, ms_idx: np.ndarray
    ) -> np.ndarray:
        """Integra aplicando ZUPT y bloquea periodos inactivos prolongados."""
        velocity = np.zeros_like(linear_acc)
        max_dur_paso = 2.0  # UMBRAL FISIOLOGICO MAXIMO (segundos)

        # ITERAR TRAMOS ZUPT
        for i in range(len(ms_idx) - 1):
            start = ms_idx[i]
            end = ms_idx[i + 1]
            duracion = (end - start) / self.config.fs

            if start >= end:
                continue

            # BLOQUEO ESTATICO PARA DESCANSOS Y HUECOS
            if duracion > max_dur_paso:
                # Congelar velocidad a 0.0 evita que el integrador acumule ruido
                velocity[start:end + 1, :] = 0.0
                continue

            # INTEGRAR PASOS VALIDOS
            for axis in range(3):
                acc_segment = linear_acc[start:end + 1, axis]
                vel_segment = cumulative_trapezoid(acc_segment, dx=self.dt, initial=0)

                # CORRECCION ZUPT LINEAL (Deriva cero en Mid-Stance)
                error_impacto = vel_segment[-1]
                tasa_deriva = error_impacto / (end - start) if (end - start) > 0 else 0.0
                vel_segment -= tasa_deriva * np.arange(len(vel_segment))

                velocity[start:end + 1, axis] = vel_segment

        return velocity

    def integrate_position(
        self, velocity_xyz: np.ndarray, ms_idx: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Integra posicion global en XY e integracion local con ZUPT en Z.

        :param velocity_xyz: Velocidad global (N, 3).
        :param ms_idx: Indices de Mid-Stance.
        :return: Posicion (N, 3) y diagnostico de deriva vertical.
        """
        position = np.zeros_like(velocity_xyz)
        max_dur_paso = 2.0  # UMBRAL FISIOLOGICO MAXIMO (segundos)

        # VARIABLES DIAGNOSTICO
        max_dur = 0.0
        max_drift = 0.0
        max_vel_z = float(np.max(np.abs(velocity_xyz[:, 2]))) if len(velocity_xyz) > 0 else 0.0

        # 1. INTEGRACION CONTINUA XY
        # (Inmune a los descansos porque velocity_xyz ya es 0 en paradas largas)
        for axis in range(2):
            position[:, axis] = cumulative_trapezoid(
                velocity_xyz[:, axis], dx=self.dt, initial=0
            )

        diagnostico = {
            "max_dur_mid_stance_s": max_dur,
            "max_drift_z_m": max_drift,
            "max_vel_z_ms": max_vel_z
        }

        if len(ms_idx) < 2:
            return position, diagnostico

        # 2. INTEGRACION LOCAL Z CON CORRECCION DE DERIVA
        for i in range(len(ms_idx) - 1):
            start = ms_idx[i]
            end = ms_idx[i + 1]
            duracion = (end - start) / self.config.fs

            if start >= end:
                continue

            if duracion > max_dur:
                max_dur = duracion

            # IGNORAR PARADAS LARGAS (Z se mantiene plano en 0.0)
            if duracion > max_dur_paso:
                continue

            # AISLAR VELOCIDAD Z Y CALCULAR LOCALMENTE
            vel_z_segment = velocity_xyz[start:end + 1, 2]
            pos_z_local = cumulative_trapezoid(vel_z_segment, dx=self.dt, initial=0)

            # EXTRACCION Y PURGA DE ERROR
            drift = pos_z_local[-1]
            if abs(drift) > max_drift:
                max_drift = abs(drift)

            rate = drift / (end - start) if (end - start) > 0 else 0.0
            correction = rate * np.arange(end - start + 1)

            # INYECTAR PASO LIMPIO
            position[start:end + 1, 2] = pos_z_local - correction

        # LIMPIAR EXTREMOS Z
        position[:ms_idx[0], 2] = 0.0
        position[ms_idx[-1]:, 2] = 0.0

        diagnostico["max_dur_mid_stance_s"] = max_dur
        diagnostico["max_drift_z_m"] = max_drift
        diagnostico["max_vel_z_ms"] = max_vel_z

        return position, diagnostico

    def compute_temporal_metrics(
        self, hs_idx: np.ndarray, to_idx: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calcula metricas temporales."""
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

    def compute_stride_length(self, position_xyz: np.ndarray, hs_idx: np.ndarray) -> np.ndarray:
        """Calcula longitud de zancada."""
        stride_lengths = []
        for i in range(len(hs_idx) - 1):
            p1 = position_xyz[hs_idx[i]]
            p2 = position_xyz[hs_idx[i + 1]]
            dist = np.linalg.norm((p2 - p1)[:2])
            stride_lengths.append(dist)
        return np.array(stride_lengths)

    def compute_gait_speed(
        self, stride_lengths: np.ndarray, stride_times: np.ndarray
    ) -> np.ndarray:
        """Calcula velocidad de marcha."""
        n = min(len(stride_lengths), len(stride_times))
        return stride_lengths[:n] / stride_times[:n] if n > 0 else np.array([])

    def compute_mtc(
        self, position_xyz: np.ndarray, to_idx: np.ndarray, hs_idx: np.ndarray
    ) -> np.ndarray:
        """Calcula Minimum Toe Clearance."""
        mtc_values = []
        z = position_xyz[:, 2]
        fraccion = self.config.mtc_window_fraction

        for to in to_idx:
            future_hs = hs_idx[hs_idx > to]
            if len(future_hs) == 0:
                continue

            hs_next = future_hs[0]
            segment = z[to:hs_next]

            if len(segment) < 2:
                continue

            rel_seg = segment - np.mean(segment[:min(5, len(segment))])
            largo = len(rel_seg)
            centro = largo / 2.0
            m_ancho = (fraccion * largo) / 2.0
            inicio, fin = max(0, int(centro - m_ancho)), min(largo, int(centro + m_ancho))

            if fin > inicio:
                mtc_values.append(np.min(rel_seg[inicio:fin]))

        return np.array(mtc_values)

    def compute_peak_swing_velocity(
        self, velocity_xyz: np.ndarray, to_idx: np.ndarray, hs_idx: np.ndarray
    ) -> np.ndarray:
        """Calcula velocidad pico en vuelo."""
        peak_vels = []
        vel_norm = np.linalg.norm(velocity_xyz, axis=1)

        # AISLAR FASE VUELO
        for to in to_idx:
            future_hs = hs_idx[hs_idx > to]
            if len(future_hs) > 0:
                hs_next = future_hs[0]

                # VALIDAR VUELO LOGICO
                if (hs_next - to) / self.config.fs <= 2.0:
                    vuelo_vel = vel_norm[to:hs_next]
                    if len(vuelo_vel) > 0:
                        peak_vels.append(np.max(vuelo_vel))

        return np.array(peak_vels)

    def run_pipeline(
        self,
        acc_xyz: np.ndarray,
        gyro_xyz: np.ndarray,
        hs_idx: np.ndarray,
        to_idx: np.ndarray,
        mag_xyz: np.ndarray = None
    ) -> dict:
        """Orquesta procesamiento cinematico completo."""
        self._validate_xyz(acc_xyz, "acc_xyz")
        self._validate_xyz(gyro_xyz, "gyro_xyz")

        # PREPARAR ACELERACION
        acc_xyz = self._convertir_aceleracion(acc_xyz, hs_idx, to_idx)

        # PREPARAR GIROSCOPIO
        gyro_rad = np.deg2rad(gyro_xyz)
        ventana_gyro = self._get_safe_stance_window(gyro_rad, hs_idx, to_idx)
        if ventana_gyro is not None:
            gyro_rad = gyro_rad - ventana_gyro.mean(axis=0)

        # FUSION SENSORIAL
        quaternions = self.estimate_orientation(gyro_rad, acc_xyz, mag_xyz)
        global_acc = self.rotate_to_global(acc_xyz, quaternions)

        # CALIBRAR BIAS
        if self.config.calibrar_bias_estatico:
            ventana_global = self._get_safe_stance_window(global_acc, hs_idx, to_idx)
            if ventana_global is not None:
                offset_estimado = ventana_global.mean(axis=0)
                gravedad_esperada = np.array([0.0, 0.0, self.config.gravity])
                global_acc = global_acc - (offset_estimado - gravedad_esperada)

        linear_acc = self.remove_gravity(global_acc)

        # INTEGRAR TRAYECTORIA
        ms_idx = self.calcular_mid_stance(hs_idx, to_idx, gyro_xyz)
        velocity = self.integrate_and_zupt(linear_acc, ms_idx)
        position, diagnostico_zupt = self.integrate_position(velocity, ms_idx)

        # DIAGNOSTICAR HUECOS ENTRE MID-STANCES
        diagnostico_huecos = self.diagnosticar_huecos_mid_stance(ms_idx, umbral_seg=2.0)

        # IMPRIMIR DIAGNOSTICO ZUPT
        print("\n" + "*" * 40)
        print("DIAGNOSTICO ZUPT INTERNO:")
        print(f"Max Tiempo entre Mid-Stances: {diagnostico_zupt['max_dur_mid_stance_s']:.2f} seg")
        print(f"Max Deriva Vertical Pura:     {diagnostico_zupt['max_drift_z_m']:.3f} m")
        print(f"Max Velocidad Vertical Z:     {diagnostico_zupt['max_vel_z_ms']:.3f} m/s")
        print(f"N Huecos > 2.0 seg:           {diagnostico_huecos['n_huecos']}")
        print(f"Duracion total en huecos:     {diagnostico_huecos['duracion_total_s']:.2f} seg")
        for h in diagnostico_huecos["huecos"][:10]:
            print(f"  Hueco: {h['inicio_s']:.2f}s -> {h['fin_s']:.2f}s (dur {h['duracion_s']:.2f}s)")
        if diagnostico_huecos["n_huecos"] > 10:
            print(f"  ... y {diagnostico_huecos['n_huecos'] - 10} huecos mas.")
        print("*" * 40 + "\n")

        # CALCULAR METRICAS
        temporal_metrics = self.compute_temporal_metrics(hs_idx, to_idx)
        stride_lengths = self.compute_stride_length(position, hs_idx)
        gait_speed = self.compute_gait_speed(stride_lengths, temporal_metrics["stride_times"])
        mtc = self.compute_mtc(position, to_idx, hs_idx)
        peak_swing_vel = self.compute_peak_swing_velocity(velocity, to_idx, hs_idx)

        return {
            "quaternions": quaternions,
            "global_acc": global_acc,
            "linear_acc": linear_acc,
            "velocity": velocity,
            "position": position,
            "stride_lengths": stride_lengths,
            "gait_speed": gait_speed,
            "mtc": mtc,
            "peak_swing_velocity": peak_swing_vel,
            "zupt_diagnostico": diagnostico_zupt,
            "huecos_diagnostico": diagnostico_huecos,
            **temporal_metrics
        }

    def run_pipeline_segmentado(
        self,
        acc_xyz: np.ndarray,
        gyro_xyz: np.ndarray,
        hs_idx: np.ndarray,
        to_idx: np.ndarray,
        mag_xyz: np.ndarray = None,
        umbral_hueco_seg: float = 2.0,
        min_pasos_segmento: int = 4
    ) -> dict:
        """
        Ejecuta el pipeline cinematico por tramos continuos de marcha,
        cortando en los huecos temporales entre HS consecutivos, para
        evitar que la integracion arrastre deriva a traves de pausas
        (giros, descansos) o perdidas de deteccion de eventos.

        :param acc_xyz: Aceleracion (N, 3).
        :param gyro_xyz: Giroscopio (N, 3).
        :param hs_idx: Indices Heel Strike globales.
        :param to_idx: Indices Toe Off globales.
        :param mag_xyz: Magnetometro (N, 3), opcional.
        :param umbral_hueco_seg: Duracion a partir de la cual se corta un tramo.
        :param min_pasos_segmento: Minimo de pasos para procesar un segmento
            (segmentos mas cortos se descartan por no ser representativos).
        :return: Diccionario con resultados concatenados y metadata de segmento
            (segmento_id, segmento_inicio_s) por cada zancada valida.
        """
        segmentos = self.segmentar_por_huecos(hs_idx, to_idx, umbral_seg=umbral_hueco_seg)

        acumulado: Dict[str, list] = {
            "stride_lengths": [], "gait_speed": [], "mtc": [],
            "peak_swing_velocity": [], "hs_times_s": [],
            "stride_times": [], "stance_times": [], "swing_times": [],
            "segmento_id": [], "position_segmentos": []
        }
        n_descartados = 0

        for seg_id, (hs_seg, to_seg) in enumerate(segmentos):
            if len(hs_seg) < min_pasos_segmento:
                n_descartados += 1
                continue

            resultado_seg = self.run_pipeline(
                acc_xyz=acc_xyz,
                gyro_xyz=gyro_xyz,
                hs_idx=hs_seg,
                to_idx=to_seg,
                mag_xyz=mag_xyz
            )

            n_pasos_seg = len(resultado_seg["stride_lengths"])
            acumulado["stride_lengths"].extend(resultado_seg["stride_lengths"])
            acumulado["gait_speed"].extend(resultado_seg["gait_speed"])
            acumulado["mtc"].extend(resultado_seg["mtc"])
            acumulado["peak_swing_velocity"].extend(resultado_seg["peak_swing_velocity"])
            acumulado["hs_times_s"].extend((hs_seg[:-1] / self.config.fs).tolist())
            acumulado["stride_times"].extend(resultado_seg["stride_times"])
            acumulado["stance_times"].extend(resultado_seg["stance_times"])
            acumulado["swing_times"].extend(resultado_seg["swing_times"])
            acumulado["segmento_id"].extend([seg_id] * n_pasos_seg)
            acumulado["position_segmentos"].append(resultado_seg["position"])

        print(
            f"[SEGMENTACION] {len(segmentos)} tramos detectados, "
            f"{len(segmentos) - n_descartados} procesados, "
            f"{n_descartados} descartados (< {min_pasos_segmento} pasos)."
        )

        return {
            "stride_lengths": np.array(acumulado["stride_lengths"]),
            "gait_speed": np.array(acumulado["gait_speed"]),
            "mtc": np.array(acumulado["mtc"]),
            "peak_swing_velocity": np.array(acumulado["peak_swing_velocity"]),
            "hs_times_s": np.array(acumulado["hs_times_s"]),
            "stride_times": np.array(acumulado["stride_times"]),
            "stance_times": np.array(acumulado["stance_times"]),
            "swing_times": np.array(acumulado["swing_times"]),
            "segmento_id": np.array(acumulado["segmento_id"]),
            "position_segmentos": acumulado["position_segmentos"],
            "n_segmentos_totales": len(segmentos),
            "n_segmentos_procesados": len(segmentos) - n_descartados
        }