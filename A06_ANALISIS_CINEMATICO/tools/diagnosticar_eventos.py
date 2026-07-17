# -*- coding: utf-8 -*-
"""
Script de diagnostico de eventos orientado a objetos.
Hereda del orquestador principal para interceptar y evaluar
el detector de eventos sin modificar el codigo original.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any

# CONFIGURAR RUTAS GLOBALES (sube dos niveles: tools/ -> A06_ANALISIS_CINEMATICO/ -> raiz proyecto)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# IMPORTAR CLASE BASE
from A06_ANALISIS_CINEMATICO.Orquestador_biomecanico import (
    BilateralPipeline, parse_args, construir_config
)


class OrquestadorDiagnostico(BilateralPipeline):
    """Clase derivada para diagnosticar perdida de pisadas."""

    def _procesar_pie(self, pie: str) -> Dict[str, Any]:
        """
        Sobrescribe el procesamiento para inyectar diagnostico.

        :param pie: Lado del cuerpo ('Right' o 'Left').
        :type pie: str
        :return: Matrices procesadas.
        :rtype: Dict[str, Any]
        """
        # EXTRAER DATOS PIE
        df_pie = self._extraer_pie(pie)
        presion = self._preparar_presion(df_pie)

        # CONFIGURAR UMBRAL CORTE (mismo criterio en cascada que el orquestador)
        umbral_manual = self.config.th_right_manual if pie == "Right" else self.config.th_left_manual
        if umbral_manual is not None:
            umbral = umbral_manual
        elif self.config.auto_calibrar_umbral:
            umbral = self._auto_calibrar_umbral(presion, pie)
        else:
            umbral = self.detector.config.threshold_fraction

        # EJECUTAR DETECTOR EVENTOS
        hs_idx, to_idx, threshold_calc = self.detector.detect_from_pressure(
            presion, threshold_fraction_override=umbral
        )

        # IMPRIMIR RADIOGRAFIA CONSOLA (incluye paciente para trazabilidad)
        fs = self.config.fs
        print(f"RADIOGRAFIA EVENTOS - {self.config.paciente} - {pie.upper()}")
        print("=" * 50)
        print(f"Umbral utilizado:    {threshold_calc:.4f}")
        print(f"Total HS detectados: {len(hs_idx)}")

        if len(hs_idx) > 1:
            duraciones_s = np.diff(hs_idx) / fs
            indices_huecos = np.where(duraciones_s > 2.0)[0]
            print(f"Huecos anomalos (> 2s): {len(indices_huecos)}\n")

            for i, idx in enumerate(indices_huecos):
                inicio = hs_idx[idx]
                fin = hs_idx[idx + 1]
                duracion = duraciones_s[idx]

                tramo_presion = presion[inicio:fin]
                pmax = np.max(tramo_presion) if len(tramo_presion) > 0 else 0.0

                print(f"  [ HUECO {i + 1} ]")
                print(f"  - Tiempo:      {inicio / fs:.1f}s  --->  {fin / fs:.1f}s")
                print(f"  - Duracion:    {duracion:.1f} s")
                print(f"  - Presion max: {pmax:.4f}")

                # EVALUAR CAUSA RAIZ
                if pmax < threshold_calc:
                    print("  -> CAUSA: Falla por UMBRAL (Presion baja).")
                else:
                    print("  -> CAUSA: Falla por FILTRO/REFRACTARIO.")
                print("-" * 40)
        print("=" * 50 + "\n")

        # CONTINUAR MOTOR CINEMATICO (respeta magnetometro real y segmentacion,
        # solo para completar el diagnostico; no se usa para graficar ni
        # exportar CSV, evitando cualquier riesgo de sobrescribir outputs
        # del orquestador oficial)
        matrices = self._extraer_matrices(df_pie)
        print(f"[{pie}] Magnetometro usado: {'SI' if matrices['mag_xyz'] is not None else 'NO'}")

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

    def ejecutar(self) -> None:
        """
        Ejecuta unicamente el diagnostico de eventos por pie.

        Deliberadamente NO llama al ejecutar() completo del orquestador
        (que grafica trayectoria 3D, exporta CSV de metricas y corre el
        analisis de fatiga), porque esos archivos comparten nombre con
        los que genera Orquestador_biomecanico.py y una ejecucion de
        diagnostico no debe sobrescribir resultados oficiales ya
        calculados para el mismo paciente.
        """
        self._procesar_pie("Right")
        self._procesar_pie("Left")


if __name__ == "__main__":
    # OBTENER ARGUMENTOS CONSOLA (mismo parser que el orquestador oficial)
    args = parse_args()

    # CONSTRUIR CONFIG REUTILIZANDO LA MISMA VALIDACION/PARSEO DE FECHAS
    config = construir_config(args)

    # INICIAR ORQUESTADOR DERIVADO
    orquestador = OrquestadorDiagnostico(config)
    orquestador.ejecutar()