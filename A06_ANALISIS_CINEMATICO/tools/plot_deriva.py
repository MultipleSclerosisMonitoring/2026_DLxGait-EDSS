# -*- coding: utf-8 -*-
"""
plot_deriva.py
==============
Herramienta de diagnostico visual: grafica la senal de presion plantar
contra el umbral de deteccion calculado, en una ventana temporal
especifica, para inspeccionar visualmente por que el detector de
eventos pudo perder o generar transiciones espurias en ese tramo.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIGURAR RUTAS GLOBALES (sube dos niveles: tools/ -> A06_ANALISIS_CINEMATICO/ -> raiz proyecto)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# IMPORTAR CLASE BASE
from A06_ANALISIS_CINEMATICO.Orquestador_biomecanico import (
    BilateralPipeline, parse_args, construir_config
)


class PlotDeriva(BilateralPipeline):
    """Clase para graficar deriva de presion frente al umbral de deteccion."""

    def ejecutar(self, pie: str = "Right", t_inicio: float = 250.0, t_fin: float = 290.0) -> None:
        """
        Ejecuta extraccion y grafica la ventana temporal solicitada.

        :param pie: Lado a graficar ("Right" o "Left").
        :param t_inicio: Segundo de inicio de la ventana a graficar.
        :param t_fin: Segundo de fin de la ventana a graficar.
        """
        # EXTRAER DATOS PIE
        df_pie = self._extraer_pie(pie)
        presion = self._preparar_presion(df_pie)
        fs = self.config.fs

        # CONFIGURAR UMBRAL CORTE (mismo criterio en cascada que el orquestador)
        umbral_manual = self.config.th_right_manual if pie == "Right" else self.config.th_left_manual
        if umbral_manual is not None:
            umbral = umbral_manual
        elif self.config.auto_calibrar_umbral:
            umbral = self._auto_calibrar_umbral(presion, pie)
        else:
            umbral = self.detector.config.threshold_fraction

        # CALCULAR UMBRAL EXACTO
        _, _, threshold_calc = self.detector.detect_from_pressure(
            presion, threshold_fraction_override=umbral
        )

        # CREAR VECTOR TIEMPO
        tiempo = np.arange(len(presion)) / fs

        # FILTRAR VENTANA TIEMPO
        mascara = (tiempo >= t_inicio) & (tiempo <= t_fin)
        tiempo_ventana = tiempo[mascara]
        presion_ventana = presion[mascara]

        # GRAFICAR RESULTADOS
        plt.figure(figsize=(10, 5))
        plt.plot(tiempo_ventana, presion_ventana, label="Presion", color="blue")
        plt.axhline(y=threshold_calc, color="red", linestyle="--", label=f"Umbral ({threshold_calc:.3f})")

        # CONFIGURAR EJES GRAFICA (titulo incluye el paciente)
        plt.title(f"Prueba de Deriva - {self.config.paciente} - Pie {pie} ({t_inicio}s - {t_fin}s)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Presion Plantar")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # GUARDAR GRAFICA EN SUBCARPETA POR PACIENTE
        carpeta_paciente = self.config.output_dir / self.config.paciente
        carpeta_paciente.mkdir(parents=True, exist_ok=True)
        ruta_imagen = carpeta_paciente / f"{self.config.paciente}_deriva_{pie}_{t_inicio}_{t_fin}.png"
        plt.savefig(ruta_imagen, dpi=300, bbox_inches="tight")
        plt.close()

        # CONFIRMAR EN CONSOLA
        print("\n" + "=" * 50)
        print("IMAGEN GUARDADA EXITOSAMENTE EN:")
        print(f"{ruta_imagen}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    # OBTENER ARGUMENTOS CONSOLA (mismo parser que el orquestador oficial)
    args = parse_args()

    # CONSTRUIR CONFIG REUTILIZANDO LA MISMA VALIDACION/PARSEO DE FECHAS
    config = construir_config(args)

    # INICIAR ORQUESTADOR PLOT
    plotter = PlotDeriva(config)
    plotter.ejecutar(pie="Right", t_inicio=250.0, t_fin=290.0)