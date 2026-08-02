# -*- coding: utf-8 -*-
"""
Genera un informe PDF corto que consolida los resultados de los 3
pipelines (Agnostic_evaluator.py, Orquestador_temporal.py de A06, y
analizar_velocidad_gps_paciente.py de A07) para UN paciente/sesion
especifico.

El informe SOLO se genera si los 3 pipelines ya corrieron exitosamente
para ese paciente (se detecta buscando los 3 archivos de salida
esperados). Si falta alguno, se informa cual falta y no se genera nada
-- para evitar reportar un informe incompleto como si fuera completo.
"""

from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import Optional, Dict, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def _localizar_outputs(
    paciente: str, modelo_agnostic: str,
    dir_agnostic: Path, dir_a06: Path, dir_a07: Path
) -> Dict[str, Optional[Path]]:
    """
    Busca los 3 archivos de salida esperados para un paciente, uno por
    pipeline. Devuelve None en cada clave cuyo archivo no exista, en vez
    de lanzar una excepcion -- para poder reportar con precision cual
    pipeline falta.

    :param paciente: CodeID del paciente.
    :param modelo_agnostic: Sufijo de modelo usado por Agnostic_evaluator.py
        (ej. "hibrido_late", "fft", "transformer", "hibrido_early").
    :param dir_agnostic: Carpeta RESULTADO_AGNOSTIC de A04.
    :param dir_a06: Carpeta RESULTADOS_TEMPORALES de A06.
    :param dir_a07: Carpeta RESULTADOS_VELOCIDAD_GPS de A07.
    :return: Diccionario {"agnostic": Path|None, "a06": Path|None, "a07": Path|None}.
    """
    ruta_agnostic = dir_agnostic / f"agnostico_{paciente}_{modelo_agnostic}.csv"
    ruta_a06 = dir_a06 / f"{paciente}_metricas_temporales.csv"
    ruta_a07 = dir_a07 / f"{paciente}_velocidad_gps.csv"

    return {
        "agnostic": ruta_agnostic if ruta_agnostic.exists() else None,
        "a06": ruta_a06 if ruta_a06.exists() else None,
        "a07": ruta_a07 if ruta_a07.exists() else None,
    }


def verificar_pipelines_completos(
    paciente: str, modelo_agnostic: str,
    dir_agnostic: Path, dir_a06: Path, dir_a07: Path
) -> Tuple[bool, Dict[str, Optional[Path]]]:
    """
    Verifica si los 3 pipelines corrieron exitosamente para un paciente
    (condicion de activacion del informe).

    :return: Tupla (todos_completos: bool, rutas: dict). Si
        todos_completos es False, revisar que claves de rutas son None
        para saber cual pipeline falta.
    """
    rutas = _localizar_outputs(paciente, modelo_agnostic, dir_agnostic, dir_a06, dir_a07)
    todos_completos = all(v is not None for v in rutas.values())
    return todos_completos, rutas


def _grafica_probabilidad_agnostic(df_agnostic: pd.DataFrame, out_path: Path) -> None:
    """Genera la grafica de linea temporal de probabilidad de marcha (Agnostic)."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(pd.to_datetime(df_agnostic["timestamp"]), df_agnostic["prob_smoothed"], linewidth=1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Prob. marcha")
    ax.set_xlabel("Tiempo")
    ax.set_title("Probabilidad de marcha en el tiempo (Agnostic)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _grafica_asimetria_a06(df_a06: pd.DataFrame, out_path: Path) -> None:
    """Genera la grafica de barras de tiempos derecha/izquierda (A06)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    metricas = ["stride_time", "stance_time", "swing_time"]
    etiquetas = ["Zancada", "Apoyo", "Vuelo"]
    x = range(len(metricas))
    ancho = 0.35

    medias_der = [df_a06[f"{m}_right"].mean() for m in metricas]
    medias_izq = [df_a06[f"{m}_left"].mean() for m in metricas]

    ax.bar([i - ancho / 2 for i in x], medias_der, ancho, label="Derecha")
    ax.bar([i + ancho / 2 for i in x], medias_izq, ancho, label="Izquierda")
    ax.set_xticks(list(x))
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("Tiempo (s)")
    ax.set_title("Metricas temporales bilaterales (A06)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generar_informe_conjunto(
    paciente: str,
    modelo_agnostic: str,
    dir_agnostic: Path,
    dir_a06: Path,
    dir_a07: Path,
    output_pdf: Path,
) -> bool:
    """
    Genera el informe PDF conjunto SOLO si los 3 pipelines corrieron
    exitosamente para el paciente indicado.

    :param paciente: CodeID del paciente.
    :param modelo_agnostic: Sufijo de modelo de Agnostic_evaluator.py.
    :param dir_agnostic: Carpeta RESULTADO_AGNOSTIC de A04.
    :param dir_a06: Carpeta RESULTADOS_TEMPORALES de A06.
    :param dir_a07: Carpeta RESULTADOS_VELOCIDAD_GPS de A07.
    :param output_pdf: Ruta de salida del PDF.
    :return: True si el informe se genero, False si falto algun pipeline
        (en cuyo caso se imprime cual falta y no se genera nada).
    """
    completos, rutas = verificar_pipelines_completos(
        paciente, modelo_agnostic, dir_agnostic, dir_a06, dir_a07
    )

    if not completos:
        faltantes = [nombre for nombre, ruta in rutas.items() if ruta is None]
        print(f"INFORME NO GENERADO: faltan resultados de: {', '.join(faltantes)} "
              f"para el paciente {paciente}.")
        return False

    df_agnostic = pd.read_csv(rutas["agnostic"])
    df_a06 = pd.read_csv(rutas["a06"])
    df_a07 = pd.read_csv(rutas["a07"])

    carpeta_temp = output_pdf.parent / "_graficas_temp_informe"
    carpeta_temp.mkdir(parents=True, exist_ok=True)
    ruta_grafica_agnostic = carpeta_temp / f"{paciente}_agnostic.png"
    ruta_grafica_a06 = carpeta_temp / f"{paciente}_a06.png"

    _grafica_probabilidad_agnostic(df_agnostic, ruta_grafica_agnostic)
    _grafica_asimetria_a06(df_a06, ruta_grafica_a06)

    # -------------------------------------------------------------------
    # METRICAS RESUMEN DE CADA PIPELINE
    # -------------------------------------------------------------------
    pct_marcha = 100.0 * (df_agnostic["pred_final_smoothed"] == 1).mean()
    prob_media = df_agnostic["prob_smoothed"].mean()

    stride_time_medio = (df_a06["stride_time_right"].mean() + df_a06["stride_time_left"].mean()) / 2
    asimetria_media = df_a06["Asimetria_stride_time_pct"].mean()
    n_zancadas = len(df_a06)

    fila_a07 = df_a07.iloc[0]
    velocidad_ms = fila_a07["velocidad_ms"]
    distancia_m = fila_a07["distancia_m"]
    n_lecturas_gps = fila_a07["n_lecturas_gps_reales"]

    # -------------------------------------------------------------------
    # CONSTRUCCION DEL PDF
    # -------------------------------------------------------------------
    styles = getSampleStyleSheet()
    titulo_style = ParagraphStyle("TituloInforme", parent=styles["Title"], fontSize=16, spaceAfter=4)
    subtitulo_style = ParagraphStyle("Subtitulo", parent=styles["Normal"], fontSize=10, textColor=colors.grey, spaceAfter=12)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)

    doc = SimpleDocTemplate(
        str(output_pdf), pagesize=LETTER,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm
    )
    elementos = []

    elementos.append(Paragraph(f"Informe Consolidado — {paciente}", titulo_style))
    elementos.append(Paragraph(
        f"Fecha de generación: {date.today().isoformat()} &nbsp;|&nbsp; "
        f"Pipelines: Clasificación (Agnostic), Biomecánica temporal (A06), Velocidad GPS (A07)",
        subtitulo_style
    ))

    elementos.append(Paragraph("1. Clasificación marcha/reposo (Agnostic)", h2_style))
    elementos.append(ListFlowable([
        ListItem(Paragraph(f"Modelo usado: {modelo_agnostic}", body_style)),
        ListItem(Paragraph(f"Porcentaje de tiempo clasificado como marcha: {pct_marcha:.1f}%", body_style)),
        ListItem(Paragraph(f"Probabilidad media de marcha: {prob_media:.3f}", body_style)),
        ListItem(Paragraph(f"Predicciones totales: {len(df_agnostic)}", body_style)),
    ], bulletType="bullet"))
    elementos.append(Image(str(ruta_grafica_agnostic), width=16 * cm, height=6.5 * cm))

    elementos.append(Paragraph("2. Biomecánica temporal bilateral (A06)", h2_style))
    elementos.append(Paragraph(
        "Métricas temporales únicamente (duración de zancada, apoyo, vuelo). "
        "No incluye trayectoria, velocidad de marcha, longitud de zancada ni MTC "
        "(descartadas por fragilidad de la reconstrucción espacial).",
        body_style
    ))
    elementos.append(ListFlowable([
        ListItem(Paragraph(f"Zancadas emparejadas: {n_zancadas}", body_style)),
        ListItem(Paragraph(f"Duración media de zancada: {stride_time_medio:.3f} s", body_style)),
        ListItem(Paragraph(f"Asimetría media (zancada): {asimetria_media:.1f}%", body_style)),
    ], bulletType="bullet"))
    elementos.append(Image(str(ruta_grafica_a06), width=14 * cm, height=8 * cm))

    elementos.append(Paragraph("3. Velocidad de desplazamiento (A07, vía GPS)", h2_style))
    elementos.append(Paragraph(
        "Velocidad promedio de sesión (distancia recorrida real, sumando tramos entre "
        "lecturas GPS reales consecutivas, no desplazamiento neto) — no equivale a "
        "velocidad de marcha instantánea ni sustituye longitud de zancada/MTC.",
        body_style
    ))
    velocidad_str = f"{velocidad_ms:.3f} m/s" if pd.notna(velocidad_ms) else "No calculable (GPS insuficiente)"
    elementos.append(ListFlowable([
        ListItem(Paragraph(f"Velocidad promedio de sesión: {velocidad_str}", body_style)),
        ListItem(Paragraph(f"Distancia recorrida (real, no neta): {distancia_m:.2f} m", body_style)),
        ListItem(Paragraph(f"Lecturas GPS reales en la sesión: {int(n_lecturas_gps)}", body_style)),
    ], bulletType="bullet"))

    doc.build(elementos)

    # LIMPIAR GRAFICAS TEMPORALES
    ruta_grafica_agnostic.unlink(missing_ok=True)
    ruta_grafica_a06.unlink(missing_ok=True)
    try:
        carpeta_temp.rmdir()
    except OSError:
        pass

    print(f"INFORME GENERADO: {output_pdf}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genera informe PDF conjunto de los 3 pipelines para un paciente.")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--modelo-agnostic", type=str, default="hibrido_late")
    parser.add_argument("--dir-agnostic", type=str, required=True)
    parser.add_argument("--dir-a06", type=str, required=True)
    parser.add_argument("--dir-a07", type=str, required=True)
    parser.add_argument("--output-pdf", type=str, required=True)
    args = parser.parse_args()

    generar_informe_conjunto(
        paciente=args.paciente,
        modelo_agnostic=args.modelo_agnostic,
        dir_agnostic=Path(args.dir_agnostic),
        dir_a06=Path(args.dir_a06),
        dir_a07=Path(args.dir_a07),
        output_pdf=Path(args.output_pdf),
    )