# -*- coding: utf-8 -*-
"""
Auditoria del dataset HDF5 jerarquico, con exportacion a PDF.

Recorre el HDF5, calcula el resumen global, el recuento por paciente,
reproduce el split fijo (train/val/test) de AA_TRANSFORMER_V1.py, y
verifica ausencia de leakage por paciente entre particiones. El
resultado se exporta como un unico archivo PDF con formato de reporte.
"""

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)


def cargar_estructura(h5_path: Path) -> list:
    """Recorre el HDF5 y devuelve una lista de registros por ventana.

    :param h5_path: Ruta al archivo HDF5 jerarquico.
    :type h5_path: Path
    :return: Lista de diccionarios con paciente, label y forma del tensor.
    :rtype: list
    """
    registros = []
    with h5py.File(h5_path, "r") as hf:
        # ORDEN DE ITERACION: igual al usado por AA_TRANSFORMER_V1.py
        # (np.unique ordena alfabeticamente los CodeID de paciente)
        for paciente in hf.keys():
            for seg_chunk in hf[paciente].keys():
                for lado in hf[paciente][seg_chunk].keys():
                    ds = hf[paciente][seg_chunk][lado]
                    registros.append({
                        "paciente": paciente,
                        "label": int(ds.attrs.get("label", -1)),
                        "forma": ds.shape,
                    })
    return registros


def resumen_por_paciente(registros: list) -> dict:
    """Agrupa el conteo de muestras y clases por paciente.

    :param registros: Lista de registros de cargar_estructura.
    :type registros: list
    :return: Diccionario paciente -> {n0, n1, total}.
    :rtype: dict
    """
    resumen = {}
    for r in registros:
        p = r["paciente"]
        if p not in resumen:
            resumen[p] = {"n0": 0, "n1": 0}
        if r["label"] == 0:
            resumen[p]["n0"] += 1
        elif r["label"] == 1:
            resumen[p]["n1"] += 1
    for p in resumen:
        resumen[p]["total"] = resumen[p]["n0"] + resumen[p]["n1"]
    return resumen


def reproducir_split(registros: list, resumen_pac: dict) -> dict:
    """Reproduce el split fijo train/val/test tal como lo define
    AA_TRANSFORMER_V1.py: el ultimo paciente (orden alfabetico) es
    test, el penultimo es validacion, el resto es entrenamiento.

    :param registros: Lista de registros de cargar_estructura.
    :type registros: list
    :param resumen_pac: Resumen por paciente ya calculado.
    :type resumen_pac: dict
    :return: Diccionario con la informacion de cada particion.
    :rtype: dict
    """
    pacientes_unicos = sorted(resumen_pac.keys())
    test_patient = pacientes_unicos[-1]
    val_patient = pacientes_unicos[-2]
    train_patients = pacientes_unicos[:-2]

    def agregado(pacientes):
        n0 = sum(resumen_pac[p]["n0"] for p in pacientes)
        n1 = sum(resumen_pac[p]["n1"] for p in pacientes)
        return {"pacientes": pacientes, "n0": n0, "n1": n1, "total": n0 + n1}

    return {
        "train": agregado(train_patients),
        "val": {"pacientes": [val_patient], **{
            "n0": resumen_pac[val_patient]["n0"],
            "n1": resumen_pac[val_patient]["n1"],
            "total": resumen_pac[val_patient]["total"],
        }},
        "test": {"pacientes": [test_patient], **{
            "n0": resumen_pac[test_patient]["n0"],
            "n1": resumen_pac[test_patient]["n1"],
            "total": resumen_pac[test_patient]["total"],
        }},
    }


def generar_pdf(h5_path: Path, salida_pdf: Path) -> None:
    """Genera el PDF de auditoria completo.

    :param h5_path: Ruta al HDF5 a auditar.
    :type h5_path: Path
    :param salida_pdf: Ruta de salida del PDF.
    :type salida_pdf: Path
    """
    registros = cargar_estructura(h5_path)
    resumen_pac = resumen_por_paciente(registros)

    total_muestras = len(registros)
    total_pacientes = len(resumen_pac)
    formas_unicas = sorted({r["forma"] for r in registros})
    n_clase0 = sum(1 for r in registros if r["label"] == 0)
    n_clase1 = sum(1 for r in registros if r["label"] == 1)

    split = reproducir_split(registros, resumen_pac)

    # VERIFICACION DE LEAKAGE ENTRE PARTICIONES
    set_train = set(split["train"]["pacientes"])
    set_val = set(split["val"]["pacientes"])
    set_test = set(split["test"]["pacientes"])
    inter_tv = set_train & set_val
    inter_tt = set_train & set_test
    inter_vt = set_val & set_test
    sin_leakage = not (inter_tv or inter_tt or inter_vt)

    # ==================== CONSTRUCCION DEL PDF ====================
    styles = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle("TituloReporte", parent=styles["Title"], fontSize=16)
    estilo_h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    estilo_normal = styles["Normal"]

    doc = SimpleDocTemplate(str(salida_pdf), pagesize=letter,
                             topMargin=2*cm, bottomMargin=2*cm)
    story = []

    # TITULO Y METADATOS
    story.append(Paragraph("Auditoría Del Dataset Actual", estilo_titulo))
    story.append(Spacer(1, 8))
    fecha_auditoria = datetime.now().strftime("%Y-%m-%d")
    story.append(Paragraph(
        f"Fecha de auditoría: {fecha_auditoria} | Dataset auditado: {h5_path.name}",
        estilo_normal
    ))
    story.append(Spacer(1, 12))

    # RESUMEN GLOBAL
    story.append(Paragraph("Resumen global", estilo_h2))
    pct0 = 100 * n_clase0 / total_muestras if total_muestras else 0
    pct1 = 100 * n_clase1 / total_muestras if total_muestras else 0
    resumen_texto = (
        f"• Muestras totales: {total_muestras:,}<br/>"
        f"• Pacientes totales: {total_pacientes}<br/>"
        f"• Forma(s) única(s) de tensor: {', '.join(str(f) for f in formas_unicas)}<br/>"
        f"• Clase 0 (no_marcha): {n_clase0:,} ({pct0:.2f}%)<br/>"
        f"• Clase 1 (marcha): {n_clase1:,} ({pct1:.2f}%)"
    )
    story.append(Paragraph(resumen_texto, estilo_normal))

    # RECUENTO POR PACIENTE
    story.append(Paragraph("Recuento por paciente", estilo_h2))
    tabla_datos = [["Paciente", "Muestras", "No marcha (0)", "Marcha (1)", "Ambas clases"]]
    for p in sorted(resumen_pac.keys()):
        d = resumen_pac[p]
        ambas = "Sí" if d["n0"] > 0 and d["n1"] > 0 else "No"
        tabla_datos.append([p, str(d["total"]), str(d["n0"]), str(d["n1"]), ambas])

    tabla = Table(tabla_datos, repeatRows=1)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(tabla)
    story.append(Spacer(1, 12))

    # SPLIT REPRODUCIDO
    story.append(Paragraph("Split actual reproducido con la lógica de AA_TRANSFORMER_V1.py", estilo_h2))
    story.append(Paragraph(
        "La implementación actual define:<br/>"
        "• test_patient = unique_patients[-1]<br/>"
        "• val_patient = unique_patients[-2]<br/>"
        "• train = resto de pacientes<br/><br/>"
        "Con el HDF5 actual eso produce:",
        estilo_normal
    ))
    story.append(Spacer(1, 6))

    for nombre, clave in [("Train", "train"), ("Validación", "val"), ("Test", "test")]:
        d = split[clave]
        pct0_d = 100 * d["n0"] / d["total"] if d["total"] else 0
        pct1_d = 100 * d["n1"] / d["total"] if d["total"] else 0
        if clave == "train":
            linea_pac = f"• Pacientes: {len(d['pacientes'])}"
        else:
            linea_pac = f"• Paciente: {d['pacientes'][0]}"
        story.append(Paragraph(
            f"<b>{nombre}</b><br/>"
            f"{linea_pac}<br/>"
            f"• Muestras: {d['total']:,}<br/>"
            f"• Clase 0: {d['n0']:,} ({pct0_d:.2f}%)<br/>"
            f"• Clase 1: {d['n1']:,} ({pct1_d:.2f}%)",
            estilo_normal
        ))
        story.append(Spacer(1, 8))

    # AUDITORIA DE LEAKAGE
    story.append(Paragraph("Auditoría de leakage", estilo_h2))
    story.append(Paragraph("Comprobación de solapamiento de pacientes entre splits", estilo_normal))
    story.append(Paragraph(
        f"• train ∩ val = {inter_tv if inter_tv else '∅'}<br/>"
        f"• train ∩ test = {inter_tt if inter_tt else '∅'}<br/>"
        f"• val ∩ test = {inter_vt if inter_vt else '∅'}",
        estilo_normal
    ))
    story.append(Spacer(1, 8))

    conclusion = (
        "No se observa leakage por paciente en el split actual. La separación por "
        "grupos/paciente evita que ventanas del mismo paciente aparezcan a la vez "
        "en entrenamiento y evaluación."
        if sin_leakage else
        "ALERTA: se detectó solapamiento de pacientes entre particiones. "
        "Esto constituye leakage y compromete la validez de la evaluación."
    )
    story.append(Paragraph(f"<b>Conclusión:</b><br/>• {conclusion}", estilo_normal))

    doc.build(story)
    print(f"PDF generado en: {salida_pdf}")


def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Auditoria del HDF5 con exportacion a PDF.")
    parser.add_argument("--h5", type=Path, required=True, help="Ruta al dataset_jerarquico.hdf5")
    parser.add_argument("--out", type=Path, default=Path("auditoria_dataset.pdf"), help="Ruta de salida del PDF")
    args = parser.parse_args()

    generar_pdf(args.h5, args.out)


if __name__ == "__main__":
    main()
