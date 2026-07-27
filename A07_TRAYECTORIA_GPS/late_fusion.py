# -*- coding: utf-8 -*-
"""
Ensamble por Late Fusion (fusion tardia) de los modelos Transformer y FFT.

Este script hace dos cosas en una sola ejecucion, SIN reentrenar ninguna red:

  1. Reconstruye x_test/y_test desde el HDF5 + test_idx.npy, carga los
     modelos YA ENTRENADOS (modelo_transformer.pth, modelo_fft.pth) y corre
     unicamente inferencia (forward pass) para obtener las probabilidades
     de test de cada modelo por separado.
  2. Combina esas probabilidades mediante 3 criterios de Late Fusion:
       - Media geometrica.
       - Voto mayoritario (con desempate por promedio de probabilidades).
       - Meta-clasificador (regresion logistica con validacion cruzada).

A diferencia de GaitHybridModel (Early Fusion: concatena features
intermedias antes de un clasificador conjunto), aqui cada modelo termina su
propia prediccion de forma independiente y solo se combinan al final.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

# ASEGURAR QUE EL PROPIO DIRECTORIO (A04_TRANSFORMER) ESTE EN EL PATH,
# necesario cuando este script se ejecuta desde otro directorio de trabajo
# (ej. runfile con wdir en la raiz del proyecto) en vez de ejecutarse
# directamente desde dentro de A04_TRANSFORMER.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from AA_TRANSFORMER_V1 import (
    GaitDatasetLoader,
    ModelConfig,
    TransformerConfig,
    GaitTransformer,
    FFTModel,
    FFTProcessor,
    AugmentedDataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# PASO 1: RECONSTRUIR PROBABILIDADES DE TEST DESDE MODELOS YA ENTRENADOS
# =============================================================================

def generar_probabilidades_test(dataset_path: Path, models_dir: Path):
    """
    Reconstruye x_test/y_test desde el HDF5 + test_idx.npy, carga los modelos
    ya entrenados y corre solo inferencia (sin entrenar) para obtener las
    probabilidades de test de Transformer y FFT.

    :param dataset_path: Ruta al HDF5 original usado para entrenar los modelos.
    :param models_dir: Ruta a A05_MODELOS_ENTRENADOS (con los .pth, .joblib, .npy).
    :return: Tupla (y_true, prob_transformer, prob_fft) como arrays de numpy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"DEVICE: {device}")

    cfg_data = ModelConfig(h5_path=dataset_path, output_dir=models_dir)
    loader = GaitDatasetLoader(cfg_data)
    loader.scaler = joblib.load(models_dir / "scaler_gait.joblib")

    x_raw, groups, labels = loader.get_all_raw_data()
    test_idx = np.load(models_dir / "test_idx.npy")

    x_test_raw = x_raw[test_idx]
    y_test = labels[test_idx].astype(np.float32)
    x_test = loader._scale_data(x_test_raw, fit=False).astype(np.float32)

    logger.info(f"X_TEST RECONSTRUIDO: {x_test.shape}, Y_TEST: {y_test.shape}")

    fft_proc = FFTProcessor()
    x_test_fft = fft_proc.get_fft_features(x_test)
    x_test_fft_flat = x_test_fft.reshape(x_test.shape[0], -1)

    t_cfg_dict = joblib.load(models_dir / "transformer_config.joblib")
    t_cfg = TransformerConfig(**t_cfg_dict)

    model_time = GaitTransformer(t_cfg).to(device)
    model_time.load_state_dict(torch.load(models_dir / "modelo_transformer.pth", map_location=device))
    model_time.eval()

    fft_dim = x_test_fft_flat.shape[1]
    model_fft = FFTModel(fft_dim).to(device)
    model_fft.load_state_dict(torch.load(models_dir / "modelo_fft.pth", map_location=device))
    model_fft.eval()

    ts_l = DataLoader(AugmentedDataset(x_test, y_test), batch_size=64)
    fft_ts_l = DataLoader(AugmentedDataset(x_test_fft_flat, y_test), batch_size=64)

    def obtener_probs(model, loader_):
        probs, y_true = [], []
        with torch.no_grad():
            for xb, yb in loader_:
                out = model(xb.to(device))
                probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                y_true.extend(yb.numpy())
        return np.array(y_true), np.array(probs)

    y_true_t, prob_t = obtener_probs(model_time, ts_l)
    y_true_f, prob_f = obtener_probs(model_fft, fft_ts_l)

    logger.info(f"PROBABILIDADES GENERADAS: {len(y_true_t)} muestras de test")
    return y_true_t, prob_t, prob_f


# =============================================================================
# PASO 2: CRITERIOS DE LATE FUSION
# =============================================================================

def fusion_media_geometrica(prob_t: np.ndarray, prob_f: np.ndarray) -> np.ndarray:
    """Combina dos probabilidades mediante media geometrica."""
    eps = 1e-7
    prob_t_safe = np.clip(prob_t, eps, 1 - eps)
    prob_f_safe = np.clip(prob_f, eps, 1 - eps)
    return np.sqrt(prob_t_safe * prob_f_safe)


def fusion_voto_mayoritario(prob_t: np.ndarray, prob_f: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Combina dos predicciones binarias por voto mayoritario. Con 2 votantes,
    un empate (1 vs 1) se rompe usando la probabilidad promedio, en vez de
    favorecer arbitrariamente a un modelo.
    """
    pred_t = (prob_t >= threshold).astype(int)
    pred_f = (prob_f >= threshold).astype(int)

    suma_votos = pred_t + pred_f
    prob_efectiva = np.where(
        suma_votos == 2, 1.0,
        np.where(suma_votos == 0, 0.0, (prob_t + prob_f) / 2.0)
    )
    return prob_efectiva


def fusion_meta_clasificador(y_true: np.ndarray, prob_t: np.ndarray, prob_f: np.ndarray, cv_folds: int = 5) -> np.ndarray:
    """
    Entrena un meta-clasificador (regresion logistica) sobre
    [prob_transformer, prob_fft] -> clase real, con validacion cruzada
    para evitar evaluar sobre los mismos datos con que se entreno.

    Nota: esto devuelve predicciones OUT-OF-FOLD para reportar metricas sin
    optimismo. El meta-clasificador final para PRODUCCION (entrenado con
    todos los datos) se ajusta y guarda por separado en
    entrenar_y_guardar_meta_clasificador().
    """
    X = np.column_stack([prob_t, prob_f])
    prob_meta = np.zeros_like(y_true, dtype=float)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=13)
    for train_idx, test_idx in skf.split(X, y_true):
        meta_clf = LogisticRegression()
        meta_clf.fit(X[train_idx], y_true[train_idx])
        prob_meta[test_idx] = meta_clf.predict_proba(X[test_idx])[:, 1]

    return prob_meta


def entrenar_y_guardar_meta_clasificador(
    y_true: np.ndarray, prob_t: np.ndarray, prob_f: np.ndarray, out_path: Path
) -> LogisticRegression:
    """
    Entrena el meta-clasificador FINAL (con todos los datos disponibles,
    sin reservar fold de validacion) y lo guarda a disco para uso en
    produccion (ej. Agnostic_evaluator.py). Distinto del entrenamiento por
    fold de fusion_meta_clasificador(), que solo sirve para reportar
    metricas sin optimismo.

    :param y_true: Etiquetas reales de todo el set de test disponible.
    :param prob_t: Probabilidades del modelo Transformer.
    :param prob_f: Probabilidades del modelo FFT.
    :param out_path: Ruta donde guardar el .joblib del meta-clasificador.
    :return: El meta-clasificador ya entrenado.
    """
    X = np.column_stack([prob_t, prob_f])
    meta_clf_final = LogisticRegression()
    meta_clf_final.fit(X, y_true)

    joblib.dump(meta_clf_final, out_path)
    logger.info(f"META-CLASIFICADOR FINAL GUARDADO EN: {out_path}")
    logger.info(f"Coeficientes: prob_transformer={meta_clf_final.coef_[0][0]:.4f}, "
                f"prob_fft={meta_clf_final.coef_[0][1]:.4f}, intercepto={meta_clf_final.intercept_[0]:.4f}")

    return meta_clf_final


def calcular_metricas(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calcula metricas de clasificacion y calibracion para una probabilidad dada."""
    y_pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    auc = roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, prob)

    return {
        "AUC": auc, "Sensibilidad": sensibilidad, "Especificidad": especificidad,
        "FPR": fpr, "Brier": brier, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }


def graficar_matriz_confusion(y_true: np.ndarray, prob: np.ndarray, titulo: str, save_path: Path) -> None:
    """Genera y guarda la matriz de confusion para una probabilidad dada."""
    y_pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    auc = roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else 0.0

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['REPOSO', 'MARCHA'], yticklabels=['REPOSO', 'MARCHA'])
    plt.title(f'{titulo}\nAUC: {auc:.4f} | Umbral: 0.50')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Genera probabilidades de test (sin reentrenar) y calcula Late Fusion."""
    parser = argparse.ArgumentParser(description="Genera probabilidades de test y calcula Late Fusion (Transformer + FFT).")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(r"C:\Users\jairi\OneDrive\Escritorio\TFMCLONFINAL\DATASET_HDF5\dataset_jerarquico.hdf5"),
        help="Ruta al HDF5 usado para entrenar los modelos"
    )
    parser.add_argument("--models-dir", type=Path, required=True, help="Ruta a A05_MODELOS_ENTRENADOS")
    args = parser.parse_args()

    eval_dir = args.models_dir / "evaluacion_final"
    out_dir = args.models_dir / "LATE_FUSION"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PASO 1: OBTENER PROBABILIDADES (SIN REENTRENAR)
    y_true, prob_t, prob_f = generar_probabilidades_test(args.dataset, args.models_dir)

    np.save(eval_dir / "y_test.npy", y_true)
    np.save(eval_dir / "prob_transformer.npy", prob_t)
    np.save(eval_dir / "prob_fft.npy", prob_f)
    logger.info(f"PROBABILIDADES GUARDADAS EN: {eval_dir}")

    # PASO 2: CALCULAR LAS 3 VARIANTES DE LATE FUSION (metricas out-of-fold)
    prob_geom = fusion_media_geometrica(prob_t, prob_f)
    prob_voto = fusion_voto_mayoritario(prob_t, prob_f)
    prob_meta = fusion_meta_clasificador(y_true, prob_t, prob_f)

    # PASO 3: ENTRENAR Y GUARDAR EL META-CLASIFICADOR FINAL PARA PRODUCCION
    # ADVERTENCIA: se entrena sobre el mismo test set usado para reportar
    # metricas (unico conjunto etiquetado disponible fuera de train/val).
    # Las metricas de "Meta_Clasificador" en el reporte son out-of-fold
    # (sin este sesgo), pero el .joblib guardado aqui SI ha visto estos
    # datos. Considerar reservar un conjunto adicional si se dispone de
    # mas pacientes en el futuro.
    meta_clf_path = out_dir / "meta_clasificador_late_fusion.joblib"
    entrenar_y_guardar_meta_clasificador(y_true, prob_t, prob_f, meta_clf_path)

    variantes = {
        "Media_Geometrica": prob_geom,
        "Voto_Mayoritario": prob_voto,
        "Meta_Clasificador": prob_meta,
    }

    reporte = []
    reporte.append({"Metodo": "Transformer (solo)", **calcular_metricas(y_true, prob_t)})
    reporte.append({"Metodo": "FFT (solo)", **calcular_metricas(y_true, prob_f)})
    for nombre, prob in variantes.items():
        reporte.append({"Metodo": nombre, **calcular_metricas(y_true, prob)})
        graficar_matriz_confusion(y_true, prob, f"LATE FUSION - {nombre}", out_dir / f"{nombre}_Matriz_Confusion.png")

    df_reporte = pd.DataFrame(reporte)
    df_reporte.to_csv(out_dir / "late_fusion_metricas.csv", index=False)

    print("\n" + "=" * 90)
    print("COMPARATIVA LATE FUSION (Transformer + FFT) — sin reentrenar")
    print("=" * 90)
    for _, row in df_reporte.iterrows():
        print(
            f"{row['Metodo']:22s} | AUC={row['AUC']:.4f} | Sens={row['Sensibilidad']:.2%} "
            f"| Spec={row['Especificidad']:.2%} | FPR={row['FPR']:.2%} | Brier={row['Brier']:.4f}"
        )
    print("=" * 90 + "\n")

    logger.info(f"RESULTADOS GUARDADOS EN: {out_dir}")


if __name__ == "__main__":
    main()
