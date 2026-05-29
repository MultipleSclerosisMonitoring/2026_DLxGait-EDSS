# -*- coding: utf-8 -*-
"""
Evaluación global de modelos entrenados.

Calcula umbrales globales precisos agregando predicciones
Out-Of-Fold (OOF) para maximizar la generalización.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.fft import rfft
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# CONFIGURAR REGISTRO LOGS
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =========================================================
# DEFINIR PROCESADOR FFT
# =========================================================

class FFTProcessor:
    """Procesador estático de señales frecuenciales."""

    @staticmethod
    def get_fft_features(data: np.ndarray) -> np.ndarray:
        """
        Aplica transformada rápida Fourier.

        :param data: Matriz de entrada temporal.
        :return: Características en dominio frecuencial.
        """
        data_fft = np.abs(rfft(data, axis=1))
        return (data_fft / data.shape[1]).astype(np.float32)

# =========================================================
# DEFINIR REDES NEURONALES
# =========================================================

class GaitTransformer(nn.Module):
    """Modelo clasificador basado en atención."""

    def __init__(self, input_dim: int, max_len: int, model_dim: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 2) -> None:
        """Inicializa arquitectura del Transformer."""
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, model_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ejecuta pase frontal temporal."""
        x = self.embedding(x)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


class FFTModel(nn.Module):
    """Clasificador espectral directo."""

    def __init__(self, input_dim: int) -> None:
        """Inicializa clasificador de frecuencias."""
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ejecuta pase frontal frecuencial."""
        return self.classifier(x)


class GaitHybridModel(nn.Module):
    """Modelo de fusión multimodal."""

    def __init__(self, transformer_model: nn.Module, fft_input_dim: int, model_dim: int = 64) -> None:
        """Inicializa arquitectura combinada."""
        super().__init__()
        self.transformer_branch = transformer_model
        self.transformer_branch.classifier = nn.Identity()
        
        self.fft_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fft_input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(model_dim + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x_t: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        """Ejecuta pase multimodal conjunto."""
        feat_t = self.transformer_branch(x_t)
        feat_f = self.fft_branch(x_f)
        return self.fusion_head(torch.cat((feat_t, feat_f), dim=1))

# =========================================================
# GESTION DE DATOS
# =========================================================

def load_h5_dataset(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga registros del dataset binario.

    :param h5_path: Ruta del archivo HDF5.
    :return: Tensores X, Y, y Grupos.
    """
    x_list, y_list, groups = [], [], []

    with h5py.File(h5_path, "r") as hf:
        for patient in hf.keys():
            for seg_chunk in hf[patient].keys():
                for foot in hf[patient][seg_chunk].keys():
                    ds = hf[patient][seg_chunk][foot]
                    x_list.append(ds[:])
                    y_list.append(ds.attrs["label"])
                    groups.append(patient)

    return np.array(x_list), np.array(y_list), np.array(groups)

# =========================================================
# CALCULO METRICAS
# =========================================================

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Identifica límite óptimo global.

    :param y_true: Etiquetas reales matriz.
    :param y_prob: Probabilidades predichas vector.
    :return: Valor numérico del umbral.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    threshold = thresholds[idx]

    if not np.isfinite(threshold):
        return 0.5

    return float(threshold)


def extract_predictions(model: nn.Module, loader: DataLoader, device: str, 
                        is_hybrid: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera predicciones continuas batch.

    :param model: Red PyTorch activa.
    :param loader: Generador datos iterables.
    :param device: Hardware cálculo asignado.
    :param is_hybrid: Bandera modelo doble.
    :return: Vectores de etiquetas y probabilidades.
    """
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in loader:
            if is_hybrid:
                xt, xf, yb = batch
                out = model(xt.to(device), xf.to(device))
            else:
                xb, yb = batch
                out = model(xb.to(device))

            probs = torch.softmax(out, dim=1)[:, 1]
            y_true.extend(yb.numpy())
            y_prob.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_prob)

# =========================================================
# FLUJO PRINCIPAL EJECUCION
# =========================================================

def main() -> None:
    """Función constructora orquestadora."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models_dir", type=Path, default=Path("A05_MODELOS_ENTRENADOS"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CARGAR DATOS SISTEMA
    x_all, y_all, groups_all = load_h5_dataset(args.dataset)
    logger.info(f"DATASET CARGADO: {len(x_all)} MUESTRAS")

    # APLICAR ESCALADOR ESTATICO
    scaler: StandardScaler = joblib.load(args.models_dir / "scaler_gait.joblib")
    x_scaled = scaler.transform(x_all.reshape(-1, x_all.shape[2])).reshape(x_all.shape)

    # EXTRAER RASGOS FRECUENCIALES
    fft_proc = FFTProcessor()
    x_fft = fft_proc.get_fft_features(x_scaled)
    fft_dim = x_fft.shape[1] * x_fft.shape[2]

    # INICIALIZAR REDES CARGADAS
    fft_model = FFTModel(fft_dim)
    fft_model.load_state_dict(torch.load(args.models_dir / "modelo_fft.pth", map_location=device))
    fft_model.to(device)

    # CONFIGURAR PARTICIONADOR LOGO
    logo = LeaveOneGroupOut()
    global_y_true_fft, global_y_prob_fft = [], []
    valid_folds = 0

    logger.info("INICIANDO EXTRACCION OOF (FFT)")

    # ITERAR EVALUACION CRUZADA
    for fold, (_, test_idx) in enumerate(logo.split(x_scaled, y_all, groups_all), 1):
        y_test = y_all[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        x_test_fft = x_fft[test_idx]
        
        # PREPARAR DATASET INDIVIDUAL
        fft_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_test_fft.reshape(x_test_fft.shape[0], -1)).float(),
                torch.from_numpy(y_test).long()
            ),
            batch_size=64, shuffle=False
        )

        # CAPTURAR PREDICCIONES PARCIALES
        y_t, y_p = extract_predictions(fft_model, fft_loader, device, is_hybrid=False)
        global_y_true_fft.extend(y_t)
        global_y_prob_fft.extend(y_p)
        
        valid_folds += 1
        auc = roc_auc_score(y_t, y_p)
        logger.info(f"FOLD {fold:02d} | AUC PARCIAL: {auc:.4f}")

    # CALCULAR UMBRAL GLOBAL
    y_true_all = np.array(global_y_true_fft)
    y_prob_all = np.array(global_y_prob_fft)
    
    fft_threshold = find_optimal_threshold(y_true_all, y_prob_all)
    
    # GUARDAR ARTEFACTO DISCO
    joblib.dump(fft_threshold, args.models_dir / "optimal_threshold_fft.joblib")

    # MOSTRAR RESUMEN LIMPIO
    y_pred_all = (y_prob_all >= fft_threshold).astype(int)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc_global = roc_auc_score(y_true_all, y_prob_all)

    print("\n" + "=" * 50)
    print("METRICAS GLOBALES MODELO FFT (LOGO OOF)")
    print("=" * 50)
    print(f"FOLDS VALIDOS EVALUADOS: {valid_folds}")
    print(f"UMBRAL OPTIMO CALCULADO: {fft_threshold:.6f}")
    print(f"AUC GLOBAL AGREGADO:     {auc_global:.4f}")
    print(f"SENSIBILIDAD GENERAL:    {sens:.4f}")
    print(f"ESPECIFICIDAD GENERAL:   {spec:.4f}")
    print("-" * 50)
    print("MATRIZ DE CONFUSION:")
    print(f"TN: {tn:5d} | FP: {fp:5d}")
    print(f"FN: {fn:5d} | TP: {tp:5d}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
