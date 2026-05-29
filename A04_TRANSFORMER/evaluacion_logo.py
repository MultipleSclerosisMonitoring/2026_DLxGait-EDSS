# -*- coding: utf-8 -*-
"""
Evaluación extra de modelos entrenados.

Este script:

- Carga modelos .pth ya entrenados
- Carga scaler existente
- Reutiliza arquitectura original
- Genera nuevos folds de evaluación
- Ejecuta SOLO inferencia

Compatible con:
- Transformer
- FFT
- Modelo Híbrido

Autor:
Jairo Eduardo Paez Leal
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, List

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn

from scipy.fft import rfft

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# =========================================================
# FFT
# =========================================================

class FFTProcessor:

    @staticmethod
    def get_fft_features(
        data: np.ndarray
    ) -> np.ndarray:

        data_fft = np.abs(
            rfft(data, axis=1)
        )

        return (
            data_fft / data.shape[1]
        ).astype(np.float32)

# =========================================================
# TRANSFORMER
# =========================================================

class GaitTransformer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        max_len: int,
        model_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ) -> None:

        super().__init__()

        self.embedding = nn.Linear(
            input_dim,
            model_dim
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_len,
                model_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = self.embedding(x)

        seq_len = x.size(1)

        x = x + self.pos_embedding[:, :seq_len, :]

        x = self.transformer(x)

        return self.classifier(
            x.mean(dim=1)
        )

# =========================================================
# FFT MODEL
# =========================================================

class FFTModel(nn.Module):

    def __init__(
        self,
        input_dim: int
    ) -> None:

        super().__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.classifier(x)

# =========================================================
# HYBRID MODEL
# =========================================================

class GaitHybridModel(nn.Module):

    def __init__(
        self,
        transformer_model: nn.Module,
        fft_input_dim: int,
        model_dim: int = 64
    ) -> None:

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

    def forward(
        self,
        x_t: torch.Tensor,
        x_f: torch.Tensor
    ) -> torch.Tensor:

        feat_t = self.transformer_branch(x_t)

        feat_f = self.fft_branch(x_f)

        return self.fusion_head(
            torch.cat(
                (feat_t, feat_f),
                dim=1
            )
        )

# =========================================================
# LOAD DATASET
# =========================================================

def load_h5_dataset(
    h5_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x_list = []
    y_list = []
    groups = []

    with h5py.File(h5_path, "r") as hf:

        for patient in hf.keys():

            for seg_chunk in hf[patient].keys():

                for foot in hf[patient][seg_chunk].keys():

                    ds = hf[patient][seg_chunk][foot]

                    x_list.append(ds[:])

                    y_list.append(ds.attrs["label"])

                    # SOLO PACIENTE
                    groups.append(patient)

    return (
        np.array(x_list),
        np.array(y_list),
        np.array(groups)
    )

# =========================================================
# THRESHOLD ROBUSTO
# =========================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:

    if len(np.unique(y_true)) < 2:

        logger.warning(
            "SOLO UNA CLASE EN FOLD -> threshold=0.5"
        )

        return 0.5

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob
    )

    idx = np.argmax(
        tpr - fpr
    )

    threshold = thresholds[idx]

    # CONTROL NUMERICO
    if not np.isfinite(threshold):

        logger.warning(
            "THRESHOLD INVALIDO -> 0.5"
        )

        threshold = 0.5

    return float(threshold)

# =========================================================
# EVALUACION ROBUSTA
# =========================================================

def evaluate_model(
    model,
    loader,
    device,
    hybrid=False
) -> Dict:

    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():

        for batch in loader:

            if hybrid:

                xt, xf, yb = batch

                out = model(
                    xt.to(device),
                    xf.to(device)
                )

            else:

                xb, yb = batch

                out = model(
                    xb.to(device)
                )

            probs = torch.softmax(
                out,
                dim=1
            )[:, 1]

            y_true.extend(
                yb.numpy()
            )

            y_prob.extend(
                probs.cpu().numpy()
            )

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # =========================================
    # VALIDACION CLASES
    # =========================================

    if len(np.unique(y_true)) < 2:

        logger.warning(
            "FOLD OMITIDO: UNA SOLA CLASE"
        )

        return None

    threshold = find_optimal_threshold(
        y_true,
        y_prob
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    return {
        "AUC": auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Threshold": threshold,
        "ConfusionMatrix": cm
    }

# =========================================================
# MAIN
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--models_dir",
        type=Path,
        default=Path(
            "A05_MODELOS_ENTRENADOS"
        )
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =====================================================
    # DATASET
    # =====================================================

    logger.info(
        "CARGANDO DATASET"
    )

    x_all, y_all, groups_all = load_h5_dataset(
        args.dataset
    )

    logger.info(
        f"Muestras: {len(x_all)}"
    )

    scaler: StandardScaler = joblib.load(
        args.models_dir /
        "scaler_gait.joblib"
    )

    x_scaled = scaler.transform(
        x_all.reshape(
            -1,
            x_all.shape[2]
        )
    ).reshape(x_all.shape)

    fft_proc = FFTProcessor()

    x_fft = fft_proc.get_fft_features(
        x_scaled
    )

    fft_dim = (
        x_fft.shape[1] *
        x_fft.shape[2]
    )

    input_dim = x_scaled.shape[2]

    seq_len = x_scaled.shape[1]

    # =====================================================
    # LOAD MODELS
    # =====================================================

    logger.info(
        "CARGANDO MODELOS"
    )

    transformer_model = GaitTransformer(
        input_dim=input_dim,
        max_len=seq_len
    )

    transformer_model.load_state_dict(
        torch.load(
            args.models_dir /
            "modelo_transformer.pth",
            map_location=device
        )
    )

    transformer_model.to(device)

    fft_model = FFTModel(
        fft_dim
    )

    fft_model.load_state_dict(
        torch.load(
            args.models_dir /
            "modelo_fft.pth",
            map_location=device
        )
    )

    fft_model.to(device)

    hybrid_model = GaitHybridModel(
        transformer_model=GaitTransformer(
            input_dim=input_dim,
            max_len=seq_len
        ),
        fft_input_dim=fft_dim
    )

    hybrid_model.load_state_dict(
        torch.load(
            args.models_dir /
            "modelo_hibrido.pth",
            map_location=device
        )
    )

    hybrid_model.to(device)

    logger.info(
        "MODELOS CARGADOS CORRECTAMENTE"
    )

    # =====================================================
    # LOGO
    # =====================================================

    logo = LeaveOneGroupOut()

    aucs_h = []

    valid_folds = 0

    logger.info(
        "INICIANDO EVALUACION LOGO"
    )

    for fold, (_, test_idx) in enumerate(

        logo.split(
            x_scaled,
            y_all,
            groups_all
        ),

        1
    ):

        logger.info(
            f"FOLD {fold}"
        )

        x_test = x_scaled[test_idx]

        y_test = y_all[test_idx]

        # =========================================
        # SALTAR FOLDS INVALIDOS
        # =========================================

        unique_classes = np.unique(y_test)

        if len(unique_classes) < 2:

            logger.warning(
                f"FOLD {fold} OMITIDO -> "
                f"UNA SOLA CLASE: {unique_classes}"
            )

            continue

        x_test_fft = x_fft[test_idx]

        hybrid_loader = DataLoader(

            TensorDataset(

                torch.from_numpy(
                    x_test
                ).float(),

                torch.from_numpy(
                    x_test_fft.reshape(
                        x_test.shape[0],
                        -1
                    )
                ).float(),

                torch.from_numpy(
                    y_test
                ).long()

            ),

            batch_size=64,
            shuffle=False
        )

        metrics = evaluate_model(
            hybrid_model,
            hybrid_loader,
            device,
            hybrid=True
        )

        if metrics is None:

            continue

        aucs_h.append(
            metrics["AUC"]
        )

        valid_folds += 1

        print("\n")
        print("=" * 60)
        print(f"FOLD {fold}")
        print("=" * 60)

        print(
            f"AUC: {metrics['AUC']:.4f}"
        )

        print(
            f"SENSITIVITY: "
            f"{metrics['Sensitivity']:.4f}"
        )

        print(
            f"SPECIFICITY: "
            f"{metrics['Specificity']:.4f}"
        )

        print(
            f"THRESHOLD: "
            f"{metrics['Threshold']:.4f}"
        )

        print("\nCONFUSION MATRIX:")

        print(
            metrics["ConfusionMatrix"]
        )

    # =====================================================
    # FINAL
    # =====================================================

    print("\n")
    print("=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)

    print(
        f"FOLDS VALIDOS: {valid_folds}"
    )

    if len(aucs_h) > 0:

        print(
            f"AUC PROMEDIO LOGO: "
            f"{np.mean(aucs_h):.4f}"
        )

        print(
            f"AUC STD LOGO: "
            f"{np.std(aucs_h):.4f}"
        )

        print(
            f"AUC MIN: "
            f"{np.min(aucs_h):.4f}"
        )

        print(
            f"AUC MAX: "
            f"{np.max(aucs_h):.4f}"
        )

    else:

        print(
            "NO EXISTEN FOLDS VALIDOS"
        )

# =========================================================

if __name__ == "__main__":

    main()