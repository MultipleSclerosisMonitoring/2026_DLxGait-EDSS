# -*- coding: utf-8 -*-
"""
Modelo de trayectoria corporal (X, Y en metros) a partir de presion
plantar + IMU de ambos pies, reutilizando (fine-tuning) los pesos ya
entrenados de GaitTransformer como extractor de caracteristicas.

Arquitectura:
    presion+IMU Right+Left -> PSD combinado "Both" (348 dim, igual que el
    dataset original) -> GaitTransformer (pesos preentrenados, UNA sola
    instancia) -> feat (64 dim) -> cabezal de regresion (nuevo) -> (X, Y)

NOTA DE DISENO (revisado): se descarto el diseno inicial de DOS instancias
de GaitTransformer (una por pie, 174 features cada una) porque, al
inspeccionar dataset_jerarquico.hdf5 directamente, se confirmo que el
modelo original fue entrenado con un UNICO tensor "Both" de 348 features
por frame (ambos pies ya apilados/combinados antes de entrar al
Transformer), no con dos tensores separados de 174. Usar dos instancias
hubiera sido arquitectonicamente incompatible con los pesos preentrenados.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "A04_TRANSFORMER"))
from AA_TRANSFORMER_V1 import GaitTransformer, TransformerConfig


class TrajectoryModel(nn.Module):
    """
    Predice la posicion (X, Y) del cuerpo en metros locales, a partir de
    un espectrograma PSD combinado de ambos pies (348 features, igual
    formato que el dataset de clasificacion original), usando una rama
    GaitTransformer (fine-tuning) seguida de un cabezal de regresion nuevo.
    """

    def __init__(self, t_cfg: TransformerConfig, congelar_transformer: bool = False) -> None:
        """
        Inicializa el modelo con una rama GaitTransformer y un cabezal
        de regresion para (X, Y).

        :param t_cfg: Configuracion estructural del Transformer, debe
            coincidir con la usada para entrenar modelo_transformer.pth
            (input_dim=348, max_len=20, model_dim=64, etc., segun
            transformer_config.joblib).
        :param congelar_transformer: Si True, congela los pesos de la rama
            GaitTransformer y solo entrena el cabezal de regresion
            (fine-tuning parcial). Si False, todos los pesos son
            entrenables (fine-tuning completo).
        """
        super(TrajectoryModel, self).__init__()

        self.transformer = GaitTransformer(t_cfg)

        # DESCARTAR EL CLASIFICADOR ORIGINAL (REPOSO/MARCHA): solo se
        # reutiliza el encoder Transformer como extractor de features
        self.transformer.classifier = nn.Identity()

        if congelar_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.cabezal_regresion = nn.Sequential(
            nn.Linear(t_cfg.model_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # salida: (X, Y) en metros
        )

    def cargar_pesos_preentrenados(self, ruta_pth: Path, device: str) -> None:
        """
        Carga los pesos de modelo_transformer.pth en la rama Transformer.

        :param ruta_pth: Ruta al archivo modelo_transformer.pth.
        :param device: Dispositivo ("cuda" o "cpu").
        """
        state_dict = torch.load(ruta_pth, map_location=device)

        # EXCLUIR EL CLASIFICADOR ORIGINAL DEL STATE_DICT (ya no existe,
        # fue reemplazado por nn.Identity)
        state_dict_encoder = {
            k: v for k, v in state_dict.items() if not k.startswith("classifier.")
        }

        self.transformer.load_state_dict(state_dict_encoder, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.

        :param x: Tensor PSD combinado (ambos pies), forma (batch, seq_len, 348).
        :return: Tensor de posicion predicha (X, Y), forma (batch, 2).
        """
        feat = self.transformer(x)
        return self.cabezal_regresion(feat)


class VentanasSecuenciaDataset(Dataset):
    """
    Arma ventanas deslizantes (stride 1) de longitud max_len sobre un
    tensor PSD combinado, para alimentar al Transformer en secuencias
    en vez de frames individuales. La posicion objetivo de cada ventana
    es la del ULTIMO frame (el mas reciente), consistente con como
    GaitTransformer usa pos_embedding sobre toda la secuencia y agrega
    con mean() al final -- se elige el ultimo frame como referencia
    temporal de la prediccion, ya que es el instante mas cercano al
    "presente" de la ventana.

    Vive en trajectory_model.py (no en entrenar_trajectory_model.py) para
    que tanto el entrenamiento de un solo segmento como el multi-segmento
    puedan importarla del mismo lugar sin depender uno del otro (evita
    imports circulares entre los dos scripts de entrenamiento).
    """

    def __init__(self, x_psd: np.ndarray, y_pos: np.ndarray, max_len: int) -> None:
        """
        :param x_psd: Tensor PSD combinado, forma (n_frames, 348).
        :param y_pos: Posicion objetivo por frame, forma (n_frames, 2).
        :param max_len: Longitud de la ventana (debe coincidir con
            TransformerConfig.max_len del checkpoint preentrenado, 20).
        """
        self.max_len = max_len
        self.ventanas_x = []
        self.ventanas_y = []

        for i in range(len(x_psd) - max_len + 1):
            self.ventanas_x.append(x_psd[i:i + max_len])
            self.ventanas_y.append(y_pos[i + max_len - 1])  # ultimo frame de la ventana

        self.ventanas_x = np.array(self.ventanas_x, dtype=np.float32)
        self.ventanas_y = np.array(self.ventanas_y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ventanas_x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.ventanas_x[idx]),
            torch.from_numpy(self.ventanas_y[idx])
        )

