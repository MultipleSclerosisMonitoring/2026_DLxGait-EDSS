# -*- coding: utf-8 -*-
"""
Script de evaluacion para modelos de marcha. Posterior al entrenamiento.
Carga modelos preentrenados, calcula metricas y genera reportes.
Evaluación con Leave-One-Patient-Out (LOPO).
"""

from __future__ import annotations
import h5py
import numpy as np
import argparse
import logging
import os
import copy
import gc
from pathlib import Path
from typing import Tuple, List, Dict, Any
from pydantic import BaseModel, PositiveInt, confloat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, brier_score_loss, ConfusionMatrixDisplay,
    balanced_accuracy_score, average_precision_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import rfft
import joblib
import pandas as pd

# CONFIGURAR LOGGING CENTRAL
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FIJAR SEMILLAS GLOBALES
def set_seed(seed: int = 13) -> None:
    """
    Fija las semillas para garantizar reproducibilidad.

    :param seed: Valor de la semilla.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Genera muestreo balanceado para compensar desbalance reposo/marcha en cada fold."""
    counts = np.bincount(labels.astype(int), minlength=2)
    class_weights = 1.0 / np.where(counts > 0, counts, 1)
    sample_weights = class_weights[labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True
    )

# CONFIGURACION DE RUTAS
class EvalConfig(BaseModel):
    """
    Contenedor Pydantic para rutas de evaluacion.
    """
    dataset_path: Path
    model_dir: Path

class TrainConfig(BaseModel):
    """Hiperparametros de reentrenamiento usados en cada fold LOPO."""
    batch_size: PositiveInt = 64
    epochs: PositiveInt = 50
    lr: confloat(gt=0) = 0.001
    patience: PositiveInt = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# CARGADOR DE DATOS
class GaitEvalLoader:
    """
    Carga y escala los conjuntos de datos biologicos.
    """
    def __init__(self, config: EvalConfig) -> None:
        """
        Inicializa el cargador con rutas convalidadas.

        :param config: Objeto de configuracion.
        """
        self.config = config
        self.scaler = joblib.load(self.config.model_dir / "scaler_gait.joblib")

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extrae y normaliza el conjunto de test.

        :return: Tupla con matrices de tiempo, etiquetas y grupos.
        """
        x_raw, groups, labels = self.get_all_raw_data()
        test_idx = np.load(self.config.model_dir / "test_idx.npy")
        x_test_raw = x_raw[test_idx]
        y_test = labels[test_idx]
        x_test = self._scale_data(x_test_raw)
        return x_test.astype(np.float32), y_test.astype(np.float32), groups[test_idx]

    def _scale_data(self, data: np.ndarray) -> np.ndarray:
        """
        Escala los datos bidimensionales intermedios.

        :param data: Matriz de entrada tridimensional.
        :return: Matriz escalada.
        """
        n_samples, n_steps, n_features = data.shape
        flat_data = data.reshape(-1, n_features)
        scaled = self.scaler.transform(flat_data)
        return scaled.reshape(n_samples, n_steps, n_features)

    def get_all_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga todo el contenido jerarquico del HDF5.

        :return: Estructuras crudas del dataset.
        """
        x_list, groups, labels = [], [], []
        with h5py.File(self.config.dataset_path, "r") as hf:
            for patient in hf.keys():
                for seg_chunk in hf[patient].keys():
                    for pie in hf[patient][seg_chunk].keys():
                        dataset = hf[patient][seg_chunk][pie]
                        x_list.append(dataset[:])
                        labels.append(dataset.attrs["label"])
                        groups.append(patient)
        return np.array(x_list), np.array(groups), np.array(labels)

# CONFIGURACION DE ENTRADA
class StandardDataset(torch.utils.data.Dataset):
    """
    Dataset general Pytorch para tensores simples.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

class MultiModalDataset(torch.utils.data.Dataset):
    """
    Dataset multimodal para procesamiento hibrido.
    """
    def __init__(self, x_time: np.ndarray, x_fft: np.ndarray, y: np.ndarray) -> None:
        self.x_time = torch.as_tensor(x_time, dtype=torch.float32)
        self.x_fft = torch.as_tensor(x_fft, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_time[idx], self.x_fft[idx], self.y[idx]

# CONFIGURACION ARQUITECTURA TRANSFORMER
class TransformerConfig(BaseModel):
    """
    Parametros estructurales de la red Transformer.
    """
    input_dim: PositiveInt
    max_len: PositiveInt
    model_dim: PositiveInt = 64
    nhead: PositiveInt = 4
    num_layers: PositiveInt = 2
    dropout: confloat(ge=0, le=0.5) = 0.3
    num_classes: PositiveInt = 2

class GaitTransformer(nn.Module):
    """
    Red Transformer para clasificar series temporales de marcha.
    """
    def __init__(self, config: TransformerConfig) -> None:
        super(GaitTransformer, self).__init__()
        self.embedding = nn.Linear(config.input_dim, config.model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_len, config.model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))

# PROCESADOR DE FRECUENCIA
class FFTProcessor:
    """
    Calcula transformadas de Fourier sobre matrices.
    """
    @staticmethod
    def get_fft_features(data: np.ndarray) -> np.ndarray:
        data_f32 = data.astype(np.float32)
        data_fft = np.abs(rfft(data_f32, axis=1, workers=1)).astype(np.float32)
        return (data_fft / data.shape[1]).astype(np.float32)

class FFTModel(nn.Module):
    """
    Red densa para clasificar espectrogramas de frecuencia.
    """
    def __init__(self, input_dim: int) -> None:
        super(FFTModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class GaitHybridModel(nn.Module):
    """
    Arquitectura multimodal con fusion de caracteristicas tardias.
    """
    def __init__(self, t_cfg: TransformerConfig, fft_input_dim: int) -> None:
        super(GaitHybridModel, self).__init__()
        self.transformer_branch = GaitTransformer(t_cfg)
        self.transformer_branch.classifier = nn.Identity()

        self.fft_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fft_input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(t_cfg.model_dim + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x_t: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        feat_t = self.transformer_branch(x_t)
        feat_f = self.fft_branch(x_f)
        return self.fusion_head(torch.cat((feat_t, feat_f), dim=1))

class GaitTrainer:
    """Bucle de entrenamiento con early stopping para cada fold LOPO."""
    def __init__(self, model: nn.Module, config: TrainConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.best_state = None

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        best_auc, no_improve = 0.0, 0
        for epoch in range(self.config.epochs):
            self.model.train()
            for xt, xf, yb in train_loader:
                self.optimizer.zero_grad()
                out = self.model(xt.to(self.config.device), xf.to(self.config.device))
                loss = self.criterion(out, yb.to(self.config.device))
                loss.backward()
                self.optimizer.step()

            val_auc = self._evaluate_auc(val_loader)
            self.scheduler.step(val_auc)

            if val_auc > best_auc:
                best_auc, no_improve = val_auc, 0
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                no_improve += 1
            if no_improve >= self.config.patience:
                break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self.model

    def _evaluate_auc(self, loader: DataLoader) -> float:
        self.model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for xt, xf, yb in loader:
                out = self.model(xt.to(self.config.device), xf.to(self.config.device))
                y_prob.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                y_true.extend(yb.numpy())
        return roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0

# MATRICES DE CALIBRACION
def calibration_metrics(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcula el error de Brier y los coeficientes de Platt.

    :param y_true: Etiquetas reales.
    :param probs: Probabilidades predichas.
    :return: Brier score, intercepto y pendiente.
    """
    brier = brier_score_loss(y_true, probs)
    logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / (1 - np.clip(probs, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    lr = LogisticRegression(penalty=None).fit(logits, y_true)
    return brier, float(lr.intercept_[0]), float(lr.coef_[0][0])

# EVALUADOR DE MODELOS
class GaitEvaluator:
    """
    Extrae predicciones e implementa graficas diagnosticas.
    """
    def __init__(self, model: nn.Module, device: str) -> None:
        self.model = model.to(device)
        self.device = device

    def extract_probabilities(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta el bucle forward para extraer probabilidades.
        """
        self.model.eval()
        probs, y_true = [], []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    out = self.model(batch[0].to(self.device), batch[1].to(self.device))
                    yb = batch[2]
                else:
                    out = self.model(batch[0].to(self.device))
                    yb = batch[1]
                probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                y_true.extend(yb.numpy())
        return np.array(y_true), np.array(probs)

    def compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float, title: str, save_dir: Path) -> Dict[str, Any]:
        """
        Calcula metricas de clasificacion y exporta curvas ROC.
        """
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        auc_score = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0

        # GUARDAR MATRIZ CONFUSION
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['REPOSO', 'MARCHA'],
                    yticklabels=['REPOSO', 'MARCHA'])
        plt.title(f'{title}\nAUC: {auc_score:.4f} | Umbral: {threshold:.2f}')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        plt.savefig(save_dir / f"{title.replace(' ', '_')}_Matriz_Confusion.png", dpi=300, bbox_inches='tight')
        plt.close('all')

        return {
            "Sensibilidad": sensitivity,
            "Especificidad": specificity,
            "FPR": fpr_val,
            "AUC": auc_score,
            "Threshold": threshold
        }

# =============================================================================
# COMPARATIVA GENERAL: resumen legible + graficas de los 5 esquemas LOPO
# =============================================================================

# Parametros entrenables de cada modelo, calculados una sola vez (no
# dependen del dataset ni de la corrida). Usados como proxy de "costo
# computacional" en la grafica de efectividad vs costo. Late Fusion no
# entrena una red nueva propia (combina Transformer + FFT ya entrenados
# por separado), asi que su costo se reporta como la suma de ambos.
PARAMETROS_MODELOS = {
    "Transformer": 586_178,
    "FFT": 245_443,
    "Hibrido": 649_907,
}
PARAMETROS_MODELOS["Late Fusion (Transformer+FFT)"] = (
    PARAMETROS_MODELOS["Transformer"] + PARAMETROS_MODELOS["FFT"]
)

NOMBRES_LEGIBLES_ESQUEMA = {
    "hibrido": ("Híbrido (Early Fusion)", PARAMETROS_MODELOS["Hibrido"]),
    "fft": ("FFT (solo)", PARAMETROS_MODELOS["FFT"]),
    "latefusion_media_geometrica": ("Late Fusion — Media Geométrica", PARAMETROS_MODELOS["Late Fusion (Transformer+FFT)"]),
    "latefusion_voto_mayoritario": ("Late Fusion — Voto Mayoritario", PARAMETROS_MODELOS["Late Fusion (Transformer+FFT)"]),
    "latefusion_meta_clasificador": ("Late Fusion — Meta-Clasificador", PARAMETROS_MODELOS["Late Fusion (Transformer+FFT)"]),
}


def generar_comparativa_general(
    resumenes: Dict[str, Dict[str, Any]], comparativa_dir: Path
) -> None:
    """
    Genera la carpeta "comparativa" dentro de LOPO/: dos archivos de
    texto legibles (resumen_general.txt y detalles_por_modelo.txt) y dos
    graficas (efectividad y efectividad-vs-costo) que resumen los 5
    esquemas evaluados, sin repetir el detalle fold-por-fold que ya vive
    en cada carpeta de modelo individual.

    :param resumenes: Diccionario {prefijo: resumen_dict}, donde cada
        resumen_dict es el diccionario devuelto por _finalizar_lopo.
    :param comparativa_dir: Carpeta LOPO/comparativa/ (se crea si no existe).
    """
    graficas_dir = comparativa_dir / "graficas"
    comparativa_dir.mkdir(parents=True, exist_ok=True)
    graficas_dir.mkdir(parents=True, exist_ok=True)

    filas = []
    for prefijo, resumen in resumenes.items():
        if not resumen:
            continue
        nombre_legible, n_params = NOMBRES_LEGIBLES_ESQUEMA.get(prefijo, (prefijo, None))
        filas.append({
            "prefijo": prefijo,
            "nombre": nombre_legible,
            "n_parametros": n_params,
            **resumen
        })

    if not filas:
        logger.warning("COMPARATIVA GENERAL: SIN RESULTADOS PARA CONSOLIDAR")
        return

    df = pd.DataFrame(filas).sort_values("auc_global", ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------------
    # resumen_general.txt: lo mas sustancial, facil de leer de un vistazo
    # -------------------------------------------------------------------
    lineas = []
    lineas.append("RESUMEN GENERAL — COMPARATIVA DE MODELOS (LOPO)")
    lineas.append(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lineas.append("=" * 92)
    lineas.append("")
    lineas.append(f"{'Modelo':32s} {'AUC':>7s} {'PR-AUC':>7s} {'BalAcc':>7s} {'MCC':>7s} {'Params':>10s}")
    lineas.append("-" * 92)
    for _, row in df.iterrows():
        params_str = f"{row['n_parametros']:,}" if row["n_parametros"] else "N/D"
        lineas.append(
            f"{row['nombre']:32s} {row['auc_global']:7.4f} {row['pr_auc_global']:7.4f} "
            f"{row['balanced_accuracy_global']:7.4f} {row['mcc_global']:7.4f} {params_str:>10s}"
        )
    lineas.append("")

    mejor = df.iloc[0]
    lineas.append(f"Mejor AUC global: {mejor['nombre']} ({mejor['auc_global']:.4f})")

    mas_barato = df.loc[df["n_parametros"].idxmin()] if df["n_parametros"].notna().any() else None
    if mas_barato is not None:
        lineas.append(
            f"Modelo mas ligero: {mas_barato['nombre']} "
            f"({mas_barato['n_parametros']:,} parametros, AUC={mas_barato['auc_global']:.4f})"
        )

    lineas.append("")
    lineas.append("Consistencia entre pacientes (menor STD = mas parejo, menos riesgo de")
    lineas.append("fallar sorpresivamente en un paciente nuevo):")
    df_std = df.sort_values("auc_std_folds")
    for _, row in df_std.iterrows():
        lineas.append(f"  {row['nombre']:32s} STD={row['auc_std_folds']:.4f}")

    lineas.append("")
    lineas.append("Peor paciente por modelo (el mas dificil de clasificar en cada esquema):")
    for _, row in df.iterrows():
        if row["peor_paciente"]:
            lineas.append(
                f"  {row['nombre']:32s} -> {row['peor_paciente']} (AUC={row['peor_auc_fold']:.4f})"
            )

    with open(comparativa_dir / "resumen_general.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    logger.info("resumen_general.txt GENERADO")

    # -------------------------------------------------------------------
    # detalles_por_modelo.txt: un poco mas de detalle por modelo, pero
    # SIN repetir la tabla fold-por-fold completa (esa ya vive en cada
    # carpeta LOPO/<modelo>/lopo_metricas_por_paciente.csv)
    # -------------------------------------------------------------------
    lineas_detalle = []
    lineas_detalle.append("DETALLES POR MODELO — COMPARATIVA LOPO")
    lineas_detalle.append(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lineas_detalle.append("=" * 92)

    for _, row in df.iterrows():
        lineas_detalle.append("")
        lineas_detalle.append(f"{row['nombre']}")
        lineas_detalle.append("-" * 92)
        lineas_detalle.append(f"  AUC global (todas las predicciones concatenadas): {row['auc_global']:.4f}")
        lineas_detalle.append(f"  AUC promedio por fold: {row['auc_promedio_folds']:.4f} (STD: {row['auc_std_folds']:.4f})")
        lineas_detalle.append(f"  PR-AUC: {row['pr_auc_global']:.4f}  |  Balanced Accuracy: {row['balanced_accuracy_global']:.4f}  |  MCC: {row['mcc_global']:.4f}")
        lineas_detalle.append(f"  Brier score (calibracion): {row['brier']:.4f}")
        lineas_detalle.append(f"  Folds evaluados: {row['n_folds']}")
        if row["peor_paciente"]:
            lineas_detalle.append(f"  Paciente mas dificil: {row['peor_paciente']} (AUC={row['peor_auc_fold']:.4f})")
        if row["n_parametros"]:
            lineas_detalle.append(f"  Parametros entrenables: {row['n_parametros']:,}")
        lineas_detalle.append(
            f"  Detalle completo (por paciente): ver LOPO/{row['prefijo']}/lopo_metricas_por_paciente.csv"
        )

    with open(comparativa_dir / "detalles_por_modelo.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_detalle))
    logger.info("detalles_por_modelo.txt GENERADO")

    # -------------------------------------------------------------------
    # Grafica 1: efectividad (AUC, PR-AUC, BalAcc, MCC) por modelo
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    metricas_plot = ["auc_global", "pr_auc_global", "balanced_accuracy_global", "mcc_global"]
    etiquetas_metricas = ["AUC", "PR-AUC", "Balanced Acc.", "MCC"]
    x = np.arange(len(df))
    ancho = 0.2

    for i, (metrica, etiqueta) in enumerate(zip(metricas_plot, etiquetas_metricas)):
        ax.bar(x + i * ancho, df[metrica], ancho, label=etiqueta)

    ax.set_xticks(x + ancho * 1.5)
    ax.set_xticklabels(df["nombre"], rotation=25, ha="right")
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Comparativa de efectividad — LOPO (todos los esquemas)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(graficas_dir / "comparativa_efectividad.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------
    # Grafica 2: efectividad (AUC) vs costo (parametros entrenables)
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    df_con_params = df.dropna(subset=["n_parametros"])
    ax.scatter(df_con_params["n_parametros"], df_con_params["auc_global"], s=120, alpha=0.8)
    for _, row in df_con_params.iterrows():
        ax.annotate(
            row["nombre"], (row["n_parametros"], row["auc_global"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8
        )
    ax.set_xlabel("Parámetros entrenables (costo computacional)")
    ax.set_ylabel("AUC global (LOPO)")
    ax.set_title("Efectividad vs. costo computacional")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(graficas_dir / "comparativa_efectividad_vs_costo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("GRAFICAS COMPARATIVAS GENERADAS (efectividad, efectividad_vs_costo)")



# CONTEXTO DE EJECUCION GENERAL
def main() -> None:
    set_seed(13)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--run-lopo", action="store_true")
    args = parser.parse_args()

    MODELS_DIR = args.models

    # -------------------------------------------------------------------
    # NUEVA ESTRUCTURA DE CARPETAS
    #
    # A05_MODELOS_ENTRENADOS/
    # ├── evaluacion_superficial/   (test independiente, 1 solo paciente,
    # │   ├── graficas/              metrica secundaria/no representativa
    # │   ├── metricas_basicas.txt   -- se guardan aqui las graficas y
    # │   ├── calibracion_eval.csv   metricas que antes vivian sueltas
    # │   └── reporte_final_evaluacion.txt en EVAL_GENERAL/)
    # └── LOPO/                     (metrica principal de referencia)
    #     ├── hibrido/{graficas/, *.csv}
    #     ├── fft/{graficas/, *.csv}
    #     ├── latefusion_media_geometrica/{graficas/, *.csv}
    #     ├── latefusion_voto_mayoritario/{graficas/, *.csv}
    #     ├── latefusion_meta_clasificador/{graficas/, *.csv}
    #     └── comparativa/
    #         ├── resumen_general.txt
    #         ├── detalles_por_modelo.txt
    #         └── graficas/
    # -------------------------------------------------------------------
    SUPERFICIAL_DIR = MODELS_DIR / "evaluacion_superficial"
    SUPERFICIAL_GRAFICAS_DIR = SUPERFICIAL_DIR / "graficas"
    SUPERFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    SUPERFICIAL_GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)

    LOPO_DIR = MODELS_DIR / "LOPO"
    LOPO_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("INICIANDO EVALUACION SUPERFICIAL (TEST INDEPENDIENTE)")

    cfg_data = EvalConfig(dataset_path=args.dataset, model_dir=MODELS_DIR)
    loader = GaitEvalLoader(cfg_data)

    # CARGAR ARREGLOS DE EVALUACION
    x_test, y_test, _ = loader.get_test_data()
    fft_proc = FFTProcessor()
    x_test_fft = fft_proc.get_fft_features(x_test)
    REAL_FFT_DIM = x_test_fft.shape[1] * x_test_fft.shape[2]

    # RECONSTRUIR DATALOADERS
    ts_l = DataLoader(StandardDataset(x_test, y_test), batch_size=64)
    fft_ts_l = DataLoader(StandardDataset(x_test_fft.reshape(x_test.shape[0], -1), y_test), batch_size=64)
    h_ts = DataLoader(MultiModalDataset(x_test, x_test_fft.reshape(x_test.shape[0], -1), y_test), batch_size=64)

    config_dict = joblib.load(MODELS_DIR / "transformer_config.joblib")
    m1_cfg = TransformerConfig(**config_dict)

    # RECONSTRUIR ARQUITECTURAS
    model_time = GaitTransformer(m1_cfg)
    model_fft = FFTModel(REAL_FFT_DIM)
    model_hybrid = GaitHybridModel(m1_cfg, REAL_FFT_DIM)

    # ACOPLAR PESOS CONGELADOS
    model_time.load_state_dict(torch.load(MODELS_DIR / "modelo_transformer.pth", map_location=DEVICE))
    model_fft.load_state_dict(torch.load(MODELS_DIR / "modelo_fft.pth", map_location=DEVICE))
    model_hybrid.load_state_dict(torch.load(MODELS_DIR / "modelo_hibrido.pth", map_location=DEVICE))

    # FORZAR UMBRAL DE CORTE NEUTRAL
    opt_thresh_t = 0.50
    opt_thresh_f = 0.50
    opt_thresh_h = 0.50
    logger.info("UMBRALES FORZADOS MANUALMENTE A 0.50")

    evaluator_t = GaitEvaluator(model_time, DEVICE)
    evaluator_f = GaitEvaluator(model_fft, DEVICE)
    evaluator_h = GaitEvaluator(model_hybrid, DEVICE)

    y_true_t, prob_t = evaluator_t.extract_probabilities(ts_l)
    y_true_f, prob_f = evaluator_f.extract_probabilities(fft_ts_l)
    y_true_h, prob_h = evaluator_h.extract_probabilities(h_ts)

    # EVALUAR METRICAS FINALES (graficas van a evaluacion_superficial/graficas/)
    metrics_t = evaluator_t.compute_metrics(y_true_t, prob_t, opt_thresh_t, "MODELO TRANSFORMER", SUPERFICIAL_GRAFICAS_DIR)
    metrics_f = evaluator_f.compute_metrics(y_true_f, prob_f, opt_thresh_f, "MODELO FFT", SUPERFICIAL_GRAFICAS_DIR)
    metrics_h = evaluator_h.compute_metrics(y_true_h, prob_h, opt_thresh_h, "MODELO HIBRIDO FINAL", SUPERFICIAL_GRAFICAS_DIR)

    # CALCULAR DISCRIMINACION Y CALIBRACION (+ Balanced Accuracy y MCC,
    # metricas adicionales pedidas para que este resumen tambien sea
    # informativo pese a ser una evaluacion superficial/no representativa)
    reporte = []
    for nombre, yt, yp in [("Transformer", y_true_t, prob_t), ("FFT", y_true_f, prob_f), ("Hibrido", y_true_h, prob_h)]:
        b, i, s = calibration_metrics(yt, yp)
        y_pred_05 = (yp >= 0.5).astype(int)
        bal_acc = balanced_accuracy_score(yt, y_pred_05)
        mcc = matthews_corrcoef(yt, y_pred_05)
        reporte.append({
            "Modelo": nombre, "Brier": b, "Intercepto": i, "Pendiente": s,
            "Balanced_Accuracy": bal_acc, "MCC": mcc
        })

    # EXPORTAR METRICAS CALIBRACION
    pd.DataFrame(reporte).to_csv(SUPERFICIAL_DIR / "calibracion_eval.csv", index=False)

    # EXPORTAR METRICAS TEXTO
    pd.DataFrame([
        {"Modelo": "Transformer", **metrics_t},
        {"Modelo": "FFT", **metrics_f},
        {"Modelo": "Hibrido", **metrics_h},
    ]).to_csv(SUPERFICIAL_DIR / "metricas_basicas.txt", sep='\t', index=False)

    # CONSTRUIR TABLA ASCII REPORTE
    lineas_reporte = []
    lineas_reporte.append("\n" + "=" * 95)
    lineas_reporte.append("RESULTADOS FINALES (TEST INDEPENDIENTE) — ADVERTENCIA: metrica")
    lineas_reporte.append("secundaria/no representativa (test = 1 solo paciente). La metrica")
    lineas_reporte.append("principal de referencia es LOPO (carpeta LOPO/comparativa/).")
    lineas_reporte.append("=" * 95)
    for modelo, met, cal in zip(["TRANSFORMER", "FFT", "HIBRIDO"], [metrics_t, metrics_f, metrics_h], reporte):
        lineas_reporte.append(
            f"{modelo:12s}"
            f" | AUC={met['AUC']:.4f}"
            f" | Sens={met['Sensibilidad']:.2%}"
            f" | Spec={met['Especificidad']:.2%}"
            f" | FPR={met['FPR']:.2%}"
            f" | BalAcc={cal['Balanced_Accuracy']:.4f}"
            f" | MCC={cal['MCC']:.4f}"
            f" | Brier={cal['Brier']:.4f}"
            f" | Int={cal['Intercepto']:.3f}"
            f" | Pend={cal['Pendiente']:.3f}"
        )
    lineas_reporte.append("=" * 95)

    # IMPRIMIR E INYECTAR REPORTE TXT
    texto_final = "\n".join(lineas_reporte)
    print(texto_final)

    with open(SUPERFICIAL_DIR / "reporte_final_evaluacion.txt", "w", encoding="utf-8") as f_txt:
        f_txt.write("REPORTE OPERACIONAL DE INFERENCIA BIOMEDICA (EVALUACION SUPERFICIAL)\n")
        f_txt.write(texto_final)
    logger.info("EVALUACION SUPERFICIAL COMPLETADA")

    # -------------------------------------------------------------------
    # LOPO (metrica principal): cada esquema guarda en su propia carpeta
    # dentro de LOPO/, y al final se genera la comparativa general.
    # -------------------------------------------------------------------
    if args.run_lopo:
        x_all, groups_all, y_all = loader.get_all_raw_data()
        train_cfg = TrainConfig(device=DEVICE)

        resumenes: Dict[str, Dict[str, Any]] = {}
        resumenes["hibrido"] = run_lopo_evaluation(x_all, y_all, groups_all, m1_cfg, train_cfg, LOPO_DIR)
        resumenes["fft"] = run_lopo_evaluation_fft(x_all, y_all, groups_all, train_cfg, LOPO_DIR)
        resumenes_lf = run_lopo_late_fusion(x_all, y_all, groups_all, m1_cfg, train_cfg, LOPO_DIR)
        for nombre_variante, resumen_variante in resumenes_lf.items():
            resumenes[f"latefusion_{nombre_variante}"] = resumen_variante

        generar_comparativa_general(resumenes, LOPO_DIR / "comparativa")
    else:
        logger.info("EVALUACION LOPO OMITIDA: Use --run-lopo para lanzar validacion cruzada paciente a paciente.")

# =============================================================================
# SECCION EXCLUSIVA: LOGICA PARA LEAVE-ONE-PATIENT-OUT (LOPO)
# =============================================================================

def run_lopo_evaluation(
    x_all: np.ndarray,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    t_cfg: TransformerConfig,
    train_cfg: TrainConfig,
    lopo_dir: Path
) -> Dict[str, Any]:
    """
    LOPO real: reentrena el modelo hibrido desde cero en cada fold,
    dejando siempre un paciente distinto fuera de train/val.

    :param x_all: Tensores crudos de todos los pacientes, forma (N, steps, features).
    :param y_all: Etiquetas binarias (reposo/marcha) por muestra.
    :param groups_all: Identificador de paciente por muestra, usado por LeaveOneGroupOut.
    :param t_cfg: Configuracion estructural del Transformer (rama temporal del hibrido).
    :param train_cfg: Hiperparametros de entrenamiento (batch_size, epochs, lr, patience, device).
    :param eval_dir: Directorio donde se exportan CSVs y la matriz de confusion global.
    """
    logo = LeaveOneGroupOut()
    fft_proc = FFTProcessor()
    unique_p = np.unique(groups_all)

    all_y_true: List[int] = []
    all_y_prob: List[float] = []
    fold_scores: List[float] = []
    patient_ids: List[str] = []
    fold_metricas: List[Dict[str, float]] = []

    logger.info(
        f"INICIANDO LOPO REAL HIBRIDO (REENTRENAMIENTO POR FOLD) — {len(unique_p)} PACIENTES"
    )

    for fold, (trainval_idx, test_idx) in enumerate(
        logo.split(x_all, y_all, groups_all), 1
    ):

        test_patient = groups_all[test_idx][0]

        trainval_patients = np.unique(groups_all[trainval_idx])
        val_patient = trainval_patients[-1]

        val_mask = groups_all[trainval_idx] == val_patient
        train_idx_f = trainval_idx[~val_mask]
        val_idx_f = trainval_idx[val_mask]

        x_tr, y_tr = x_all[train_idx_f], y_all[train_idx_f]
        x_val, y_val = x_all[val_idx_f], y_all[val_idx_f]
        x_ts, y_ts = x_all[test_idx], y_all[test_idx]

        # Entrenamiento requiere ambas clases (no se puede entrenar un
        # clasificador binario con una sola clase en train). Un paciente
        # de TEST monoclase (ej. 02548893X-118, MGM-202406-79) YA NO se
        # descarta con continue: se entrena igual el fold y se evalua,
        # registrando NaN en AUC/PR-AUC (no calculables con una sola
        # clase real) pero calculando balanced_accuracy y MCC, que si
        # aportan informacion util sobre ese paciente.
        if len(np.unique(y_tr)) < 2:
            logger.warning(
                f"FOLD {fold} ({test_patient}) OMITIDO — TRAIN CON CLASE UNICA"
            )
            continue

        # ---------------------------------------------------------------------
        # Escalado (fit unicamente con entrenamiento)
        # ---------------------------------------------------------------------
        sc = StandardScaler()

        x_tr_s = sc.fit_transform(
            x_tr.reshape(-1, x_tr.shape[2])
        ).reshape(x_tr.shape).astype(np.float32)

        x_val_s = sc.transform(
            x_val.reshape(-1, x_val.shape[2])
        ).reshape(x_val.shape).astype(np.float32)

        x_ts_s = sc.transform(
            x_ts.reshape(-1, x_ts.shape[2])
        ).reshape(x_ts.shape).astype(np.float32)

        # ---------------------------------------------------------------------
        # FFT
        # ---------------------------------------------------------------------
        x_tr_fft = fft_proc.get_fft_features(x_tr_s).reshape(x_tr_s.shape[0], -1)
        x_val_fft = fft_proc.get_fft_features(x_val_s).reshape(x_val_s.shape[0], -1)
        x_ts_fft = fft_proc.get_fft_features(x_ts_s).reshape(x_ts_s.shape[0], -1)

        fft_dim = x_tr_fft.shape[1]

        sampler = make_weighted_sampler(y_tr)

        tr_l = DataLoader(
            MultiModalDataset(x_tr_s, x_tr_fft, y_tr),
            batch_size=train_cfg.batch_size,
            sampler=sampler,
            drop_last=True
        )

        val_l = DataLoader(
            MultiModalDataset(x_val_s, x_val_fft, y_val),
            batch_size=train_cfg.batch_size
        )

        ts_l = DataLoader(
            MultiModalDataset(x_ts_s, x_ts_fft, y_ts),
            batch_size=train_cfg.batch_size
        )

        # ---------------------------------------------------------------------
        # Entrenamiento desde cero
        # ---------------------------------------------------------------------
        model_fold = GaitHybridModel(t_cfg, fft_dim).to(train_cfg.device)

        trainer = GaitTrainer(model_fold, train_cfg)
        model_fold = trainer.train(tr_l, val_l)

        # ---------------------------------------------------------------------
        # Evaluacion del paciente reservado
        # ---------------------------------------------------------------------
        model_fold.eval()

        probs_fold = []
        y_true_fold = []

        with torch.no_grad():
            for xt, xf, yb in ts_l:
                out = model_fold(
                    xt.to(train_cfg.device),
                    xf.to(train_cfg.device)
                )

                probs_fold.extend(
                    torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                )

                y_true_fold.extend(yb.numpy())

        metricas_fold = _metricas_por_fold(y_true_fold, probs_fold)
        auc = metricas_fold["auc"]

        fold_scores.append(auc if not np.isnan(auc) else 0.0)
        patient_ids.append(str(test_patient))
        fold_metricas.append(metricas_fold)

        all_y_true.extend(y_true_fold)
        all_y_prob.extend(probs_fold)

        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "NaN (monoclase)"
        logger.info(
            f"FOLD {fold:02d} | PACIENTE={test_patient} | "
            f"AUC={auc_str} | BalAcc={metricas_fold['balanced_accuracy']:.4f} | "
            f"MCC={metricas_fold['mcc']:.4f} | N_TEST={len(y_true_fold)}"
        )

        del model_fold, trainer, tr_l, val_l, ts_l

        torch.cuda.empty_cache()
        gc.collect()

    resumen = _finalizar_lopo(
        all_y_true=all_y_true,
        all_y_prob=all_y_prob,
        fold_scores=fold_scores,
        patient_ids=patient_ids,
        lopo_dir=lopo_dir,
        prefijo="hibrido",
        titulo_grafica="LOPO REAL HIBRIDO — Reentrenado por Fold",
        fold_metricas=fold_metricas
    )
    return resumen


def run_lopo_evaluation_fft(
    x_all: np.ndarray,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    train_cfg: TrainConfig,
    lopo_dir: Path
) -> Dict[str, Any]:
    """
    LOPO real para el modelo FFT puro: reentrena FFTModel desde cero en
    cada fold, dejando siempre un paciente distinto fuera de train/val.

    Analoga a run_lopo_evaluation, pero usa unicamente la rama frecuencial
    (FFTModel) en vez del modelo hibrido, sin la rama temporal Transformer.
    Se guarda en archivos separados (prefijo "fft") para no sobreescribir
    los resultados LOPO del hibrido.

    :param x_all: Tensores crudos de todos los pacientes, forma (N, steps, features).
    :param y_all: Etiquetas binarias (reposo/marcha) por muestra.
    :param groups_all: Identificador de paciente por muestra, usado por LeaveOneGroupOut.
    :param train_cfg: Hiperparametros de entrenamiento (batch_size, epochs, lr, patience, device).
    :param eval_dir: Directorio donde se exportan CSVs y la matriz de confusion global.
    """
    logo = LeaveOneGroupOut()
    fft_proc = FFTProcessor()
    unique_p = np.unique(groups_all)

    all_y_true: List[int] = []
    all_y_prob: List[float] = []
    fold_scores: List[float] = []
    patient_ids: List[str] = []
    fold_metricas: List[Dict[str, float]] = []

    logger.info(
        f"INICIANDO LOPO REAL FFT (REENTRENAMIENTO POR FOLD) — {len(unique_p)} PACIENTES"
    )

    for fold, (trainval_idx, test_idx) in enumerate(
        logo.split(x_all, y_all, groups_all), 1
    ):

        test_patient = groups_all[test_idx][0]

        trainval_patients = np.unique(groups_all[trainval_idx])
        val_patient = trainval_patients[-1]

        val_mask = groups_all[trainval_idx] == val_patient
        train_idx_f = trainval_idx[~val_mask]
        val_idx_f = trainval_idx[val_mask]

        x_tr, y_tr = x_all[train_idx_f], y_all[train_idx_f]
        x_val, y_val = x_all[val_idx_f], y_all[val_idx_f]
        x_ts, y_ts = x_all[test_idx], y_all[test_idx]

        # Entrenamiento requiere ambas clases. Un paciente de TEST
        # monoclase ya NO se descarta con continue: se entrena y evalua
        # igual, registrando NaN en AUC/PR-AUC pero calculando
        # balanced_accuracy y MCC (ver _metricas_por_fold).
        if len(np.unique(y_tr)) < 2:
            logger.warning(
                f"FOLD {fold} ({test_patient}) OMITIDO — TRAIN CON CLASE UNICA"
            )
            continue

        # ---------------------------------------------------------------------
        # Escalado (fit unicamente con entrenamiento)
        # ---------------------------------------------------------------------
        sc = StandardScaler()

        x_tr_s = sc.fit_transform(
            x_tr.reshape(-1, x_tr.shape[2])
        ).reshape(x_tr.shape).astype(np.float32)

        x_val_s = sc.transform(
            x_val.reshape(-1, x_val.shape[2])
        ).reshape(x_val.shape).astype(np.float32)

        x_ts_s = sc.transform(
            x_ts.reshape(-1, x_ts.shape[2])
        ).reshape(x_ts.shape).astype(np.float32)

        # ---------------------------------------------------------------------
        # FFT (unica representacion de entrada para este modelo)
        # ---------------------------------------------------------------------
        x_tr_fft = fft_proc.get_fft_features(x_tr_s).reshape(x_tr_s.shape[0], -1)
        x_val_fft = fft_proc.get_fft_features(x_val_s).reshape(x_val_s.shape[0], -1)
        x_ts_fft = fft_proc.get_fft_features(x_ts_s).reshape(x_ts_s.shape[0], -1)

        fft_dim = x_tr_fft.shape[1]

        sampler = make_weighted_sampler(y_tr)

        # NOTA: StandardDataset (2 elementos: x, y), no MultiModalDataset,
        # ya que FFTModel solo recibe una entrada (forward(self, x))
        tr_l = DataLoader(
            StandardDataset(x_tr_fft, y_tr),
            batch_size=train_cfg.batch_size,
            sampler=sampler,
            drop_last=True
        )

        val_l = DataLoader(
            StandardDataset(x_val_fft, y_val),
            batch_size=train_cfg.batch_size
        )

        ts_l = DataLoader(
            StandardDataset(x_ts_fft, y_ts),
            batch_size=train_cfg.batch_size
        )

        # ---------------------------------------------------------------------
        # Entrenamiento desde cero
        # ---------------------------------------------------------------------
        model_fold = FFTModel(fft_dim).to(train_cfg.device)

        trainer = GaitTrainerFFT(model_fold, train_cfg)
        model_fold = trainer.train(tr_l, val_l)

        # ---------------------------------------------------------------------
        # Evaluacion del paciente reservado
        # ---------------------------------------------------------------------
        model_fold.eval()

        probs_fold = []
        y_true_fold = []

        with torch.no_grad():
            for xb, yb in ts_l:
                out = model_fold(xb.to(train_cfg.device))

                probs_fold.extend(
                    torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                )

                y_true_fold.extend(yb.numpy())

        metricas_fold = _metricas_por_fold(y_true_fold, probs_fold)
        auc = metricas_fold["auc"]

        fold_scores.append(auc if not np.isnan(auc) else 0.0)
        patient_ids.append(str(test_patient))
        fold_metricas.append(metricas_fold)

        all_y_true.extend(y_true_fold)
        all_y_prob.extend(probs_fold)

        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "NaN (monoclase)"
        logger.info(
            f"FOLD {fold:02d} | PACIENTE={test_patient} | "
            f"AUC={auc_str} | BalAcc={metricas_fold['balanced_accuracy']:.4f} | "
            f"MCC={metricas_fold['mcc']:.4f} | N_TEST={len(y_true_fold)}"
        )

        del model_fold, trainer, tr_l, val_l, ts_l

        torch.cuda.empty_cache()
        gc.collect()

    resumen = _finalizar_lopo(
        all_y_true=all_y_true,
        all_y_prob=all_y_prob,
        fold_scores=fold_scores,
        patient_ids=patient_ids,
        lopo_dir=lopo_dir,
        prefijo="fft",
        titulo_grafica="LOPO REAL FFT — Reentrenado por Fold",
        fold_metricas=fold_metricas
    )
    return resumen


def run_lopo_late_fusion(
    x_all: np.ndarray,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    t_cfg: TransformerConfig,
    train_cfg: TrainConfig,
    lopo_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """
    LOPO real para Late Fusion: en cada fold, reentrena Transformer y FFT
    POR SEPARADO desde cero (dejando un paciente distinto fuera de train/val
    en cada iteracion), obtiene las probabilidades de cada uno sobre el
    paciente reservado, y las combina con 3 criterios de Late Fusion
    (media geometrica, voto mayoritario, meta-clasificador entrenado
    dentro del propio fold de validacion, nunca visto en el paciente de test).

    A diferencia de run_lopo_evaluation (Hibrido, Early Fusion), aqui NO se
    entrena ninguna red conjunta: cada modelo aprende de forma completamente
    independiente y solo se combinan sus predicciones finales.

    :param x_all: Tensores crudos de todos los pacientes, forma (N, steps, features).
    :param y_all: Etiquetas binarias (reposo/marcha) por muestra.
    :param groups_all: Identificador de paciente por muestra.
    :param t_cfg: Configuracion estructural del Transformer.
    :param train_cfg: Hiperparametros de entrenamiento.
    :param eval_dir: Directorio de salida para CSVs y matriz de confusion.
    """
    logo = LeaveOneGroupOut()
    fft_proc = FFTProcessor()
    unique_p = np.unique(groups_all)

    resultados_por_variante: Dict[str, Dict[str, List]] = {
        "media_geometrica": {"y_true": [], "y_prob": [], "fold_scores": [], "patient_ids": [], "fold_metricas": []},
        "voto_mayoritario": {"y_true": [], "y_prob": [], "fold_scores": [], "patient_ids": [], "fold_metricas": []},
        "meta_clasificador": {"y_true": [], "y_prob": [], "fold_scores": [], "patient_ids": [], "fold_metricas": []},
    }

    logger.info(
        f"INICIANDO LOPO REAL LATE FUSION (REENTRENAMIENTO POR FOLD) — {len(unique_p)} PACIENTES"
    )

    for fold, (trainval_idx, test_idx) in enumerate(
        logo.split(x_all, y_all, groups_all), 1
    ):
        test_patient = groups_all[test_idx][0]

        trainval_patients = np.unique(groups_all[trainval_idx])
        val_patient = trainval_patients[-1]

        val_mask = groups_all[trainval_idx] == val_patient
        train_idx_f = trainval_idx[~val_mask]
        val_idx_f = trainval_idx[val_mask]

        x_tr, y_tr = x_all[train_idx_f], y_all[train_idx_f]
        x_val, y_val = x_all[val_idx_f], y_all[val_idx_f]
        x_ts, y_ts = x_all[test_idx], y_all[test_idx]

        # Entrenamiento requiere ambas clases. Un paciente de TEST
        # monoclase ya NO se descarta: se entrena y evalua igual,
        # registrando NaN en AUC/PR-AUC de esa variante para ese fold
        # (ver _metricas_por_fold), pero calculando balanced_accuracy y MCC.
        if len(np.unique(y_tr)) < 2:
            logger.warning(f"FOLD {fold} ({test_patient}) OMITIDO — TRAIN CON CLASE UNICA")
            continue

        # ESCALADO (fit unicamente con entrenamiento)
        sc = StandardScaler()
        x_tr_s = sc.fit_transform(x_tr.reshape(-1, x_tr.shape[2])).reshape(x_tr.shape).astype(np.float32)
        x_val_s = sc.transform(x_val.reshape(-1, x_val.shape[2])).reshape(x_val.shape).astype(np.float32)
        x_ts_s = sc.transform(x_ts.reshape(-1, x_ts.shape[2])).reshape(x_ts.shape).astype(np.float32)

        # FFT
        x_tr_fft = fft_proc.get_fft_features(x_tr_s).reshape(x_tr_s.shape[0], -1)
        x_val_fft = fft_proc.get_fft_features(x_val_s).reshape(x_val_s.shape[0], -1)
        x_ts_fft = fft_proc.get_fft_features(x_ts_s).reshape(x_ts_s.shape[0], -1)
        fft_dim = x_tr_fft.shape[1]

        # ---------------------------------------------------------------
        # ENTRENAR TRANSFORMER (independiente)
        # ---------------------------------------------------------------
        sampler_t = make_weighted_sampler(y_tr)
        tr_l_t = DataLoader(StandardDataset(x_tr_s, y_tr), batch_size=train_cfg.batch_size, sampler=sampler_t, drop_last=True)
        val_l_t = DataLoader(StandardDataset(x_val_s, y_val), batch_size=train_cfg.batch_size)
        ts_l_t = DataLoader(StandardDataset(x_ts_s, y_ts), batch_size=train_cfg.batch_size)

        model_transformer_fold = GaitTransformer(t_cfg).to(train_cfg.device)
        trainer_t = GaitTrainerFFT(model_transformer_fold, train_cfg)  # bucle de 2 elementos (x, y)
        model_transformer_fold = trainer_t.train(tr_l_t, val_l_t)

        # ---------------------------------------------------------------
        # ENTRENAR FFT (independiente)
        # ---------------------------------------------------------------
        sampler_f = make_weighted_sampler(y_tr)
        tr_l_f = DataLoader(StandardDataset(x_tr_fft, y_tr), batch_size=train_cfg.batch_size, sampler=sampler_f, drop_last=True)
        val_l_f = DataLoader(StandardDataset(x_val_fft, y_val), batch_size=train_cfg.batch_size)
        ts_l_f = DataLoader(StandardDataset(x_ts_fft, y_ts), batch_size=train_cfg.batch_size)

        model_fft_fold = FFTModel(fft_dim).to(train_cfg.device)
        trainer_f = GaitTrainerFFT(model_fft_fold, train_cfg)
        model_fft_fold = trainer_f.train(tr_l_f, val_l_f)

        # ---------------------------------------------------------------
        # OBTENER PROBABILIDADES DE VAL (para entrenar meta-clasificador
        # SIN usar el paciente de test) Y DE TEST (paciente reservado)
        # ---------------------------------------------------------------
        def probs_de(model, loader_):
            model.eval()
            probs = []
            with torch.no_grad():
                for xb, yb in loader_:
                    out = model(xb.to(train_cfg.device))
                    probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            return np.array(probs)

        prob_val_t = probs_de(model_transformer_fold, val_l_t)
        prob_val_f = probs_de(model_fft_fold, val_l_f)
        prob_ts_t = probs_de(model_transformer_fold, ts_l_t)
        prob_ts_f = probs_de(model_fft_fold, ts_l_f)

        # ---------------------------------------------------------------
        # COMBINAR: MEDIA GEOMETRICA
        # ---------------------------------------------------------------
        eps = 1e-7
        prob_geom = np.sqrt(np.clip(prob_ts_t, eps, 1 - eps) * np.clip(prob_ts_f, eps, 1 - eps))

        # ---------------------------------------------------------------
        # COMBINAR: VOTO MAYORITARIO
        # ---------------------------------------------------------------
        pred_t = (prob_ts_t >= 0.5).astype(int)
        pred_f = (prob_ts_f >= 0.5).astype(int)
        suma_votos = pred_t + pred_f
        prob_voto = np.where(suma_votos == 2, 1.0, np.where(suma_votos == 0, 0.0, (prob_ts_t + prob_ts_f) / 2.0))

        # ---------------------------------------------------------------
        # COMBINAR: META-CLASIFICADOR (entrenado SOLO con el fold de val,
        # nunca con el paciente de test reservado)
        # ---------------------------------------------------------------
        meta_clf_fold = LogisticRegression()
        X_val_meta = np.column_stack([prob_val_t, prob_val_f])
        meta_clf_fold.fit(X_val_meta, y_val)
        X_ts_meta = np.column_stack([prob_ts_t, prob_ts_f])
        prob_meta = meta_clf_fold.predict_proba(X_ts_meta)[:, 1]

        # ---------------------------------------------------------------
        # REGISTRAR RESULTADOS DE CADA VARIANTE
        # ---------------------------------------------------------------
        auc_log_partes = []
        for nombre, prob_fold in [
            ("media_geometrica", prob_geom),
            ("voto_mayoritario", prob_voto),
            ("meta_clasificador", prob_meta),
        ]:
            metricas_fold = _metricas_por_fold(y_ts, prob_fold)
            auc_fold = metricas_fold["auc"]

            resultados_por_variante[nombre]["y_true"].extend(y_ts.tolist())
            resultados_por_variante[nombre]["y_prob"].extend(prob_fold.tolist())
            resultados_por_variante[nombre]["fold_scores"].append(auc_fold if not np.isnan(auc_fold) else 0.0)
            resultados_por_variante[nombre]["patient_ids"].append(str(test_patient))
            resultados_por_variante[nombre]["fold_metricas"].append(metricas_fold)

            auc_str = f"{auc_fold:.4f}" if not np.isnan(auc_fold) else "NaN"
            auc_log_partes.append(f"AUC_{nombre}={auc_str}")

        logger.info(
            f"FOLD {fold:02d} | PACIENTE={test_patient} | "
            + " | ".join(auc_log_partes) +
            f" | N_TEST={len(y_ts)}"
        )

        del model_transformer_fold, model_fft_fold, trainer_t, trainer_f
        del tr_l_t, val_l_t, ts_l_t, tr_l_f, val_l_f, ts_l_f
        torch.cuda.empty_cache()
        gc.collect()

    # CONSOLIDAR CADA VARIANTE POR SEPARADO
    resumenes_late_fusion: Dict[str, Dict[str, Any]] = {}
    for nombre, datos in resultados_por_variante.items():
        resumen = _finalizar_lopo(
            all_y_true=datos["y_true"],
            all_y_prob=datos["y_prob"],
            fold_scores=datos["fold_scores"],
            patient_ids=datos["patient_ids"],
            lopo_dir=lopo_dir,
            prefijo=f"latefusion_{nombre}",
            titulo_grafica=f"LOPO REAL LATE FUSION ({nombre}) — Reentrenado por Fold",
            fold_metricas=datos["fold_metricas"]
        )
        resumenes_late_fusion[nombre] = resumen

    return resumenes_late_fusion


class GaitTrainerFFT:
    """
    Bucle de entrenamiento con early stopping para modelos de una sola
    entrada (2 elementos por batch: x, y). Se usa tanto para FFTModel
    como para GaitTransformer cuando se entrena de forma independiente
    (Late Fusion), analogo a GaitTrainer pero sin la rama del hibrido.
    """
    def __init__(self, model: nn.Module, config: TrainConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.best_state = None

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        """Entrena con early stopping segun AUC de validacion."""
        best_auc, no_improve = 0.0, 0
        for epoch in range(self.config.epochs):
            self.model.train()
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                out = self.model(xb.to(self.config.device))
                loss = self.criterion(out, yb.to(self.config.device))
                loss.backward()
                self.optimizer.step()

            val_auc = self._evaluate_auc(val_loader)
            self.scheduler.step(val_auc)

            if val_auc > best_auc:
                best_auc, no_improve = val_auc, 0
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                no_improve += 1
            if no_improve >= self.config.patience:
                break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self.model

    def _evaluate_auc(self, loader: DataLoader) -> float:
        """Calcula AUC de validacion, o 0.0 si solo hay una clase presente."""
        self.model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for xb, yb in loader:
                out = self.model(xb.to(self.config.device))
                y_prob.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                y_true.extend(yb.numpy())
        return roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0


def _metricas_por_fold(y_true: List[int], y_prob: List[float], threshold: float = 0.50) -> Dict[str, float]:
    """
    Calcula AUC, PR-AUC, balanced accuracy y MCC para un unico fold,
    con proteccion explicita contra pacientes monoclase.

    Algunos pacientes del dataset (ej. 02548893X-118, MGM-202406-79) solo
    tienen una clase presente (unicamente "marcha"). En ese caso, AUC y
    PR-AUC no estan matematicamente definidos (roc_auc_score/
    average_precision_score lanzan ValueError con una sola clase en
    y_true) y se registran como NaN en vez de intentar calcularlos o de
    detener el script. Balanced accuracy y MCC si se calculan siempre,
    ya que no requieren ambas clases en y_true para tener sentido (aunque
    su interpretacion con una sola clase real es limitada, no producen
    una excepcion).

    :param y_true: Etiquetas reales del fold (puede ser monoclase).
    :param y_prob: Probabilidades predichas del fold.
    :param threshold: Umbral de decision para binarizar y calcular
        balanced accuracy y MCC.
    :return: Diccionario con auc, pr_auc, balanced_accuracy, mcc. Los
        valores no calculables (por monoclase) quedan como float('nan').
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    es_monoclase = len(np.unique(y_true_arr)) < 2

    auc = roc_auc_score(y_true_arr, y_prob_arr) if not es_monoclase else float("nan")
    pr_auc = average_precision_score(y_true_arr, y_prob_arr) if not es_monoclase else float("nan")

    # balanced_accuracy_score y matthews_corrcoef no requieren ambas clases
    # en y_true para ejecutar sin excepcion, pero con y_pred tambien
    # monoclase (ej. el fold entero predicho igual) devuelven un valor
    # degenerado (0.5 o 0.0) en vez de fallar -- se calculan siempre.
    balanced_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    mcc = matthews_corrcoef(y_true_arr, y_pred_arr)

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "es_monoclase": es_monoclase,
    }


def _finalizar_lopo(
    all_y_true: List[int],
    all_y_prob: List[float],
    fold_scores: List[float],
    patient_ids: List[str],
    lopo_dir: Path,
    prefijo: str,
    titulo_grafica: str,
    fold_metricas: List[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Consolida resultados de un esquema LOPO (fold-por-fold ya ejecutados)
    en una carpeta DEDICADA por modelo (lopo_dir / prefijo /), con una
    subcarpeta "graficas" para la matriz de confusion. Compartido por
    run_lopo_evaluation, run_lopo_evaluation_fft y run_lopo_late_fusion.

    Estructura generada (dentro de lopo_dir / prefijo /):
        lopo_fold_aucs.csv
        lopo_metricas_por_paciente.csv
        metricas_lopo.csv
        calibracion_lopo.csv
        graficas/LOPO_Global_Eval_Matriz.png

    :param all_y_true: Etiquetas reales concatenadas de todos los folds.
    :param all_y_prob: Probabilidades predichas concatenadas de todos los folds.
    :param fold_scores: AUC individual de cada fold (paciente dejado fuera).
    :param patient_ids: Identificador de paciente correspondiente a cada fold.
    :param lopo_dir: Carpeta base LOPO/ (se crea lopo_dir/prefijo/ dentro).
    :param prefijo: Nombre del esquema (ej. "hibrido", "fft",
        "latefusion_media_geometrica"), usado como nombre de subcarpeta.
    :param titulo_grafica: Titulo mostrado en la matriz de confusion exportada.
    :param fold_metricas: Lista (un dict por fold, mismo orden que
        patient_ids) con las metricas devueltas por _metricas_por_fold.
    :return: Diccionario resumen (usado despues por la comparativa
        general): auc_global, pr_auc_global, balanced_accuracy_global,
        mcc_global, auc_promedio_folds, auc_std_folds, n_folds,
        peor_paciente, peor_auc_fold, n_parametros (se completa fuera).
    """
    modelo_dir = lopo_dir / prefijo
    graficas_dir = modelo_dir / "graficas"
    modelo_dir.mkdir(parents=True, exist_ok=True)
    graficas_dir.mkdir(parents=True, exist_ok=True)

    if not all_y_true:
        logger.warning(f"LOPO ({prefijo.upper()}): SIN RESULTADOS PARA CONSOLIDAR")
        return {}

    auc_global = roc_auc_score(all_y_true, all_y_prob)

    logger.info(
        f"LOPO REAL {prefijo.upper()} — "
        f"AUC PROMEDIO={np.mean(fold_scores):.4f} "
        f"STD={np.std(fold_scores):.4f} "
        f"| GLOBAL={auc_global:.4f}"
    )

    pd.DataFrame({
        "fold": range(1, len(fold_scores) + 1),
        "paciente": patient_ids,
        "auc": fold_scores
    }).to_csv(
        modelo_dir / "lopo_fold_aucs.csv",
        index=False
    )

    # TABLA EXPANDIDA POR PACIENTE (doble entrada): AUC, PR-AUC, balanced
    # accuracy y MCC por cada sujeto dejado fuera, con fila final de
    # media +/- desviacion estandar de toda la poblacion.
    peor_paciente, peor_auc_fold = None, None
    if fold_metricas is not None:
        filas_por_paciente = []
        for paciente, m in zip(patient_ids, fold_metricas):
            filas_por_paciente.append({
                "paciente": paciente,
                "monoclase": m["es_monoclase"],
                "AUC": m["auc"],
                "PR_AUC": m["pr_auc"],
                "Balanced_Accuracy": m["balanced_accuracy"],
                "MCC": m["mcc"],
            })

        df_por_paciente = pd.DataFrame(filas_por_paciente)

        # IDENTIFICAR EL PEOR PACIENTE POR AUC (ignorando monoclase, que
        # no tiene AUC calculable), para la comparativa legible despues.
        df_con_auc = df_por_paciente.dropna(subset=["AUC"])
        if not df_con_auc.empty:
            fila_peor = df_con_auc.loc[df_con_auc["AUC"].idxmin()]
            peor_paciente = fila_peor["paciente"]
            peor_auc_fold = float(fila_peor["AUC"])

        fila_media = {
            "paciente": "MEDIA_POBLACION",
            "monoclase": "",
            "AUC": np.nanmean(df_por_paciente["AUC"]),
            "PR_AUC": np.nanmean(df_por_paciente["PR_AUC"]),
            "Balanced_Accuracy": np.nanmean(df_por_paciente["Balanced_Accuracy"]),
            "MCC": np.nanmean(df_por_paciente["MCC"]),
        }
        fila_std = {
            "paciente": "STD_POBLACION",
            "monoclase": "",
            "AUC": np.nanstd(df_por_paciente["AUC"]),
            "PR_AUC": np.nanstd(df_por_paciente["PR_AUC"]),
            "Balanced_Accuracy": np.nanstd(df_por_paciente["Balanced_Accuracy"]),
            "MCC": np.nanstd(df_por_paciente["MCC"]),
        }
        df_por_paciente = pd.concat(
            [df_por_paciente, pd.DataFrame([fila_media, fila_std])],
            ignore_index=True
        )

        df_por_paciente.to_csv(
            modelo_dir / "lopo_metricas_por_paciente.csv",
            index=False
        )
        logger.info(f"TABLA POR PACIENTE (AUC/PR-AUC/BalAcc/MCC) EXPORTADA: {prefijo}")

    umbral_global = 0.50

    y_pred_global = (np.array(all_y_prob) >= umbral_global).astype(int)

    cm_global = confusion_matrix(
        all_y_true,
        y_pred_global,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm_global.ravel()

    balanced_acc_global = balanced_accuracy_score(all_y_true, y_pred_global)
    mcc_global = matthews_corrcoef(all_y_true, y_pred_global)
    pr_auc_global = average_precision_score(all_y_true, all_y_prob)

    pd.DataFrame([{
        "AUC": auc_global,
        "PR_AUC": pr_auc_global,
        "Balanced_Accuracy": balanced_acc_global,
        "MCC": mcc_global,
        "Sensibilidad": tp / (tp + fn),
        "Especificidad": tn / (tn + fp),
        "FPR": fp / (fp + tn),
        "Threshold": umbral_global
    }]).to_csv(
        modelo_dir / "metricas_lopo.csv",
        index=False
    )

    brier, inter, pend = calibration_metrics(
        np.array(all_y_true),
        np.array(all_y_prob)
    )

    pd.DataFrame([{
        "AUC": auc_global,
        "Brier": brier,
        "Intercepto": inter,
        "Pendiente": pend
    }]).to_csv(
        modelo_dir / "calibracion_lopo.csv",
        index=False
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_global,
        display_labels=["Reposo", "Marcha"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")

    plt.title(
        f"{titulo_grafica}\n"
        f"AUC: {auc_global:.4f} | Umbral: {umbral_global:.2f}"
    )

    plt.savefig(
        graficas_dir / "LOPO_Global_Eval_Matriz.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    logger.info(f"MATRIZ GRAFICA LOPO ({prefijo.upper()}) EXPORTADA CON EXITO")

    return {
        "prefijo": prefijo,
        "auc_global": auc_global,
        "pr_auc_global": pr_auc_global,
        "balanced_accuracy_global": balanced_acc_global,
        "mcc_global": mcc_global,
        "brier": brier,
        "auc_promedio_folds": float(np.mean(fold_scores)),
        "auc_std_folds": float(np.std(fold_scores)),
        "n_folds": len(fold_scores),
        "peor_paciente": peor_paciente,
        "peor_auc_fold": peor_auc_fold,
    }


if __name__ == "__main__":
    main()
