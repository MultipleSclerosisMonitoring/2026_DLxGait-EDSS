# -*- coding: utf-8 -*-
"""
Script de evaluacion para modelos de marcha.
Carga modelos preentrenados, calcula metricas y genera reportes.
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
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, ConfusionMatrixDisplay
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

# CONTEXTO DE EJECUCION GENERAL
def main() -> None:
    set_seed(13)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--run-lopo", action="store_true")
    args = parser.parse_args()

    MODELS_DIR = args.models
    # Carpeta principal
    EVAL_GENERAL_DIR = MODELS_DIR / "EVAL_GENERAL"
    EVAL_GENERAL_DIR.mkdir(parents=True, exist_ok=True)

    # Todas las figuras normales
    GRAFICAS_DIR = EVAL_GENERAL_DIR / "graficas"
    GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)

    # Resultados del test independiente
    EVAL_DIR = EVAL_GENERAL_DIR / "evaluacion_final"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("INICIANDO EVALUACION METRICA")
    
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

    # EVALUAR METRICAS FINALES
    metrics_t = evaluator_t.compute_metrics(y_true_t, prob_t, opt_thresh_t, "MODELO TRANSFORMER", GRAFICAS_DIR)
    metrics_f = evaluator_f.compute_metrics(y_true_f, prob_f, opt_thresh_f, "MODELO FFT", GRAFICAS_DIR)
    metrics_h = evaluator_h.compute_metrics(y_true_h, prob_h, opt_thresh_h, "MODELO HIBRIDO FINAL", GRAFICAS_DIR)

    # CALCULAR DISCRIMINACION Y CALIBRACION
    reporte = []
    for nombre, yt, yp in [("Transformer", y_true_t, prob_t), ("FFT", y_true_f, prob_f), ("Hibrido", y_true_h, prob_h)]:
        b, i, s = calibration_metrics(yt, yp)
        reporte.append({"Modelo": nombre, "Brier": b, "Intercepto": i, "Pendiente": s})
    
    # EXPORTAR METRICAS CALIBRACION
    pd.DataFrame(reporte).to_csv(EVAL_DIR / "calibracion_eval.csv", index=False)
    
    # EXPORTAR METRICAS TEXTO
    pd.DataFrame([
        {"Modelo": "Transformer", **metrics_t},
        {"Modelo": "FFT", **metrics_f},
        {"Modelo": "Hibrido", **metrics_h},
    ]).to_csv(EVAL_DIR / "metricas_basicas.txt", sep='\t', index=False)

    # CONSTRUIR TABLA ASCII REPORTE
    lineas_reporte = []
    lineas_reporte.append("\n" + "=" * 95)
    lineas_reporte.append("RESULTADOS FINALES (TEST INDEPENDIENTE)")
    lineas_reporte.append("=" * 95)
    for modelo, met, cal in zip(["TRANSFORMER", "FFT", "HIBRIDO"], [metrics_t, metrics_f, metrics_h], reporte):
        lineas_reporte.append(
            f"{modelo:12s}"
            f" | AUC={met['AUC']:.4f}"
            f" | Sens={met['Sensibilidad']:.2%}"
            f" | Spec={met['Especificidad']:.2%}"
            f" | FPR={met['FPR']:.2%}"
            f" | Brier={cal['Brier']:.4f}"
            f" | Int={cal['Intercepto']:.3f}"
            f" | Pend={cal['Pendiente']:.3f}"
        )
    lineas_reporte.append("=" * 95)

    # IMPRIMIR E INYECTAR REPORTE TXT
    texto_final = "\n".join(lineas_reporte)
    print(texto_final)
    
    with open(EVAL_DIR / "reporte_final_evaluacion.txt", "w", encoding="utf-8") as f_txt:
        f_txt.write("REPORTE OPERACIONAL DE INFERENCIA BIOMEDICA\n")
        f_txt.write(texto_final)
    logger.info("DOCUMENTO DE TEXTO GENERADO CORRECTAMENTE")

    # DISPARAR ENTORNO LOPO OPCIONAL
    
    LOPO_DIR = EVAL_GENERAL_DIR / "LOPO_HIBRIDO"
    LOPO_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.run_lopo:
        x_all, groups_all, y_all = loader.get_all_raw_data()
        train_cfg = TrainConfig(device=DEVICE)
        run_lopo_evaluation(x_all, y_all, groups_all, m1_cfg, train_cfg, EVAL_DIR)
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
    eval_dir: Path
) -> None:
    """
    LOPO real: reentrena el modelo híbrido desde cero en cada fold,
    dejando siempre un paciente distinto fuera de train/val.
    """
    logo = LeaveOneGroupOut()
    fft_proc = FFTProcessor()
    unique_p = np.unique(groups_all)

    all_y_true: List[int] = []
    all_y_prob: List[float] = []
    fold_scores: List[float] = []
    patient_ids: List[str] = []

    logger.info(
        f"INICIANDO LOPO REAL (REENTRENAMIENTO POR FOLD) — {len(unique_p)} PACIENTES"
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

        # Debe haber ambas clases en entrenamiento y prueba
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_ts)) < 2:
            logger.warning(
                f"FOLD {fold} ({test_patient}) OMITIDO — CLASE ÚNICA"
            )
            continue

        # ---------------------------------------------------------------------
        # Escalado (fit únicamente con entrenamiento)
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
        # Evaluación del paciente reservado
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

        auc = roc_auc_score(y_true_fold, probs_fold)

        fold_scores.append(auc)
        patient_ids.append(str(test_patient))

        all_y_true.extend(y_true_fold)
        all_y_prob.extend(probs_fold)

        logger.info(
            f"FOLD {fold:02d} | PACIENTE={test_patient} | "
            f"AUC={auc:.4f} | N_TEST={len(y_true_fold)}"
        )

        del model_fold, trainer, tr_l, val_l, ts_l

        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================================
    # RESULTADOS GLOBALES
    # =========================================================================

    if all_y_true:

        auc_global = roc_auc_score(all_y_true, all_y_prob)

        logger.info(
            f"LOPO REAL — "
            f"AUC PROMEDIO={np.mean(fold_scores):.4f} "
            f"STD={np.std(fold_scores):.4f} "
            f"| GLOBAL={auc_global:.4f}"
        )

        pd.DataFrame({
            "fold": range(1, len(fold_scores) + 1),
            "paciente": patient_ids,
            "auc": fold_scores
        }).to_csv(
            eval_dir / "lopo_fold_aucs_eval.csv",
            index=False
        )

        umbral_global = 0.50

        y_pred_global = (np.array(all_y_prob) >= umbral_global).astype(int)

        cm_global = confusion_matrix(
            all_y_true,
            y_pred_global,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm_global.ravel()

        pd.DataFrame([{
            "AUC": auc_global,
            "Sensibilidad": tp / (tp + fn),
            "Especificidad": tn / (tn + fp),
            "FPR": fp / (fp + tn),
            "Threshold": umbral_global
        }]).to_csv(
            eval_dir / "metricas_lopo.csv",
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
            eval_dir / "calibracion_lopo.csv",
            index=False
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_global,
            display_labels=["Reposo", "Marcha"]
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues")

        plt.title(
            f"LOPO REAL — Reentrenado por Fold\n"
            f"AUC: {auc_global:.4f} | Umbral: {umbral_global:.2f}"
        )

        plt.savefig(
            eval_dir / "LOPO_Global_Eval_Matriz.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        logger.info("MATRIZ GRAFICA LOPO EXPORTADA CON EXITO")


if __name__ == "__main__":
    main()
