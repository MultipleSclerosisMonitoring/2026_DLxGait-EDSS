# -*- coding: utf-8 -*-
"""
Motor de inferencia continua basado en deep learning.
Incluye modelos en tiempo, frecuencia e híbridos.

"""

from __future__ import annotations
import h5py
import numpy as np
import argparse
import logging
import copy
import random
import os
from pathlib import Path
from typing import Tuple, Optional, Dict
from pydantic import BaseModel, FilePath, PositiveInt, confloat, Field
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels.astype(int), minlength=2)
    class_weights = 1.0 / np.where(counts > 0, counts, 1)
    sample_weights = class_weights[labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True
    )


# CARGAR DATOS HDF5

class ModelConfig(BaseModel):
    h5_path: FilePath
    output_dir: Path
    random_state: PositiveInt = 13


class GaitDatasetLoader:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()

    def get_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Partición LOPO: el último paciente único es test,
        el penúltimo es val, el resto train.
        Los índices se guardan en disco.
        """
        x_raw, groups, labels = self._load_data()
        unique_patients = np.unique(groups)

        # LOPO: reservar ultimo paciente como test
        test_patient  = unique_patients[-1]
        val_patient   = unique_patients[-2]

        test_mask  = groups == test_patient
        val_mask   = groups == val_patient
        train_mask = ~test_mask & ~val_mask

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]
        test_idx  = np.where(test_mask)[0]

        np.save(self.config.output_dir / "train_idx.npy", train_idx)
        np.save(self.config.output_dir / "val_idx.npy",   val_idx)
        np.save(self.config.output_dir / "test_idx.npy",  test_idx)

        # GUARDAR TAMBIEN LOS NOMBRES DE PACIENTE (no solo los indices
        # numericos), para poder auditar rapidamente "quien quedo en
        # test/val" sin tener que recargar el HDF5 completo y cruzar
        # indices manualmente cada vez que se reentrena con un dataset
        # distinto (ej. tras anadir pacientes nuevos).
        import json
        info_particion = {
            "test_patient": str(test_patient),
            "val_patient": str(val_patient),
            "train_patients": sorted(str(p) for p in unique_patients if p not in (test_patient, val_patient)),
            "n_train_patients": len(unique_patients) - 2,
        }
        with open(self.config.output_dir / "particion_pacientes.json", "w", encoding="utf-8") as f:
            json.dump(info_particion, f, indent=4, ensure_ascii=False)

        logger.info(f"LOPO TEST={test_patient} VAL={val_patient} TRAIN={len(unique_patients)-2} pacientes")
        logger.info(f"TRAIN={len(train_idx)} VAL={len(val_idx)} TEST={len(test_idx)}")

        x_train_raw = x_raw[train_idx]
        x_val_raw   = x_raw[val_idx]
        x_test_raw  = x_raw[test_idx]
        y_train     = labels[train_idx]
        y_val       = labels[val_idx]
        y_test      = labels[test_idx]

        x_train = self._scale_data(x_train_raw, fit=True)
        x_val   = self._scale_data(x_val_raw,   fit=False)
        x_test  = self._scale_data(x_test_raw,  fit=False)

        return (
            x_train.astype(np.float32),
            x_val.astype(np.float32),
            x_test.astype(np.float32),
            y_train.astype(np.float32),
            y_val.astype(np.float32),
            y_test.astype(np.float32)
        )

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_list, groups, labels = [], [], []
        with h5py.File(self.config.h5_path, "r") as hf:
            for patient in hf.keys():
                for seg_chunk in hf[patient].keys():
                    for pie in hf[patient][seg_chunk].keys():
                        dataset = hf[patient][seg_chunk][pie]
                        x_list.append(dataset[:])
                        labels.append(dataset.attrs["label"])
                        groups.append(patient)
        return np.array(x_list), np.array(groups), np.array(labels)

    def _scale_data(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        n_samples, n_steps, n_features = data.shape
        flat_data = data.reshape(-1, n_features)
        if fit:
            self.scaler.fit(flat_data)
        scaled = self.scaler.transform(flat_data)
        return scaled.reshape(n_samples, n_steps, n_features)

    def get_all_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._load_data()


# DEFINIR ARQUITECTURAS MODELOS

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, is_train: bool = False) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.x[idx].clone()
        if self.is_train and torch.rand(1).item() > 0.5:
            sample += torch.randn_like(sample) * 0.05
        return sample, self.y[idx]


class MultiModalAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, x_time: np.ndarray, x_fft: np.ndarray, y: np.ndarray, is_train: bool = False) -> None:
        self.x_time   = torch.as_tensor(x_time, dtype=torch.float32)
        self.x_fft    = torch.as_tensor(x_fft, dtype=torch.float32)
        self.y        = torch.as_tensor(y, dtype=torch.long)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_sample = self.x_time[idx].clone()
        if self.is_train and torch.rand(1).item() > 0.5:
            t_sample += torch.randn_like(t_sample) * 0.05
        return t_sample, self.x_fft[idx], self.y[idx]


class TransformerConfig(BaseModel):
    input_dim:   PositiveInt
    max_len:     PositiveInt
    model_dim:   PositiveInt = 64
    nhead:       PositiveInt = 4
    num_layers:  PositiveInt = 2
    dropout:     confloat(ge=0, le=0.5) = 0.3
    num_classes: PositiveInt = 2


class GaitTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super(GaitTransformer, self).__init__()
        self.embedding     = nn.Linear(config.input_dim, config.model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_len, config.model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x       = self.embedding(x)
        seq_len = x.size(1)
        if seq_len > self.pos_embedding.size(1):
            raise ValueError(f"Seq_len {seq_len} excede max_len {self.pos_embedding.size(1)}")
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


class FFTProcessor:
    @staticmethod
    def get_fft_features(data: np.ndarray) -> np.ndarray:
        data_f32 = data.astype(np.float32)
        data_fft = np.abs(rfft(data_f32, axis=1, workers=1)).astype(np.float32)
        return (data_fft / data.shape[1]).astype(np.float32)


class FFTModel(nn.Module):
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
    Modelo Hibrido (Early Fusion): concatena la rama Transformer (temporal)
    con la rama FFT (frecuencial) antes de un cabezal de clasificacion final.

    NOTA IMPORTANTE DE DISENO: si se proporciona pretrained_transformer,
    la rama Transformer se inicializa con esos pesos YA ENTRENADOS y
    luego se CONGELA por completo (requires_grad=False en todos sus
    parametros). Es decir, "Hibrido" en este caso no entrena su rama
    temporal desde cero junto con la rama FFT -- solo aprende la rama
    FFT y el cabezal de fusion final, reutilizando el Transformer ya
    optimizado como extractor de features fijo. Esto reduce riesgo de
    sobreajuste y acelera el entrenamiento, pero significa que el
    Hibrido es menos "conjunto" de lo que su nombre sugiere: no hay
    coadaptacion entre ambas ramas durante el entrenamiento, solo fusion
    de una rama fija (Transformer) con una rama entrenable (FFT).
    """
    def __init__(self, t_cfg: TransformerConfig, fft_input_dim: int, pretrained_transformer: Optional[nn.Module] = None) -> None:
        super(GaitHybridModel, self).__init__()
        self.transformer_branch = GaitTransformer(t_cfg)

        if pretrained_transformer is not None:
            self.transformer_branch.load_state_dict(pretrained_transformer.state_dict())
            for param in self.transformer_branch.parameters():
                param.requires_grad = False

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


# CONFIGURAR ENTRENAMIENTO ROBUSTO

class TrainConfig(BaseModel):
    batch_size: PositiveInt = 64
    epochs:     PositiveInt = 100
    lr:         confloat(gt=0) = 0.001
    patience:   PositiveInt = 20
    device:     str = Field(default="cuda" if torch.cuda.is_available() else "cpu")


class GaitTrainer:
    def __init__(self, model: nn.Module, config: TrainConfig, class_weights: Optional[torch.Tensor] = None) -> None:
        self.model     = model.to(config.device)
        self.config    = config
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.best_model_state = None
        self.use_amp   = torch.cuda.is_available()
        # API NUEVA (torch.amp, no torch.cuda.amp -- deprecado en
        # versiones recientes de PyTorch, aunque aun funciona como alias
        # de compatibilidad).
        self.scaler    = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = True) -> nn.Module:
        if verbose:
            logger.info(f"ENTRENANDO DISPOSITIVO {self.config.device.upper()} AMP {self.use_amp}")

        best_val_auc      = 0.0
        epochs_no_improve = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    if len(batch) == 3:
                        xt  = batch[0].to(self.config.device)
                        xf  = batch[1].to(self.config.device)
                        yb  = batch[2].to(self.config.device)
                        out = self.model(xt, xf)
                    else:
                        x_batch = batch[0].to(self.config.device)
                        yb      = batch[1].to(self.config.device)
                        out     = self.model(x_batch)

                    loss = self.criterion(out, yb)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()

            val_auc = self.evaluate_auc(val_loader)
            self.scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc      = val_auc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"EPOCA {epoch+1:03d} LOSS {total_loss/len(train_loader):.4f} AUC {val_auc:.4f} LR {self.optimizer.param_groups[0]['lr']:.6f}")

            if epochs_no_improve >= self.config.patience:
                if verbose:
                    logger.info(f"PARADA TEMPRANA EPOCA {epoch+1} MEJOR {best_val_auc:.4f}")
                break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.model

    def evaluate_auc(self, loader: DataLoader) -> float:
        self.model.eval()
        y_true, y_prob = [], []

        with torch.no_grad():
            for batch in loader:
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    if len(batch) == 3:
                        xt  = batch[0].to(self.config.device)
                        xf  = batch[1].to(self.config.device)
                        yb  = batch[2].to(self.config.device)
                        out = self.model(xt, xf)
                    else:
                        x_batch = batch[0].to(self.config.device)
                        yb      = batch[1].to(self.config.device)
                        out     = self.model(x_batch)

                probs = torch.softmax(out, dim=1)[:, 1]
                y_true.extend(yb.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        if len(set(y_true)) > 1:
            return roc_auc_score(y_true, y_prob)
        return 0.0


# EVALUAR METRICAS CLINICAS

class GaitEvaluator:
    def __init__(self, model: nn.Module, device: str) -> None:
        self.model  = model
        self.device = device

    def plot_results(self, test_loader: DataLoader, val_loader: DataLoader, title: str, save_dir: Optional[Path] = None) -> Tuple[float, Dict[str, float]]:
        # UMBRAL FIJO POR DISENO (no calibrado tipo Youden's J). No se
        # calcula el punto optimo sobre val_loader -- aunque el parametro
        # se recibe, se usa 0.50 fijo. Ver documentacion del proyecto
        # para el razonamiento (evitar que un .joblib de umbral corrupto
        # o mal generado altere la clasificacion sin que se note).
        opt_thresh = 0.50
        self.model.eval()
        y_true, y_prob = [], []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    out = self.model(batch[0].to(self.device), batch[1].to(self.device))
                    yb  = batch[2]
                else:
                    out = self.model(batch[0].to(self.device))
                    yb  = batch[1]
                y_true.extend(yb.cpu().numpy())
                y_prob.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())

        y_pred = (np.array(y_prob) >= opt_thresh).astype(int)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val     = fp / (fp + tn) if (fp + tn) > 0 else 0
        auc_score   = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0

        metrics = {
            "Sensibilidad": sensitivity,
            "Especificidad": specificity,
            "FPR": fpr_val,
            "AUC": auc_score,
            "Threshold": opt_thresh
        }

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['REPOSO', 'MARCHA'],
                    yticklabels=['REPOSO', 'MARCHA'])
        plt.title(f'{title}\nAUC: {auc_score:.2f} | Umbral Fijo: {opt_thresh:.2f}')
        plt.ylabel('Real')
        plt.xlabel('Predicho')

        if save_dir:
            fname = f"{title.replace(' ', '_')}_Matriz_Confusion.png"
            plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
            logger.info("GRAFICA ALMACENADA RUTA LOCAL")

        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

        return opt_thresh, metrics

def obtener_probabilidades(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, y_true = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                out = model(batch[0].to(device), batch[1].to(device))
                yb  = batch[2]
            else:
                out = model(batch[0].to(device))
                yb  = batch[1]
            probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            y_true.extend(yb.numpy())
    return np.array(y_true), np.array(probs)


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float, float]:
    brier  = brier_score_loss(y_true, probs)
    logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / (1 - np.clip(probs, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    lr     = LogisticRegression(penalty=None).fit(logits, y_true)
    return brier, float(lr.intercept_[0]), float(lr.coef_[0][0])


# FUNCION PRINCIPAL MAIN

def main() -> None:
    set_seed(13)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    BASE_DIR        = Path(__file__).resolve().parent.parent
    OUTPUT_BASE_DIR = BASE_DIR / "A05_MODELOS_ENTRENADOS"
    GRAFICAS_DIR    = OUTPUT_BASE_DIR / "graficas"
    EVAL_DIR        = OUTPUT_BASE_DIR / "evaluacion_final"
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        cfg_data = ModelConfig(h5_path=args.dataset, output_dir=OUTPUT_BASE_DIR)
        loader   = GaitDatasetLoader(cfg_data)
        x_train, x_val, x_test, y_train, y_val, y_test = loader.get_train_test_data()

        REAL_INPUT_DIM = x_train.shape[2]
        REAL_SEQ_LEN   = x_train.shape[1]

        fft_proc    = FFTProcessor()
        x_train_fft = fft_proc.get_fft_features(x_train)
        x_val_fft   = fft_proc.get_fft_features(x_val)
        x_test_fft  = fft_proc.get_fft_features(x_test)

        REAL_FFT_DIM = x_train_fft.shape[1] * x_train_fft.shape[2]
        t_cfg        = TrainConfig()

        sampler_train = make_weighted_sampler(y_train)

        # ENTRENAR MODELO TRANSFORMER
        logger.info("ENTRENAR MODELO TEMPORAL")
        m1_cfg     = TransformerConfig(input_dim=REAL_INPUT_DIM, max_len=REAL_SEQ_LEN)
        model_time = GaitTransformer(m1_cfg)

        tr_l  = DataLoader(AugmentedDataset(x_train, y_train, is_train=True), batch_size=t_cfg.batch_size, sampler=sampler_train, drop_last=True)
        val_l = DataLoader(AugmentedDataset(x_val,   y_val),  batch_size=t_cfg.batch_size)
        ts_l  = DataLoader(AugmentedDataset(x_test,  y_test), batch_size=t_cfg.batch_size)

        trainer_time = GaitTrainer(model_time, t_cfg)
        model_time   = trainer_time.train(tr_l, val_l)
        opt_thresh_t, metrics_t = GaitEvaluator(model_time, t_cfg.device).plot_results(ts_l, val_l, title="MODELO TRANSFORMER", save_dir=GRAFICAS_DIR)

        # ENTRENAR MODELO FFT
        logger.info("ENTRENAR MODELO FFT")
        model_fft   = FFTModel(REAL_FFT_DIM).to(t_cfg.device)
        trainer_fft = GaitTrainer(model_fft, t_cfg)

        sampler_fft = make_weighted_sampler(y_train)
        fft_tr_l  = DataLoader(AugmentedDataset(x_train_fft.reshape(x_train.shape[0], -1), y_train, is_train=True), batch_size=t_cfg.batch_size, sampler=sampler_fft, drop_last=True)
        fft_val_l = DataLoader(AugmentedDataset(x_val_fft.reshape(x_val.shape[0],   -1), y_val),   batch_size=t_cfg.batch_size)
        fft_ts_l  = DataLoader(AugmentedDataset(x_test_fft.reshape(x_test.shape[0], -1), y_test),  batch_size=t_cfg.batch_size)

        model_fft   = trainer_fft.train(fft_tr_l, fft_val_l)
        opt_thresh_f, metrics_f = GaitEvaluator(model_fft, t_cfg.device).plot_results(fft_ts_l, fft_val_l, title="MODELO FFT", save_dir=GRAFICAS_DIR)

        # ENTRENAR MODELO HIBRIDO
        logger.info("ENTRENAR MODELO HIBRIDO")
        sampler_hybrid = make_weighted_sampler(y_train)
        h_tr  = DataLoader(MultiModalAugmentedDataset(x_train, x_train_fft.reshape(x_train.shape[0], -1), y_train, is_train=True), batch_size=t_cfg.batch_size, sampler=sampler_hybrid, drop_last=True)
        h_val = DataLoader(MultiModalAugmentedDataset(x_val,   x_val_fft.reshape(x_val.shape[0],   -1), y_val),   batch_size=t_cfg.batch_size)
        h_ts  = DataLoader(MultiModalAugmentedDataset(x_test,  x_test_fft.reshape(x_test.shape[0], -1), y_test),  batch_size=t_cfg.batch_size)

        model_hybrid   = GaitHybridModel(m1_cfg, REAL_FFT_DIM, pretrained_transformer=model_time).to(t_cfg.device)
        trainer_hybrid = GaitTrainer(model_hybrid, t_cfg)
        model_hybrid   = trainer_hybrid.train(h_tr, h_val)
        opt_thresh_h, metrics_h = GaitEvaluator(model_hybrid, t_cfg.device).plot_results(h_ts, h_val, title="MODELO HIBRIDO FINAL", save_dir=GRAFICAS_DIR)

        # GUARDAR PROBABILIDADES
        y_true_t, prob_t = obtener_probabilidades(model_time,   ts_l,     t_cfg.device)
        y_true_f, prob_f = obtener_probabilidades(model_fft,    fft_ts_l, t_cfg.device)
        y_true_h, prob_h = obtener_probabilidades(model_hybrid, h_ts,     t_cfg.device)

        np.save(EVAL_DIR / "y_test.npy",           y_true_h)
        np.save(EVAL_DIR / "prob_transformer.npy",  prob_t)
        np.save(EVAL_DIR / "prob_fft.npy",          prob_f)
        np.save(EVAL_DIR / "prob_hibrido.npy",      prob_h)

        pd.DataFrame([
            {"Modelo": "Transformer", **metrics_t},
            {"Modelo": "FFT",         **metrics_f},
            {"Modelo": "Hibrido",     **metrics_h},
        ]).to_csv(EVAL_DIR / "metricas_basicas.csv", index=False)

        reporte = []
        for nombre, yt, yp in [("Transformer", y_true_t, prob_t), ("FFT", y_true_f, prob_f), ("Hibrido", y_true_h, prob_h)]:
            b, i, s = calibration_metrics(yt, yp)
            reporte.append({"Modelo": nombre, "Brier": b, "Intercepto": i, "Pendiente": s, "AUC": roc_auc_score(yt, yp)})
        pd.DataFrame(reporte).to_csv(EVAL_DIR / "calibracion.csv", index=False)

        # GUARDAR MODELOS
        torch.save(model_time.state_dict(),   OUTPUT_BASE_DIR / "modelo_transformer.pth")
        torch.save(model_fft.state_dict(),    OUTPUT_BASE_DIR / "modelo_fft.pth")
        torch.save(model_hybrid.state_dict(), OUTPUT_BASE_DIR / "modelo_hibrido.pth")
        joblib.dump(loader.scaler,            OUTPUT_BASE_DIR / "scaler_gait.joblib")
        joblib.dump(m1_cfg.model_dump(),      OUTPUT_BASE_DIR / "transformer_config.joblib")

        logger.info(f"MODELOS GUARDADOS EN {OUTPUT_BASE_DIR}")

        # RESUMEN
        print("\nRESUMEN FINAL DE METRICAS (CONJUNTO DE TEST — PACIENTE RESERVADO)")
        for name, mets in [("TRANSFORMER", metrics_t), ("FFT", metrics_f), ("HIBRIDO", metrics_h)]:
            print(f"[{name}] AUC: {mets['AUC']:.4f} SENS: {mets['Sensibilidad']:.2%} SPEC: {mets['Especificidad']:.2%} UMBRAL: {mets['Threshold']:.4f}")
        print("\n")

    except Exception as e:
        logger.critical(f"ERROR EXCEPCION CRITICA DETECTADA: {e}", exc_info=True)


if __name__ == "__main__":
    main()
