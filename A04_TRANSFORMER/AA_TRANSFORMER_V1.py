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
from typing import Tuple, List, Optional, Dict
from pydantic import BaseModel, FilePath, PositiveInt, confloat, Field
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import rfft
import joblib

# CONFIGURAR LOGGING CENTRAL
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FIJAR SEMILLAS GLOBALES
def set_seed(seed: int = 13) -> None:
    """
    Fija semillas para reproducibilidad.

    :param seed: Valor de la semilla.
    :type seed: int
    :return: Nada.
    :rtype: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# CARGAR DATOS HDF5

class ModelConfig(BaseModel):
    """
    Configuración de rutas principales.
    """
    h5_path: FilePath
    output_dir: Path
    random_state: PositiveInt = 13

class GaitDatasetLoader:
    """
    Cargador de datos biomecánicos.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        Inicializa cargador de datos.

        :param config: Configuración del modelo.
        :type config: ModelConfig
        """
        self.config = config
        self.scaler = StandardScaler()

    def get_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera particiones de datos.

        :return: Tupla con tensores divididos.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        x_raw, groups, labels = self._load_data()

        # DIVIDIR CONJUNTO TEST
        sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)
        train_val_idx, test_idx = next(sgkf_test.split(x_raw, labels, groups))

        x_tv = x_raw[train_val_idx]
        y_tv = labels[train_val_idx]
        groups_tv = groups[train_val_idx]

        x_test_raw = x_raw[test_idx]
        y_test = labels[test_idx]

        # DIVIDIR CONJUNTO VALIDACION
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)
        train_idx, val_idx = next(sgkf_val.split(x_tv, y_tv, groups_tv))

        x_train_raw = x_tv[train_idx]
        x_val_raw = x_tv[val_idx]

        y_train = y_tv[train_idx]
        y_val = y_tv[val_idx]

        # ESCALAR DATOS CONTINUOS
        x_train = self._scale_data(x_train_raw, fit=True)
        x_val = self._scale_data(x_val_raw, fit=False)
        x_test = self._scale_data(x_test_raw, fit=False)

        return (
            x_train.astype(np.float32),
            x_val.astype(np.float32),
            x_test.astype(np.float32),
            y_train.astype(np.float32),
            y_val.astype(np.float32),
            y_test.astype(np.float32)
        )

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lee datos desde HDF5.

        :return: Matrices de datos crudos.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        x_list, groups, labels = [], [], []

        # LEER ARCHIVO BINARIO
        with h5py.File(self.config.h5_path, "r") as hf:
            for patient in hf.keys():
                for seg_chunk in hf[patient].keys():
                    for pie in hf[patient][seg_chunk].keys():
                        dataset = hf[patient][seg_chunk][pie]
                        x_list.append(dataset[:])
                        labels.append(dataset.attrs["label"])
                        groups.append(f"{patient}_{seg_chunk.split('_CH_')[0]}")

        return np.array(x_list), np.array(groups), np.array(labels)

    def _scale_data(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Escala tensores temporalmente.

        :param data: Datos a escalar.
        :type data: np.ndarray
        :param fit: Ajustar escalador interno.
        :type fit: bool
        :return: Datos escalados numéricamente.
        :rtype: np.ndarray
        """
        n_samples, n_steps, n_features = data.shape
        flat_data = data.reshape(-1, n_features)

        if fit:
            self.scaler.fit(flat_data)

        scaled = self.scaler.transform(flat_data)
        return scaled.reshape(n_samples, n_steps, n_features)

    def get_all_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Devuelve matriz cruda total.

        :return: Todos los datos extraídos.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        return self._load_data()

# DEFINIR ARQUITECTURAS MODELOS

class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset con aumento gaussiano.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, is_train: bool = False) -> None:
        """
        Inicializa dataset base.

        :param x: Tensor de características.
        :type x: np.ndarray
        :param y: Tensor de etiquetas.
        :type y: np.ndarray
        :param is_train: Habilitar aumento estocástico.
        :type is_train: bool
        """
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.is_train = is_train

    def __len__(self) -> int:
        """
        Calcula longitud total dataset.

        :return: Cantidad de muestras.
        :rtype: int
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extrae y aumenta muestra.

        :param idx: Indice de muestra.
        :type idx: int
        :return: Tensor aumentado y etiqueta.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        sample = self.x[idx].clone()
        # APLICAR RUIDO GAUSSIANO
        if self.is_train and torch.rand(1).item() > 0.5:
            sample += torch.randn_like(sample) * 0.05
        return sample, self.y[idx]

class MultiModalAugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset multimodal con aumento.
    """
    def __init__(self, x_time: np.ndarray, x_fft: np.ndarray, y: np.ndarray, is_train: bool = False) -> None:
        """
        Inicializa dataset doble entrada.

        :param x_time: Tensor temporal crudo.
        :type x_time: np.ndarray
        :param x_fft: Tensor frecuencial procesado.
        :type x_fft: np.ndarray
        :param y: Etiquetas binarias.
        :type y: np.ndarray
        :param is_train: Habilitar modo entrenamiento.
        :type is_train: bool
        """
        self.x_time = torch.from_numpy(x_time).float()
        self.x_fft = torch.from_numpy(x_fft).float()
        self.y = torch.from_numpy(y).long()
        self.is_train = is_train

    def __len__(self) -> int:
        """
        Longitud de dataset multimodal.

        :return: Total de elementos.
        :rtype: int
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extrae tensores multimodales alineados.

        :param idx: Indice de lectura.
        :type idx: int
        :return: Tiempo, frecuencia y etiqueta.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        t_sample = self.x_time[idx].clone()
        # APLICAR AUMENTO TEMPORAL
        if self.is_train and torch.rand(1).item() > 0.5:
            t_sample += torch.randn_like(t_sample) * 0.05
        return t_sample, self.x_fft[idx], self.y[idx]

class TransformerConfig(BaseModel):
    """
    Configuración de red transformer.
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
    Modelo secuencial atención temporal.
    """
    def __init__(self, config: TransformerConfig) -> None:
        """
        Construye red de atención.

        :param config: Parametros de red.
        :type config: TransformerConfig
        """
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
        """
        Pase frontal del modelo.

        :param x: Tensor temporal entrada.
        :type x: torch.Tensor
        :return: Predicción de clases.
        :rtype: torch.Tensor
        """
        x = self.embedding(x)
        seq_len = x.size(1)

        # COMPROBAR LONGITUD MAXIMA
        if seq_len > self.pos_embedding.size(1):
            raise ValueError(f"Seq_len {seq_len} excede max_len {self.pos_embedding.size(1)}")

        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))

class FFTProcessor:
    """
    Procesador estático de transformada.
    """
    @staticmethod
    def get_fft_features(data: np.ndarray) -> np.ndarray:
        """
        Calcula espectro de frecuencia.

        :param data: Matriz continua temporal.
        :type data: np.ndarray
        :return: Magnitud de espectro.
        :rtype: np.ndarray
        """
        data_fft = np.abs(rfft(data, axis=1))
        return (data_fft / data.shape[1]).astype(np.float32)

class FFTModel(nn.Module):
    """
    Modelo de clasificación espectral.
    """
    def __init__(self, input_dim: int) -> None:
        """
        Inicializa red frecuencial.

        :param input_dim: Dimensión características FFT.
        :type input_dim: int
        """
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
        """
        Pase frontal frecuencial.

        :param x: Espectro de entrada.
        :type x: torch.Tensor
        :return: Salida binaria.
        :rtype: torch.Tensor
        """
        return self.classifier(x)

class GaitHybridModel(nn.Module):
    """
    Fusión multimodal tiempo frecuencia.
    """
    def __init__(self, t_cfg: TransformerConfig, fft_input_dim: int, pretrained_transformer: Optional[nn.Module] = None) -> None:
        """
        Ensambla ramas de extracción.

        :param t_cfg: Configuración transformer base.
        :type t_cfg: TransformerConfig
        :param fft_input_dim: Dimensiones transformada entrada.
        :type fft_input_dim: int
        :param pretrained_transformer: Pesos previos pre-entrenados.
        :type pretrained_transformer: Optional[nn.Module]
        """
        super(GaitHybridModel, self).__init__()
        self.transformer_branch = GaitTransformer(t_cfg)

        # CONGELAR PESOS TEMPORALES
        if pretrained_transformer is not None:
            self.transformer_branch.load_state_dict(pretrained_transformer.state_dict())
            for param in self.transformer_branch.parameters():
                param.requires_grad = False

        self.transformer_branch.classifier = nn.Identity()

        # COMPRIMIR RAMA FFT
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
        """
        Calcula predicción multimodal conjunta.

        :param x_t: Tensor dominio tiempo.
        :type x_t: torch.Tensor
        :param x_f: Tensor dominio frecuencia.
        :type x_f: torch.Tensor
        :return: Valores clasificación final.
        :rtype: torch.Tensor
        """
        feat_t = self.transformer_branch(x_t)
        feat_f = self.fft_branch(x_f)
        return self.fusion_head(torch.cat((feat_t, feat_f), dim=1))

# CONFIGURAR ENTRENAMIENTO ROBUSTO

class TrainConfig(BaseModel):
    """
    Parámetros proceso de optimización.
    """
    batch_size: PositiveInt = 64
    epochs: PositiveInt = 100
    lr: confloat(gt=0) = 0.001
    patience: PositiveInt = 10
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")

class GaitTrainer:
    """
    Clase supervisora del entrenamiento.
    """
    def __init__(self, model: nn.Module, config: TrainConfig, class_weights: Optional[torch.Tensor] = None) -> None:
        """
        Inicializa motor de optimización.

        :param model: Red neuronal objetivo.
        :type model: nn.Module
        :param config: Configuración del bucle.
        :type config: TrainConfig
        :param class_weights: Pesos balanceo clases.
        :type class_weights: Optional[torch.Tensor]
        """
        self.model = model.to(config.device)
        self.config = config
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        self.best_model_state = None
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = True) -> nn.Module:
        """
        Ejecuta iteraciones de mejora.

        :param train_loader: Lotes de entrenamiento.
        :type train_loader: DataLoader
        :param val_loader: Lotes de validación.
        :type val_loader: DataLoader
        :param verbose: Nivel de impresión.
        :type verbose: bool
        :return: Modelo ajustado óptimo.
        :rtype: nn.Module
        """
        if verbose:
            logger.info(f"ENTRENANDO DISPOSITIVO {self.config.device.upper()} AMP {self.use_amp}")

        best_val_auc = 0.0
        epochs_no_improve = 0

        # INICIAR BUCLE EPOCAS
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0

            # ITERAR LOTES DATOS
            for batch in train_loader:
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if len(batch) == 3:
                        xt = batch[0].to(self.config.device)
                        xf = batch[1].to(self.config.device)
                        yb = batch[2].to(self.config.device)
                        out = self.model(xt, xf)
                    else:
                        x_batch = batch[0].to(self.config.device)
                        yb = batch[1].to(self.config.device)
                        out = self.model(x_batch)

                    loss = self.criterion(out, yb)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()

            val_auc = self.evaluate_auc(val_loader)
            self.scheduler.step(val_auc)

            # COMPROBAR MEJORA RENDIMIENTO
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"EPOCA {epoch+1:03d} LOSS {total_loss/len(train_loader):.4f} AUC {val_auc:.4f} LR {self.optimizer.param_groups[0]['lr']:.6f}")

            # PARAR ENTRENAMIENTO PREMATURO
            if epochs_no_improve >= self.config.patience:
                if verbose:
                    logger.info(f"PARADA TEMPRANA EPOCA {epoch+1} MEJOR {best_val_auc:.4f}")
                break

        # RESTAURAR MEJOR MODELO
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.model

    def evaluate_auc(self, loader: DataLoader) -> float:
        """
        Calcula área bajo curva.

        :param loader: Dataset a evaluar.
        :type loader: DataLoader
        :return: Valor de métrica.
        :rtype: float
        """
        self.model.eval()
        y_true = []
        y_prob = []

        with torch.no_grad():
            for batch in loader:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if len(batch) == 3:
                        xt = batch[0].to(self.config.device)
                        xf = batch[1].to(self.config.device)
                        yb = batch[2].to(self.config.device)
                        out = self.model(xt, xf)
                    else:
                        x_batch = batch[0].to(self.config.device)
                        yb = batch[1].to(self.config.device)
                        out = self.model(x_batch)

                probs = torch.softmax(out, dim=1)[:, 1]
                y_true.extend(yb.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        if len(set(y_true)) > 1:
            return roc_auc_score(y_true, y_prob)

        return 0.0

# EVALUAR METRICAS CLINICAS

class GaitEvaluator:
    """
    Evaluador clínico umbral analítico.
    """
    def __init__(self, model: nn.Module, device: str) -> None:
        """
        Inicializa entorno evaluación.

        :param model: Red neuronal entrenada.
        :type model: nn.Module
        :param device: Dispositivo ejecución.
        :type device: str
        """
        self.model = model
        self.device = device

    def find_optimal_threshold(self, val_loader: DataLoader) -> float:
        """
        Obtiene umbral de decisión.

        :param val_loader: Set de calibración.
        :type val_loader: DataLoader
        :return: Probabilidad de corte.
        :rtype: float
        """
        self.model.eval()
        y_true = []
        y_prob = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    xt = batch[0].to(self.device)
                    xf = batch[1].to(self.device)
                    yb = batch[2].to(self.device)
                    out = self.model(xt, xf)
                else:
                    xb = batch[0].to(self.device)
                    yb = batch[1].to(self.device)
                    out = self.model(xb)

                probs = torch.softmax(out, dim=1)[:, 1]
                y_true.extend(yb.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        # VALIDAR MULTIPLES CLASES
        if len(set(y_true)) <= 1:
            logger.warning("CLASE UNICA DETECTADA UMBRAL 0.5")
            return 0.5

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        return float(thresholds[optimal_idx])

    def plot_results(self, test_loader: DataLoader, val_loader: DataLoader, title: str, save_dir: Optional[Path] = None) -> Tuple[float, Dict[str, float]]:
        """
        Dibuja y guarda análisis.

        :param test_loader: Lotes evaluación.
        :type test_loader: DataLoader
        :param val_loader: Lotes validación.
        :type val_loader: DataLoader
        :param title: Titulo del gráfico.
        :type title: str
        :param save_dir: Directorio exportación imagen.
        :type save_dir: Optional[Path]
        :return: Umbral y diccionario de resultados.
        :rtype: Tuple[float, Dict[str, float]]
        """
        opt_thresh = self.find_optimal_threshold(val_loader)
        self.model.eval()

        y_true = []
        y_prob = []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    xt = batch[0].to(self.device)
                    xf = batch[1].to(self.device)
                    yb = batch[2].to(self.device)
                    out = self.model(xt, xf)
                else:
                    xb = batch[0].to(self.device)
                    yb = batch[1].to(self.device)
                    out = self.model(xb)

                probs = torch.softmax(out, dim=1)[:, 1]
                y_true.extend(yb.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        y_pred = (np.array(y_prob) >= opt_thresh).astype(int)

        # DIBUJAR MATRIZ CONFUSION
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

        if len(set(y_true)) > 1:
            auc_score = roc_auc_score(y_true, y_prob)
        else:
            auc_score = 0.0

        metrics = {
            "Sensibilidad": sensitivity,
            "Especificidad": specificity,
            "FPR": fpr_val,
            "AUC": auc_score,
            "Threshold": opt_thresh
        }

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REPOSO', 'MARCHA'], yticklabels=['REPOSO', 'MARCHA'])
        plt.title(f'{title}\nAUC: {auc_score:.2f} | Umbral Optimo: {opt_thresh:.2f}')
        plt.ylabel('Real')
        plt.xlabel('Predicho')

        # GUARDAR GRAFICA DISCO
        if save_dir:
            file_name = f"{title.replace(' ', '_')}_Matriz_Confusion.png"
            plt.savefig(save_dir / file_name, dpi=300, bbox_inches='tight')
            logger.info(f"GRAFICA ALMACENADA RUTA LOCAL")

        plt.show(block=False)
        plt.pause(1)
        plt.close('all')
        
        return opt_thresh, metrics

# VALIDACION CRUZADA KFOLD

def run_hybrid_stress_test(x_all: np.ndarray, y_all: np.ndarray, groups_all: np.ndarray, t_cfg: TrainConfig, m1_cfg: TransformerConfig) -> None:
    """
    Aplica K-Fold al híbrido.

    :param x_all: Características absolutas.
    :type x_all: np.ndarray
    :param y_all: Etiquetas absolutas.
    :type y_all: np.ndarray
    :param groups_all: Pacientes de agrupamiento.
    :type groups_all: np.ndarray
    :param t_cfg: Configuración de bucle.
    :type t_cfg: TrainConfig
    :param m1_cfg: Arquitectura tiempo real.
    :type m1_cfg: TransformerConfig
    :return: Nada, salida consola.
    :rtype: None
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
    fft_proc = FFTProcessor()
    scores: List[float] = []

    logger.info("EJECUTANDO TEST ESTRES GLOBAL")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(x_all, y_all, groups_all), 1):
        x_tr_raw = x_all[train_idx]
        x_val_raw = x_all[val_idx]
        y_tr = y_all[train_idx]
        y_val = y_all[val_idx]

        sc = StandardScaler()
        x_tr_s = sc.fit_transform(x_tr_raw.reshape(-1, x_tr_raw.shape[2])).reshape(x_tr_raw.shape)
        x_val_s = sc.transform(x_val_raw.reshape(-1, x_val_raw.shape[2])).reshape(x_val_raw.shape)

        x_tr_fft = fft_proc.get_fft_features(x_tr_s)
        x_val_fft = fft_proc.get_fft_features(x_val_s)
        real_fft_dim = x_tr_fft.shape[1] * x_tr_fft.shape[2]

        c_counts = np.bincount(y_tr.astype(int), minlength=2)
        w0 = len(y_tr) / (2 * c_counts[0]) if c_counts[0] > 0 else 1.0
        w1 = len(y_tr) / (2 * c_counts[1]) if c_counts[1] > 0 else 1.0
        w = torch.tensor([w0, w1]).float().to(t_cfg.device)

        tr_time_loader = DataLoader(TensorDataset(torch.from_numpy(x_tr_s).float(), torch.from_numpy(y_tr)), batch_size=t_cfg.batch_size, shuffle=True)
        val_time_loader = DataLoader(TensorDataset(torch.from_numpy(x_val_s).float(), torch.from_numpy(y_val)), batch_size=t_cfg.batch_size)

        cv_cfg = t_cfg.model_copy()
        cv_cfg.epochs = 25
        cv_cfg.patience = 5

        model_time_cv = GaitTransformer(m1_cfg).to(cv_cfg.device)
        trainer_time_cv = GaitTrainer(model_time_cv, cv_cfg, class_weights=w)
        model_time_cv = trainer_time_cv.train(tr_time_loader, val_time_loader, verbose=False)

        model_hybrid_cv = GaitHybridModel(m1_cfg, real_fft_dim, pretrained_transformer=model_time_cv).to(cv_cfg.device)

        h_tr_loader = DataLoader(MultiModalAugmentedDataset(x_tr_s, x_tr_fft.reshape(x_tr_s.shape[0], -1), y_tr), batch_size=cv_cfg.batch_size, shuffle=True)
        h_val_loader = DataLoader(MultiModalAugmentedDataset(x_val_s, x_val_fft.reshape(x_val_s.shape[0], -1), y_val), batch_size=cv_cfg.batch_size)

        trainer_h_cv = GaitTrainer(model_hybrid_cv, cv_cfg, class_weights=w)
        model_hybrid_cv = trainer_h_cv.train(h_tr_loader, h_val_loader, verbose=False)

        model_hybrid_cv.eval()
        probs: List[float] = []

        with torch.no_grad():
            for xt, xf, _ in h_val_loader:
                out = model_hybrid_cv(xt.to(cv_cfg.device).float(), xf.to(cv_cfg.device).float())
                probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())

        if len(set(y_val)) > 1:
            auc = roc_auc_score(y_val, probs)
            scores.append(auc)
            logger.info(f"PLIEGUE CRUZADO {fold} RESULTADO {auc:.4f}")
        else:
            logger.warning(f"PLIEGUE CRUZADO {fold} OMITIDO")

    if scores:
        logger.info(f"EVALUACION ROBUSTA FINAL PROMEDIO {np.mean(scores):.4f}")

# FUNCION PRINCIPAL MAIN

def main() -> None:
    """
    Ejecuta pipeline completo inferencia.

    :return: Nada, guarda modelos disco.
    :rtype: None
    """
    set_seed(13)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    # CONFIGURAR RUTAS RELATIVAS
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_BASE_DIR = BASE_DIR / "A05_MODELOS_ENTRENADOS"
    GRAFICAS_DIR = OUTPUT_BASE_DIR / "graficas"
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        cfg_data = ModelConfig(h5_path=args.dataset, output_dir=OUTPUT_BASE_DIR)
        loader = GaitDatasetLoader(cfg_data)
        x_train, x_val, x_test, y_train, y_val, y_test = loader.get_train_test_data()

        REAL_INPUT_DIM = x_train.shape[2]
        REAL_SEQ_LEN = x_train.shape[1]

        fft_proc = FFTProcessor()
        x_train_fft = fft_proc.get_fft_features(x_train)
        x_val_fft = fft_proc.get_fft_features(x_val)
        x_test_fft = fft_proc.get_fft_features(x_test)

        REAL_FFT_DIM = x_train_fft.shape[1] * x_train_fft.shape[2]
        t_cfg = TrainConfig()

        class_counts = np.bincount(y_train.astype(int), minlength=2)
        w0 = len(y_train) / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0
        w1 = len(y_train) / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
        w = torch.tensor([w0, w1]).float().to(t_cfg.device)

        # ENTRENAR MODELO TRANSFORMER
        logger.info("ENTRENAR MODELO TEMPORAL")
        m1_cfg = TransformerConfig(input_dim=REAL_INPUT_DIM, max_len=REAL_SEQ_LEN)
        model_time = GaitTransformer(m1_cfg)

        tr_l = DataLoader(AugmentedDataset(x_train, y_train, is_train=True), batch_size=t_cfg.batch_size, shuffle=True)
        val_l = DataLoader(AugmentedDataset(x_val, y_val), batch_size=t_cfg.batch_size)
        ts_l = DataLoader(AugmentedDataset(x_test, y_test), batch_size=t_cfg.batch_size)

        trainer_time = GaitTrainer(model_time, t_cfg, class_weights=w)
        model_time = trainer_time.train(tr_l, val_l)
        _, metrics_t = GaitEvaluator(model_time, t_cfg.device).plot_results(ts_l, val_l, title="MODELO TRANSFORMER", save_dir=GRAFICAS_DIR)
        
        # ENTRENAR MODELO FFT (INDEPENDIENTE)
        logger.info("ENTRENAR MODELO FFT")
        model_fft = FFTModel(REAL_FFT_DIM).to(t_cfg.device)
        trainer_fft = GaitTrainer(model_fft, t_cfg, class_weights=w)
        
        fft_tr_l = DataLoader(AugmentedDataset(x_train_fft.reshape(x_train.shape[0], -1), y_train, is_train=True), batch_size=t_cfg.batch_size, shuffle=True)
        fft_val_l = DataLoader(AugmentedDataset(x_val_fft.reshape(x_val.shape[0], -1), y_val), batch_size=t_cfg.batch_size)
        fft_ts_l = DataLoader(AugmentedDataset(x_test_fft.reshape(x_test.shape[0], -1), y_test), batch_size=t_cfg.batch_size)
        
        model_fft = trainer_fft.train(fft_tr_l, fft_val_l)
        _, metrics_f = GaitEvaluator(model_fft, t_cfg.device).plot_results(fft_ts_l, fft_val_l, title="MODELO FFT", save_dir=GRAFICAS_DIR)

        # ENTRENAR MODELO HIBRIDO
        logger.info("ENTRENAR MODELO HIBRIDO")
        h_tr = DataLoader(MultiModalAugmentedDataset(x_train, x_train_fft.reshape(x_train.shape[0], -1), y_train, is_train=True), batch_size=t_cfg.batch_size, shuffle=True)
        h_val = DataLoader(MultiModalAugmentedDataset(x_val, x_val_fft.reshape(x_val.shape[0], -1), y_val), batch_size=t_cfg.batch_size)
        h_ts = DataLoader(MultiModalAugmentedDataset(x_test, x_test_fft.reshape(x_test.shape[0], -1), y_test), batch_size=t_cfg.batch_size)

        model_hybrid = GaitHybridModel(m1_cfg, REAL_FFT_DIM, pretrained_transformer=model_time).to(t_cfg.device)
        trainer_hybrid = GaitTrainer(model_hybrid, t_cfg, class_weights=w)
        model_hybrid = trainer_hybrid.train(h_tr, h_val)

        opt_thresh_h, metrics_h = GaitEvaluator(model_hybrid, t_cfg.device).plot_results(h_ts, h_val, title="MODELO HIBRIDO FINAL", save_dir=GRAFICAS_DIR)

        # GUARDAR ARTEFACTOS MODELO
        torch.save(model_time.state_dict(), OUTPUT_BASE_DIR / "modelo_transformer.pth")
        torch.save(model_fft.state_dict(), OUTPUT_BASE_DIR / "modelo_fft.pth")
        torch.save(model_hybrid.state_dict(), OUTPUT_BASE_DIR / "modelo_hibrido.pth")
        joblib.dump(loader.scaler, OUTPUT_BASE_DIR / "scaler_gait.joblib")
        joblib.dump(opt_thresh_h, OUTPUT_BASE_DIR / "optimal_threshold_hibrido.joblib")
        joblib.dump(m1_cfg.model_dump(), OUTPUT_BASE_DIR / "transformer_config.joblib")

        logger.info(f"MODELOS DESCARGADOS RUTA LOCAL EN {OUTPUT_BASE_DIR}")

        # IMPRIMIR REPORTE FINAL
        print("\nRESUMEN FINAL DE METRICAS (CONJUNTO DE TEST)")
        for name, mets in [("TRANSFORMER", metrics_t), ("FFT", metrics_f), ("HIBRIDO", metrics_h)]:
            print(f"[{name}] AUC: {mets['AUC']:.4f} SENS: {mets['Sensibilidad']:.2%} SPEC: {mets['Especificidad']:.2%} UMBRAL: {mets['Threshold']:.4f}")
        print("\n")

        # EJECUTAR PRUEBA ESTRES
        x_all, groups_all, y_all = loader.get_all_raw_data()
        run_hybrid_stress_test(x_all, y_all, groups_all, t_cfg, m1_cfg)

    except Exception as e:
        logger.critical(f"ERROR EXCEPCION CRITICA DETECTADA: {e}")

if __name__ == "__main__":
    main()