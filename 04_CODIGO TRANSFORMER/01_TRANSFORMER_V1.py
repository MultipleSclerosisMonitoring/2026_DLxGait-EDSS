# -*- coding: utf-8 -*-
"""
Motor de Inferencia Continua basado en Deep Learning para Análisis Biomecánico.

Implementa arquitecturas en el dominio del tiempo (Transformer), frecuencia (FFT) 
e híbridas. Incluye validación cruzada estratificada por grupos para evaluar 
la robustez y garantizar la ausencia de fuga de datos (Data Leakage).
"""

import h5py
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from pydantic import BaseModel, FilePath, PositiveInt, confloat, Field
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import rfft
import joblib

# CONFIGURACION LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# EXCEPCIONES PERSONALIZADAS
class ModelExecutionError(Exception):
    """Excepción general para errores durante el entrenamiento o evaluación."""
    pass

############################# FASE 1: CARGA Y PREPARACION DE TENSORES

class ModelConfig(BaseModel):
    """
    Configuración para las rutas y proporciones de división de datos.
    """
    h5_path: FilePath 
    output_dir: Path
    test_size: confloat(gt=0, lt=1) = 0.15
    val_size: confloat(gt=0, lt=1) = 0.15 
    random_state: PositiveInt = 13 

class GaitDatasetLoader:
    """
    Clase encargada de cargar, dividir y escalar los datos desde el HDF5.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        Inicializa el cargador con la configuración y el escalador Z-score.

        :param config: Parámetros de configuración del dataset.
        :type config: ModelConfig
        """
        self.config = config
        self.scaler = StandardScaler()

    def get_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza la carga, división y normalización asegurando la identidad del paciente.

        :return: Tupla con tensores x_train, x_val, x_test, y_train, y_val, y_test.
        :rtype: Tuple[np.ndarray, ...]
        """
        try:
            # CARGAR HDF5
            x_raw, groups, labels = self._load_data()

            # DIVISION 1 TEST
            gss_test = GroupShuffleSplit(n_splits=1, test_size=self.config.test_size, random_state=self.config.random_state)
            tv_idx, test_idx = next(gss_test.split(x_raw, labels, groups))
            
            x_tv, y_tv, groups_tv = x_raw[tv_idx], labels[tv_idx], groups[tv_idx]
            x_test_raw, y_test = x_raw[test_idx], labels[test_idx]

            # DIVISION 2 VALIDACION 
            gss_val = GroupShuffleSplit(n_splits=1, test_size=self.config.val_size, random_state=self.config.random_state)
            train_idx, val_idx = next(gss_val.split(x_tv, y_tv, groups_tv))

            x_train_raw, x_val_raw = x_tv[train_idx], x_tv[val_idx]
            y_train, y_val = y_tv[train_idx], y_tv[val_idx]

            # NORMALIZACION Z-SCORE 
            x_train = self._scale_data(x_train_raw, fit=True)
            x_val = self._scale_data(x_val_raw, fit=False)
            x_test = self._scale_data(x_test_raw, fit=False)

            return (x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32), 
                    y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32))
        except Exception as e:
            logger.error(f"FALLO AL CARGAR DATOS: {e}")
            raise ModelExecutionError("Error particionando los datos.") from e

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lee internamente el archivo HDF5 iterando por paciente.

        :return: Datos (X), grupos y etiquetas (y).
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        x_list: List[np.ndarray] = []
        groups: List[str] = []
        labels: List[int] = []
        
        with h5py.File(self.config.h5_path, "r") as hf:
            for patient in hf.keys():
                for seg_chunk in hf[patient].keys():
                    for pie in hf[patient][seg_chunk].keys():
                        dataset = hf[patient][seg_chunk][pie]
                        x_list.append(dataset[:])
                        labels.append(dataset.attrs["label"])
                        base_segment = seg_chunk.split('_CH_')[0]
                        groups.append(f"{patient}_{base_segment}")
                        
        return np.array(x_list), np.array(groups), np.array(labels)

    def _scale_data(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Aplica StandardScaler a un tensor 3D.
        """
        n_samples, n_steps, n_features = data.shape
        flat_data = data.reshape(-1, n_features)
        if fit:
            self.scaler.fit(flat_data)
        scaled_flat = self.scaler.transform(flat_data)
        return scaled_flat.reshape(n_samples, n_steps, n_features)

    def get_all_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recupera dataset completo sin procesar para pruebas de estrés."""
        return self._load_data()

############################## FASE 2: ARQUITECTURA DEL MODELO

class TransformerConfig(BaseModel):
    """Hiperparámetros arquitectura Transformer."""
    input_dim: PositiveInt = 290
    model_dim: PositiveInt = 128
    nhead: PositiveInt = 8
    num_layers: PositiveInt = 4
    dropout: confloat(ge=0, le=0.5) = 0.1
    num_classes: PositiveInt = 2
    max_len: PositiveInt = 100

class GaitTransformer(nn.Module):
    """Modelo Transformer para series temporales."""
    def __init__(self, config: TransformerConfig) -> None:
        super(GaitTransformer, self).__init__()
        self.embedding = nn.Linear(config.input_dim, config.model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_len, config.model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim, nhead=config.nhead, dropout=config.dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

############################## FASE 3: ENTRENAMIENTO

class TrainConfig(BaseModel):
    """Configuración bucles de entrenamiento."""
    batch_size: PositiveInt = 32
    epochs: PositiveInt = 50
    lr: confloat(gt=0) = 0.001
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")

class GaitTrainer:
    """Motor optimizador de modelos PyTorch."""
    def __init__(self, model: nn.Module, config: TrainConfig, class_weights: Optional[torch.Tensor] = None) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Ejecuta épocas y actualiza gradientes."""
        logger.info(f"ENTRENANDO EN DISPOSITIVO: {self.config.device.upper()}")
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss: float = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device).long()
                
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                acc = self.evaluate(val_loader)
                avg_loss = total_loss / len(train_loader)
                logger.info(f"EPOCA {epoch+1:02d} | LOSS: {avg_loss:.4f} | VAL ACC: {acc:.2%}")

    def evaluate(self, loader: DataLoader) -> float:
        """Evalúa accuracy sin actualizar gradientes."""
        self.model.eval()
        correct: int = 0
        total: int = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)
                outputs = self.model(x_batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        return correct / total if total > 0 else 0.0

############################## FASE 4: EVALUACION

class GaitEvaluator:
    """Reporte visual y estadístico."""
    def __init__(self, model: nn.Module, device: str) -> None:
        self.model = model
        self.device = device

    def plot_results(self, loader: DataLoader, title: str = "MODELO") -> None:
        """Genera reporte clasificación y matriz confusión."""
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        y_prob: List[float] = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                y_true.extend(y_batch.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        auc_score = roc_auc_score(y_true, y_prob)
        logger.info(f"METRICAS {title} | ROC AUC: {auc_score:.4f}")
        print(classification_report(y_true, y_pred, target_names=['NO MARCHA', 'MARCHA']))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NO MARCHA', 'MARCHA'], yticklabels=['NO MARCHA', 'MARCHA'])
        plt.title(f'MATRIZ CONFUSION - {title}\nAUC: {auc_score:.4f}')
        plt.xlabel('PREDICHO')
        plt.ylabel('REAL')
        plt.show()

############################## FASE 5: LOGICA FFT

class FFTProcessor:
    """Procesador dominio frecuencia."""
    @staticmethod
    def get_fft_features(data: np.ndarray) -> np.ndarray:
        """Extrae magnitudes FFT."""
        data_fft = np.abs(rfft(data, axis=1))
        return (data_fft / data.shape[1]).astype(np.float32)

class FFTModel(nn.Module):
    """Clasificador espectral lineal."""
    def __init__(self, input_dim: int = 51 * 290) -> None:
        super(FFTModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

############################## FASE 6: MODELO HIBRIDO 

class GaitHybridModel(nn.Module):
    """Fusión de Transformer y FFT."""
    def __init__(self, t_cfg: TransformerConfig, pretrained_transformer: Optional[nn.Module] = None) -> None:
        super(GaitHybridModel, self).__init__()
        self.transformer_branch = GaitTransformer(t_cfg)
        
        if pretrained_transformer is not None:
            self.transformer_branch.load_state_dict(pretrained_transformer.state_dict())
            
        self.transformer_branch.classifier = nn.Identity() 
        self.fft_branch = nn.Sequential(
            nn.Flatten(), nn.Linear(51 * 290, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 128, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x_t: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        feat_t = self.transformer_branch(x_t)
        feat_f = self.fft_branch(x_f)
        return self.fusion_head(torch.cat((feat_t, feat_f), dim=1))

############################## FASE 7: DATASET MULTIMODAL
    
class MultiModalDataset(torch.utils.data.Dataset):
    """Dataset para suministrar tiempo y frecuencia simultáneamente."""
    def __init__(self, x_time: np.ndarray, x_fft: np.ndarray, y: np.ndarray) -> None:
        self.x_time = torch.from_numpy(x_time)
        self.x_fft = torch.from_numpy(x_fft)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_time[idx], self.x_fft[idx], self.y[idx]

############################## FASE 8: STRESS TEST

def run_fft_stress_test(x_all: np.ndarray, y_all: np.ndarray, groups_all: np.ndarray, t_cfg: TrainConfig) -> None:
    """Validación cruzada estratificada K-Fold."""
    sgkf = StratifiedGroupKFold(n_splits=5)
    fft_proc = FFTProcessor()
    scores: List[float] = []

    logger.info("INICIANDO STRESS TEST: 5-FOLD STRATIFIED GROUP VALIDATION")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(x_all, y_all, groups_all), 1):
        x_tr_raw, x_val_raw = x_all[train_idx], x_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        sc = StandardScaler()
        x_tr_s = sc.fit_transform(x_tr_raw.reshape(-1, 290)).reshape(x_tr_raw.shape)
        x_val_s = sc.transform(x_val_raw.reshape(-1, 290)).reshape(x_val_raw.shape)
        
        x_tr_fft = fft_proc.get_fft_features(x_tr_s)
        x_val_fft = fft_proc.get_fft_features(x_val_s)

        c_counts = np.bincount(y_tr.astype(int))
        w = torch.tensor([len(y_tr)/(2*c_counts[0]), len(y_tr)/(2*c_counts[1])]).float().to(t_cfg.device)
        
        t_cfg_cv = t_cfg.model_copy()
        t_cfg_cv.epochs = 20 
        
        model_cv = FFTModel()
        trainer_cv = GaitTrainer(model_cv, t_cfg_cv, class_weights=w)
        
        val_l = DataLoader(TensorDataset(torch.from_numpy(x_val_fft).float(), torch.from_numpy(y_val)), batch_size=t_cfg.batch_size)
        tr_l = DataLoader(TensorDataset(torch.from_numpy(x_tr_fft).float(), torch.from_numpy(y_tr)), batch_size=t_cfg.batch_size, shuffle=True)

        trainer_cv.train(tr_l, val_l)
        
        model_cv.eval()
        probs: List[float] = []
        with torch.no_grad():
            for xb, _ in val_l:
                out = model_cv(xb.to(t_cfg.device).float())
                probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
        
        auc = roc_auc_score(y_val, probs)
        scores.append(auc)
        logger.info(f"FOLD {fold} COMPLETADO | AUC: {auc:.4f}")

    logger.info(f"RESULTADO GLOBAL FINAL: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

def main() -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Entrenamiento Modelos Biomecánicos")
    parser.add_argument("--dataset", type=Path, required=True, help="Ruta al archivo HDF5")
    parser.add_argument("--output", type=Path, required=True, help="Directorio para guardar modelos")
    args = parser.parse_args()

    try:
        # INICIAR GESTOR
        cfg_data = ModelConfig(h5_path=args.dataset, output_dir=args.output)
        loader = GaitDatasetLoader(cfg_data)
        
        # PREPARAR DATOS
        x_train, x_val, x_test, y_train, y_val, y_test = loader.get_train_test_data()
        t_cfg = TrainConfig() 

        num_nomarcha = np.sum(y_train == 0)
        num_marcha = np.sum(y_train == 1)
        total_samples = len(y_train)
        w_no = total_samples / (2.0 * num_nomarcha) if num_nomarcha > 0 else 1.0
        w_yes = total_samples / (2.0 * num_marcha) if num_marcha > 0 else 1.0
        class_weights = torch.tensor([w_no, w_yes], dtype=torch.float32).to(t_cfg.device)

        logger.info(f"DISTRIBUCION - TRAIN: {x_train.shape[0]}, VAL: {x_val.shape[0]}, TEST: {x_test.shape[0]}")

        # MODELO 1: TRANSFORMER
        logger.info("INICIANDO MODELO 1: TRANSFORMER (TIEMPO)")
        m1_cfg = TransformerConfig()
        model_time = GaitTransformer(m1_cfg)
        
        train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=t_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=t_cfg.batch_size)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=t_cfg.batch_size)

        trainer_time = GaitTrainer(model_time, t_cfg, class_weights=class_weights)
        trainer_time.train(train_loader, val_loader) 
        GaitEvaluator(model_time, t_cfg.device).plot_results(test_loader, title="TRANSFORMER (TIEMPO)")

        # MODELO 2: FFT 
        logger.info("INICIANDO MODELO 2: FFT (FRECUENCIA)")
        fft_proc = FFTProcessor()
        x_train_fft = fft_proc.get_fft_features(x_train)
        x_val_fft = fft_proc.get_fft_features(x_val)
        x_test_fft = fft_proc.get_fft_features(x_test)

        train_fft_loader = DataLoader(TensorDataset(torch.from_numpy(x_train_fft), torch.from_numpy(y_train)), batch_size=t_cfg.batch_size, shuffle=True)
        val_fft_loader = DataLoader(TensorDataset(torch.from_numpy(x_val_fft), torch.from_numpy(y_val)), batch_size=t_cfg.batch_size)
        test_fft_loader = DataLoader(TensorDataset(torch.from_numpy(x_test_fft), torch.from_numpy(y_test)), batch_size=t_cfg.batch_size)

        model_fft = FFTModel()
        trainer_fft = GaitTrainer(model_fft, t_cfg, class_weights=class_weights)
        trainer_fft.train(train_fft_loader, val_fft_loader)
        GaitEvaluator(model_fft, t_cfg.device).plot_results(test_fft_loader, title="FFT (FRECUENCIA)")

        # MODELO 3: HIBRIDO 
        logger.info("INICIANDO MODELO 3: HIBRIDO")
        h_loader = DataLoader(MultiModalDataset(x_train, x_train_fft, y_train), batch_size=t_cfg.batch_size, shuffle=True)
        h_test_loader = DataLoader(MultiModalDataset(x_test, x_test_fft, y_test), batch_size=t_cfg.batch_size)

        model_hybrid = GaitHybridModel(m1_cfg, pretrained_transformer=model_time).to(t_cfg.device)
        optimizer_h = optim.Adam(model_hybrid.parameters(), lr=t_cfg.lr)
        criterion_h = nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(t_cfg.epochs):
            model_hybrid.train()
            epoch_loss: float = 0.0
            for xt, xf, y in h_loader:
                xt, xf, y = xt.to(t_cfg.device), xf.to(t_cfg.device), y.to(t_cfg.device)
                optimizer_h.zero_grad()
                out = model_hybrid(xt, xf)
                loss = criterion_h(out, y)
                loss.backward()
                optimizer_h.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"HIBRIDO EPOCA {epoch+1:02d} | LOSS: {epoch_loss/len(h_loader):.4f}")

        # EVALUAR HIBRIDO
        model_hybrid.eval()
        y_true_h, y_pred_h, y_prob_h = [], [], []
        with torch.no_grad():
            for xt, xf, y in h_test_loader:
                xt, xf = xt.to(t_cfg.device), xf.to(t_cfg.device)
                out = model_hybrid(xt, xf)
                probs = torch.softmax(out, dim=1)[:, 1]
                y_pred_h.extend(out.argmax(dim=1).cpu().numpy())
                y_prob_h.extend(probs.cpu().numpy())
                y_true_h.extend(y.numpy())

        auc_score_h = roc_auc_score(y_true_h, y_prob_h)
        logger.info(f"METRICAS MODELO HIBRIDO | ROC AUC: {auc_score_h:.4f}")
        print(classification_report(y_true_h, y_pred_h, target_names=['NO MARCHA', 'MARCHA']))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_true_h, y_pred_h), annot=True, fmt='d', cmap='Greens', xticklabels=['NO MARCHA', 'MARCHA'], yticklabels=['NO MARCHA', 'MARCHA'])
        plt.title(f'MATRIZ CONFUSION - MODELO HIBRIDO\nAUC: {auc_score_h:.4f}')
        plt.show()

        # GUARDAR MODELOS
        cfg_data.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_time.state_dict(), cfg_data.output_dir / "modelo_tiempo.pth")
        torch.save(model_fft.state_dict(), cfg_data.output_dir / "modelo_frecuencia.pth")
        torch.save(model_hybrid.state_dict(), cfg_data.output_dir / "modelo_hibrido.pth")
        joblib.dump(loader.scaler, cfg_data.output_dir / "scaler_gait.joblib")
        logger.info(f"MODELOS Y SCALER GUARDADOS EN: {cfg_data.output_dir}")

        # STRESS TEST FINAL
        x_all, groups_all, y_all = loader.get_all_raw_data()
        run_fft_stress_test(x_all, y_all, groups_all, t_cfg)

    except (OSError, RuntimeError, ValueError) as e:
        logger.critical(f"EJECUCION INTERRUMPIDA: {e}")

if __name__ == "__main__":
    main()

        x_all, groups_all, y_all = loader.get_all_raw_data()
        run_fft_stress_test(x_all, y_all, groups_all, t_cfg)

    except Exception as e:
        print(f"\n# ERROR: {e}")
