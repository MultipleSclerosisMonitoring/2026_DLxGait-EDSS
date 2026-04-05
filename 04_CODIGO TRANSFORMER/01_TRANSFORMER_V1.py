# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 15:28:53 2026
@author: jairi
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, List
from pydantic import BaseModel, FilePath
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pydantic import PositiveInt, confloat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from scipy.fft import rfft
import joblib

############################# FASE 1: CARGA Y PREPARACION DE TENSORES

class ModelConfig(BaseModel):
    h5_path: FilePath 
    test_size: float = 0.15 # 15% PARA TEST
    val_size: float = 0.15  # 15% PARA VALIDACION 
    random_state: int = 13 

class GaitDatasetLoader:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = StandardScaler()

    def get_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 1 CARGA DESDE HDF5
        x_raw, groups, labels = self._load_data()

        # 2 DIVISION 1: SEPARAR TEST
        gss_test = GroupShuffleSplit(n_splits=1, test_size=self.config.test_size, random_state=self.config.random_state)
        tv_idx, test_idx = next(gss_test.split(x_raw, labels, groups))
        
        x_tv, y_tv, groups_tv = x_raw[tv_idx], labels[tv_idx], groups[tv_idx]
        x_test_raw, y_test = x_raw[test_idx], labels[test_idx]

        # 3 DIVISION 2: SEPARAR TRAIN Y VALIDACION 
        gss_val = GroupShuffleSplit(n_splits=1, test_size=self.config.val_size, random_state=self.config.random_state)
        train_idx, val_idx = next(gss_val.split(x_tv, y_tv, groups_tv))

        x_train_raw, x_val_raw = x_tv[train_idx], x_tv[val_idx]
        y_train, y_val = y_tv[train_idx], y_tv[val_idx]

        # 4 NORMALIZACION Z-SCORE 
        x_train = self._scale_data(x_train_raw, fit=True)
        x_val = self._scale_data(x_val_raw, fit=False)
        x_test = self._scale_data(x_test_raw, fit=False)

        return (x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32), 
                y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32))

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_list, groups, labels = [], [], []
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
        n_samples, n_steps, n_features = data.shape
        flat_data = data.reshape(-1, n_features)
        if fit:
            self.scaler.fit(flat_data)
        scaled_flat = self.scaler.transform(flat_data)
        return scaled_flat.reshape(n_samples, n_steps, n_features)

    def get_all_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._load_data()

############################## FASE 2: ARQUITECTURA DEL MODELO

class TransformerConfig(BaseModel):
    input_dim: PositiveInt = 290
    model_dim: PositiveInt = 128
    nhead: PositiveInt = 8
    num_layers: PositiveInt = 4
    dropout: confloat(ge=0, le=0.5) = 0.1
    num_classes: PositiveInt = 2
    max_len: PositiveInt = 100

class GaitTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
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
    batch_size: PositiveInt = 32
    epochs: PositiveInt = 50
    lr: confloat(gt=0) = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class GaitTrainer:
    def __init__(self, model: nn.Module, config: TrainConfig, class_weights: torch.Tensor = None):
        self.model = model.to(config.device)
        self.config = config
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        print(f"# ENTRENANDO EN {self.config.device.upper()}")
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.config.device), y_batch.to(self.config.device).long()
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                acc = self.evaluate(val_loader)
                print(f"# EPOCA {epoch+1:02d} | LOSS: {total_loss/len(train_loader):.4f} | VAL ACC: {acc:.2%}")

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.config.device), y_batch.to(self.config.device)
                outputs = self.model(x_batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        return correct / total

############################## FASE 4: EVALUACION

class GaitEvaluator:
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device

    def plot_results(self, loader: DataLoader, title: str = "MODELO"):
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)
                
                preds = torch.argmax(outputs, dim=1)
                # OBTENER PROBABILIDADES PARA LA CLASE POSITIVA (MARCHA) PARA EL AUC
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                y_true.extend(y_batch.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        # METRICAS (ROC AUC)
        auc_score = roc_auc_score(y_true, y_prob)
        print(f"\n# METRICAS DETALLADAS - {title}")
        print(f"  - ROC AUC SCORE: {auc_score:.4f}")
        print("-" * 40)
        print(classification_report(y_true, y_pred, target_names=['NO MARCHA', 'MARCHA']))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NO MARCHA', 'MARCHA'], yticklabels=['NO MARCHA', 'MARCHA'])
        plt.title(f'MATRIZ CONFUSION - {title}\nAUC: {auc_score:.4f}')
        plt.xlabel('PREDICHO')
        plt.ylabel('REAL')
        plt.show()

############################## FASE 5: LOGICA DE FRECUENCIA (FFT)

class FFTProcessor:
    @staticmethod
    def get_fft_features(data: np.ndarray) -> np.ndarray:
        data_fft = np.abs(rfft(data, axis=1))
        return (data_fft / data.shape[1]).astype(np.float32)

class FFTModel(nn.Module):
    def __init__(self, input_dim: int = 51 * 290):
        super(FFTModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

############################## FASE 6: ARQUITECTURA MODELO HIBRIDO 

class GaitHybridModel(nn.Module):
    def __init__(self, t_cfg: TransformerConfig, pretrained_transformer: nn.Module = None):
        super(GaitHybridModel, self).__init__()
        
        self.transformer_branch = GaitTransformer(t_cfg)
        if pretrained_transformer is not None:
            self.transformer_branch.load_state_dict(pretrained_transformer.state_dict())
            
        self.transformer_branch.classifier = nn.Identity() 
        
        self.fft_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(51 * 290, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_t: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        feat_t = self.transformer_branch(x_t)
        feat_f = self.fft_branch(x_f)
        combined = torch.cat((feat_t, feat_f), dim=1)
        return self.fusion_head(combined)

############################## FASE 7: BLOQUE PRINCIPAL DE EJECUCION
    
class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, x_time: np.ndarray, x_fft: np.ndarray, y: np.ndarray):
        self.x_time = torch.from_numpy(x_time)
        self.x_fft = torch.from_numpy(x_fft)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_time[idx], self.x_fft[idx], self.y[idx]

############################## FASE 8: STRESS TEST (ROBUSTEZ FINAL)

def run_fft_stress_test(x_all: np.ndarray, y_all: np.ndarray, groups_all: np.ndarray, t_cfg: TrainConfig):
    """
    VALIDACION CRUZADA ESTRATIFICADA PARA ASEGURAR REPRESENTACION DE CLASES POR FOLD.
    """
    # CAMBIAMOS A StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5)
    fft_proc = FFTProcessor()
    scores = []

    print("\n" + "!"*45)
    print("# INICIANDO STRESS TEST: 5-FOLD STRATIFIED GROUP VALIDATION")
    print("!"*45)

    # AHORA PASAMOS y_all PARA QUE PUEDA ESTRATIFICAR
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
        probs = []
        with torch.no_grad():
            for xb, _ in val_l:
                out = model_cv(xb.to(t_cfg.device).float())
                probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
        
        auc = roc_auc_score(y_val, probs)
        scores.append(auc)
        print(f"  >> FOLD {fold} COMPLETADO | AUC: {auc:.4f}")

    print("-" * 45)
    print(f"# RESULTADO GLOBAL FINAL: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print("!"*45)

if __name__ == "__main__":

    try:
        H5_PATH = Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\03_CODIGOS PREPROCESAMIENTO\DATASET_LISTO\dataset_jerarquico.hdf5")
        cfg_data = ModelConfig(h5_path=H5_PATH)
        loader = GaitDatasetLoader(cfg_data)
        
        # TRAIN, VAL Y TEST
        x_train, x_val, x_test, y_train, y_val, y_test = loader.get_train_test_data()
        t_cfg = TrainConfig() 

        num_nomarcha = np.sum(y_train == 0)
        num_marcha = np.sum(y_train == 1)
        total_samples = len(y_train)
        weight_nomarcha = total_samples / (2.0 * num_nomarcha) if num_nomarcha > 0 else 1.0
        weight_marcha = total_samples / (2.0 * num_marcha) if num_marcha > 0 else 1.0
        class_weights = torch.tensor([weight_nomarcha, weight_marcha], dtype=torch.float32).to(t_cfg.device)

        print("\n" + "="*40)
        print("# DISTRIBUCION (TRAIN / VAL / TEST)")
        print(f"  TRAIN {x_train.shape[0]} muestras")
        print(f"  VAL   {x_val.shape[0]} muestras")
        print(f"  TEST  {x_test.shape[0]} muestras")
        print("="*40)

        # ==========================================
        # MODELO 1 TRANSFORMER TIEMPO
        # ==========================================
        print("\n# INICIANDO MODELO 1: TRANSFORMER TIEMPO")
        m1_cfg = TransformerConfig()
        model_time = GaitTransformer(m1_cfg)
        
        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
        test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        
        train_loader = DataLoader(train_ds, batch_size=t_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=t_cfg.batch_size)
        test_loader = DataLoader(test_ds, batch_size=t_cfg.batch_size)

        trainer_time = GaitTrainer(model_time, t_cfg, class_weights=class_weights)
        # ENTRENAMOS USANDO VAL EN VEZ DE TEST
        trainer_time.train(train_loader, val_loader) 
        
        # EXAMEN FINAL USANDO TEST
        evaluator_time = GaitEvaluator(model_time, t_cfg.device)
        evaluator_time.plot_results(test_loader, title="TRANSFORMER (TIEMPO)")

        # ==========================================
        # MODELO 2: FFT 
        # ==========================================
        print("\n# INICIANDO MODELO 2: FFT")
        fft_proc = FFTProcessor()
        x_train_fft = fft_proc.get_fft_features(x_train)
        x_val_fft = fft_proc.get_fft_features(x_val)
        x_test_fft = fft_proc.get_fft_features(x_test)

        train_fft_ds = TensorDataset(torch.from_numpy(x_train_fft), torch.from_numpy(y_train))
        val_fft_ds = TensorDataset(torch.from_numpy(x_val_fft), torch.from_numpy(y_val))
        test_fft_ds = TensorDataset(torch.from_numpy(x_test_fft), torch.from_numpy(y_test))
        
        train_fft_loader = DataLoader(train_fft_ds, batch_size=t_cfg.batch_size, shuffle=True)
        val_fft_loader = DataLoader(val_fft_ds, batch_size=t_cfg.batch_size)
        test_fft_loader = DataLoader(test_fft_ds, batch_size=t_cfg.batch_size)

        model_fft = FFTModel()
        trainer_fft = GaitTrainer(model_fft, t_cfg, class_weights=class_weights)
        trainer_fft.train(train_fft_loader, val_fft_loader)

        evaluator_fft = GaitEvaluator(model_fft, t_cfg.device)
        evaluator_fft.plot_results(test_fft_loader, title="FFT (FRECUENCIA)")

        # ==========================================
        # MODELO 3: HIBRIDO 
        # ==========================================
        print("\n# INICIANDO MODELO 3: HIBRIDO")
        train_h_ds = MultiModalDataset(x_train, x_train_fft, y_train)
        val_h_ds = MultiModalDataset(x_val, x_val_fft, y_val)
        test_h_ds = MultiModalDataset(x_test, x_test_fft, y_test)
        
        h_loader = DataLoader(train_h_ds, batch_size=t_cfg.batch_size, shuffle=True)
        h_val_loader = DataLoader(val_h_ds, batch_size=t_cfg.batch_size)
        h_test_loader = DataLoader(test_h_ds, batch_size=t_cfg.batch_size)

        model_hybrid = GaitHybridModel(m1_cfg, pretrained_transformer=model_time).to(t_cfg.device)
        optimizer_h = optim.Adam(model_hybrid.parameters(), lr=t_cfg.lr)
        criterion_h = nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(t_cfg.epochs):
            model_hybrid.train()
            epoch_loss = 0
            for xt, xf, y in h_loader:
                xt, xf, y = xt.to(t_cfg.device), xf.to(t_cfg.device), y.to(t_cfg.device)
                optimizer_h.zero_grad()
                out = model_hybrid(xt, xf)
                loss = criterion_h(out, y)
                loss.backward()
                optimizer_h.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"# EPOCA {epoch+1:02d} | LOSS: {epoch_loss/len(h_loader):.4f}")

        # EVALUACION HIBRIDA CON TEST 
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
        print("\n# METRICAS DETALLADAS - MODELO HIBRIDO")
        print(f"  - ROC AUC SCORE: {auc_score_h:.4f}")
        print("-" * 40)
        print(classification_report(y_true_h, y_pred_h, target_names=['NO MARCHA', 'MARCHA']))
        
        plt.figure(figsize=(8, 6))
        cm_h = confusion_matrix(y_true_h, y_pred_h)
        sns.heatmap(cm_h, annot=True, fmt='d', cmap='Greens', xticklabels=['NO MARCHA', 'MARCHA'], yticklabels=['NO MARCHA', 'MARCHA'])
        plt.title(f'MATRIZ CONFUSION - MODELO HIBRIDO\nAUC: {auc_score_h:.4f}')
        plt.show()

        # GUARDAR MODELOS
        print("\n" + "="*40)
        save_path = Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\MODELOS_ENTRENADOS")
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(model_time.state_dict(), save_path / "modelo_tiempo.pth")
        torch.save(model_fft.state_dict(), save_path / "modelo_frecuencia.pth")
        torch.save(model_hybrid.state_dict(), save_path / "modelo_hibrido.pth")
        joblib.dump(loader.scaler, save_path / "scaler_gait.joblib")

        print(f"# MODELOS Y SCALER GUARDADOS EN {save_path}")
        print("="*40)

        x_all, groups_all, y_all = loader.get_all_raw_data()
        run_fft_stress_test(x_all, y_all, groups_all, t_cfg)

    except Exception as e:
        print(f"\n# ERROR: {e}")