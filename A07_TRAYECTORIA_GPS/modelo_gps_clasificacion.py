# -*- coding: utf-8 -*-
"""
A07 REDISEÑADO — Modelo de clasificación marcha/reposo con GPS como rama
discreta de entrada (cross-attention), en un único script.

Contiene:
  1. Conector: arma el tensor PSD (rama IMU/presión, reutilizando
     GaitFeatureExtractor de A01 sin cambios) + la secuencia GPS discreta
     (delta_t, x, y, máscara) para cada segmento.
  2. Arquitectura: GaitTransformer (rama densa, reutilizado de A04) +
     encoder GPS (rama discreta) + módulo de cross-attention + cabezal de
     clasificación binaria (marcha/reposo).
  3. Entrenamiento LOPO real (mismo criterio que A04/EVALUACION_MODELOS.py):
     reentrena desde cero dejando un paciente fuera en cada fold.
  4. Reporte de métricas (AUC, PR-AUC, balanced accuracy, MCC) con
     protección para pacientes monoclase.

Requiere: preparar_dataset_clasificacion_gps.py (mismo directorio) y
segmentos_A07_v2.xlsx (lista de segmentos con mov_type).
"""

from __future__ import annotations
import sys
import gc
import copy
import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "A01_EXTRACCION_DATOS"))
sys.path.insert(0, str(PROJECT_ROOT / "A04_TRANSFORMER"))

from extract_data_plus import ExtractionParams, GaitFeatureExtractor
from AA_TRANSFORMER_V1 import GaitTransformer, TransformerConfig

import preparar_dataset_clasificacion_gps as pdc


# =============================================================================
# 1. CONECTOR: arma tensores de ambas ramas para un segmento
# =============================================================================

def _dataframe_a_dict_por_pie(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Convierte el DataFrame combinado (columnas sufijadas _Right/_Left) al formato {"Left": df, "Right": df} que espera GaitFeatureExtractor."""
    resultado = {}
    for pie in ["Left", "Right"]:
        columnas_pie = {
            c.replace(f"_{pie}", ""): c
            for c in df.columns if c.endswith(f"_{pie}")
        }
        df_pie = df[list(columnas_pie.values())].rename(columns={v: k for k, v in columnas_pie.items()})
        resultado[pie] = df_pie
    return resultado


def generar_tensores_segmento(
    df: pd.DataFrame, params: ExtractionParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera, para un segmento ya extraído (con columnas gps_* añadidas),
    el tensor PSD combinado (rama IMU/presión, 348 dim) alineado en el
    tiempo con la secuencia GPS discreta correspondiente a cada frame.

    :param df: DataFrame de un segmento, con columnas de sensores
        sufijadas _Left/_Right y las columnas gps_delta_t, gps_x_m,
        gps_y_m, gps_mascara ya calculadas.
    :param params: ExtractionParams (freq_psd_hz=75, f_start_hz=0.25,
        f_stop_hz=29.0 -- mismos usados en A04/A07 original).
    :return: Tupla (x_psd, gps_dt, gps_xy, gps_mask):
        x_psd: (n_frames, 348) tensor PSD combinado, normalizado [0,1].
        gps_dt: (n_frames,) delta_t de cada frame.
        gps_xy: (n_frames, 2) posicion (x,y) de cada frame (forward-filled).
        gps_mask: (n_frames,) mascara de observacion real (1.0/0.0).
    """
    dict_por_pie = _dataframe_a_dict_por_pie(df)
    lfeat = ["acc_mag", "gyro_mag", "mag_mag", "S0", "S1", "S2"]
    extractor = GaitFeatureExtractor(params=params, lfeat=lfeat)
    tensores = extractor.process_interval(dict_por_pie)

    if "Left" not in tensores or "Right" not in tensores:
        raise ValueError("GaitFeatureExtractor no genero tensores para ambos pies.")

    x_right_raw, t_frames, _f = tensores["Right"]
    x_left_raw, _t2, _f2 = tensores["Left"]
    min_len = min(x_right_raw.shape[1], x_left_raw.shape[1])

    x_both = np.vstack([x_right_raw[:, :min_len], x_left_raw[:, :min_len]]).T
    x_both = (x_both.astype(np.float32)) / 255.0

    t_absolutos = (df.index - df.index[0]).total_seconds().values
    n_frames = min_len

    gps_dt = np.zeros(n_frames, dtype=np.float32)
    gps_xy = np.zeros((n_frames, 2), dtype=np.float32)
    gps_mask = np.zeros(n_frames, dtype=np.float32)

    for i, t_centro in enumerate(t_frames[:n_frames]):
        idx_cercano = int(np.argmin(np.abs(t_absolutos - t_centro)))
        gps_dt[i] = df["gps_delta_t"].iloc[idx_cercano]
        gps_xy[i, 0] = df["gps_x_m"].iloc[idx_cercano]
        gps_xy[i, 1] = df["gps_y_m"].iloc[idx_cercano]
        gps_mask[i] = df["gps_mascara"].iloc[idx_cercano]

    return x_both, gps_dt, gps_xy, gps_mask


def preparar_todos_los_segmentos(
    config_path: str, ruta_excel: Path, params: ExtractionParams
):
    """
    Itera sobre todos los segmentos del Excel, extrayendo y generando
    los tensores de ambas ramas + label. Los segmentos que fallen (por
    conexion, datos insuficientes, etc.) se omiten con advertencia.

    :param config_path: Ruta al config.yaml de InfluxDB.
    :param ruta_excel: Ruta a segmentos_A07_v2.xlsx.
    :param params: ExtractionParams para la rama PSD.
    :return: Listas paralelas (x_psd, gps_dt, gps_xy, gps_mask, labels,
        pacientes), una entrada por segmento exitoso.
    """
    df_segmentos = pdc.cargar_segmentos_desde_excel(ruta_excel)

    lista_x_psd, lista_gps_dt, lista_gps_xy, lista_gps_mask = [], [], [], []
    lista_labels, lista_pacientes = [], []

    for _, fila in df_segmentos.iterrows():
        paciente = fila["Reference"]
        print(f"\n{'='*70}\nPROCESANDO: {paciente} | {fila['datefrom']} -> {fila['dateuntil']} | mov={fila['mov_type']}\n{'='*70}")
        try:
            df_seg, label = pdc.preparar_segmento_clasificacion(
                config_path, paciente, fila["datefrom"].to_pydatetime(),
                fila["dateuntil"].to_pydatetime(), int(fila["mov_type"]),
                es_utc=bool(fila["es_utc"])
            )
            x_psd, gps_dt, gps_xy, gps_mask = generar_tensores_segmento(df_seg, params)

            lista_x_psd.append(x_psd)
            lista_gps_dt.append(gps_dt)
            lista_gps_xy.append(gps_xy)
            lista_gps_mask.append(gps_mask)
            lista_labels.append(label)
            lista_pacientes.append(paciente)

            print(f"OK: {len(x_psd)} frames, label={label}")
        except Exception as e:
            print(f"FALLO: {e}")

    print(f"\n\nRESUMEN: {len(lista_x_psd)} de {len(df_segmentos)} segmentos exitosos")
    return lista_x_psd, lista_gps_dt, lista_gps_xy, lista_gps_mask, lista_labels, lista_pacientes


# =============================================================================
# 2. DATASET: ventanas fijas (max_len) para el Transformer, con rama GPS
# =============================================================================

class VentanasClasificacionGPS(Dataset):
    """
    Arma ventanas deslizantes de longitud max_len sobre cada segmento
    (mismo criterio que A04: cada ventana es una muestra), incluyendo
    la secuencia GPS discreta alineada, y repite el label del segmento
    para cada ventana generada.
    """

    def __init__(
        self, lista_x_psd: list, lista_gps_dt: list, lista_gps_xy: list,
        lista_gps_mask: list, lista_labels: list, lista_pacientes: list,
        max_len: int, step: int = 5
    ) -> None:
        """
        :param lista_x_psd: Lista de tensores PSD por segmento, (n_frames, 348) cada uno.
        :param lista_gps_dt, lista_gps_xy, lista_gps_mask: Listas paralelas de la rama GPS.
        :param lista_labels: Label (0/1) por segmento.
        :param lista_pacientes: Paciente por segmento (para LOPO).
        :param max_len: Longitud de ventana (20, igual que A04).
        :param step: Paso entre ventanas consecutivas.
        """
        self.muestras = []
        self.pacientes = []

        for x_psd, gps_dt, gps_xy, gps_mask, label, paciente in zip(
            lista_x_psd, lista_gps_dt, lista_gps_xy, lista_gps_mask, lista_labels, lista_pacientes
        ):
            n_frames = len(x_psd)
            if n_frames < max_len:
                continue
            for start in range(0, n_frames - max_len + 1, step):
                self.muestras.append({
                    "x_psd": x_psd[start:start + max_len],
                    "gps_dt": gps_dt[start:start + max_len],
                    "gps_xy": gps_xy[start:start + max_len],
                    "gps_mask": gps_mask[start:start + max_len],
                    "label": label,
                })
                self.pacientes.append(paciente)

    def __len__(self) -> int:
        return len(self.muestras)

    def __getitem__(self, idx: int):
        m = self.muestras[idx]
        gps_seq = np.stack([m["gps_dt"], m["gps_xy"][:, 0], m["gps_xy"][:, 1], m["gps_mask"]], axis=1)
        return (
            torch.from_numpy(m["x_psd"].astype(np.float32)),
            torch.from_numpy(gps_seq.astype(np.float32)),
            torch.tensor(m["label"], dtype=torch.long),
        )


# =============================================================================
# 3. ARQUITECTURA: rama IMU/presion + rama GPS + cross-attention
# =============================================================================

class EncoderGPS(nn.Module):
    """
    Encoder ligero para la secuencia GPS discreta (delta_t, x, y, mascara
    por paso). Proyecta cada paso a la misma dimension que la rama densa,
    para poder aplicar cross-attention entre ambas ramas.
    """
    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim: Dimension de embedding compartida con la rama
            IMU/presion (para que cross-attention pueda operar entre ambas).
        """
        super(EncoderGPS, self).__init__()
        self.proyeccion = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, model_dim),
        )

    def forward(self, gps_seq: torch.Tensor) -> torch.Tensor:
        """
        :param gps_seq: (batch, seq_len, 4) -- dt, x, y, mascara por paso.
        :return: (batch, seq_len, model_dim) embeddings de la rama GPS.
        """
        return self.proyeccion(gps_seq)


class ModeloClasificacionGPS(nn.Module):
    """
    Modelo de 2 ramas con cross-attention: rama densa (GaitTransformer,
    IMU/presion) + rama discreta (GPS), fusionadas mediante atencion
    cruzada antes del cabezal de clasificacion binaria (marcha/reposo).

    Diseno acorde a la recomendacion del director del TFM: el GPS entra
    como secuencia irregular de baja frecuencia con su propia mascara de
    observacion, no como target denso interpolado.
    """

    def __init__(self, t_cfg: TransformerConfig) -> None:
        """
        :param t_cfg: Configuracion del GaitTransformer base (reutilizado
            de A04, incluyendo sus pesos preentrenados si se cargan).
        """
        super(ModeloClasificacionGPS, self).__init__()

        self.transformer_imu = GaitTransformer(t_cfg)
        self.transformer_imu.classifier = nn.Identity()

        self.encoder_gps = EncoderGPS(model_dim=t_cfg.model_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=t_cfg.model_dim, num_heads=4, batch_first=True
        )

        self.cabezal_clasificacion = nn.Sequential(
            nn.Linear(t_cfg.model_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def cargar_pesos_transformer_preentrenado(self, ruta_pth: Path, device: str) -> None:
        """Carga los pesos de modelo_transformer.pth (A04) en la rama IMU/presion."""
        state_dict = torch.load(ruta_pth, map_location=device, weights_only=True)
        state_dict_encoder = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
        self.transformer_imu.load_state_dict(state_dict_encoder, strict=False)

    def forward(self, x_psd: torch.Tensor, gps_seq: torch.Tensor) -> torch.Tensor:
        """
        :param x_psd: (batch, seq_len, 348) rama IMU/presion.
        :param gps_seq: (batch, seq_len, 4) rama GPS discreta.
        :return: (batch, 2) logits de clasificacion.

        Replica manualmente el forward() de GaitTransformer hasta ANTES
        de la agregacion final (x.mean(dim=1)), para obtener los
        embeddings POR PASO (no el vector agregado) y poder aplicar
        cross-attention token-a-token contra la rama GPS. Los nombres de
        atributos (embedding, pos_embedding, transformer) coinciden
        exactamente con la implementacion real de GaitTransformer en
        AA_TRANSFORMER_V1.py.
        """
        x = self.transformer_imu.embedding(x_psd)
        seq_len = x.size(1)
        if seq_len > self.transformer_imu.pos_embedding.size(1):
            raise ValueError(
                f"Seq_len {seq_len} excede max_len {self.transformer_imu.pos_embedding.size(1)}"
            )
        x = x + self.transformer_imu.pos_embedding[:, :seq_len, :]
        emb_imu_seq = self.transformer_imu.transformer(x)

        emb_gps_seq = self.encoder_gps(gps_seq)

        fusion, _ = self.cross_attention(query=emb_imu_seq, key=emb_gps_seq, value=emb_gps_seq)

        fusion_agregada = fusion.mean(dim=1)
        return self.cabezal_clasificacion(fusion_agregada)


# =============================================================================
# 4. ENTRENAMIENTO LOPO REAL
# =============================================================================

def _metricas_fold(y_true: list, y_prob: list) -> Dict[str, float]:
    """Calcula AUC/PR-AUC/BalAcc/MCC con proteccion monoclase (mismo criterio que A04)."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred_arr = (y_prob_arr >= 0.5).astype(int)
    es_monoclase = len(np.unique(y_true_arr)) < 2

    return {
        "auc": roc_auc_score(y_true_arr, y_prob_arr) if not es_monoclase else float("nan"),
        "pr_auc": average_precision_score(y_true_arr, y_prob_arr) if not es_monoclase else float("nan"),
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
        "mcc": matthews_corrcoef(y_true_arr, y_pred_arr),
        "es_monoclase": es_monoclase,
    }


def entrenar_lopo(
    dataset_completo: VentanasClasificacionGPS,
    t_cfg: TransformerConfig,
    ruta_pesos_transformer: Path,
    device: str,
    epochs: int = 60,
    lr: float = 0.0001,
    patience: int = 10,
    ruta_guardado_incremental: Path = None,
) -> pd.DataFrame:
    """
    Entrena y evalua con LOPO real (LeaveOneGroupOut por paciente),
    reentrenando el modelo completo desde cero en cada fold.

    :param ruta_guardado_incremental: Si se indica, guarda el CSV parcial
        despues de CADA fold (no solo al final), para no perder el
        resultado ya calculado si algo falla a mitad del LOPO (ej. el
        archivo de salida esta bloqueado, o se interrumpe la ejecucion).
    :return: DataFrame con metricas por paciente + fila de media/std.
    """
    pacientes_arr = np.array(dataset_completo.pacientes)
    labels_arr = np.array([m["label"] for m in dataset_completo.muestras])
    indices = np.arange(len(dataset_completo))

    logo = LeaveOneGroupOut()
    filas_resultado = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(indices, labels_arr, pacientes_arr), 1):
        paciente_test = pacientes_arr[test_idx][0]
        y_test = labels_arr[test_idx]

        if len(np.unique(labels_arr[train_idx])) < 2:
            print(f"FOLD {fold} ({paciente_test}) OMITIDO — TRAIN CON CLASE UNICA")
            continue

        subset_train = torch.utils.data.Subset(dataset_completo, train_idx.tolist())
        subset_test = torch.utils.data.Subset(dataset_completo, test_idx.tolist())

        loader_train = DataLoader(subset_train, batch_size=16, shuffle=True)
        loader_test = DataLoader(subset_test, batch_size=16)

        modelo = ModeloClasificacionGPS(t_cfg).to(device)
        if ruta_pesos_transformer.exists():
            modelo.cargar_pesos_transformer_preentrenado(ruta_pesos_transformer, device)

        criterio = nn.CrossEntropyLoss()
        optimizador = optim.Adam(modelo.parameters(), lr=lr)

        mejor_loss, sin_mejora, mejor_estado = float("inf"), 0, None

        for epoch in range(epochs):
            modelo.train()
            perdida_epoch = 0.0
            for x_psd, gps_seq, y in loader_train:
                optimizador.zero_grad()
                out = modelo(x_psd.to(device), gps_seq.to(device))
                loss = criterio(out, y.to(device))
                loss.backward()
                optimizador.step()
                perdida_epoch += loss.item()

            if perdida_epoch < mejor_loss:
                mejor_loss, sin_mejora = perdida_epoch, 0
                mejor_estado = copy.deepcopy(modelo.state_dict())
            else:
                sin_mejora += 1
            if sin_mejora >= patience:
                break

        if mejor_estado is not None:
            modelo.load_state_dict(mejor_estado)

        modelo.eval()
        probs_fold = []
        with torch.no_grad():
            for x_psd, gps_seq, y in loader_test:
                out = modelo(x_psd.to(device), gps_seq.to(device))
                probs_fold.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())

        metricas = _metricas_fold(y_test.tolist(), probs_fold)
        filas_resultado.append({"paciente": paciente_test, **metricas})

        auc_str = f"{metricas['auc']:.4f}" if not np.isnan(metricas["auc"]) else "NaN"
        pr_auc_str = f"{metricas['pr_auc']:.4f}" if not np.isnan(metricas["pr_auc"]) else "NaN"
        print(f"FOLD {fold:02d} | {paciente_test} | AUC={auc_str} | PR-AUC={pr_auc_str} | "
              f"BalAcc={metricas['balanced_accuracy']:.4f} | MCC={metricas['mcc']:.4f}")

        # GUARDADO INCREMENTAL: si se indico una ruta, se persiste el
        # resultado parcial tras cada fold. Un fallo de guardado aqui
        # (ej. archivo bloqueado) se reporta como advertencia pero NUNCA
        # detiene el LOPO -- el entrenamiento ya invertido no debe perderse
        # ni interrumpirse por un problema de escritura en disco.
        if ruta_guardado_incremental is not None:
            try:
                pd.DataFrame(filas_resultado).to_csv(ruta_guardado_incremental, index=False)
            except Exception as e:
                print(f"  (advertencia: no se pudo guardar el progreso incremental: {e})")

        del modelo, loader_train, loader_test
        torch.cuda.empty_cache()
        gc.collect()

    df_resultado = pd.DataFrame(filas_resultado)
    if not df_resultado.empty:
        media = {"paciente": "MEDIA_POBLACION", "es_monoclase": "",
                 "auc": df_resultado["auc"].mean(skipna=True),
                 "pr_auc": df_resultado["pr_auc"].mean(skipna=True),
                 "balanced_accuracy": df_resultado["balanced_accuracy"].mean(),
                 "mcc": df_resultado["mcc"].mean()}
        df_resultado = pd.concat([df_resultado, pd.DataFrame([media])], ignore_index=True)

    return df_resultado


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena/evalua el modelo de clasificacion GPS+IMU con cross-attention (LOPO).")
    parser.add_argument("--config-yaml", type=str, default=str(PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml"))
    parser.add_argument("--excel", type=str, default=str(Path(__file__).resolve().parent / "segmentos_A07_v2.xlsx"))
    parser.add_argument("--models-dir", type=str, default=str(PROJECT_ROOT / "A05_MODELOS_ENTRENADOS"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "RESULTADOS_CLASIFICACION_GPS"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)

    t_cfg_dict = joblib.load(models_dir / "transformer_config.joblib")
    t_cfg = TransformerConfig(**t_cfg_dict)

    params = ExtractionParams(f_start_hz=0.25, f_stop_hz=29.0)

    x_psd, gps_dt, gps_xy, gps_mask, labels, pacientes = preparar_todos_los_segmentos(
        args.config_yaml, Path(args.excel), params
    )

    dataset = VentanasClasificacionGPS(
        x_psd, gps_dt, gps_xy, gps_mask, labels, pacientes, max_len=t_cfg.max_len
    )
    print(f"\nDataset total: {len(dataset)} ventanas de {len(set(pacientes))} pacientes")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_csv = out_dir / "lopo_clasificacion_gps.csv"

    df_resultado = entrenar_lopo(
        dataset, t_cfg, models_dir / "modelo_transformer.pth", device,
        epochs=args.epochs, lr=args.lr,
        ruta_guardado_incremental=ruta_csv
    )

    try:
        df_resultado.to_csv(ruta_csv, index=False)
        print(f"\nResultados guardados en: {ruta_csv}")
    except PermissionError:
        # El archivo probablemente esta abierto en Excel u otro programa.
        # En vez de perder el resultado ya calculado (que costo horas de
        # entrenamiento LOPO), se guarda en una ruta alternativa con
        # timestamp, y se avisa explicitamente al usuario.
        from datetime import datetime
        sufijo = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_alternativa = out_dir / f"lopo_clasificacion_gps_{sufijo}.csv"
        df_resultado.to_csv(ruta_alternativa, index=False)
        print(f"\nADVERTENCIA: '{ruta_csv}' estaba bloqueado (probablemente abierto "
              f"en Excel u otro programa). Resultado guardado en ruta alternativa: "
              f"{ruta_alternativa}")

    print(df_resultado.to_string(index=False))