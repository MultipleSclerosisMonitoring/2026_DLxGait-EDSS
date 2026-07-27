# -*- coding: utf-8 -*-
"""
Entrena TrajectoryModel (fine-tuning de GaitTransformer) para predecir la
posicion (X, Y) del cuerpo a partir de espectrogramas PSD combinados de
presion+IMU de ambos pies, usando como ground truth de entrenamiento la
interpolacion PCHIP de los puntos GPS reales, y evaluando el error final
contra esos mismos puntos GPS reales (nunca vistos como interpolacion
durante el entrenamiento).
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "A01_EXTRACCION_DATOS"))
sys.path.insert(0, str(PROJECT_ROOT / "A04_TRANSFORMER"))

from extract_data_plus import ExtractionParams
from AA_TRANSFORMER_V1 import TransformerConfig

import preparar_dataset_trayectoria as pdt
import conector_trayectoria as ct
from trajectory_model import TrajectoryModel, VentanasSecuenciaDataset


def entrenar_trajectory_model(
    df_train,
    df_val,
    params: ExtractionParams,
    t_cfg: TransformerConfig,
    ruta_pesos_preentrenados: Path,
    device: str,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 15,
    congelar_transformer: bool = False
) -> tuple:
    """
    Entrena TrajectoryModel con fine-tuning sobre modelo_transformer.pth,
    usando el ground truth interpolado (PCHIP) de df_train, y evalua el
    error final contra el GPS real de df_val.

    :param df_train: DataFrame de entrenamiento (con X_m_gt, Y_m_gt).
    :param df_val: DataFrame de validacion (con X_m, Y_m reales, GPS).
    :param params: ExtractionParams para generar los espectrogramas PSD.
    :param t_cfg: TransformerConfig, debe coincidir con el checkpoint.
    :param ruta_pesos_preentrenados: Ruta a modelo_transformer.pth.
    :param device: "cuda" o "cpu".
    :param epochs: Numero maximo de epocas.
    :param lr: Tasa de aprendizaje.
    :param patience: Paciencia para early stopping.
    :param congelar_transformer: Si True, solo entrena el cabezal de regresion.
    :return: Tupla (modelo entrenado, historial de perdidas, error_val_metros).
    """
    # GENERAR TENSORES DE ENTRENAMIENTO (ground truth interpolado PCHIP)
    x_train_psd, y_train_pos = ct.generar_tensores_psd_y_posicion(
        df_train, params, columna_x="X_m_gt", columna_y="Y_m_gt"
    )

    # GENERAR TENSORES DE VALIDACION (GPS real, SIN interpolar)
    x_val_psd, y_val_pos = ct.generar_tensores_psd_y_posicion(
        df_val, params, columna_x="X_m", columna_y="Y_m"
    )

    max_len = t_cfg.max_len
    dataset_train = VentanasSecuenciaDataset(x_train_psd, y_train_pos, max_len)

    if len(dataset_train) == 0:
        raise ValueError(
            f"Sin ventanas de entrenamiento: se necesitan al menos {max_len} frames, "
            f"se obtuvieron {len(x_train_psd)}."
        )

    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    # MODELO: FINE-TUNING DE GaitTransformer
    modelo = TrajectoryModel(t_cfg, congelar_transformer=congelar_transformer).to(device)
    modelo.cargar_pesos_preentrenados(ruta_pesos_preentrenados, device)

    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="min", factor=0.5, patience=5)

    mejor_loss = float("inf")
    sin_mejora = 0
    mejor_estado = None
    historial = []

    for epoch in range(epochs):
        modelo.train()
        perdida_epoch = 0.0

        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizador.zero_grad()
            pred = modelo(xb)
            loss = criterio(pred, yb)
            loss.backward()
            optimizador.step()
            perdida_epoch += loss.item() * xb.size(0)

        perdida_epoch /= len(dataset_train)
        historial.append(perdida_epoch)
        scheduler.step(perdida_epoch)

        if perdida_epoch < mejor_loss:
            mejor_loss = perdida_epoch
            sin_mejora = 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1

        if epoch % 10 == 0 or sin_mejora >= patience:
            print(f"Epoch {epoch:03d} | Loss (MSE m^2): {perdida_epoch:.4f} | Sin mejora: {sin_mejora}")

        if sin_mejora >= patience:
            print(f"Early stopping en epoch {epoch}")
            break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)

    # EVALUACION FINAL: error de trayectoria en metros, contra GPS REAL
    # (nunca contra la interpolacion), usando ventanas de los 18 puntos val
    dataset_val = VentanasSecuenciaDataset(x_val_psd, y_val_pos, min(max_len, len(x_val_psd)))

    error_val_metros = None
    if len(dataset_val) > 0:
        modelo.eval()
        errores = []
        with torch.no_grad():
            for i in range(len(dataset_val)):
                xb, yb = dataset_val[i]
                xb = xb.unsqueeze(0).to(device)
                pred = modelo(xb).cpu().numpy()[0]
                error_euclidiano = np.sqrt(np.sum((pred - yb.numpy()) ** 2))
                errores.append(error_euclidiano)
        error_val_metros = float(np.mean(errores))
        print(f"\nError medio de trayectoria vs GPS real: {error_val_metros:.3f} m "
              f"(sobre {len(errores)} puntos de validacion)")
    else:
        print(
            "\nADVERTENCIA: no hay suficientes frames de validacion contiguos "
            f"para formar ni una ventana de longitud {max_len}. "
            "No se pudo calcular el error final contra GPS real."
        )

    return modelo, historial, error_val_metros


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de trayectoria (fine-tuning GaitTransformer).")
    parser.add_argument("--paciente", type=str, required=True)
    parser.add_argument("--inicio", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--fin", type=str, required=True, help="'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--config-yaml", type=str, default=str(PROJECT_ROOT / "A01_EXTRACCION_DATOS" / "config.yaml"))
    parser.add_argument("--models-dir", type=str, default=str(PROJECT_ROOT / "A05_MODELOS_ENTRENADOS"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--congelar-transformer", action="store_true")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "A07_TRAYECTORIA_GPS" / "RESULTADOS_TRAYECTORIA")
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)

    import joblib
    t_cfg_dict = joblib.load(models_dir / "transformer_config.joblib")
    t_cfg = TransformerConfig(**t_cfg_dict)

    params = ExtractionParams(f_start_hz=0.25, f_stop_hz=29.0)

    inicio_dt = datetime.strptime(args.inicio, "%Y-%m-%d %H:%M:%S")
    fin_dt = datetime.strptime(args.fin, "%Y-%m-%d %H:%M:%S")

    df_train, df_val = pdt.preparar_dataset_trayectoria(
        args.config_yaml, args.paciente, inicio_dt, fin_dt
    )

    modelo, historial, error_val_metros = entrenar_trajectory_model(
        df_train, df_val, params, t_cfg,
        ruta_pesos_preentrenados=models_dir / "modelo_transformer.pth",
        device=device, epochs=args.epochs, lr=args.lr,
        congelar_transformer=args.congelar_transformer
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), out_dir / f"{args.paciente}_trajectory_model.pth")

    print(f"\nModelo guardado en: {out_dir / f'{args.paciente}_trajectory_model.pth'}")
    if error_val_metros is not None:
        print(f"ERROR FINAL DE TRAYECTORIA vs GPS REAL: {error_val_metros:.3f} metros")
