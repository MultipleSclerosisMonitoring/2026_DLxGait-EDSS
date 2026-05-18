# -*- coding: utf-8 -*-
"""
Agnostic Evaluation System for Continuous Biomechanical Gait Analysis.

This module provides a unified pipeline to query continuous sensor data from
InfluxDB, compute the PSD Spectrograms (290 features) matching the training phase,
and yield second-by-second predictions smoothed via rolling window.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.fft import rfft

# =========================================================
# AÑADIR RUTAS DEL PROYECTO
# =========================================================
sys.path.append(str(Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\01_EXTRACCION DE DATOS")))
sys.path.append(str(Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\04_CODIGO TRANSFORMER")))

# =========================================================
# IMPORTS DEL PROYECTO
# =========================================================
from extract_data_plus import (
    cInfluxDB,
    uniform_timebase,
    SENSOR_FIELDS_INFLUXDBMS,
    FEET,
    process_interval,
    load_param_config
)

from TRANSFORMER_V1 import (
    GaitTransformer,
    TransformerConfig,
    FFTModel,
    GaitHybridModel,
)

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# AGNOSTIC EVALUATOR
# =========================================================
class AgnosticEvaluator:
    """Deploys end-to-end continuous inference over an isolated time interval."""

    def __init__(self, config_path: Path, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"DEVICE DETECTED: {self.device}")

        # 1. CLIENTE Y YAML CONFIG
        self.client = cInfluxDB(config_path=str(config_path))
        yaml_cfg = load_param_config(config_path)
        self.extraction_params = yaml_cfg.get('params', yaml_cfg)

        # 2. SCALER
        scaler_path = model_dir / "scaler_gait.joblib"
        if not scaler_path.exists(): raise FileNotFoundError("Scaler not found")
        self.scaler = joblib.load(scaler_path)

        # 3. TRANSFORMER CONFIG
        cfg_path = model_dir / "transformer_config.joblib"
        if not cfg_path.exists(): raise FileNotFoundError("Transformer config missing")
        self.t_cfg = TransformerConfig(**joblib.load(cfg_path))
        logger.info("TRANSFORMER CONFIGURATION LOADED")

        # 4. INICIALIZAR REDES
        self._init_models()

    def _init_models(self) -> None:
        """Loads trained PyTorch networks and thresholds/calibration."""
        # Tiempos y Frecuencias (290 features)
        self.model_time = GaitTransformer(self.t_cfg).to(self.device)
        self.model_time.load_state_dict(torch.load(self.model_dir / "modelo_transformer.pth", map_location=self.device))
        self.model_time.eval()

        fft_dim = ((self.t_cfg.max_len // 2) + 1) * self.t_cfg.input_dim
        self.model_fft = FFTModel(fft_dim).to(self.device)
        self.model_fft.load_state_dict(torch.load(self.model_dir / "modelo_fft.pth", map_location=self.device))
        self.model_fft.eval()

        self.model_hybrid = GaitHybridModel(self.t_cfg, fft_dim).to(self.device)
        self.model_hybrid.load_state_dict(torch.load(self.model_dir / "modelo_hibrido.pth", map_location=self.device))
        self.model_hybrid.eval()
        logger.info("TODOS LOS MODELOS CARGADOS (TIEMPO, FFT, HIBRIDO)")

        # THRESHOLD Y TEMPERATURA
        self.threshold = joblib.load(self.model_dir / "optimal_threshold_hibrido.joblib")
        try:
            self.temperature = joblib.load(self.model_dir / "optimal_temperature_hibrido.joblib")
            logger.info(f"CALIBRACION (TEMPERATURE SCALING) CARGADA: T={self.temperature:.4f}")
        except FileNotFoundError:
            self.temperature = 1.0
            logger.warning("Temperatura no encontrada. Usando T=1.0")
        logger.info(f"OPTIMAL THRESHOLD: {self.threshold:.4f}")

    def fetch_and_align_stream(self, reference: str, start: datetime, end: datetime, freq_target: float = 75.0) -> Dict[str, pd.DataFrame]:
        """Queries continuous sensor streams and normalizes timebase."""
        raw_data_by_foot: Dict[str, pd.DataFrame] = {}

        for foot in FEET:
            logger.info(f"QUERYING FOOT: {foot}")
            df = self.client.query_data(start, end, reference, foot)
            if df.empty:
                logger.warning(f"No records found for patient {reference} foot {foot}")
                continue
            df = df.set_index("_time").sort_index()
            df.index = pd.to_datetime(df.index)
            df.rename(columns=SENSOR_FIELDS_INFLUXDBMS, inplace=True)
            raw_data_by_foot[foot] = df

        if len(raw_data_by_foot) < 2:
            raise ValueError("Both feet are required for inference.")

        common_start = max(df.index.min() for df in raw_data_by_foot.values())
        common_end = min(df.index.max() for df in raw_data_by_foot.values())
        if common_start >= common_end: raise ValueError("Invalid common time interval.")

        target_idx = pd.date_range(start=common_start, end=common_end, freq=pd.Timedelta(seconds=1/freq_target), name="_time")
        aligned_data: Dict[str, pd.DataFrame] = {}

        for foot, df_raw in raw_data_by_foot.items():
            logger.info(f"ALIGNING FOOT: {foot}")
            df_resampled = uniform_timebase(df_raw, freq_target, target_idx_override=target_idx)
            if df_resampled.isnull().values.any():
                df_resampled = df_resampled.interpolate(method="cubicspline").bfill().ffill()
            aligned_data[foot] = df_resampled

        logger.info("SIGNALS ALIGNED SUCCESSFULLY")
        return aligned_data

    def run_inference(self, aligned_data: Dict[str, pd.DataFrame], start: datetime) -> pd.DataFrame:
        """Computes PSD Spectrograms and runs rolling window inference."""
        logger.info("EXTRAYENDO ESPECTROGRAMAS (290 FEATURES) PARA INFERENCIA...")
        lfeat = ['acc_mag','gyro_mag','S0','S1','S2']
        
        # Generar espectrogramas tal y como se entrenó el modelo
        tensors = process_interval(aligned_data, self.extraction_params, lfeat)
        if "Left" not in tensors or "Right" not in tensors:
            raise ValueError("Fallo en la extracción de características (process_interval).")

        tensor_l = tensors["Left"][0]
        tensor_r = tensors["Right"][0]
        t_frames = tensors["Left"][1] # Eje temporal en segundos

        # Emparejar longitudes
        min_len = min(tensor_l.shape[1], tensor_r.shape[1])
        tensor_all = np.vstack([tensor_l[:, :min_len], tensor_r[:, :min_len]]).T # Forma: (frames_tiempo, 290)

        if tensor_all.shape[1] != self.t_cfg.input_dim:
            raise ValueError(f"Dimensión incorrecta. Modelo espera {self.t_cfg.input_dim}, se generaron {tensor_all.shape[1]}")

        # Crear DataFrame temporal con los frames del espectrograma
        df_specs = pd.DataFrame(tensor_all)
        df_specs.index = [start + timedelta(seconds=float(t)) for t in t_frames[:min_len]]
        
        seq_len = self.t_cfg.max_len
        results_log = []
        
        logger.info(f"STARTING SLIDING WINDOW OVER {len(df_specs)} SPECTROGRAM FRAMES")

        # Ventana deslizante sobre los frames del espectrograma
        for i in range(0, len(df_specs) - seq_len + 1):
            window = df_specs.iloc[i : i + seq_len].values # (100, 290)
            current_time = df_specs.index[i + (seq_len // 2)] # Marca de tiempo del centro

            # 1. Escalar (100, 290)
            flat_scaled = self.scaler.transform(window)
            x_time_np = np.expand_dims(flat_scaled, axis=0).astype(np.float32)

            # 2. Generar FFT interna para la rama híbrida
            x_fft_np = np.abs(rfft(x_time_np, axis=1))
            x_fft_np = (x_fft_np / x_time_np.shape[1]).astype(np.float32).reshape(1, -1)

            # 3. A tensores CUDA/CPU
            x_time_tensor = torch.from_numpy(x_time_np).to(self.device)
            x_fft_tensor = torch.from_numpy(x_fft_np).to(self.device)

            # 4. Inferencia con Temperature Scaling
            with torch.no_grad():
                out_hybrid_logits = self.model_hybrid(x_time_tensor, x_fft_tensor)
                prob_hybrid = torch.softmax(out_hybrid_logits / self.temperature, dim=1)[0, 1].item()
                pred_hybrid = int(prob_hybrid >= self.threshold)

            results_log.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "prob_hybrid": prob_hybrid,
                "pred_hybrid": pred_hybrid
            })

            # Imprimir en consola cada 50 frames para ver avance
            if i % 50 == 0:
                logger.info(f"{current_time.strftime('%H:%M:%S')} | HYBRID PROB: {prob_hybrid:.3f} | PRED: {pred_hybrid}")

        self.client.close()

        # =================================================
        # PUNTO 6: REGULARIZACION / SUAVIZADO DE HISTÉRESIS
        # =================================================
        if results_log:
            df_res = pd.DataFrame(results_log)
            # Media móvil para evitar "parpadeos" en la marcha
            df_res['prob_smoothed'] = df_res['prob_hybrid'].rolling(window=10, min_periods=1, center=True).mean()
            df_res['pred_final_smoothed'] = (df_res['prob_smoothed'] >= self.threshold).astype(int)
            return df_res
        
        return pd.DataFrame()

# =========================================================
# ENTRYPOINT
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-m", "--models", type=Path, required=True)
    parser.add_argument("-r", "--reference", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("-o", "--output", type=Path, default=Path("./continuous_eval.csv"))
    args = parser.parse_args()

    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        logger.critical(f"DATE FORMAT ERROR: {e}")
        sys.exit(1)

    logger.info(f"STARTING AGNOSTIC EVALUATION FOR SUBJECT: {args.reference}")

    try:
        evaluator = AgnosticEvaluator(config_path=args.config, model_dir=args.models)
        
        logger.info("FETCHING CONTINUOUS STREAMS...")
        aligned_streams = evaluator.fetch_and_align_stream(args.reference, start_dt, end_dt)

        inference_report = evaluator.run_inference(aligned_streams, start_dt)

        if not inference_report.empty:
            inference_report.to_csv(args.output, index=False)
            logger.info(f"RESULTS SAVED AT: {args.output}")
            logger.info(f"TOTAL PREDICTIONS: {len(inference_report)}")
        else:
            logger.warning("No se generaron predicciones (rango temporal demasiado corto).")

    except Exception as e:
        logger.critical(f"AGNOSTIC EVALUATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()