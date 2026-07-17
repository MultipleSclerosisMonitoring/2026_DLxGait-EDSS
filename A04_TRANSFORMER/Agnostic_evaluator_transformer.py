# -*- coding: utf-8 -*-
"""
Sistema de Evaluacion Agnostico para el Analisis Biomecanico Continuo de la Marcha.

Este modulo proporciona un pipeline unificado para consultar datos continuos 
de sensores desde InfluxDB, calcular los espectrogramas PSD correspondientes 
a la fase de entrenamiento, y generar predicciones segundo a segundo suavizadas 
mediante una ventana movil.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import rfft

# =========================================================
# IMPORTS DEL PROYECTO REFACTORIZADO
# =========================================================

from A01_EXTRACCION_DATOS.extract_data_plus import (
    cInfluxDB,
    SignalAligner,
    GaitFeatureExtractor,
    ExtractionParams,
    SENSOR_FIELDS_INFLUXDBMS,
    FEET,
    load_param_config
)

from A04_TRANSFORMER.AA_TRANSFORMER_V1 import (
    GaitTransformer,
    TransformerConfig,
    FFTModel,
    GaitHybridModel,
)

# =========================================================
# CONFIGURAR LOGGING
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# MOTOR DE EVALUACION CONTINUA
# =========================================================
class AgnosticEvaluator:
    """Deploys end-to-end continuous inference over an isolated time interval."""

    def __init__(self, config_path: Path, model_dir: Path) -> None:
        """Inicializa entorno y carga artefactos."""
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"DEVICE DETECTED: {self.device}")

        # 1. CLIENTE Y CONFIGURACION PYDANTIC
        self.client = cInfluxDB(config_path=str(config_path))
        yaml_cfg = load_param_config(config_path)
        params_dict = yaml_cfg.get('params', yaml_cfg)
        self.extraction_params = ExtractionParams(**params_dict)

        # 2. CARGAR ESCALADOR
        scaler_path = model_dir / "scaler_gait.joblib"
        if not scaler_path.exists(): raise FileNotFoundError("Scaler not found")
        self.scaler = joblib.load(scaler_path)

        # 3. CARGAR CONFIGURACION TRANSFORMER
        cfg_path = model_dir / "transformer_config.joblib"
        if not cfg_path.exists(): raise FileNotFoundError("Transformer config missing")
        self.t_cfg = TransformerConfig(**joblib.load(cfg_path))
        logger.info("TRANSFORMER CONFIGURATION LOADED")

        # 4. INICIALIZAR REDES
        self._init_models()

    def _init_models(self) -> None:
        """Loads trained PyTorch networks and thresholds/calibration."""
        # CARGAR MODELO TEMPORAL
        self.model_time = GaitTransformer(self.t_cfg).to(self.device)
        self.model_time.load_state_dict(torch.load(self.model_dir / "modelo_transformer.pth", map_location=self.device))
        self.model_time.eval()

        # CARGAR MODELO FFT
        fft_dim = ((self.t_cfg.max_len // 2) + 1) * self.t_cfg.input_dim
        self.model_fft = FFTModel(fft_dim).to(self.device)
        self.model_fft.load_state_dict(torch.load(self.model_dir / "modelo_fft.pth", map_location=self.device))
        self.model_fft.eval()

        # CARGAR MODELO HIBRIDO
        self.model_hybrid = GaitHybridModel(self.t_cfg, fft_dim).to(self.device)
        self.model_hybrid.load_state_dict(torch.load(self.model_dir / "modelo_hibrido.pth", map_location=self.device))
        self.model_hybrid.eval()
        logger.info("TODOS LOS MODELOS CARGADOS (TIEMPO, FFT, HIBRIDO)")

        # UMBRAL FIJO POR DECISION DE DISENO (no calibrado tipo Youden's J).
        # No se carga desde disco: un .joblib de umbral mal generado o
        # sobrescrito accidentalmente (ej. con un valor casi cero) haria que
        # el modelo clasifique casi todo como "marcha" sin ningun aviso.
        self.threshold = 0.5
        logger.info(f"UMBRAL FIJO (POR DISENO): {self.threshold:.6f}")

    def fetch_and_align_stream(self, reference: str, start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """Queries continuous sensor streams and normalizes timebase via Object Oriented Aligner."""
        raw_data_by_foot: Dict[str, pd.DataFrame] = {}
        freq_target = self.extraction_params.freq_target_hz

        # CONSULTAR INFLUXDB
        for foot in FEET:
            logger.info(f"CONSULTANDO PIE: {foot}")
            
            try:
                # EJECUTA CONSULTA API
                df = self.client.query_data(start, end, reference, foot)
                
            except Exception as e:
                # EXTRAE TOKEN PARCIAL
                tk = self.client.token
                tk_safe = f"{tk[:5]}...{tk[-5:]}" if len(tk) > 10 else "INVALID"
                
                # REGISTRA CREDENCIALES USADAS
                logger.error(f"ERROR CONSULTA: {e}")
                logger.critical(
                    f"FALLO EN EVALUACION AGNOSTICA. "
                    f"CREDENCIALES USADAS -> "
                    f"URL: {self.client.url} | "
                    f"ORG: {self.client.org} | "
                    f"BUCKET: {self.client.bucket} | "
                    f"TOKEN: {tk_safe}"
                )
                raise SystemExit(1)

            if df.empty:
                logger.warning(f"SIN DATOS: Paciente {reference} Pie {foot}")
                continue
                
            df = df.set_index("_time").sort_index()
            df.index = pd.to_datetime(df.index)
            df.rename(columns=SENSOR_FIELDS_INFLUXDBMS, inplace=True)
            raw_data_by_foot[foot] = df

        # VALIDAR AMBOS PIES
        if len(raw_data_by_foot) < 2:
            raise ValueError("AMBOS PIES SON REQUERIDOS PARA INFERENCIA.")

        # CALCULAR RANGO TEMPORAL
        common_start = max(df.index.min() for df in raw_data_by_foot.values())
        common_end = min(df.index.max() for df in raw_data_by_foot.values())
        if common_start >= common_end: raise ValueError("INTERVALO TEMPORAL INVALIDO.")

        target_idx = pd.date_range(start=common_start, end=common_end, freq=pd.Timedelta(seconds=1/freq_target), name="_time")
        aligned_data: Dict[str, pd.DataFrame] = {}

        # ALINEACION FISICA
        for foot, df_raw in raw_data_by_foot.items():
            logger.info(f"ALINEANDO PIE: {foot}")
            df_resampled = SignalAligner.uniform_timebase(df_raw, freq_target, target_idx_override=target_idx)
            if df_resampled.isnull().values.any():
                df_resampled = df_resampled.interpolate(method="cubicspline").bfill().ffill()
            aligned_data[foot] = df_resampled

        logger.info("SEÑALES ALINEADAS CON EXITO")
        return aligned_data

    def run_inference(self, aligned_data: Dict[str, pd.DataFrame], start: datetime) -> pd.DataFrame:
        """Computes PSD Spectrograms and runs rolling window inference (modelo Transformer)."""
        logger.info("EXTRAYENDO ESPECTROGRAMAS PARA INFERENCIA CONTINUA...")
        # INCLUYE mag_mag: GaitFeatureExtractor.stack_features (extract_data_plus.py)
        # tiene el orden de features hardcodeado como
        # ["acc_mag", "gyro_mag", "mag_mag", "S0", "S1", "S2"], por lo que
        # omitir "mag_mag" aqui provoca KeyError: 'mag_mag' en stack_features,
        # independientemente de si el modelo entrenado realmente lo usa.
        lfeat = ['acc_mag', 'gyro_mag', 'mag_mag', 'S0', 'S1', 'S2']
        
        # EXTRACCION USANDO LA CLASE REFACTORIZADA
        extractor = GaitFeatureExtractor(params=self.extraction_params, lfeat=lfeat)
        tensors = extractor.process_interval(aligned_data)
        
        if "Left" not in tensors or "Right" not in tensors:
            raise ValueError("FALLO EN LA EXTRACCION DE CARACTERISTICAS.")

        tensor_l = tensors["Left"][0]
        tensor_r = tensors["Right"][0]
        t_frames = tensors["Left"][1] 

        # EMPAREJAR LONGITUDES
        min_len = min(tensor_l.shape[1], tensor_r.shape[1])
        tensor_all = np.vstack([tensor_r[:, :min_len], tensor_l[:, :min_len]]).T 

        if tensor_all.shape[1] != self.t_cfg.input_dim:
            raise ValueError(f"DIMENSION INCORRECTA: Esperaba {self.t_cfg.input_dim}, obtuvo {tensor_all.shape[1]}")

        # CREAR MALLA TEMPORAL
        df_specs = pd.DataFrame(tensor_all)
        df_specs.index = [start + timedelta(seconds=float(t)) for t in t_frames[:min_len]]
        
        seq_len = self.t_cfg.max_len
        results_log = []
        
        logger.info(f"INICIANDO VENTANA DESLIZANTE SOBRE {len(df_specs)} FRAMES")

        # BUCLE DE INFERENCIA
        for i in range(0, len(df_specs) - seq_len + 1):
            window = df_specs.iloc[i : i + seq_len].values 
            current_time = df_specs.index[i + (seq_len // 2)] 

            # ESCALADO TEMPORAL
            flat_scaled = self.scaler.transform(window)
            x_time_np = np.expand_dims(flat_scaled, axis=0).astype(np.float32)

            # CARGA A DISPOSITIVO
            x_time_tensor = torch.from_numpy(x_time_np).to(self.device)

            # PREDICCION TRANSFORMER (usa directamente la serie temporal escalada,
            # sin pasar por FFT, ya que GaitTransformer.forward opera sobre x_t)
            with torch.no_grad():
                out_trans_logits = self.model_time(x_time_tensor)
                prob_trans = torch.softmax(out_trans_logits, dim=1)[0, 1].item()
                pred_trans = int(prob_trans >= self.threshold)

            results_log.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "prob_trans": prob_trans,
                "pred_trans": pred_trans
            })

            if i % 50 == 0:
                logger.info(f"{current_time.strftime('%H:%M:%S')} | PROB: {prob_trans:.3f} | PRED: {pred_trans}")

        self.client.close()

        # SUAVIZADO POR HISTERESIS
        if results_log:
            df_res = pd.DataFrame(results_log)
            df_res['prob_smoothed'] = df_res['prob_trans'].rolling(window=10, min_periods=1, center=True).mean()
            df_res['pred_final_smoothed'] = (df_res['prob_smoothed'] >= self.threshold).astype(int)
            return df_res
        
        return pd.DataFrame()

def graficar_timeline(df: pd.DataFrame, threshold: float, reference: str, output_dir: Path) -> Path:
    """
    Genera la grafica de linea de tiempo de inferencia (probabilidad
    suavizada vs umbral) y la guarda en output_dir.

    :param df: DataFrame con columnas 'timestamp' y 'prob_smoothed'.
    :param threshold: Umbral de decision usado, para la linea de referencia.
    :param reference: Identificador del paciente, usado en el titulo/nombre.
    :param output_dir: Carpeta donde se guarda la imagen.
    :return: Ruta del archivo PNG generado.
    """
    df_plot = df.copy()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_plot["timestamp"], df_plot["prob_smoothed"], label="Probabilidad", linewidth=1.2)
    ax.axhline(threshold, color="black", linestyle="--", label=f"Umbral {threshold:.2f}")

    ax.set_title("Linea de Tiempo Inferencia")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Probabilidad de Marcha")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{reference}_agnostico_timeline.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


# =========================================================
# ENTRYPOINT CLI
# =========================================================
def main() -> None:
    """Punto de entrada ejecutable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-m", "--models", type=Path, required=True)
    parser.add_argument("-r", "--reference", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "RESULTADO_AGNOSTIC",
        help="Carpeta donde se guardan el CSV y la grafica de resultados."
    )
    args = parser.parse_args()

    try:
    # Intentar ISO 8601
        start_dt = datetime.fromisoformat(
            args.start.replace("Z", "+00:00")
        ).replace(tzinfo=None)

        end_dt = datetime.fromisoformat(
            args.end.replace("Z", "+00:00")
        ).replace(tzinfo=None)

    except Exception:
        try:
            # Intentar formato clásico
            start_dt = datetime.strptime(
                args.start,
                "%Y-%m-%d %H:%M:%S"
            )

            end_dt = datetime.strptime(
                args.end,
                "%Y-%m-%d %H:%M:%S"
        )

        except Exception as e:
            logger.critical(f"ERROR FORMATO FECHA: {e}")
            sys.exit(1)

    logger.info(f"INICIANDO EVALUACION AGNOSTICA: {args.reference}")

    try:
        evaluator = AgnosticEvaluator(config_path=args.config, model_dir=args.models)
        
        logger.info("RECUPERANDO FLUJOS INFLUXDB...")
        aligned_streams = evaluator.fetch_and_align_stream(args.reference, start_dt, end_dt)

        inference_report = evaluator.run_inference(aligned_streams, start_dt)

        if not inference_report.empty:
            args.output_dir.mkdir(parents=True, exist_ok=True)

            csv_path = args.output_dir / f"agnostico_{args.reference}_transformer.csv"
            inference_report.to_csv(csv_path, index=False)
            logger.info(f"RESULTADOS GUARDADOS EN: {csv_path}")
            logger.info(f"TOTAL PREDICCIONES GENERADAS: {len(inference_report)}")

            png_path = graficar_timeline(
                inference_report, evaluator.threshold, f"{args.reference}_transformer", args.output_dir
            )
            logger.info(f"GRAFICA GUARDADA EN: {png_path}")
        else:
            logger.warning("SIN PREDICCIONES: INTERVALO TEMPORAL MUY CORTO.")

    except Exception as e:
        logger.critical(f"FALLO EN EVALUACION AGNOSTICA: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
