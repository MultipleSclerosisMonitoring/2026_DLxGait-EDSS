"""
Sistema de Evaluacion Agnostico UNIFICADO para el Analisis Biomecanico
Continuo de la Marcha. Permite elegir, por consola, cual de los 4 modelos
usar para la inferencia continua sobre un segmento (paciente + rango de
fechas), sin necesidad de mantener scripts separados por modelo:
    1. FFT       -> FFTModel solo (modelo_fft.pth)
    2. Transf    -> GaitTransformer solo (modelo_transformer.pth)
    3. Hibrido Early Fusion -> GaitHybridModel (modelo_hibrido.pth)
    4. Hibrido Late Fusion (Media Geometrica) -> combina FFT + Transformer
       por separado, sin .pth adicional (la combinacion es una formula,
       no un modelo entrenado) -- se eligio Media Geometrica por ser la
       variante de Late Fusion con mejor AUC en LOPO real (0.9002),
       superando a Voto Mayoritario (0.8651) y Meta-Clasificador (0.8918).
Todos los modos comparten la misma extraccion de features (presion+IMU
de ambos pies -> PSD combinado "Both" de 348 dimensiones, mismo esquema
que el dataset de entrenamiento original) y, cuando corresponde, el mismo
calculo de FFT sobre la ventana temporal escalada.
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
MODELOS_DISPONIBLES = {
    "1": "fft",
    "2": "transformer",
    "3": "hibrido_early",
    "4": "hibrido_late",
}
NOMBRES_LEGIBLES = {
    "fft": "FFT",
    "transformer": "Transformer",
    "hibrido_early": "Hibrido Early Fusion",
    "hibrido_late": "Hibrido Late Fusion (Media Geometrica)",
}
class AgnosticEvaluatorUnificado:
    """Motor de inferencia continua que soporta los 4 modos de modelo."""
    def __init__(self, config_path: Path, model_dir: Path, modelo: str) -> None:
        """
        Inicializa entorno y carga UNICAMENTE los artefactos necesarios
        para el modelo elegido (no carga los 3 .pth siempre, solo los
        que requiere el modo seleccionado).
        :param config_path: Ruta al config.yaml de InfluxDB.
        :param model_dir: Ruta a A05_MODELOS_ENTRENADOS.
        :param modelo: Uno de "fft", "transformer", "hibrido_early", "hibrido_late".
        """
        self.model_dir = model_dir
        self.modelo = modelo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"DEVICE DETECTED: {self.device}")
        logger.info(f"MODELO SELECCIONADO: {NOMBRES_LEGIBLES[modelo]}")
        self.client = cInfluxDB(config_path=str(config_path))
        yaml_cfg = load_param_config(config_path)
        params_dict = yaml_cfg.get('params', yaml_cfg)
        self.extraction_params = ExtractionParams(**params_dict)
        scaler_path = model_dir / "scaler_gait.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError("Scaler not found")
        self.scaler = joblib.load(scaler_path)
        cfg_path = model_dir / "transformer_config.joblib"
        if not cfg_path.exists():
            raise FileNotFoundError("Transformer config missing")
        self.t_cfg = TransformerConfig(**joblib.load(cfg_path))
        # UMBRAL FIJO POR DISENO (no calibrado tipo Youden's J). No se
        # carga desde disco: un .joblib de umbral mal generado o
        # sobrescrito accidentalmente haria que el modelo clasifique casi
        # todo como "marcha" sin ningun aviso (ver historial de depuracion).
        self.threshold = 0.5
        logger.info(f"UMBRAL FIJO (POR DISENO): {self.threshold:.6f}")
        self._init_modelos_necesarios()
    def _init_modelos_necesarios(self) -> None:
        """Carga solo los .pth que requiere el modo elegido."""
        fft_dim = ((self.t_cfg.max_len // 2) + 1) * self.t_cfg.input_dim
        if self.modelo in ("transformer", "hibrido_late"):
            self.model_time = GaitTransformer(self.t_cfg).to(self.device)
            self.model_time.load_state_dict(
                torch.load(self.model_dir / "modelo_transformer.pth", map_location=self.device)
            )
            self.model_time.eval()
            logger.info("MODELO CARGADO: modelo_transformer.pth")
        if self.modelo in ("fft", "hibrido_late"):
            self.model_fft = FFTModel(fft_dim).to(self.device)
            self.model_fft.load_state_dict(
                torch.load(self.model_dir / "modelo_fft.pth", map_location=self.device)
            )
            self.model_fft.eval()
            logger.info("MODELO CARGADO: modelo_fft.pth")
        if self.modelo == "hibrido_early":
            self.model_hybrid = GaitHybridModel(self.t_cfg, fft_dim).to(self.device)
            self.model_hybrid.load_state_dict(
                torch.load(self.model_dir / "modelo_hibrido.pth", map_location=self.device)
            )
            self.model_hybrid.eval()
            logger.info("MODELO CARGADO: modelo_hibrido.pth")

    def fetch_and_align_stream(self, reference: str, start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """
        Queries continuous sensor streams and normalizes timebase via
        Object Oriented Aligner.

        CORRECCION DE REPRODUCIBILIDAD (2026-08-05): la malla temporal
        uniforme (target_idx) ya NO se ancla a common_start (el minimo
        real de los datos recibidos), porque ese valor depende de que
        rango exacto se le pidio a InfluxDB -- pedir una ventana mas
        amplia alrededor del mismo instante real desplaza ligeramente
        common_start, lo cual cambia la FASE de la malla de remuestreo
        (los timestamps objetivo caen en instantes distintos respecto a
        los datos crudos), y esa diferencia de fase se propaga a traves
        de la interpolacion cubic spline, el extractor de features (PSD)
        y finalmente la prediccion del modelo. Se confirmo empiricamente:
        el mismo paciente, mismo instante real, con dos ventanas de
        consulta distintas (una anidada dentro de la otra), dio
        predicciones de marcha/reposo opuestas en el tramo compartido.

        Ahora la malla se ancla a un ORIGEN FIJO E INDEPENDIENTE del
        rango consultado (medianoche del dia de common_start), de modo
        que un mismo instante real cae siempre en la misma fase de la
        malla sin importar que ventana de consulta se use alrededor de
        el. common_start/common_end siguen usandose para RECORTAR la
        malla al tramo con datos de ambos pies, pero no para fijar su fase.
        """
        raw_data_by_foot: Dict[str, pd.DataFrame] = {}
        freq_target = self.extraction_params.freq_target_hz
        for foot in FEET:
            logger.info(f"CONSULTANDO PIE: {foot}")
            try:
                df = self.client.query_data(start, end, reference, foot)
            except Exception as e:
                tk = self.client.token
                tk_safe = f"{tk[:5]}...{tk[-5:]}" if len(tk) > 10 else "INVALID"
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
        if len(raw_data_by_foot) < 2:
            raise ValueError("AMBOS PIES SON REQUERIDOS PARA INFERENCIA.")
        common_start = max(df.index.min() for df in raw_data_by_foot.values())
        common_end = min(df.index.max() for df in raw_data_by_foot.values())
        if common_start >= common_end:
            raise ValueError("INTERVALO TEMPORAL INVALIDO.")

        # ANCLAJE FIJO DE FASE (independiente del rango consultado)
        origen_fijo = pd.Timestamp(common_start).normalize()
        periodo = pd.Timedelta(seconds=1 / freq_target)
        n_periodos_desde_origen = np.ceil((common_start - origen_fijo) / periodo)
        primer_punto_malla = origen_fijo + n_periodos_desde_origen * periodo

        target_idx = pd.date_range(
            start=primer_punto_malla, end=common_end,
            freq=periodo, name="_time"
        )
        aligned_data: Dict[str, pd.DataFrame] = {}
        for foot, df_raw in raw_data_by_foot.items():
            logger.info(f"ALINEANDO PIE: {foot}")
            df_resampled = SignalAligner.uniform_timebase(df_raw, freq_target, target_idx_override=target_idx)
            if df_resampled.isnull().values.any():
                df_resampled = df_resampled.interpolate(method="cubicspline").bfill().ffill()
            aligned_data[foot] = df_resampled
        logger.info("SEÑALES ALINEADAS CON EXITO (malla con fase fija, reproducible entre rangos)")
        return aligned_data

    def _calibrar_duracion_ventana_local(
        self, aligned_data: Dict[str, pd.DataFrame], freq_target: float
    ) -> int:
        """
        Determina EMPIRICAMENTE (sin asumir formulas de n_per_seg/overlap
        a mano) cuantas muestras crudas se necesitan para que
        GaitFeatureExtractor.process_interval produzca al menos seq_len
        frames de PSD, probando con duraciones crecientes hasta lograrlo.

        Esto evita el error cometido en un intento anterior, donde se
        calculo manualmente n_per_seg = fs_psd * fact_hz asumiendo
        valores incorrectos (fact_hz=1.0 en vez del real 4.0) -- aqui se
        prueba directamente contra la funcion real, sin replicar su
        matematica interna.

        :param aligned_data: Señales crudas alineadas (Left/Right).
        :param freq_target: Frecuencia de muestreo de aligned_data (Hz).
        :return: Numero de muestras crudas necesarias por ventana local.
        """
        lfeat = ['acc_mag', 'gyro_mag', 'mag_mag', 'S0', 'S1', 'S2']
        extractor_prueba = GaitFeatureExtractor(params=self.extraction_params, lfeat=lfeat)
        seq_len = self.t_cfg.max_len

        n_total_disponible = min(len(df) for df in aligned_data.values())
        # PUNTO DE PARTIDA: 10s en vez del minimo tecnico -- un ciclo de
        # marcha (paso completo, apoyo+vuelo, ambos pies) dura tipicamente
        # mas de 1s; una ventana calibrada al minimo que apenas alcanza
        # para el PSD (ej. 5s) captura pocos ciclos completos y produce
        # predicciones mas sensibles a micro-variaciones puntuales de la
        # señal, en vez de reconocer el patron de marcha de forma estable.
        duracion_prueba_seg = 10.0
        max_intentos = 20

        for _ in range(max_intentos):
            n_muestras_prueba = int(duracion_prueba_seg * freq_target)
            if n_muestras_prueba > n_total_disponible:
                raise ValueError(
                    f"NO HAY SUFICIENTES DATOS PARA CALIBRAR: se necesitarian mas de "
                    f"{duracion_prueba_seg:.1f}s pero solo hay {n_total_disponible / freq_target:.1f}s disponibles."
                )
            datos_prueba = {foot: df.iloc[:n_muestras_prueba] for foot, df in aligned_data.items()}
            tensors_prueba = extractor_prueba.process_interval(datos_prueba, min_samples=1)
            if "Left" in tensors_prueba and "Right" in tensors_prueba:
                n_frames_producidos = min(tensors_prueba["Left"][0].shape[1], tensors_prueba["Right"][0].shape[1])
                if n_frames_producidos >= seq_len:
                    logger.info(
                        f"CALIBRACION VENTANA LOCAL: {duracion_prueba_seg:.2f}s "
                        f"({n_muestras_prueba} muestras) -> {n_frames_producidos} frames PSD (>= {seq_len} requeridos)"
                    )
                    return n_muestras_prueba
            duracion_prueba_seg *= 1.5

        raise ValueError("NO SE PUDO CALIBRAR LA DURACION DE VENTANA LOCAL TRAS VARIOS INTENTOS.")

    def run_inference_local(self, aligned_data: Dict[str, pd.DataFrame], start: datetime) -> pd.DataFrame:
        """
        Extrae espectrogramas y corre inferencia CON NORMALIZACION LOCAL:
        para cada punto de tiempo, se recorta una ventana PEQUEÑA y
        centrada de datos CRUDOS, se calcula su espectrograma de forma
        AISLADA (normalizado solo con esos datos), y se predice sobre
        ese resultado -- en vez de normalizar sobre toda la consulta de
        una vez (metodo original, sensible al contexto) o dividir en
        tramos fijos que aun pueden mezclar marcha/reposo (division en
        tramos de 6 minutos, insuficiente si la mezcla ocurre dentro del
        propio tramo).

        La duracion de la ventana local se calibra EMPIRICAMENTE al
        inicio (ver _calibrar_duracion_ventana_local), sin asumir
        formulas de conversion manuales.

        :param aligned_data: Señales crudas alineadas (Left/Right).
        :param start: Inicio real de la consulta (referencia de tiempo).
        :return: DataFrame con timestamp, prob, pred, prob_smoothed,
            pred_final_smoothed -- una fila por ventana de inferencia.
        """
        logger.info("EXTRAYENDO ESPECTROGRAMAS CON NORMALIZACION LOCAL (ventana por ventana)...")
        lfeat = ['acc_mag', 'gyro_mag', 'mag_mag', 'S0', 'S1', 'S2']
        extractor = GaitFeatureExtractor(params=self.extraction_params, lfeat=lfeat)
        seq_len = self.t_cfg.max_len
        freq_target = self.extraction_params.freq_target_hz

        n_muestras_por_ventana = self._calibrar_duracion_ventana_local(aligned_data, freq_target)
        n_total_disponible = min(len(df) for df in aligned_data.values())

        # PASO ENTRE VENTANAS CONSECUTIVAS: se avanza una fraccion de la
        # ventana (25%) para mantener resolucion temporal razonable sin
        # recalcular el PSD en exceso.
        paso_muestras = max(1, n_muestras_por_ventana // 4)
        n_ventanas = 1 + max(0, (n_total_disponible - n_muestras_por_ventana) // paso_muestras)
        logger.info(f"INICIANDO {n_ventanas} VENTANAS LOCALES "
                    f"({n_muestras_por_ventana} muestras c/u, paso {paso_muestras} muestras)")

        results_log = []
        for i in range(n_ventanas):
            inicio_crudo = i * paso_muestras
            fin_crudo = inicio_crudo + n_muestras_por_ventana
            if fin_crudo > n_total_disponible:
                break

            datos_ventana = {foot: df.iloc[inicio_crudo:fin_crudo] for foot, df in aligned_data.items()}
            tensors_ventana = extractor.process_interval(datos_ventana, min_samples=1)
            if "Left" not in tensors_ventana or "Right" not in tensors_ventana:
                continue

            tensor_l = tensors_ventana["Left"][0]
            tensor_r = tensors_ventana["Right"][0]
            min_len = min(tensor_l.shape[1], tensor_r.shape[1])
            if min_len < seq_len:
                continue

            window = np.vstack([tensor_r[:, :seq_len], tensor_l[:, :seq_len]]).T
            if window.shape[1] != self.t_cfg.input_dim:
                raise ValueError(f"DIMENSION INCORRECTA: Esperaba {self.t_cfg.input_dim}, obtuvo {window.shape[1]}")

            idx_centro = inicio_crudo + n_muestras_por_ventana // 2
            current_time = start + timedelta(seconds=idx_centro / freq_target)

            flat_scaled = self.scaler.transform(window)
            x_time_np = np.expand_dims(flat_scaled, axis=0).astype(np.float32)
            x_time_tensor = torch.from_numpy(x_time_np).to(self.device)
            x_fft_tensor = None
            if self.modelo in ("fft", "hibrido_early", "hibrido_late"):
                x_fft_np = np.abs(rfft(x_time_np, axis=1))
                x_fft_np = (x_fft_np / x_time_np.shape[1]).astype(np.float32).reshape(1, -1)
                x_fft_tensor = torch.from_numpy(x_fft_np).to(self.device)
            with torch.no_grad():
                if self.modelo == "fft":
                    out = self.model_fft(x_fft_tensor)
                    prob = torch.softmax(out, dim=1)[0, 1].item()
                elif self.modelo == "transformer":
                    out = self.model_time(x_time_tensor)
                    prob = torch.softmax(out, dim=1)[0, 1].item()
                elif self.modelo == "hibrido_early":
                    out = self.model_hybrid(x_time_tensor, x_fft_tensor)
                    prob = torch.softmax(out, dim=1)[0, 1].item()
                elif self.modelo == "hibrido_late":
                    out_t = self.model_time(x_time_tensor)
                    prob_t = torch.softmax(out_t, dim=1)[0, 1].item()
                    out_f = self.model_fft(x_fft_tensor)
                    prob_f = torch.softmax(out_f, dim=1)[0, 1].item()
                    eps = 1e-7
                    prob = float(np.sqrt(np.clip(prob_t, eps, 1 - eps) * np.clip(prob_f, eps, 1 - eps)))
            pred = int(prob >= self.threshold)
            results_log.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "prob": prob,
                "pred": pred
            })
            if i % 20 == 0:
                logger.info(f"{current_time.strftime('%H:%M:%S')} | PROB: {prob:.3f} | PRED: {pred}")

        if results_log:
            df_res = pd.DataFrame(results_log)
            df_res['prob_smoothed'] = df_res['prob'].rolling(window=5, min_periods=1, center=True).mean()
            df_res['pred_final_smoothed'] = (df_res['prob_smoothed'] >= self.threshold).astype(int)
            return df_res
        return pd.DataFrame()

    def run_inference(self, aligned_data: Dict[str, pd.DataFrame], start: datetime) -> pd.DataFrame:
        """Extrae espectrogramas y corre inferencia con el modelo elegido, ventana por ventana."""
        logger.info("EXTRAYENDO ESPECTROGRAMAS PARA INFERENCIA CONTINUA...")
        lfeat = ['acc_mag', 'gyro_mag', 'mag_mag', 'S0', 'S1', 'S2']
        extractor = GaitFeatureExtractor(params=self.extraction_params, lfeat=lfeat)
        tensors = extractor.process_interval(aligned_data)
        if "Left" not in tensors or "Right" not in tensors:
            raise ValueError("FALLO EN LA EXTRACCION DE CARACTERISTICAS.")
        tensor_l = tensors["Left"][0]
        tensor_r = tensors["Right"][0]
        t_frames = tensors["Left"][1]
        min_len = min(tensor_l.shape[1], tensor_r.shape[1])
        tensor_all = np.vstack([tensor_r[:, :min_len], tensor_l[:, :min_len]]).T
        if tensor_all.shape[1] != self.t_cfg.input_dim:
            raise ValueError(f"DIMENSION INCORRECTA: Esperaba {self.t_cfg.input_dim}, obtuvo {tensor_all.shape[1]}")
        df_specs = pd.DataFrame(tensor_all)
        df_specs.index = [start + timedelta(seconds=float(t)) for t in t_frames[:min_len]]
        seq_len = self.t_cfg.max_len
        results_log = []
        logger.info(f"INICIANDO VENTANA DESLIZANTE SOBRE {len(df_specs)} FRAMES")
        for i in range(0, len(df_specs) - seq_len + 1):
            window = df_specs.iloc[i: i + seq_len].values
            current_time = df_specs.index[i + (seq_len // 2)]
            flat_scaled = self.scaler.transform(window)
            x_time_np = np.expand_dims(flat_scaled, axis=0).astype(np.float32)
            x_time_tensor = torch.from_numpy(x_time_np).to(self.device)
            x_fft_tensor = None
            if self.modelo in ("fft", "hibrido_early", "hibrido_late"):
                x_fft_np = np.abs(rfft(x_time_np, axis=1))
                x_fft_np = (x_fft_np / x_time_np.shape[1]).astype(np.float32).reshape(1, -1)
                x_fft_tensor = torch.from_numpy(x_fft_np).to(self.device)
            with torch.no_grad():
                if self.modelo == "fft":
                    out = self.model_fft(x_fft_tensor)
                    prob = torch.softmax(out, dim=1)[0, 1].item()
                elif self.modelo == "transformer":
                    out = self.model_time(x_time_tensor)
                    prob = torch.softmax(out, dim=1)[0, 1].item()
                elif self.modelo == "hibrido_early":
                    out = self.model_hybrid(x_time_tensor, x_fft_tensor)
                    prob = torch.softmax(out, dim=1)[0, 1].item()
                elif self.modelo == "hibrido_late":
                    out_t = self.model_time(x_time_tensor)
                    prob_t = torch.softmax(out_t, dim=1)[0, 1].item()
                    out_f = self.model_fft(x_fft_tensor)
                    prob_f = torch.softmax(out_f, dim=1)[0, 1].item()
                    eps = 1e-7
                    prob = float(np.sqrt(np.clip(prob_t, eps, 1 - eps) * np.clip(prob_f, eps, 1 - eps)))
            pred = int(prob >= self.threshold)
            results_log.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "prob": prob,
                "pred": pred
            })
            if i % 50 == 0:
                logger.info(f"{current_time.strftime('%H:%M:%S')} | PROB: {prob:.3f} | PRED: {pred}")
        self.client.close()
        if results_log:
            df_res = pd.DataFrame(results_log)
            df_res['prob_smoothed'] = df_res['prob'].rolling(window=10, min_periods=1, center=True).mean()
            df_res['pred_final_smoothed'] = (df_res['prob_smoothed'] >= self.threshold).astype(int)
            return df_res
        return pd.DataFrame()
def graficar_timeline(df: pd.DataFrame, threshold: float, reference: str, modelo: str, output_dir: Path) -> Path:
    """
    Genera la grafica de linea de tiempo de inferencia (probabilidad
    suavizada vs umbral), con el fondo pintado de amarillo en los tramos
    donde la prediccion final es marcha (pred_final_smoothed == 1), y
    ticks del eje X calculados dinamicamente para garantizar ~10
    etiquetas legibles sin importar la duracion real del rango
    consultado (desde segundos hasta horas), alineadas a horas
    "redondas" y sin el prefijo de dia que antepone matplotlib por
    defecto.
    """
    import matplotlib.dates as mdates

    df_plot = df.copy()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
    tiempos = df_plot["timestamp"].reset_index(drop=True)
    predicciones = df_plot["pred_final_smoothed"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    # PINTAR TRAMOS DE MARCHA: agrupar filas consecutivas con
    # pred_final_smoothed == 1 en un unico rectangulo por tramo.
    es_marcha = (predicciones == 1).values
    cambios = np.where(np.diff(es_marcha.astype(int)) != 0)[0] + 1
    limites = np.concatenate(([0], cambios, [len(es_marcha)]))
    for i in range(len(limites) - 1):
        inicio_idx, fin_idx = limites[i], limites[i + 1] - 1
        if es_marcha[inicio_idx]:
            ax.axvspan(tiempos.iloc[inicio_idx], tiempos.iloc[fin_idx], color="gold", alpha=0.35, linewidth=0, zorder=1)

    ax.plot(tiempos, df_plot["prob_smoothed"], label="Probabilidad", linewidth=1.2, color="tab:blue", zorder=3)
    ax.axhline(threshold, color="black", linestyle="--", label=f"Umbral {threshold:.2f}", zorder=2)
    ax.set_title(f"Linea de Tiempo Inferencia - {NOMBRES_LEGIBLES[modelo]}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Probabilidad de Marcha")
    ax.set_ylim(-0.02, 1.02)
    # ELIMINAR MARGEN FANTASMA: sin esto, matplotlib deja un espacio en
    # blanco antes del primer dato y despues del ultimo, que visualmente
    # se confunde con "reposo" (sin linea azul ni pintado amarillo) aunque
    # no representa ningun dato real.
    ax.set_xlim(tiempos.iloc[0], tiempos.iloc[-1])

    # LEYENDA: se agrega manualmente un "parche" para el area amarilla de
    # marcha, ya que axvspan no aparece en la leyenda por defecto.
    from matplotlib.patches import Patch
    handles_leyenda, labels_leyenda = ax.get_legend_handles_labels()
    handles_leyenda.append(Patch(facecolor="gold", alpha=0.35, label="Marcha (predicha)"))
    ax.legend(handles=handles_leyenda)
    ax.grid(True, alpha=0.3)

    # TICKS DINAMICOS: al menos ~10 marcas sin importar la duracion real
    # del rango, alineadas a multiplos exactos del intervalo elegido
    # desde medianoche (no al primer dato de la consulta).
    duracion_total_seg = (tiempos.iloc[-1] - tiempos.iloc[0]).total_seconds()
    intervalo_ideal_seg = max(duracion_total_seg / 10, 0.001)
    opciones_intervalo_seg = [1, 2, 5, 10, 15, 30, 60, 120, 150, 300, 600, 900, 1800, 3600, 7200]
    intervalo_elegido_seg = min(opciones_intervalo_seg, key=lambda x: abs(x - intervalo_ideal_seg))

    inicio_dia = tiempos.iloc[0].normalize()
    primer_offset = np.ceil((tiempos.iloc[0] - inicio_dia).total_seconds() / intervalo_elegido_seg) * intervalo_elegido_seg
    primer_tick = inicio_dia + pd.Timedelta(seconds=primer_offset)
    ticks_manuales = pd.date_range(
        start=primer_tick, end=tiempos.iloc[-1], freq=pd.Timedelta(seconds=intervalo_elegido_seg)
    )
    if len(ticks_manuales) >= 2:
        ax.set_xticks(ticks_manuales)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{reference}_{modelo}_agnostico_timeline.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
def elegir_modelo_interactivo() -> str:
    """Pide al usuario elegir el modelo por numero, si no se paso --modelo por consola."""
    print("\nSeleccione el modelo a usar:")
    print("  1. FFT")
    print("  2. Transformer")
    print("  3. Hibrido Early Fusion")
    print("  4. Hibrido Late Fusion (Media Geometrica)")
    opcion = input("Opcion (1-4): ").strip()
    if opcion not in MODELOS_DISPONIBLES:
        raise ValueError(f"Opcion invalida: {opcion}. Debe ser 1, 2, 3 o 4.")
    return MODELOS_DISPONIBLES[opcion]
def main() -> None:
    """Punto de entrada ejecutable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-m", "--models", type=Path, required=True)
    parser.add_argument("-r", "--reference", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument(
        "--modelo", type=str, default=None,
        choices=["1", "2", "3", "4", "fft", "transformer", "hibrido_early", "hibrido_late"],
        help="1/fft, 2/transformer, 3/hibrido_early, 4/hibrido_late. Si se omite, se pregunta interactivamente."
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "RESULTADO_AGNOSTIC",
        help="Carpeta donde se guardan el CSV y la grafica de resultados."
    )
    args = parser.parse_args()
    if args.modelo is None:
        modelo = elegir_modelo_interactivo()
    elif args.modelo in MODELOS_DISPONIBLES:
        modelo = MODELOS_DISPONIBLES[args.modelo]
    else:
        modelo = args.modelo
    try:
        start_dt = datetime.fromisoformat(args.start.replace("Z", "+00:00")).replace(tzinfo=None)
        end_dt = datetime.fromisoformat(args.end.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        try:
            start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.critical(f"ERROR FORMATO FECHA: {e}")
            sys.exit(1)
    logger.info(f"INICIANDO EVALUACION AGNOSTICA: {args.reference} | MODELO: {NOMBRES_LEGIBLES[modelo]}")
    try:
        evaluator = AgnosticEvaluatorUnificado(config_path=args.config, model_dir=args.models, modelo=modelo)

        # NORMALIZACION LOCAL (2026-08-05, v2): la division en tramos
        # fijos de 6 minutos resulto INSUFICIENTE -- si el propio tramo
        # sigue mezclando marcha y reposo (ej. 1 minuto de reposo seguido
        # de 5 de marcha dentro del mismo tramo de 6 min), la
        # normalizacion global del tramo sigue diluyendo la marcha real.
        # Se reemplaza por run_inference_local: cada ventana de
        # inferencia (duracion calibrada empiricamente, no asumida) se
        # normaliza de forma COMPLETAMENTE AISLADA, sin depender de
        # cuanto contexto adicional (marcha o reposo) haya alrededor.
        logger.info("RECUPERANDO FLUJOS INFLUXDB...")
        aligned_streams = evaluator.fetch_and_align_stream(args.reference, start_dt, end_dt)
        inference_report = evaluator.run_inference_local(aligned_streams, start_dt)
        evaluator.client.close()

        if not inference_report.empty:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = args.output_dir / f"agnostico_{args.reference}_{modelo}.csv"
            inference_report.to_csv(csv_path, index=False)
            logger.info(f"RESULTADOS GUARDADOS EN: {csv_path}")
            logger.info(f"TOTAL PREDICCIONES GENERADAS: {len(inference_report)}")
            png_path = graficar_timeline(
                inference_report, evaluator.threshold, args.reference, modelo, args.output_dir
            )
            logger.info(f"GRAFICA GUARDADA EN: {png_path}")
        else:
            logger.warning("SIN PREDICCIONES: INTERVALO TEMPORAL MUY CORTO.")
    except Exception as e:
        logger.critical(f"FALLO EN EVALUACION AGNOSTICA: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()
