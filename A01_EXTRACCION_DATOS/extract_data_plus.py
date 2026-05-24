# -*- coding: utf-8 -*-
"""
extract_data_plus.py
=================================
Pipeline de extracción MLOps para InfluxDB.
Incluye validación Pydantic, POO estricta, interpolación física
y auditoría de reproducibilidad (Hashing y Seeds).
"""
from __future__ import annotations

import argparse
import sys
import logging
import json
import hashlib
import subprocess
import platform
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from imageio import imwrite
from scipy import signal
import urllib3
import yaml
from pydantic import BaseModel, PositiveInt, confloat

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from influxdb_client import InfluxDBClient  # type: ignore
except ImportError:
    InfluxDBClient = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore

# CONFIGURAR LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# EXCEPCIONES PERSONALIZADAS
class InfluxExtractionError(Exception):
    """Excepción para fallos InfluxDB."""
    pass

class ExcelMetadataError(Exception):
    """Excepción para fallos Excel."""
    pass

# ============================================================
# MODELOS PYDANTIC INMUTABLES
# ============================================================

class ExtractionParams(BaseModel):
    """
    Parámetros inmutables de extracción matemática.
    """
    freq_target_hz: confloat(gt=0) = 75.0
    freq_psd_hz: PositiveInt = 75
    fact_hz: confloat(gt=0) = 1.0
    f_start_hz: confloat(ge=0) = 0.5
    f_stop_hz: confloat(gt=0) = 5.0
    window_overlap: confloat(ge=0, lt=1) = 0.5

# ============================================================
# AUDITORIA Y REPRODUCIBILIDAD (MLOps)
# ============================================================

class ExperimentAuditor:
    """
    Auditor de reproducibilidad científica y determinismo.
    """
    def __init__(self, output_dir: Path) -> None:
        """
        Inicializa gestor de metadatos.

        :param output_dir: Directorio de exportación.
        :type output_dir: Path
        """
        self.output_dir = output_dir

    @staticmethod
    def enforce_determinism(seed: int = 42) -> None:
        """
        Fija semillas pseudoaleatorias globales.

        :param seed: Valor de anclaje pseudoaleatorio.
        :type seed: int
        :return: Nada.
        :rtype: None
        """
        # FIJAR SEMILLAS
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        logger.info(f"DETERMINISMO FORZADO: SEED {seed}")

    def snapshot_experiment(self, config: BaseModel) -> Dict[str, Any]:
        """
        Genera registro inmutable del experimento.

        :param config: Configuración validada Pydantic.
        :type config: BaseModel
        :return: Diccionario de auditoría.
        :rtype: Dict[str, Any]
        """
        # SERIALIZAR CONFIGURACION
        params_dict = config.model_dump()
        serialized = json.dumps(params_dict, sort_keys=True, default=str)
        
        # GENERAR HASH UNICO
        exp_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        # LEER REPOSITORIO GIT
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            git_commit = "untracked"

        # EMPAQUETAR METADATA
        metadata = {
            "experiment_hash": exp_hash,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "git_commit": git_commit,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "effective_params": params_dict
        }

        # GUARDAR JSON
        meta_path = self.output_dir / f"experiment_metadata_{exp_hash[:8]}.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=4, ensure_ascii=False)

        logger.info(f"SNAPSHOT GUARDADO: {meta_path.name}")
        logger.info(f"HASH EXPERIMENTAL: {exp_hash}")

        return metadata

# ============================================================
# CLASES DE PROCESAMIENTO
# ============================================================

class SignalAligner:
    """
    Gestor de alineamiento y remuestreo temporal.
    """
    @staticmethod
    def uniform_timebase(
        df: pd.DataFrame, 
        freq_hz: float, 
        *, 
        method: str = "time", 
        agg: str = "mean", 
        enforce_unique_index: bool = True,
        target_idx_override: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Interpola dataframe temporal a malla uniforme.

        :param df: Señal en crudo indexada por fecha.
        :type df: pd.DataFrame
        :param freq_hz: Frecuencia de remuestreo objetivo.
        :type freq_hz: float
        :param method: Método de interpolación pandas.
        :type method: str
        :param agg: Método agregación duplicados.
        :type agg: str
        :param enforce_unique_index: Forzar unicidad índices.
        :type enforce_unique_index: bool
        :param target_idx_override: Indice temporal precalculado.
        :type target_idx_override: Optional[pd.DatetimeIndex]
        :return: DataFrame remuestreado y alineado.
        :rtype: pd.DataFrame
        """
        if df.shape[0] == 0:
            return df

        df = df.sort_index()
        df.index = pd.to_datetime(df.index) 

        # ELIMINAR DUPLICADOS
        if enforce_unique_index and df.index.has_duplicates:
            df = df.groupby(level=0).agg(agg)

        # GENERAR MALLA
        if target_idx_override is not None: 
            target_idx = target_idx_override
        else:
            step = pd.Timedelta(seconds=1.0 / freq_hz)
            target_idx = pd.date_range(
                start=df.index[0], end=df.index[-1], freq=step, name=df.index.name
            )

        full_idx = target_idx.union(df.index)
        df_interp = df.reindex(full_idx).interpolate(method=method, limit_direction="both")
        
        return df_interp.loc[target_idx]

class GaitFeatureExtractor:
    """
    Extractor de características físicas y frecuenciales.
    """
    def __init__(self, params: ExtractionParams, lfeat: List[str]) -> None:
        """
        Inicializa parámetros de transformación espectral.

        :param params: Parámetros matemáticos validados.
        :type params: ExtractionParams
        :param lfeat: Lista de características a calcular.
        :type lfeat: List[str]
        """
        self.params = params
        self.lfeat = lfeat

    @staticmethod
    def magnitude(df: pd.DataFrame, cols: List[str]) -> pd.Series:
        """
        Calcula norma euclidiana combinada de sensores espaciales.

        :param df: Matriz de datos fuente.
        :type df: pd.DataFrame
        :param cols: Lista de ejes a combinar (ej. X, Y, Z).
        :type cols: List[str]
        :return: Serie con la magnitud absoluta.
        :rtype: pd.Series
        """
        return np.sqrt((df[cols].astype(float) ** 2).sum(axis=1))

    def compute_psd_spectrogram(
        self, series: pd.Series, n_per_seg: int, n_overlap: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula Densidad Espectral de Potencia acotada.

        :param series: Señal temporal 1D.
        :type series: pd.Series
        :param n_per_seg: Tamaño de ventana FFT.
        :type n_per_seg: int
        :param n_overlap: Muestras superpuestas entre ventanas.
        :type n_overlap: int
        :return: Tupla con Frecuencias, Tiempos y Espectro (Imagen 2D).
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        fs_hz = self.params.freq_psd_hz
        fhz = int(self.params.freq_target_hz)

        # CALCULAR ESPECTROGRAMA
        f, t, Sxx = signal.spectrogram(
            series.astype(float).values, fs=fs_hz, window="hamming",
            nperseg=n_per_seg, noverlap=n_overlap, detrend=False,
            scaling="density", mode="psd",
        )
        psd_db = 10 * np.log10(Sxx + 1e-12)
        min_val, max_val = np.min(psd_db), np.max(psd_db)
        
        # NORMALIZAR MATRIZ
        if max_val == min_val:
            normalized_psd = np.zeros_like(psd_db)
        else:
            normalized_psd = (psd_db - min_val) / (max_val - min_val)
            
        scaled_psd = (normalized_psd * 255).astype(np.uint8)
        mask = (f >= self.params.f_start_hz) & (f <= self.params.f_stop_hz + 1e-9)
        return f[mask], t * fs_hz / fhz, scaled_psd[mask]

    def stack_features(self, specs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apila ejes sensores en tensor combinado.

        :param specs: Diccionario con espectrogramas individuales.
        :type specs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        :return: Matriz apilada, frames de tiempo, vector de frecuencias.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        order = ["acc_mag", "gyro_mag", "S0", "S1", "S2"]
        imgs = [specs[k][2] for k in order]
        stacked = np.vstack(imgs)
        return stacked, specs[order[0]][1], specs[order[0]][0] 

    def process_interval(
        self, data: Dict[str, pd.DataFrame], min_samples: Optional[int] = None, drop_short: bool = False
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Ejecuta transformación frecuencial sobre intervalo.

        :param data: Diccionario con dataframes de cada pie.
        :type data: Dict[str, pd.DataFrame]
        :param min_samples: Umbral mínimo de validación.
        :type min_samples: Optional[int]
        :param drop_short: Descartar intervalos menores al umbral.
        :type drop_short: bool
        :return: Tensors espectrales por cada extremidad.
        :rtype: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        """
        fs = self.params.freq_psd_hz
        n_per_seg_nom = int(round(fs * self.params.fact_hz))
        n_overlap_nom = int(n_per_seg_nom * self.params.window_overlap)
        min_samples = min_samples or n_per_seg_nom
        results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # ITERAR EXTREMIDADES
        for foot, df in data.items():
            n_samples_df = len(df)
            if n_samples_df < min_samples:
                if drop_short: continue  
                n_per_seg = n_samples_df                                  
                n_overlap = max(0, int(n_per_seg * self.params.window_overlap))
            else:
                n_per_seg = n_per_seg_nom
                n_overlap = n_overlap_nom

            if df.shape[0] == 0: continue 
            
            # MAGNITUDES
            acc_mag = self.magnitude(df, ["Ax", "Ay", "Az"])
            gyro_mag = self.magnitude(df, ["Gx", "Gy", "Gz"])
            press = [df[c] for c in ("S0", "S1", "S2")]

            specs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            
            # EXTRACCION INDIVIDUAL
            for feature in self.lfeat:
                if feature == "acc_mag":
                    psd = self.compute_psd_spectrogram(acc_mag, n_per_seg, n_overlap)
                elif feature == "gyro_mag":
                    psd = self.compute_psd_spectrogram(gyro_mag, n_per_seg, n_overlap)
                elif feature == "S0":
                    psd = self.compute_psd_spectrogram(press[0], n_per_seg, n_overlap)
                elif feature == "S1":
                    psd = self.compute_psd_spectrogram(press[1], n_per_seg, n_overlap)
                elif feature == "S2":
                    psd = self.compute_psd_spectrogram(press[2], n_per_seg, n_overlap)
                else:
                    logger.error(f"FEATURE DESCONOCIDA: {feature}")
                    raise ValueError(f"Feature {feature} not found")
                specs[feature] = psd

            results[foot] = self.stack_features(specs)

        return results

# ============================================================
# CLIENTE INFLUXDB (INFRAESTRUCTURA RED)
# ============================================================

class cInfluxDB:
    """
    Gestor de conexión segura InfluxDB.
    """
    def __init__(self, config_path: str, timeout: int = 500_000) -> None:
        """
        Inicializa cliente autenticado.

        :param config_path: Ruta archivo credenciales.
        :type config_path: str
        :param timeout: Tiempo máximo espera respuesta.
        :type timeout: int
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config: Dict[str, Any] = yaml.safe_load(file)

            self.bucket: str = config['influxdb']['bucket']
            self.org: str = config['influxdb']['org']
            self.token: str = config['influxdb']['token']
            self.url: str = config['influxdb']['url']
            self.tzval: str = config['influxdb'].get('tzval', 'Europe/Madrid')

            self.client = InfluxDBClient(
                url=self.url, token=self.token, org=self.org, 
                verify_ssl=False, timeout=timeout
            )
            self.measurement: str = self.bucket.split("/")[0] if '/' in self.bucket else self.bucket
            logger.info("CONEXION INFLUXDB ESTABLECIDA")
        except FileNotFoundError as e:
            logger.error(f"YAML NO ENCONTRADO: {e}")
            raise InfluxExtractionError("Configuración faltante.") from e
        except KeyError as e:
            logger.error(f"CLAVE YAML INVALIDA: {e}")
            raise InfluxExtractionError("Estructura inválida.") from e
    
    def query_data(
        self, 
        from_date: datetime, 
        to_date: datetime, 
        qtok: str, 
        pie: str, 
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Ejecuta consulta filtrada por sujeto e intervalo.

        :param from_date: Marca de tiempo inicio.
        :type from_date: datetime
        :param to_date: Marca de tiempo fin.
        :type to_date: datetime
        :param qtok: Identificador sujeto paciente.
        :type qtok: str
        :param pie: Lateralidad extremidad inferior.
        :type pie: str
        :param metrics: Nombres campos sensores deseados.
        :type metrics: Optional[List[str]]
        :return: Registros de serie temporal.
        :rtype: pd.DataFrame
        """
        if to_date <= from_date:
            logger.warning("FECHAS INVALIDAS.")
            return pd.DataFrame() 
        
        # FORMATEAR ZONAS HORARIAS
        from_date_str = from_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') 
        to_date_str = to_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') 

        if metrics is None:
            metrics = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S0', 'S1', 'S2']
        
        metrics_str = ' or '.join([f'r._field == "{metric}"' for metric in metrics])
        columns_str = ', '.join([f'"{metric}"' for metric in metrics])

        # CONSULTA FLUX
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: {from_date_str}, stop: {to_date_str})
        |> filter(fn: (r) => r._measurement == "{self.measurement}")
        |> filter(fn: (r) => {metrics_str})
        |> filter(fn: (r) => r["CodeID"] == "{qtok}" and r["type"] == "SCKS" and r["Foot"] == "{pie}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", {columns_str}])
        '''

        try:
            result = self.client.query_api().query(org=self.org, query=query)
        except Exception as e:
            logger.error(f"ERROR CONSULTA: {str(e)}")
            raise InfluxExtractionError("Fallo API.") from e

        data: List[List[Any]] = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        # PARSEAR RESPUESTA
        if len(data) > 0:
            df = pd.DataFrame(data).drop(['result', 'table'], axis=1)
            df["_time"] = pd.to_datetime(df["_time"], unit='ns')
            df['_time'] = df['_time'].dt.tz_convert(self.tzval).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            return df.sort_values(by="_time", ascending=False).reset_index(drop=True)
        else:
            return pd.DataFrame() 

    def close(self) -> None:
        """Cierra conexión cliente de red."""
        self.client.close()

# ============================================================
# VARIABLES GLOBALES
# ============================================================

SENSOR_FIELDS_INFLUXDB_CLIENT: Dict[str, List[str]] = {
    "acc": ["Ax", "Ay", "Az"],
    "gyro": ["Gx", "Gy", "Gz"],
    "press": ["S0", "S1", "S2"],
}

SENSOR_FIELDS_INFLUXDBMS: Dict[str, str] = {
    "Ax": "Ax", "Ay": "Ay", "Az": "Az",
    "Gx": "Gx", "Gy": "Gy", "Gz": "Gz",
    "S0": "S0", "S1": "S1", "S2": "S2",
}

FEET: Tuple[str, str] = ("Left", "Right") 

# ============================================================
# FUNCIONES AUXILIARES DISCO
# ============================================================

def load_param_config(path: Optional[Path]) -> Dict[str, Any]:
    """Carga config YAML."""
    if path is None: return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError as e:
        logger.error(f"YAML NO ENCONTRADO: {e}")
        raise

def merge_params(cli_ns: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Combina CLI YAML."""
    params: Dict[str, Any] = {}
    params.update(cfg)
    if 'params' not in params:
        params['params'] = {}
    for key in ["freq_target_hz", "f_start_hz", "f_stop_hz", "f_step_hz", "window_overlap"]:
        val = getattr(cli_ns, key, None)
        if val is not None:
            params['params'][key] = val
    return params

def read_time_windows(excel_path: Path) -> pd.DataFrame:
    """Lee metadatos Excel."""
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
        df.columns = [str(col).lower() for col in df.columns]
        
        for col in ("start", "datefrom"):
            if col in df.columns:
                df["start"] = pd.to_datetime(df[col], utc=True, format="mixed", dayfirst=True)
                break
        for col in ("end", "dateuntil"):
            if col in df.columns:
                df["end"] = pd.to_datetime(df[col], utc=True, format="mixed", dayfirst=True)
                break
                
        if "reference" in df.columns:
            df["reference"] = df["reference"].astype(str)
        else:
            raise ExcelMetadataError("Columna reference obligatoria.")
            
        return df
    except Exception as e:
        logger.error(f"FALLO METADATOS EXCEL: {e}")
        raise ExcelMetadataError("Error procesando solicitud.") from e

def save_pgm(img: np.ndarray, out_path: Path) -> None:
    """Guarda imagen PGM."""
    imwrite(out_path, img)

# ============================================================
# FLUJO PRINCIPAL
# ============================================================

def cli() -> None: 
    """Inicia proceso CLI general."""
    parser = argparse.ArgumentParser(
        description="Dual-foot extractor InfluxDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument("--backend", choices=["influxdb-client", "influxdbms"], help="Database backend")

    parser.add_argument("-c", "--config", type=Path, help="Path config.yaml")
    parser.add_argument("-o", "--output", type=Path, help="Output directory")
    parser.add_argument("-e", "--excel", type=Path, help="Metadata Excel", required=True)

    args = parser.parse_args()
    lfeat = ['acc_mag','gyro_mag','S0','S1','S2']

    try:
        # FIJAR DETERMINISMO GLOBAL
        ExperimentAuditor.enforce_determinism(42)

        param_config_from_yaml = load_param_config(args.config)
        params_dict = merge_params(args, param_config_from_yaml)
        
        if 'io' not in params_dict: params_dict['io'] = {}
        if 'params' not in params_dict: params_dict['params'] = {}

        params_dict['io']['excel'] = args.excel
        windows = read_time_windows(params_dict['io']['excel'])

        out_path = args.output if args.output else Path(params_dict['io'].get('out', './resultados'))
        out_path.mkdir(parents=True, exist_ok=True)

        # VALIDAR PARAMETROS CON PYDANTIC
        valid_params = ExtractionParams(**params_dict['params'])

        # INICIAR AUDITORIA Y GENERAR SNAPSHOT
        auditor = ExperimentAuditor(out_path)
        auditor.snapshot_experiment(valid_params)
        logger.info(f"CONFIGURACION APROBADA: {valid_params.model_dump_json()}")

        # INSTANCIAR EXTRACTOR SEGURO
        extractor = GaitFeatureExtractor(params=valid_params, lfeat=lfeat)

        if args.backend == "influxdbms":
            if args.config is None:
                raise ValueError("Config required for influxdbms")
            client = cInfluxDB(config_path=str(args.config))
        else:
            raise NotImplementedError("Backend not implemented.")

        # ITERAR VENTANAS EXCEL
        for idx, row in windows.iterrows():
            label = f"segment_{idx:03d}"
            start, end = row["start"].to_pydatetime(), row["end"].to_pydatetime()
            
            qtok = str(row["reference"]).strip()
            mov  = row.get("mov_type", None)
            
            if len(qtok) < 5: 
                logger.warning(f"[{label}] ID corto. Saltando.")
                continue

            raw_data_by_foot: Dict[str, pd.DataFrame] = {} 

            # CONSULTA BD
            for foot in FEET:
                df = client.query_data(start, end, qtok, foot)
                logger.info(f"{label}:{foot} => {len(qtok)} : {df.shape}")
                
                if not df.empty:
                    df = df.set_index("_time").sort_index()
                    df.index = pd.to_datetime(df.index)
                    df.rename(columns=SENSOR_FIELDS_INFLUXDBMS, inplace=True)
                    raw_data_by_foot[foot] = df
            
            if len(raw_data_by_foot) == len(FEET): 
                common_start = max(df.index.min() for df in raw_data_by_foot.values())
                common_end = min(df.index.max() for df in raw_data_by_foot.values())

                if common_start >= common_end:
                    logger.warning(f"[{label}] Rango temporal inválido.")
                    continue

                target_idx = pd.date_range(
                    start=common_start, end=common_end,
                    freq=pd.Timedelta(seconds=1/valid_params.freq_target_hz), name="_time" 
                )
                
                if (end.replace(tzinfo=None) - common_end).total_seconds() > 0.1:
                    logger.warning(f"[{label}] Diferencia cortes temporal.")
                    continue

                data_by_foot: Dict[str, pd.DataFrame] = {}
                for foot, df_raw in raw_data_by_foot.items():
                    df_resampled = SignalAligner.uniform_timebase(df_raw, valid_params.freq_target_hz, target_idx_override=target_idx)

                    if not df_resampled.empty:
                        if df_resampled.isnull().values.any():
                            # INTERPOLACION CUBICA FISICA
                            try:
                                df_resampled = df_resampled.interpolate(method="cubicspline").bfill().ffill()
                            except Exception:
                                df_resampled = df_resampled.interpolate(method="linear").bfill().ffill()
                        data_by_foot[foot] = df_resampled
            
                if len(data_by_foot) == len(FEET) and not any(df.empty for df in data_by_foot.values()):
                    # EXTRACCION ESPECTROGRAMAS
                    tensors = extractor.process_interval(data_by_foot)

                    for foot, (img, t_frames, frqs) in tensors.items():
                        mov_str = f"-{mov}" if mov is not None else ""
                        png_path = out_path / f"{label}_{foot}{mov_str}.pgm"
                        save_pgm(img, png_path)

                    if len(tensors.keys()) == 2: 
                        if len(tensors["Left"][1]) == len(tensors["Right"][1]):
                            tensor_all = np.vstack([tensors["Left"][0], tensors["Right"][0]])
                            parquet_path = out_path / f"{label}_tensor{mov_str}.parquet"
                            ta_pd = pd.DataFrame(tensor_all)
                            
                            combined_datetimes = [start + timedelta(seconds=float(t)) for t in tensors["Left"][1]]
                            lcls = [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for dt in combined_datetimes]
                            ta_pd.columns = lcls
                            
                            lidx = [f"{foot}-{feat}-{ifrq:.2f}Hz" for foot in FEET for feat in lfeat for ifrq in tensors["Left"][2]]
                            ta_pd.index = lidx
                            
                            if ta_pd.isnull().any().any():
                                logger.warning(f"[{label}] TENSOR CON NANS.")
                            ta_pd.to_parquet(parquet_path)
                            logger.info(f"TENSOR GUARDADO: {parquet_path.name}")

        client.close() 
        logger.info("EXTRACCION COMPLETADA CON EXITO")

    except Exception as e:
        logger.critical(f"ABORTADO POR ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli()
