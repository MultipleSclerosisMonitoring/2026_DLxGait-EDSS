# -*- coding: utf-8 -*-
"""
extract_data_plus.py
=================================
Enhanced end-to-end pipeline to extract dual-foot sensor data from InfluxDB.
Refactorizado: Tipado estricto, logging estructurado, excepciones seguras y POO.
"""
from __future__ import annotations

import argparse
import sys
import logging
import numpy as np
import pandas as pd
from imageio import imwrite
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
from scipy import signal
from zoneinfo import ZoneInfo
import urllib3
import yaml

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from influxdb_client import InfluxDBClient  # type: ignore
except ImportError:
    InfluxDBClient = None  # type: ignore

# CONFIGURACION LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# EXCEPCIONES PERSONALIZADAS
class InfluxExtractionError(Exception):
    """Excepción para fallos de conexión o consulta a InfluxDB."""
    pass

class ExcelMetadataError(Exception):
    """Excepción para errores al procesar el archivo Excel de solicitudes."""
    pass

###############################################################################
# ----------------------------- helper funcs -------------------------------- #
###############################################################################

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
    Interpola un DataFrame a una malla temporal uniforme sin perder datos.
    """
    if df.shape[0] == 0:
        return df

    df = df.sort_index()
    df.index = pd.to_datetime(df.index) 

    if enforce_unique_index and df.index.has_duplicates:
        df = df.groupby(level=0).agg(agg)

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

class cInfluxDB:
    """
    Cliente personalizado para la conexión y extracción de datos desde InfluxDB.
    """
    def __init__(self, config_path: str, timeout: int = 500_000) -> None:
        """
        Inicializa la conexión con InfluxDB leyendo credenciales desde YAML.
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
            logger.info("CONEXION INFLUXDB INICIALIZADA CORRECTAMENTE")
        except FileNotFoundError as e:
            logger.error(f"ARCHIVO CONFIG YAML NO ENCONTRADO: {e}")
            raise InfluxExtractionError("Configuración faltante.") from e
        except KeyError as e:
            logger.error(f"CLAVE FALTANTE EN YAML: {e}")
            raise InfluxExtractionError("Estructura YAML inválida.") from e
    
    def query_data(
        self, 
        from_date: datetime, 
        to_date: datetime, 
        qtok: str, 
        pie: str, 
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Realiza una consulta a la base de datos filtrando estrictamente por paciente y pie.
        """
        if to_date <= from_date:
            logger.warning("FECHA DE FIN ANTERIOR A FECHA DE INICIO. ABORTANDO QUERY.")
            return pd.DataFrame() 
        
        from_date_str = from_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') 
        to_date_str = to_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') 

        if metrics is None:
            metrics = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S0', 'S1', 'S2']
        
        metrics_str = ' or '.join([f'r._field == "{metric}"' for metric in metrics])
        columns_str = ', '.join([f'"{metric}"' for metric in metrics])

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
            logger.error(f"ERROR EN LA CONSULTA INFLUXDB: {str(e)}")
            raise InfluxExtractionError("Fallo al consultar la API de InfluxDB.") from e

        data: List[List[Any]] = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        if len(data) > 0:
            df = pd.DataFrame(data).drop(['result', 'table'], axis=1)
            df["_time"] = pd.to_datetime(df["_time"], unit='ns')
            df['_time'] = df['_time'].dt.tz_convert(self.tzval).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            return df.sort_values(by="_time", ascending=False).reset_index(drop=True)
        else:
            return pd.DataFrame() 

    def close(self) -> None:
        """Cierra la conexión con el cliente de InfluxDB."""
        self.client.close()

###############################################################################
# ------------------------------- defaults ---------------------------------- #
###############################################################################

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

###############################################################################
# ----------------------------- data processing ----------------------------- #
###############################################################################

def load_param_config(path: Optional[Path]) -> Dict[str, Any]:
    """Carga configuración base desde archivo YAML."""
    if path is None: return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError as e:
        logger.error(f"ARCHIVO CONFIG NO ENCONTRADO: {e}")
        raise

def merge_params(cli_ns: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Combina parámetros de línea de comandos con el diccionario YAML."""
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
    """Lee y estandariza los metadatos de las ventanas temporales desde Excel."""
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
        df.columns = [str(col).lower() for col in df.columns]
        
        for col in ("start", "datefrom"):
            if col in df.columns:
                df["start"] = pd.to_datetime(df[col], utc=True)
                break
        for col in ("end", "dateuntil"):
            if col in df.columns:
                df["end"] = pd.to_datetime(df[col], utc=True)
                break
                
        if "reference" in df.columns:
            df["reference"] = df["reference"].astype(str)
        else:
            raise ExcelMetadataError("La columna 'reference' es obligatoria en el Excel.")
            
        return df
    except Exception as e:
        logger.error(f"FALLO LEYENDO METADATOS EXCEL: {e}")
        raise ExcelMetadataError("Error procesando solicitud.xlsx") from e

def magnitude(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Calcula la magnitud euclidiana combinada de ejes (X,Y,Z)."""
    return np.sqrt((df[cols].astype(float) ** 2).sum(axis=1))

def compute_psd_spectrogram(
    series: pd.Series, fhz: int, fs_hz: int, n_per_seg: int, n_overlap: int,
    f_start_hz: float, f_stop_hz: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula la Densidad Espectral de Potencia (PSD)."""
    f, t, Sxx = signal.spectrogram(
        series.astype(float).values, fs=fs_hz, window="hamming",
        nperseg=n_per_seg, noverlap=n_overlap, detrend=False,
        scaling="density", mode="psd",
    )
    psd_db = 10 * np.log10(Sxx + 1e-12)
    min_val, max_val = np.min(psd_db), np.max(psd_db)
    
    if max_val == min_val:
        normalized_psd = np.zeros_like(psd_db)
    else:
        normalized_psd = (psd_db - min_val) / (max_val - min_val)
        
    scaled_psd = (normalized_psd * 255).astype(np.uint8)
    mask = (f >= f_start_hz) & (f <= f_stop_hz + 1e-9)
    return f[mask], t*fs_hz/fhz, scaled_psd[mask]

def stack_features(specs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apila las características espectrales en un tensor combinado."""
    order = ["acc_mag", "gyro_mag", "S0", "S1", "S2"]
    imgs = [specs[k][2] for k in order]
    stacked = np.vstack(imgs)
    return stacked, specs[order[0]][1], specs[order[0]][0] 

def save_pgm(img: np.ndarray, out_path: Path) -> None:
    """Guarda matriz en formato PGM (escala de grises)."""
    imwrite(out_path, img)

###############################################################################
# ----------------------------- main routine ------------------------------- #
###############################################################################

def process_interval(
    data: Dict[str, pd.DataFrame], 
    params: Dict[str, Any], 
    lfeat: List[str], 
    *,
    min_samples: Optional[int] = None, 
    drop_short: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Procesa un intervalo temporal aplicando espectrogramas."""
    fhz = int(params["freq_target_hz"])  
    fs = int(params["freq_psd_hz"])     
    f0 = float(params["f_start_hz"])
    f1 = float(params["f_stop_hz"])

    n_per_seg_nom = int(round(fs * float(params["fact_hz"])))
    n_overlap_nom = int(n_per_seg_nom * float(params["window_overlap"]))
    min_samples = min_samples or n_per_seg_nom
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for foot, df in data.items():
        n_samples = len(df)
        if n_samples < min_samples:
            if drop_short: continue  
            n_per_seg = n_samples                          
            n_overlap = max(0, int(n_per_seg * float(params["window_overlap"])))
        else:
            n_per_seg = n_per_seg_nom
            n_overlap = n_overlap_nom

        if df.shape[0] == 0: continue 
        
        acc_mag = magnitude(df, ["Ax", "Ay", "Az"])
        gyro_mag = magnitude(df, ["Gx", "Gy", "Gz"])
        press = [df[c] for c in ("S0", "S1", "S2")]

        specs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for feature in lfeat:
            if feature == "acc_mag":
                psd = compute_psd_spectrogram(acc_mag, fhz, fs, n_per_seg, n_overlap, f0, f1)
            elif feature == "gyro_mag":
                psd = compute_psd_spectrogram(gyro_mag, fhz, fs, n_per_seg, n_overlap, f0, f1)
            elif feature == "S0":
                psd = compute_psd_spectrogram(press[0], fhz, fs, n_per_seg, n_overlap, f0, f1)
            elif feature == "S1":
                psd = compute_psd_spectrogram(press[1], fhz, fs, n_per_seg, n_overlap, f0, f1)
            elif feature == "S2":
                psd = compute_psd_spectrogram(press[2], fhz, fs, n_per_seg, n_overlap, f0, f1)
            else:
                logger.error(f"CARACTERISTICA NO RECONOCIDA: {feature}")
                raise ValueError(f"Feature {feature} not found")
            specs[feature] = psd

        results[foot] = stack_features(specs)

    return results

def cli() -> None: 
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(
        description="Dual-foot data extractor with backend-agnostic InfluxDB support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument("--backend", choices=["influxdb-client", "influxdbms"], help="Select database backend")

    parser.add_argument("-c", "--config", type=Path, help="Path to config.yaml")
    parser.add_argument("-o", "--output", type=Path, help="Path to output directory")
    parser.add_argument("-e", "--excel", type=Path, help="Path to metadata Excel", required=True)

    args = parser.parse_args()
    lfeat = ['acc_mag','gyro_mag','S0','S1','S2']

    try:
        param_config_from_yaml = load_param_config(args.config)
        params = merge_params(args, param_config_from_yaml)
        
        if 'io' not in params: params['io'] = {}
        if 'params' not in params: params['params'] = {}

        params['io']['excel'] = args.excel
        windows = read_time_windows(params['io']['excel'])

        out_path = args.output if args.output else Path(params['io'].get('out', './resultados'))
        out_path.mkdir(parents=True, exist_ok=True)

        if args.backend == "influxdbms":
            if args.config is None:
                raise ValueError("--config is required when backend=influxdbms")
            client = cInfluxDB(config_path=str(args.config))
        else:
            raise NotImplementedError("influxdb-client backend refactoring pending.")

        # PROCESAR VENTANAS
        for idx, row in windows.iterrows():
            label = f"segment_{idx:03d}"
            start, end = row["start"].to_pydatetime(), row["end"].to_pydatetime()
            
            qtok = str(row["reference"]).strip()
            mov  = row.get("mov_type", None)
            
            if len(qtok) < 5: 
                logger.warning(f"[{label}] CodeID '{qtok}' muy corto. Saltando.")
                continue

            raw_data_by_foot: Dict[str, pd.DataFrame] = {} 

            for foot in FEET:
                df = client.query_data(start, end, qtok, foot)
                logger.info(f"{label}:{foot} => {len(qtok)} : Shape {df.shape}")
                
                if not df.empty:
                    df = df.set_index("_time").sort_index()
                    df.index = pd.to_datetime(df.index)
                    df.rename(columns=SENSOR_FIELDS_INFLUXDBMS, inplace=True)
                    raw_data_by_foot[foot] = df
            
            if len(raw_data_by_foot) == len(FEET): 
                common_start = max(df.index.min() for df in raw_data_by_foot.values())
                common_end = min(df.index.max() for df in raw_data_by_foot.values())

                if common_start >= common_end:
                    logger.warning(f"[{label}] Rango de tiempo inválido. Saltando.")
                    continue

                freq_target = float(params['params'].get('freq_target_hz', 75.0))
                target_idx = pd.date_range(
                    start=common_start, end=common_end,
                    freq=pd.Timedelta(seconds=1/freq_target), name="_time" 
                )
                
                if (end.replace(tzinfo=None) - common_end).total_seconds() > 0.1:
                    logger.warning(f"[{label}] Segmento común demasiado corto. Saltando.")
                    continue

                data_by_foot: Dict[str, pd.DataFrame] = {}
                for foot, df_raw in raw_data_by_foot.items():
                    df_resampled = uniform_timebase(df_raw, freq_target, target_idx_override=target_idx)

                    if not df_resampled.empty:
                        if df_resampled.isnull().values.any():
                            df_resampled = df_resampled.fillna(0) 
                        data_by_foot[foot] = df_resampled
            
                if len(data_by_foot) == len(FEET) and not any(df.empty for df in data_by_foot.values()):
                    tensors = process_interval(data_by_foot, params['params'], lfeat)

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
                                logger.warning(f"SEGMENTO {idx+1} PARQUET TIENE NANS.")
                            ta_pd.to_parquet(parquet_path)
                            logger.info(f"TENSOR GUARDADO: {parquet_path.name} | Shape: {tensor_all.shape}")

        client.close() 
        logger.info("PIPELINE DE EXTRACCION FINALIZADO CORRECTAMENTE")

    except Exception as e:
        logger.critical(f"EJECUCION ABORTADA CRITICAMENTE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli()
