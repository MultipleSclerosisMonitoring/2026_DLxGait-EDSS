"""extract_data_plus.py
=================================
Enhanced end‑to‑end pipeline to extract dual‑foot sensor data from InfluxDB **using either** the generic *influxdb-client* **or** the project‑specific wrapper `InfluxDBms` 
"""
from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio import imwrite

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from scipy import signal

# Imports específicos para la clase cInfluxDB
from zoneinfo import ZoneInfo
import urllib3
import yaml

# Desactiva las advertencias de SSL para evitar mensajes al usar InfluxDBClient
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Intenta importar InfluxDBClient. Si no está instalado, se manejará
# más adelante en la función cli().
try:
    from influxdb_client import InfluxDBClient  # type: ignore
except ImportError:
    InfluxDBClient = None  # type: ignore

# Definición de la clase cInfluxDB (copiada del archivo cInfluxDB.py original)
# Esta clase debe estar definida antes de ser utilizada en la función cli.
class cInfluxDB:
    def __init__(self, config_path: str, timeout: int = 500_000):
        """
        Initializes the connection to InfluxDB using a YAML configuration file.

        :param config_path: Path to the YAML configuration file.
        :type config_path: str
        :param timeout: Connection timeout in milliseconds.
        :type timeout: int

        """
        # Cargar la configuración desde el archivo YAML
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Extraer los valores necesarios
        self.bucket = config['influxdb']['bucket']
        self.org = config['influxdb']['org']
        self.token = config['influxdb']['token']
        self.url = config['influxdb']['url']
        self.tzval = config['influxdb'].get('tzval', 'Europe/Madrid')

        # Inicializar el cliente de InfluxDB
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, \
                                     verify_ssl=False, timeout=timeout)
        self.measurement = self.bucket.split("/")[0] if '/' in self.bucket else \
                           self.bucket
    
    def query_data(self, from_date: datetime, to_date: datetime, qtok: str, pie: str, metrics=None) -> pd.DataFrame:
        """
        Query data in InfluxDB, pivoting the results to get the metrics in columns.

        :param from_date: Start date (ISO 8601 format: 'YYYYY'-MM-DDTHH:MM:SSZ).
        :type from_date: datetime
        :param to_date: End date (ISO 8601 format: 'YYYYY'-MM-DDTHH:MM:SSZ).
        :type to_date: datetime
        :param qtok: CodeID 
        :type qtok: str
        :param pie: Left or Right foot ('Right', 'Left')
        :type pie: str
        :param metrics: List of metrics to query (default: predefined set)
        :type metrics: list[str], optional

        :return: DataFrame with the metrics pivoted on columns, ordered by _time descending.
        :rtype: pd.DataFrame
        """

        if to_date <= from_date:
            df = pd.DataFrame() 
            return df
        
        from_date_str = from_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(
                        timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')  # UTC con 'Z'
        to_date_str = to_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(
                        timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')  # UTC con 'Z'

        # Métricas por defecto
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
            print(f"Error in the query: {str(e)}")
            raise

        # Procesar los resultados en un DataFrame
        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        if len(data) > 0:
            df = pd.DataFrame(data).drop(['result', 'table'], axis=1)
            df["_time"] = pd.to_datetime(df["_time"], unit='ns')
            df['_time'] = df['_time'].dt.tz_convert(self.tzval).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            return df.sort_values(by="_time", ascending=False).reset_index(drop=True)
        else:
            df = pd.DataFrame() 
            return df


    def query_with_aggregate_window(self, from_date: datetime, to_date: datetime, window_size: str = "20ms", 
                                    qtok: str = None, pie: str = None, metrics=None) -> pd.DataFrame:
        """
        Query data in InfluxDB with aggregateWindow, pivoting the results to get metrics as columns.

        :param from_date: Start datetime (ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ).
        :type from_date: datetime
        :param to_date: End datetime (ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ).
        :type to_date: datetime
        :param window_size: Aggregation window size (default: '20ms').
        :type window_size: str
        :param qtok: CodeID (required).
        :type qtok: str
        :param pie: Left or Right foot ('Right', 'Left') (required).
        :type pie: str
        :param metrics: List of metrics to query (default: predefined set).
        :type metrics: list[str], optional
        :return: DataFrame with metrics as columns, ordered by _time.
        :rtype: pd.DataFrame
        """

        if not qtok or not pie:
            raise ValueError("Los argumentos 'qtok' y 'pie' son obligatorios para esta consulta.")

        if to_date <= from_date:
            df = pd.DataFrame() 
            return df
        
        from_date_str = from_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(
                        timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')  # UTC con 'Z'
        to_date_str = to_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(
                        timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')  # UTC con 'Z'


        # Métricas por defecto
        if metrics is None:
            metrics = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'S0', 'S1', 'S2']

        metrics_str = ' or '.join([f'r._field == "{metric}"' for metric in metrics])
        columns_str = ', '.join([f'"{metric}"' for metric in metrics])

        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: time(v: "{from_date_str}"), stop: time(v: "{to_date_str}"))
            |> filter(fn: (r) => r._measurement == "{self.measurement}")
            |> filter(fn: (r) => {metrics_str})
            |> filter(fn: (r) => r["CodeID"] == "{qtok}" and r["type"] == "SCKS" and r["Foot"] == "{pie}")
            |> group(columns: ["_field"])
            |> aggregateWindow(every: {window_size}, fn: last, createEmpty: true)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> keep(columns: ["_time", {columns_str}])
        '''

        try:
            result = self.client.query_api().query(org=self.org, query=query)
        except Exception as e:
            print(f"Error en la consulta: {str(e)}")
            raise

        # Procesar los resultados en un DataFrame
        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        if len(data) > 0:
            df = pd.DataFrame(data).drop(['result', 'table'], axis=1)
            df['_time']=df['_time'].dt.tz_convert(self.tzval).dt.strftime("%Y-%M-%D %H:%m:%S")
        else:
            df = pd.DataFrame()

        # Asegurar que todas las métricas están presentes en el DataFrame
        for col in ["_time"] + metrics:
            if col not in df:
                df[col] = None  # Rellenar con None si falta alguna columna

        return df.sort_values(by="_time", ascending=False).reset_index(drop=True)
    
    def uniform_timebase(self, df: pd.DataFrame,
        freq_hz: float, *, method: str = "time", agg: str = "mean",
        enforce_unique_index: bool = True,
        target_idx_override: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """Interpola un ``DataFrame`` a una malla temporal uniforme sin perder datos.
        ...
        """
        if df.shape[0] == 0:
            return df
    
        # 1) Orden cronológico
        df = df.sort_index()
        df.index = pd.to_datetime(df.index) # fuerza dtype datetime64[ns]

        # 2) Opcional: colapsar duplicados
        if enforce_unique_index and df.index.has_duplicates:
            df = df.groupby(level=0).agg(agg)

        # 3) Construir malla uniforme O usar la proporcionada
        if target_idx_override is not None: # <--- USO DEL NUEVO PARÁMETRO
            target_idx = target_idx_override
        else:
            step = pd.Timedelta(seconds=1.0 / freq_hz)
            target_idx = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq=step,
                name=df.index.name,
            )

        # 4) Unión de índices (muestras + grid)
        full_idx = target_idx.union(df.index)

        # 5) Reindexar sobre la unión e interpolar
        df_interp = (
            df.reindex(full_idx)
            .interpolate(method=method, limit_direction="both")
        )

        # 6) Extraer solo los nodos del grid uniforme
        return df_interp.loc[target_idx]

    def close(self):
        self.client.close()        

# Se asume que InfluxDBms.cInfluxDB es un módulo externo o que cInfluxDB
# está directamente disponible. Para este archivo único, la clase cInfluxDB
# se ha incluido directamente.
HAS_INFLUXDBMS = True # Se considera True ya que la clase está definida aquí.


###############################################################################
# ------------------------------- defaults ---------------------------------- #
###############################################################################

SENSOR_FIELDS_INFLUXDB_CLIENT: Dict[str, List[str]] = {
    "acc": ["Ax", "Ay", "Az"],
    "gyro": ["Gx", "Gy", "Gz"],
    "press": ["S0", "S1", "S2"],
}

# Mapeo de nombres de métricas de InfluxDBms → nombres genéricos usados downstream
SENSOR_FIELDS_INFLUXDBMS: Dict[str, str] = {
    "Ax": "Ax",
    "Ay": "Ay",
    "Az": "Az",
    "Gx": "Gx",
    "Gy": "Gy",
    "Gz": "Gz",
    "S0": "S0",
    "S1": "S1",
    "S2": "S2",
}

FEET: Tuple[str, str] = ("Left", "Right")  # Capitalizado para coincidir con InfluxDBms

###############################################################################
# ----------------------------- helper funcs -------------------------------- #
###############################################################################

def load_param_config(path: Optional[Path]) -> Dict[str, object]:
    """Load a YAML parameter config if provided, else return empty dict.

    Args:
        path: Path to a YAML file. If *None*, returns an empty dict.

    Returns:
        Dict[str, object]: Parsed parameter dictionary.
    """
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to use --param-config")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def merge_params(cli_ns: argparse.Namespace, cfg: Dict[str, object]) -> Dict[str, object]:
    """Merge CLI args with YAML parameters (CLI has priority).

    Args:
        cli_ns: Argparse namespace from CLI.
        cfg: Parameters loaded from YAML.

    Returns:
        Dict[str, object]: Effective parameters for the run.
    """
    params = {}
    params.update(cfg)
    # Bucle de sobrescritura de CLI
    # Asegurarse de que 'params' tiene la clave 'params' anidada
    if 'params' not in params:
        params['params'] = {}
    for key in ["freq_target_hz", "f_start_hz", "f_stop_hz", "f_step_hz", "window_overlap"]:
        val = getattr(cli_ns, key, None)
        if val is not None:
            params['params'][key] = val
    return params


def nperseg(freq_target_hz: int, f_step_hz: float) -> int:
    """Compute *nperseg* so that Δf ≈ *f_step_hz*.

    Args:
        freq_target_hz: Resampling frequency in Hz.
        fact_hz: Factor for sizing the hamming window.

    Returns:
        int: Number of samples per STFT segment.
    """
    return int(round(freq_target_hz * f_step_hz))


def read_time_windows(excel_path: Path) -> pd.DataFrame:
    """Read the first sheet and coerce *start* & *end* columns to UTC.

    The sheet can optionally include a **Reference** (CodeID) column which is
    required when using `--backend influxdbms`.

    Args:
        excel_path: Path to the Excel workbook.

    Returns:
        pd.DataFrame: DataFrame with at least `start`, `end`, and optionally `Reference`.
    """
    df = pd.read_excel(excel_path, sheet_name=0)
    # Aceptar múltiples nombres de columna posibles
    df.rename(
        columns={
            col: col.lower() for col in df.columns  # unificar mayúsculas/minúsculas
        }, inplace=True
    )
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
    return df


def magnitude(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Compute Euclidean norm for three component columns.

    Args:
        df: DataFrame containing the component columns.
        cols: List with exactly three column names.

    Returns:
        pd.Series: Series with the magnitude values.
    """
    return np.sqrt((df[cols].astype(float) ** 2).sum(axis=1))


def compute_psd_spectrogram(
    series: pd.Series,fhz:int, fs_hz: int,n_per_seg: int,n_overlap: int,
    f_start_hz: float,f_stop_hz: float,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a PSD spectrogram centred on each windowed instant.

    Args:
        series: 1‑D signal.
        fhz: Sampling frequency of data
        fs_hz: Interesting frequency for the FFT.
        n_per_seg: Samples per STFT segment.
        n_overlap: Overlap in samples.
        f_start_hz: Lower frequency bound.
        f_stop_hz: Upper frequency bound.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (frequencies, times, PSD) where
        PSD is *frequency×time*.
    """
    f, t, Sxx = signal.spectrogram(
        series.astype(float).values,
        fs=fs_hz,
        window="hamming",
        nperseg=n_per_seg,
        noverlap=n_overlap,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    # Se añade un pequeño valor para evitar log(0)
    psd_db = 10 * np.log10(Sxx + 1e-12)
    # Normalización [0,1]
    min_val = np.min(psd_db)
    max_val = np.max(psd_db)
    normalized_psd = (psd_db - min_val) / (max_val - min_val)
    # Escalar a 255
    scaled_psd = (normalized_psd * 255).astype(np.uint8)
    
    mask = (f >= f_start_hz) & (f <= f_stop_hz + 1e-9)
    return f[mask], t*fs_hz/fhz, scaled_psd[mask]


def stack_features(specs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack 5 feature spectrograms into a single image.

    Args:
        specs: Mapping *feature_name → (f, t, PSD)*.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            * image – *(125, T)* merged spectrogram.
            * t – STFT frame centres.
    """
    order = ["acc_mag", "gyro_mag", "S0", "S1", "S2"]
    imgs = [specs[k][2] for k in order]
    stacked = np.vstack(imgs)
    return stacked, specs[order[0]][1], specs[order[0]][0]  # compartir el mismo t


def save_png(img: np.ndarray, t: np.ndarray, out_path: Path, title: str) -> None:
    """Persist the dB‑scaled spectrogram as PNG.

    Args:
        img: 2‑D array *(freq_bins, time_frames)*.
        t: 1‑D array with frame centres (s) – actualmente no usado pero reservado para
           futuras marcas del eje x.
        out_path: Destination path (including *.png*).
        title: Title text for the figure.
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(10 * np.log10(img + 1e-12), aspect="auto", origin="lower")
    plt.title(title)
    plt.ylabel("Frequency bin (Δf = 0.05 Hz)")
    plt.xlabel("STFT frame index")
    plt.colorbar(label="Power / dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_pgm(img: np.ndarray, out_path: Path) -> None:
    """Persist the dB‑scaled spectrogram as PGM.

    Args:
        img: 2‑D array *(freq_bins stacked as per 5 vars, time_frames)* .
        out_path: Destination path (including *.png*).
    """
    imwrite(out_path, img)
    
    
###############################################################################
# ----------------------------- backends ------------------------------------ #
###############################################################################

def query_segment_influxdb_client(
    client: InfluxDBClient, bucket: str, measurement: str,
    foot: str, start: datetime, stop: datetime) -> pd.DataFrame:
    """Download a segment using the *influxdb‑client* package."""
    
    fields = (
        SENSOR_FIELDS_INFLUXDB_CLIENT["acc"]
        + SENSOR_FIELDS_INFLUXDB_CLIENT["gyro"]
        + SENSOR_FIELDS_INFLUXDB_CLIENT["press"]
    )
    
    # 1. Calculamos los strings fuera de la query para evitar backslashes en la f-string
    field_filter = " or ".join([f'r["_field"] == "{f}"' for f in fields])
    cols_str = ', '.join([f'"{f}"' for f in fields])
    
    # 2. Construimos la query limpia
    query = f"""
from(bucket: "{bucket}")
  |> range(start: {start.isoformat()}, stop: {stop.isoformat()})
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => r["Foot"] == "{foot}")
  |> filter(fn: (r) => {field_filter})
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> keep(columns: ["_time", {cols_str}])
"""
    # Ejecutamos la query
    df = client.query_api().query_data_frame(query)  # type: ignore
    df = df.set_index("_time").sort_index()
    return df


def query_segment_influxdbms(db: cInfluxDB,start: datetime,
    stop: datetime,qtok: str,foot: str) -> pd.DataFrame:
    """Download a segment using the **InfluxDBms** wrapper.

    Args:
        db: An instantiated `cInfluxDB` object.
        start: Interval start (UTC).
        stop: Interval end (UTC).
        qtok: CodeID (*Reference*) for the subject/session.
        foot: Foot identifier ("Left"/"Right").

    Returns:
        pd.DataFrame: DataFrame with columns renamed to generic names.
    """
    metrics = list(SENSOR_FIELDS_INFLUXDBMS.keys())
    raw = db.query_data(start, stop, qtok=qtok, pie=foot, metrics=metrics)
    if raw.shape[0] == 0:
        return pd.DataFrame()
    
    raw = raw.set_index("_time").sort_index()
    # Renombrar a nombres genéricos para que el código downstream sea agnóstico al backend
    raw.rename(columns=SENSOR_FIELDS_INFLUXDBMS, inplace=True)
    return raw

###############################################################################
# ----------------------------- main routine ------------------------------- #
###############################################################################

def process_interval(
            data: Dict[str, pd.DataFrame], params: Dict[str, object], lfeat: List[str] , *,
            min_samples: int | None = None, drop_short: bool = False,
        ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Calcula espectrogramas multi-señal para el intervalo de ambos pies.

    Ajusta `nperseg`/`noverlap` si el número de muestras no alcanza la
    configuración nominal o, opcionalmente, descarta el pie.

    Args:
        data:
            Diccionario *foot → DataFrame* ya remuestreados a la malla uniforme.
            Deben contener las columnas ``["Ax","Ay","Az","Gx","Gy","Gz","S0","S1","S2"]``.
        params:
            Parámetros globales del pipeline. Deben incluir al menos:

                - ``"freq_target_hz"`` (int)
                - ``"f_start_hz"``, ``"f_stop_hz"``, ``"f_step_hz"`` (float)
                - ``"window_overlap"``  (0 – 1)
        lfeat:
            List of features in ['acc_mag','gyro_mag','S0','S1','S2']

        min_samples:
            Mínimo de muestras requeridas para procesar un pie.  
            Si *None*, se usará `nperseg` nominal.
        drop_short:
            Si *True* y el pie no alcanza `min_samples`, se descarta.
            Si *False* (por defecto) se reduce dinámicamente `nperseg`
            y `noverlap` al tamaño admisible.

    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]:  
            Mapeo *foot → (tensor_espectrograma, t_frames)*; los pies descartados
            no aparecen en el diccionario.
    """
    # --- parámetros nominales ---
    fhz= int(params["freq_target_hz"])  # 70 Hz
    fs = int(params["freq_psd_hz"])     # 20 Hz
    f0 = float(params["f_start_hz"])
    f1 = float(params["f_stop_hz"])
    df_hz = float(params["fact_hz"])  # 3 => 20*3 = 60

    n_per_seg_nom = nperseg(fs, df_hz) # 3 => 20*3 = 60
    n_overlap_nom = int(n_per_seg_nom * float(params["window_overlap"]))

    # Umbral por defecto: una ventana completa
    min_samples = min_samples or n_per_seg_nom

    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # --- bucle por pie -------------------------------------------------------
    for foot, df in data.items():
        n_samples = len(df)
        if n_samples < min_samples:
            if drop_short:
                continue  # -- descartar pie corto --
            # -- ajustar tamaño de ventana/overlap --
            n_per_seg = n_samples                          # todo el bloque
            n_overlap = max(0, int(n_per_seg * params["window_overlap"]))
        else:
            n_per_seg = n_per_seg_nom
            n_overlap = n_overlap_nom

        if df.shape[0] == 0:
            continue # Saltar si el DataFrame está vacío después del remuestreo/recorte
        
        # Magnitudes vectoriales
        acc_mag = magnitude(df, ["Ax", "Ay", "Az"])
        gyro_mag = magnitude(df, ["Gx", "Gy", "Gz"])
        press = [df[c] for c in ("S0", "S1", "S2")]

        specs = {}
        # Espectrogramas PSD
        for j in range(len(lfeat)):
            if lfeat[j] == "acc_mag":
                psd = compute_psd_spectrogram(acc_mag, fhz, fs, n_per_seg, n_overlap, f0, f1)
            elif lfeat[j] == "gyro_mag":
                psd = compute_psd_spectrogram(gyro_mag, fhz, fs, n_per_seg, n_overlap, f0, f1)
            elif lfeat[j] == "S0":
                psd = compute_psd_spectrogram(press[0], fhz, fs, n_per_seg, n_overlap, f0, f1)
            elif lfeat[j] == "S1":
                psd = compute_psd_spectrogram(press[1], fhz, fs, n_per_seg, n_overlap, f0, f1)
            elif lfeat[j] == "S2":
                psd = compute_psd_spectrogram(press[2], fhz, fs, n_per_seg, n_overlap, f0, f1)
            else:
                print("Error: Feature not found calculating the PSD")
                sys.exit(10)
            specs[lfeat[j]] = psd

        img, t_frames, frqs = stack_features(specs)
        results[foot] = (img, t_frames, frqs)

    return results


def cli() -> None:  # noqa: C901 – keep flat for clarity
    """Command‑line interface dispatcher."""
    parser = argparse.ArgumentParser(
        description="Dual‑foot data extractor with backend‑agnostic InfluxDB support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument("--backend", choices=["influxdb-client", "influxdbms"], help="Select database backend")

    parser.add_argument("-c", "--config", type=Path, help="Path to config.yaml (backend=influxdbms)")
    parser.add_argument("-o","--output", type=Path, help="Path to output directory")
    
    # Argumentos específicos para influxdb-client si no se usa param-config
    parser.add_argument("--url", type=str, help="InfluxDB URL (for influxdb-client backend)")
    parser.add_argument("--token", type=str, help="InfluxDB Token (for influxdb-client backend)")
    parser.add_argument("--org", type=str, help="InfluxDB Organization (for influxdb-client backend)")
    parser.add_argument("--bucket", type=str, help="InfluxDB Bucket (for influxdb-client backend)")

    args = parser.parse_args()
    lfeat = ['acc_mag','gyro_mag','S0','S1','S2']

    # ---------------- params ---------------- #
    # Cargar configuración de parámetros desde YAML si se proporciona
    param_config_from_yaml = load_param_config(args.config)
    # Fusionar argumentos CLI con parámetros YAML (CLI tiene prioridad)
    params = merge_params(args, param_config_from_yaml)
    
    # Asegurarse de que las claves 'io' y 'params' existan en el diccionario params
    if 'io' not in params:
        params['io'] = {}
    if 'params' not in params:
        params['params'] = {}

    # Asignar el path del excel desde args si no está en params['io']
    if 'excel' not in params['io']:
        params['io']['excel'] = args.excel


    # ---------------- excel ----------------- #
    windows = read_time_windows(params['io']['excel'])
    if args.backend == "influxdbms" and "reference" not in windows.columns:
        sys.exit("Excel must include a 'Reference' column when backend=influxdbms")

    Path(params['io']['out']).mkdir(parents=True, exist_ok=True)

    # ---------------- backend init ---------- #
    client = None
    if args.backend == "influxdb-client":
        if InfluxDBClient is None:
            sys.exit("influxdb-client package not installed. `pip install influxdb-client`.")
        
        # Obtener los detalles de conexión, priorizando CLI sobre YAML
        client_url = args.url or params.get('influxdb', {}).get('url')
        client_token = args.token or params.get('influxdb', {}).get('token')
        client_org = args.org or params.get('influxdb', {}).get('org')
        client_bucket = args.bucket or params.get('influxdb', {}).get('bucket')

        missing = []
        if client_url is None: missing.append("url")
        if client_token is None: missing.append("token")
        if client_org is None: missing.append("org")
        if client_bucket is None: missing.append("bucket")

        if missing:
            sys.exit(f"Missing required arguments for influxdb-client: {', '.join(missing)} (check CLI args or param-config)")
        
        client = InfluxDBClient(url=client_url, token=client_token, org=client_org)  # type: ignore
        # Aseguramos que bucket y measurement estén accesibles para query_segment_influxdb_client
        # Esto es un workaround para pasar los valores a la función, ya que args no los tiene si vienen de YAML
        args.bucket = client_bucket
        args.measurement = client_bucket.split("/")[0] if '/' in client_bucket else client_bucket # Derivar measurement del bucket
    else: # backend is influxdbms
        # HAS_INFLUXDBMS es True porque la clase cInfluxDB está definida en este mismo archivo
        if args.config is None:
            sys.exit("--config is required when backend=influxdbms")
        client = cInfluxDB(config_path=str(args.config))

    # ---------------- iterate windows ------- #
    for idx, row in windows.iterrows():
        label = f"segment_{idx:03d}"
        start, end = row["start"].to_pydatetime(), row["end"].to_pydatetime()

        raw_data_by_foot: Dict[str, pd.DataFrame] = {} # Usamos un nombre diferente para los datos crudos

        for foot in FEET:
            df = pd.DataFrame() # Inicializar df para cada pie
            if args.backend == "influxdb-client":
                df = query_segment_influxdb_client(client,
                    bucket=args.bucket, measurement=args.measurement,
                    foot=foot, start=start, stop=end)
            else:  # influxdbms
                qtok = row["reference"]
                mov  = row["mov_type"] if "mov_type" in row else None
                if len(qtok) < 5: # Saltar si el CodeID es demasiado corto
                    print(f"[{label}] CodeID '{qtok}' para el pie {foot} es demasiado corto. Saltando.")
                    continue

                df = query_segment_influxdbms(client, start, end, qtok, foot)
                print(f"{label}:{foot} => {len(qtok)} : Size {df.shape}")
            
            # Almacenar los DataFrames crudos con sus índices de tiempo originales
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                raw_data_by_foot[foot] = df
        
        # --- NUEVO: Sincronizar y remuestrear ambos pies a una base de tiempo común ---
        data_by_foot: Dict[str, pd.DataFrame] = {}
        # Solo proceder si hemos obtenido datos para ambos pies en este intervalo
        if len(raw_data_by_foot) == len(FEET): 
            # 1. Encontrar el inicio más tardío y el fin más temprano entre ambos pies
            common_start = max(df.index.min() for df in raw_data_by_foot.values())
            common_end = min(df.index.max() for df in raw_data_by_foot.values())

            # Asegurarse de que el rango de tiempo común sea válido
            if common_start >= common_end:
                print(f"[{label}] Rango de tiempo común inválido ({common_start} a " +
                      f"{common_end}) para ambos pies. Saltando este segmento.")
                continue

            # 2. Crear una base de tiempo uniforme para ambos pies
            freq_target = float(params['params']['freq_target_hz'])
            target_idx = pd.date_range(
                start=common_start,
                end=common_end,
                freq=pd.Timedelta(seconds=1/freq_target),
                name="_time" # Aseguramos el nombre del índice
            )
            # Si la duración calculada del target_idx es significativamente menor que la esperada
            # O si el número de puntos en target_idx es menor que un mínimo aceptable
            if (end.replace(tzinfo=None) - common_end).total_seconds() > 0.1:
                 print(f"[{label}] Segmento de tiempo común demasiado corto " +
                       f"((end.replace(tzinfo=None) - common_end).total_seconds()). "+
                       f"Saltando este segmento.")
                 continue

            # 3. Remuestrear y recortar cada DataFrame a la base de tiempo común
            for foot, df_raw in raw_data_by_foot.items():
                # Remuestrear individualmente (la función uniform_timebase ya hace esto)
                df_resampled = client.uniform_timebase(df_raw, freq_target, target_idx_override=target_idx)

                if not df_resampled.empty:
                    if df_resampled.isnull().values.any():
                        print(f"[{label}] ADVERTENCIA: NaNs en el DataFrame remuestreado para " +
                              f"el pie {foot} (antes de PSD). Rellenando con 0.")
                        df_resampled = df_resampled.fillna(0) # Rellenar con 0 para evitar problemas en PSD
                    data_by_foot[foot] = df_resampled
                else:
                    print(f"[{label}] El DataFrame para el pie {foot} está vacío después " +
                          f"del remuestreo y recorte. No se procesará este pie.")
        else:
            print(f"[{label}] No se obtuvieron datos para ambos pies en este " +
                  f"intervalo. Saltando el procesamiento de tensores.")

        # Solo procesar si ambos pies tienen datos después del remuestreo y recorte
        if len(data_by_foot) == len(FEET) and not any(df.empty for df in data_by_foot.values()):
            num_cols_left = data_by_foot["Left"].shape[1] if "Left" in data_by_foot else 0
            num_cols_right = data_by_foot["Right"].shape[1] if "Right" in data_by_foot else 0
            if len(data_by_foot["Left"].index) != len(data_by_foot["Right"].index):
                print(f"Lenght of both Legs dataset is different: {num_cols_left} / {num_cols_right}")    
            tensors = process_interval(data_by_foot, params['params'], lfeat)

            # ---------- persistencia ---------- #
            for foot, (img, t_frames, frqs) in tensors.items():
                # Asegúrate de que mov sea una cadena válida o maneja el caso None
                mov_str = f"-{mov}" if mov is not None else ""
                png_path = args.output / f"{label}_{foot}{mov_str}.pgm"
                save_pgm(img, png_path)
                print(f"[✓] {png_path.relative_to(args.output.parent)} : {foot} => {len(t_frames)}. Shape:{img.shape}")

            if len(tensors.keys()) == 2: # Si ambos pies fueron procesados y tienen tensores
                # Verificación adicional de longitud para asegurar que son idénticos
                if len(tensors["Left"][1]) == len(tensors["Right"][1]):
                    tensor_all = np.vstack([tensors["Left"][0], tensors["Right"][0]])
                    parquet_path = args.output / f"{label}_tensor{mov_str}.parquet"
                    ta_pd = pd.DataFrame(tensor_all)
                    #
                    combined_datetimes = [start + timedelta(seconds=float(t)) for t in t_frames]
                    lcls = [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for dt in combined_datetimes]
                    ta_pd.columns = lcls
                    tapd = ta_pd.copy()
                    #
                    lidx = []
                    for foot in FEET:
                        for feat in lfeat:
                            for ifrq in frqs:
                                itm = f"{foot}-{feat}-{ifrq:.2f}Hz"
                                lidx.append(itm)
                    tapd.index = lidx
                    if tapd.isnull().any().any():
                        print(f"Segment:{idx+1} the parquet file has NaNs. Alert!!")
                    tapd.to_parquet(parquet_path)
                    print(f"[✓] tensor → {parquet_path.relative_to(args.output.parent)} => Size:{tensor_all.shape}")
                else:
                    print(f"[{label}] Las longitudes de tiempo de los tensores Left y Right no coinciden después del procesamiento. No se guardará el parquet combinado.")
            else:
                print(f"[{label}] No se generaron tensores para ambos pies. No se guardará el parquet combinado.")

    # Cerrar cliente si es necesario
    client.close()  # type: ignore

    print("Todos los intervalos procesados ✔")


if __name__ == "__main__":
    cli()
