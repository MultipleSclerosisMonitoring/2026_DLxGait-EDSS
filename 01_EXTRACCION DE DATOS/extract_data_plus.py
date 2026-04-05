# -*- coding: utf-8 -*-
"""extract_data_plus.py
=================================
Enhanced end‑to‑end pipeline to extract dual‑foot sensor data from InfluxDB.
MODIFICADO: Corrección de Data Leakage, unificación de timebase y CLI bugs.
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

from zoneinfo import ZoneInfo
import urllib3
import yaml

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from influxdb_client import InfluxDBClient  # type: ignore
except ImportError:
    InfluxDBClient = None  # type: ignore

###############################################################################
# ----------------------------- helper funcs -------------------------------- #
###############################################################################

def uniform_timebase(df: pd.DataFrame, freq_hz: float, *, method: str = "time", 
                     agg: str = "mean", enforce_unique_index: bool = True,
                     target_idx_override: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """Interpola un DataFrame a una malla temporal uniforme sin perder datos."""
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
    def __init__(self, config_path: str, timeout: int = 500_000):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.bucket = config['influxdb']['bucket']
        self.org = config['influxdb']['org']
        self.token = config['influxdb']['token']
        self.url = config['influxdb']['url']
        self.tzval = config['influxdb'].get('tzval', 'Europe/Madrid')

        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, 
                                     verify_ssl=False, timeout=timeout)
        self.measurement = self.bucket.split("/")[0] if '/' in self.bucket else self.bucket
    
    def query_data(self, from_date: datetime, to_date: datetime, qtok: str, pie: str, metrics=None) -> pd.DataFrame:
        if to_date <= from_date:
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
            print(f"Error in the query: {str(e)}")
            raise

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
            return pd.DataFrame() 

    def query_with_aggregate_window(self, from_date: datetime, to_date: datetime, window_size: str = "20ms", 
                                    qtok: str = None, pie: str = None, metrics=None) -> pd.DataFrame:
        if not qtok or not pie:
            raise ValueError("Los argumentos 'qtok' y 'pie' son obligatorios.")

        if to_date <= from_date:
            return pd.DataFrame() 
        
        from_date_str = from_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') 
        to_date_str = to_date.replace(tzinfo=ZoneInfo(self.tzval)).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') 

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

        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)

        if len(data) > 0:
            df = pd.DataFrame(data).drop(['result', 'table'], axis=1)
            # CORRECCION DEL BUG DE FECHA AQUI:
            df['_time'] = df['_time'].dt.tz_convert(self.tzval).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        else:
            df = pd.DataFrame()

        for col in ["_time"] + metrics:
            if col not in df:
                df[col] = None

        return df.sort_values(by="_time", ascending=False).reset_index(drop=True)

    def close(self):
        self.client.close()        

HAS_INFLUXDBMS = True

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

def load_param_config(path: Optional[Path]) -> Dict[str, object]:
    if path is None: return {}
    if yaml is None: raise RuntimeError("PyYAML is required")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

def merge_params(cli_ns: argparse.Namespace, cfg: Dict[str, object]) -> Dict[str, object]:
    params = {}
    params.update(cfg)
    if 'params' not in params:
        params['params'] = {}
    for key in ["freq_target_hz", "f_start_hz", "f_stop_hz", "f_step_hz", "window_overlap"]:
        val = getattr(cli_ns, key, None)
        if val is not None:
            params['params'][key] = val
    return params

def nperseg(freq_target_hz: int, f_step_hz: float) -> int:
    return int(round(freq_target_hz * f_step_hz))

def read_time_windows(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=0)
    df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)
    
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
    return np.sqrt((df[cols].astype(float) ** 2).sum(axis=1))

def compute_psd_spectrogram(
    series: pd.Series,fhz:int, fs_hz: int,n_per_seg: int,n_overlap: int,
    f_start_hz: float,f_stop_hz: float,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    f, t, Sxx = signal.spectrogram(
        series.astype(float).values, fs=fs_hz, window="hamming",
        nperseg=n_per_seg, noverlap=n_overlap, detrend=False,
        scaling="density", mode="psd",
    )
    psd_db = 10 * np.log10(Sxx + 1e-12)
    min_val, max_val = np.min(psd_db), np.max(psd_db)
    
    # Prevenir division por cero
    if max_val == min_val:
        normalized_psd = np.zeros_like(psd_db)
    else:
        normalized_psd = (psd_db - min_val) / (max_val - min_val)
        
    scaled_psd = (normalized_psd * 255).astype(np.uint8)
    mask = (f >= f_start_hz) & (f <= f_stop_hz + 1e-9)
    return f[mask], t*fs_hz/fhz, scaled_psd[mask]

def stack_features(specs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    order = ["acc_mag", "gyro_mag", "S0", "S1", "S2"]
    imgs = [specs[k][2] for k in order]
    stacked = np.vstack(imgs)
    return stacked, specs[order[0]][1], specs[order[0]][0] 

def save_pgm(img: np.ndarray, out_path: Path) -> None:
    imwrite(out_path, img)
    
###############################################################################
# ----------------------------- backends ------------------------------------ #
###############################################################################

def query_segment_influxdb_client(
    client: InfluxDBClient, bucket: str, measurement: str,
    foot: str, start: datetime, stop: datetime, qtok: str) -> pd.DataFrame:
    """MODIFICADO: Se añade el filtrado estricto por qtok (CodeID) para evitar fugas."""
    
    fields = (SENSOR_FIELDS_INFLUXDB_CLIENT["acc"] + 
              SENSOR_FIELDS_INFLUXDB_CLIENT["gyro"] + 
              SENSOR_FIELDS_INFLUXDB_CLIENT["press"])
    
    field_filter = " or ".join([f'r["_field"] == "{f}"' for f in fields])
    cols_str = ', '.join([f'"{f}"' for f in fields])
    
    query = f"""
from(bucket: "{bucket}")
  |> range(start: {start.isoformat()}, stop: {stop.isoformat()})
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => r["CodeID"] == "{qtok}") 
  |> filter(fn: (r) => r["Foot"] == "{foot}")
  |> filter(fn: (r) => {field_filter})
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> keep(columns: ["_time", {cols_str}])
"""
    df = client.query_api().query_data_frame(query)  # type: ignore
    
    # InfluxDBClient devuelve lista si hay múltiples tablas
    if isinstance(df, list):
        if len(df) == 0: return pd.DataFrame()
        df = pd.concat(df)
        
    if df.empty:
        return df
        
    if "_time" in df.columns:
        df = df.set_index("_time").sort_index()
    return df

def query_segment_influxdbms(db: cInfluxDB,start: datetime, stop: datetime,qtok: str,foot: str) -> pd.DataFrame:
    metrics = list(SENSOR_FIELDS_INFLUXDBMS.keys())
    raw = db.query_data(start, stop, qtok=qtok, pie=foot, metrics=metrics)
    if raw.shape[0] == 0:
        return pd.DataFrame()
    
    raw = raw.set_index("_time").sort_index()
    raw.rename(columns=SENSOR_FIELDS_INFLUXDBMS, inplace=True)
    return raw

###############################################################################
# ----------------------------- main routine ------------------------------- #
###############################################################################

def process_interval(
        data: Dict[str, pd.DataFrame], params: Dict[str, object], lfeat: List[str] , *,
        min_samples: int | None = None, drop_short: bool = False,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    
    fhz= int(params["freq_target_hz"])  
    fs = int(params["freq_psd_hz"])     
    f0 = float(params["f_start_hz"])
    f1 = float(params["f_stop_hz"])

    n_per_seg_nom = int(round(fs * float(params["fact_hz"])))
    n_overlap_nom = int(n_per_seg_nom * float(params["window_overlap"]))
    min_samples = min_samples or n_per_seg_nom
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

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

        specs = {}
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
                print("Error: Feature not found")
                sys.exit(10)
            specs[lfeat[j]] = psd

        img, t_frames, frqs = stack_features(specs)
        results[foot] = (img, t_frames, frqs)

    return results


def cli() -> None: 
    parser = argparse.ArgumentParser(
        description="Dual‑foot data extractor with backend‑agnostic InfluxDB support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument("--backend", choices=["influxdb-client", "influxdbms"], help="Select database backend")

    parser.add_argument("-c", "--config", type=Path, help="Path to config.yaml")
    parser.add_argument("-o", "--output", type=Path, help="Path to output directory")
    parser.add_argument("-e", "--excel", type=Path, help="Path to metadata Excel", required=True) # MODIFICADO: Añadido para arreglar el bug de AttributeError

    parser.add_argument("--url", type=str, help="InfluxDB URL (for influxdb-client)")
    parser.add_argument("--token", type=str, help="InfluxDB Token (for influxdb-client)")
    parser.add_argument("--org", type=str, help="InfluxDB Organization (for influxdb-client)")
    parser.add_argument("--bucket", type=str, help="InfluxDB Bucket (for influxdb-client)")

    args = parser.parse_args()
    lfeat = ['acc_mag','gyro_mag','S0','S1','S2']

    param_config_from_yaml = load_param_config(args.config)
    params = merge_params(args, param_config_from_yaml)
    
    if 'io' not in params: params['io'] = {}
    if 'params' not in params: params['params'] = {}

    if 'excel' not in params['io']:
        params['io']['excel'] = args.excel

    windows = read_time_windows(params['io']['excel'])
    
    # MODIFICADO: Ahora exigimos la columna reference para TODOS los backends
    if "reference" not in windows.columns:
        sys.exit("Excel must include a 'reference' column to avoid Patient Data Leakage.")

    Path(params['io']['out']).mkdir(parents=True, exist_ok=True)
    if args.output is None: args.output = Path(params['io']['out'])

    client = None
    if args.backend == "influxdb-client":
        if InfluxDBClient is None:
            sys.exit("influxdb-client package not installed.")
        
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
            sys.exit(f"Missing required arguments for influxdb-client: {', '.join(missing)}")
        
        client = InfluxDBClient(url=client_url, token=client_token, org=client_org)  # type: ignore
        args.bucket = client_bucket
        args.measurement = client_bucket.split("/")[0] if '/' in client_bucket else client_bucket 
    else: 
        if args.config is None:
            sys.exit("--config is required when backend=influxdbms")
        client = cInfluxDB(config_path=str(args.config))

    # ---------------- iterate windows ------- #
    for idx, row in windows.iterrows():
        label = f"segment_{idx:03d}"
        start, end = row["start"].to_pydatetime(), row["end"].to_pydatetime()
        
        # MODIFICADO: Extraer identidad SIEMPRE antes de la query
        qtok = str(row["reference"])
        mov  = row["mov_type"] if "mov_type" in row else None
        
        if len(qtok) < 5: 
            print(f"[{label}] CodeID '{qtok}' es demasiado corto. Saltando para evitar fugas.")
            continue

        raw_data_by_foot: Dict[str, pd.DataFrame] = {} 

        for foot in FEET:
            df = pd.DataFrame() 
            if args.backend == "influxdb-client":
                df = query_segment_influxdb_client(client,
                    bucket=args.bucket, measurement=args.measurement,
                    foot=foot, start=start, stop=end, qtok=qtok) # MODIFICADO: Se pasa qtok
            else:  
                df = query_segment_influxdbms(client, start, end, qtok, foot)
                print(f"{label}:{foot} => {len(qtok)} : Size {df.shape}")
            
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                raw_data_by_foot[foot] = df
        
        data_by_foot: Dict[str, pd.DataFrame] = {}
        if len(raw_data_by_foot) == len(FEET): 
            common_start = max(df.index.min() for df in raw_data_by_foot.values())
            common_end = min(df.index.max() for df in raw_data_by_foot.values())

            if common_start >= common_end:
                print(f"[{label}] Rango de tiempo inválido. Saltando.")
                continue

            freq_target = float(params['params']['freq_target_hz'])
            target_idx = pd.date_range(
                start=common_start, end=common_end,
                freq=pd.Timedelta(seconds=1/freq_target), name="_time" 
            )
            
            if (end.replace(tzinfo=None) - common_end).total_seconds() > 0.1:
                 print(f"[{label}] Segmento común demasiado corto. Saltando.")
                 continue

            for foot, df_raw in raw_data_by_foot.items():
                # MODIFICADO: Usar la función global en lugar de client.uniform_timebase
                df_resampled = uniform_timebase(df_raw, freq_target, target_idx_override=target_idx)

                if not df_resampled.empty:
                    if df_resampled.isnull().values.any():
                        df_resampled = df_resampled.fillna(0) 
                    data_by_foot[foot] = df_resampled
        
        if len(data_by_foot) == len(FEET) and not any(df.empty for df in data_by_foot.values()):
            tensors = process_interval(data_by_foot, params['params'], lfeat)

            for foot, (img, t_frames, frqs) in tensors.items():
                mov_str = f"-{mov}" if mov is not None else ""
                png_path = args.output / f"{label}_{foot}{mov_str}.pgm"
                save_pgm(img, png_path)

            if len(tensors.keys()) == 2: 
                if len(tensors["Left"][1]) == len(tensors["Right"][1]):
                    tensor_all = np.vstack([tensors["Left"][0], tensors["Right"][0]])
                    parquet_path = args.output / f"{label}_tensor{mov_str}.parquet"
                    ta_pd = pd.DataFrame(tensor_all)
                    
                    combined_datetimes = [start + timedelta(seconds=float(t)) for t in t_frames]
                    lcls = [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for dt in combined_datetimes]
                    ta_pd.columns = lcls
                    tapd = ta_pd.copy()
                    
                    lidx = []
                    for foot in FEET:
                        for feat in lfeat:
                            for ifrq in frqs:
                                itm = f"{foot}-{feat}-{ifrq:.2f}Hz"
                                lidx.append(itm)
                    tapd.index = lidx
                    if tapd.isnull().any().any():
                        print(f"Segment:{idx+1} parquet has NaNs. Alert!!")
                    tapd.to_parquet(parquet_path)
                    print(f"[] tensor → {parquet_path.relative_to(args.output.parent)} => Size:{tensor_all.shape}")

    if hasattr(client, 'close'):
        client.close() 
    print("# EXTRACCION FINALIZADA")

if __name__ == "__main__":
    cli()