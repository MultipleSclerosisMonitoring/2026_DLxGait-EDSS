
"""
CSV Data Extraction 

"""

import argparse
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from influxdb_client import InfluxDBClient

# --- CONFIGURATION ---
#Modificar horas segun invierno y verano

LOCAL_TIMEZONE = "Europe/Madrid"
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_OUTPUT_DIR = Path("../DATOS_1_EXACTOS")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config file")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    
    # Optional overrides
    parser.add_argument("--url", type=str, help="InfluxDB URL")
    parser.add_argument("--token", type=str, help="InfluxDB Token")
    parser.add_argument("--org", type=str, help="InfluxDB Org")
    parser.add_argument("--bucket", type=str, help="InfluxDB Bucket")
    
    return parser.parse_args()

def load_config(config_path: Path):
    """Loads and validates the YAML configuration."""
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def fetch_gait_data(client, bucket: str, start_utc: pd.Timestamp, stop_utc: pd.Timestamp, patient_id: str) -> pd.DataFrame:
    """
    Queries InfluxDB for Gait data within a specific UTC range.
    """
    start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_str = stop_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start_str}, stop: {stop_str})
      |> filter(fn: (r) => r["_measurement"] == "Gait")
      |> filter(fn: (r) => r["CodeID"] == "{patient_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    try:
        result = client.query_api().query_data_frame(query)
        
        if isinstance(result, list):
            df = pd.concat(result)
        else:
            df = result

        if df.empty or "_time" not in df.columns:
            return pd.DataFrame()

        # Cleanup dataframe
        df = df.drop(columns=["result", "table"], errors="ignore")
        df = df.set_index("_time").sort_index()
        return df

    except Exception as e:
        logger.error(f"Query failed for {patient_id}: {e}")
        return pd.DataFrame()

def calculate_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the acceleration magnitude (modA) if components exist."""
    required_cols = {"Ax", "Ay", "Az"}
    
    if required_cols.issubset(df.columns):
        df["modA"] = np.sqrt(
            df["Ax"]**2 + 
            df["Ay"]**2 + 
            df["Az"]**2
        )
    return df

def main():
    args = parse_arguments()
    config = load_config(args.config)

    # InfluxDB Connection Details
    conf_influx = config.get("influxdb", config.get("influx", {}))
    url = args.url or conf_influx.get("url")
    token = args.token or conf_influx.get("token")
    org = args.org or conf_influx.get("org")
    bucket = args.bucket or conf_influx.get("bucket")

    # Output setup
    args.output.mkdir(parents=True, exist_ok=True)

    # Load Excel Request
    excel_path = Path(config.get("io", {}).get("excel", "solicitud.xlsx"))
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        sys.exit(1)

    df_request = pd.read_excel(excel_path)
    
    # Normalize columns
    df_request = df_request.rename(columns={
        "Reference": "CodeID",
        "ry to use": "CodeID",
        "mov_type": "move_type",
        "Tag": "tag"
    })

    logger.info(f"Starting extraction process. Timezone: {LOCAL_TIMEZONE}")

    # Connect and Process
    try:
        with InfluxDBClient(url=url, token=token, org=org, verify_ssl=False, timeout=60000, retries=3) as client:
            
            total_rows = len(df_request)
            
            for idx, row in df_request.iterrows():
                patient_id = row.get("CodeID")
                move_type = row.get("move_type")
                
                try:
                    # Timezone conversion: Local -> UTC
                    start_naive = pd.to_datetime(row["datefrom"])
                    stop_naive = pd.to_datetime(row["dateuntil"])
                    
                    start_utc = start_naive.tz_localize(LOCAL_TIMEZONE).tz_convert("UTC")
                    stop_utc = stop_naive.tz_localize(LOCAL_TIMEZONE).tz_convert("UTC")
                    
                except Exception as e:
                    logger.warning(f"Skipping row {idx}: Date parsing error for {patient_id}. Error: {e}")
                    continue
                
                logger.info(f"Processing [{idx+1}/{total_rows}]: {patient_id} | {move_type}")

                # Fetch Data
                df_sensor = fetch_gait_data(client, bucket, start_utc, stop_utc, patient_id)

                if df_sensor.empty:
                    logger.warning(f"No data found for {patient_id} in specified range.")
                    continue

                # Process Data
                df_sensor = calculate_magnitude(df_sensor)
                
                # Save Data
                filename = f"{patient_id}_{move_type}_{idx}.csv".replace(" ", "_")
                output_path = args.output / filename
                
                df_sensor.to_csv(output_path)
                logger.info(f"Saved: {output_path.name} ({len(df_sensor)} rows)")

    except Exception as e:
        logger.critical(f"Fatal error during execution: {e}")
        sys.exit(1)

    logger.info("Process completed successfully.")

if __name__ == "__main__":
    main()