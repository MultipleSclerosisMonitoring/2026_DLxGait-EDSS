# -*- coding: utf-8 -*-
"""
Data Quality Audit Script
Performs temporal and integrity checks on Gait Analysis CSV files.
"""

import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# --- CONFIGURATION ---
POSSIBLE_DIRS = [
    Path("../DATOS 1_CSV"),
    Path("DATOS 1_CSV"),
    Path("../../DATOS 1_CSV"),
    Path("../DATOS_1_EXACTOS"),
    Path("DATOS_FINALES")
]

OUTPUT_REPORT_DIR = Path("REPORTES_CALIDAD")
GAP_THRESHOLD_SEC = 0.05

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

def get_input_directory():
    """Locates the input directory from possible paths."""
    for directory in POSSIBLE_DIRS:
        if directory.exists():
            return directory
    return None

def setup_environment():
    """ prepares directories and validates input."""
    input_dir = get_input_directory()
    if input_dir is None:
        logger.error(f"Input directory not found. Searched in: {[str(d) for d in POSSIBLE_DIRS]}")
        sys.exit(1)
    
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return input_dir

def load_and_clean_data(file_path):
    """Loads CSV and handles date parsing robustly."""
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        if "_time" not in df.columns:
            return None, "Missing '_time' column"

        # FIX: Handle mixed ISO formats (with and without microseconds)
        try:
            df["_time"] = pd.to_datetime(df["_time"], format='mixed')
        except ValueError:
            # Fallback for older pandas versions or legacy formats
            df["_time"] = pd.to_datetime(df["_time"], utc=True)

        df = df.sort_values("_time")
        return df, None
    except Exception as e:
        return None, str(e)

def analyze_dataset(file_path):
    """Calculates statistics for the dataset."""
    df, error = load_and_clean_data(file_path)
    if error:
        return None, error

    # 1. Temporal Analysis
    time_deltas = df["_time"].diff().dt.total_seconds()
    
    # Frequency estimation (median to ignore gaps)
    median_dt = time_deltas.median()
    fs_est = 1 / median_dt if (median_dt > 0 and not np.isnan(median_dt)) else 0.0

    # Gap detection
    gaps = time_deltas[time_deltas > GAP_THRESHOLD_SEC]
    num_gaps = len(gaps)
    max_gap = gaps.max() if num_gaps > 0 else 0.0

    # 2. Data Integrity (Magnitude)
    has_acc_cols = {"Ax", "Ay", "Az"}.issubset(df.columns)
    mean_modA = 0.0
    
    if "modA" in df.columns:
        mean_modA = df["modA"].mean()
    elif has_acc_cols:
        # Calculate resultant on the fly
        modA = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)
        mean_modA = modA.mean()

    duration = (df["_time"].iloc[-1] - df["_time"].iloc[0]).total_seconds()

    report = {
        "file": file_path.name,
        "samples": len(df),
        "duration_sec": round(duration, 2),
        "fs_hz": round(fs_est, 2),
        "num_gaps": num_gaps,
        "max_gap_sec": round(max_gap, 3),
        "mean_g_force": round(mean_modA, 3),
        "integrity": "OK" if has_acc_cols else "MISSING_SENSORS"
    }

    return df, report

def generate_quality_plot(df, report, output_path):
    """Generates visual report."""
    if df is None:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot 1: Signal
    time_axis = df["_time"]
    if "modA" in df.columns:
        signal = df["modA"]
    elif {"Ax", "Ay", "Az"}.issubset(df.columns):
        signal = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)
    else:
        signal = np.zeros(len(df))

    ax1.plot(time_axis, signal, linewidth=0.8, color="#1f77b4", label="Acceleration (g)")
    ax1.set_title(f"Quality Check: {report['file']}")
    ax1.set_ylabel("Acc (g)")
    ax1.grid(True, alpha=0.3)

    # Mark gaps
    deltas = df["_time"].diff().dt.total_seconds()
    gap_indices = deltas[deltas > GAP_THRESHOLD_SEC].index
    for idx in gap_indices:
        ax1.axvline(x=df.loc[idx, "_time"], color="red", alpha=0.5, linestyle="--")

    # Plot 2: Sampling Stability
    ax2.plot(time_axis, deltas * 1000, '.', markersize=2, color="#ff7f0e", label="Delta T")
    ax2.set_ylabel("Interval (ms)")
    ax2.set_xlabel("Time (UTC)")
    ax2.grid(True, alpha=0.3)
    
    # Format X axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def main():
    input_dir = setup_environment()
    logger.info(f"Input directory: {input_dir.resolve()}")

    files = list(input_dir.glob("*.csv"))
    if not files:
        logger.error("No CSV files found in directory.")
        sys.exit(0)

    logger.info(f"Processing {len(files)} files...")
    
    # Header for console output
    header = f"{'FILENAME':<35} | {'DUR(s)':<8} | {'Hz':<6} | {'GAPS':<5} | {'MAX GAP':<8} | {'STATUS'}"
    logger.info("-" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    results = []

    for file_path in files:
        df, report = analyze_dataset(file_path)

        if report is None: # Error case
            logger.error(f"{file_path.name:<35} | ERROR: {df}") # df contains error string here
            continue

        # Log row
        logger.info(
            f"{report['file']:<35} | "
            f"{report['duration_sec']:<8} | "
            f"{report['fs_hz']:<6} | "
            f"{report['num_gaps']:<5} | "
            f"{report['max_gap_sec']:<8} | "
            f"{report['integrity']}"
        )
        
        results.append(report)
        generate_quality_plot(df, report, OUTPUT_REPORT_DIR / f"QC_{file_path.stem}.png")

    # Save summary
    if results:
        csv_path = OUTPUT_REPORT_DIR / "summary_report.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)
        logger.info("-" * len(header))
        logger.info(f"Summary saved to: {csv_path}")

if __name__ == "__main__":
    main()