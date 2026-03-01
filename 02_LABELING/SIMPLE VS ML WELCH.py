
# -*- coding: utf-8 -*-
"""

Created on Mon Jan 12 20:18:44 2026

@author: jairi

HEURISTICO VS ML
===============================================

"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, List

# --- CONFIGURACION ---
INPUT_DIR = Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\CODIGOS EXTRACCION\DATOS_1_CSV")
OUTPUT_DIR = Path("RESULTADOS_ROBUSTOS")
FS = 100            # Hz
WINDOW_SEC = 4.0    # Ventana de análisis
LOWER_FREQ = 0.5    # Umbral Heurístico
UPPER_FREQ = 3.5

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustComparator:
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def load_and_process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga datos, elimina referencias temporales y extrae características de frecuencia.
        """
        X_features = []
        y_labels = []
        
        files = list(INPUT_DIR.glob("*.csv"))
        logger.info(f"Procesando {len(files)} archivos para extracción de características...")

        for f in files:
            # 1. ETIQUETADO AMPLIO (BROAD CLASS 0)
            fname = f.name.lower()
            if "walking" in fname and "not_walking" not in fname and "rest" not in fname:
                label = 1 # Marcha
            else:
                label = 0 # No Marcha (Rest, Raw, Static, Temblores, etc.)

            # 2. PROCESAMIENTO
            try:
                df = pd.read_csv(f)
                sig = self._preprocess_signal(df)
                if sig is None: continue

                # 3. VENTANEO (ANONIMIZACION TEMPORAL)
                
                features = self._extract_windows(sig)
                
                if len(features) > 0:
                    X_features.append(features)
                    y_labels.extend([label] * len(features))
                    
            except Exception as e:
                logger.warning(f"Error en {fname}: {e}")

        if not X_features:
            return np.array([]), np.array([])

        return np.concatenate(X_features), np.array(y_labels)

    def _preprocess_signal(self, df: pd.DataFrame) -> np.ndarray:
        # Filtrar pie izquierdo si existe duplicidad
        if "Foot" in df.columns and "Left" in df["Foot"].values:
            df = df[df["Foot"] == "Left"]
        
        # Calcular Magnitud
        if "modA" in df.columns:
            val = df["modA"].values
        elif {"Ax", "Ay", "Az"}.issubset(df.columns):
            val = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)
        else:
            return None
            
        # INTERPOLACION (100Hz)
        
        if "_time" not in df.columns: return None
        df["_time"] = pd.to_datetime(df["_time"], format='mixed', utc=True)
        t_sec = (df["_time"] - df["_time"].iloc[0]).dt.total_seconds().values
        t_new = np.arange(0, t_sec[-1], 1/FS)
        val_interp = np.interp(t_new, t_sec, val)
        
        return val_interp - np.mean(val_interp)

    def _extract_windows(self, signal_arr: np.ndarray) -> np.ndarray:
        """Genera matriz de características [Frecuencia, Potencia]."""
        win_samp = int(WINDOW_SEC * FS)
        feats = []
        
        for i in range(0, len(signal_arr) - win_samp, win_samp):
            segment = signal_arr[i:i+win_samp]
            
            # WELCH 
            freqs, psd = signal.welch(segment, fs=FS, nperseg=min(len(segment), 256))
            
            dom_freq = freqs[np.argmax(psd)]
            total_power = np.sum(psd)
            
            feats.append([dom_freq, total_power])
            
        return np.array(feats)

    def run_comparison(self):
        # 1. Cargar Datos
        X, y = self.load_and_process_data()
        logger.info(f"Dataset Final: {X.shape} muestras. Distribución: {np.bincount(y)}")

        # 2. Configurar Validación Cruzada (5-Fold Stratified)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 3. Evaluar Random Forest (ML)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Score precisión 
        scores_rf = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
        y_pred_cv = cross_val_predict(rf, X, y, cv=cv)

        # 4. COMPARAR HEURISTICA EN LOS MISMOS CASOS
        y_pred_heur = ((X[:, 0] >= LOWER_FREQ) & (X[:, 0] <= UPPER_FREQ)).astype(int)
        acc_heur = accuracy_score(y, y_pred_heur)

        # 5. RESULTADOS
        
        print("\n" + "="*50)
        print(" RESULTADOS ROBUSTOS (VALIDACION CRUZADA)")
        print("="*50)
        print(f"Heurística (Global):      {acc_heur:.2%}")
        print(f"Random Forest (CV Media): {scores_rf.mean():.2%} (+/- {scores_rf.std()*2:.2%})")
        print("-" * 50)
        print("Matriz de Confusión (Random Forest Acumulada):")
        print(confusion_matrix(y, y_pred_cv))
        print("\nReporte de Clasificación:")
        print(classification_report(y, y_pred_cv, target_names=["No Marcha (Clase 0)", "Marcha (Clase 1)"]))

        # 6. GRAFICA
        self._plot_results(X, y, y_pred_cv)

    def _plot_results(self, X, y_true, y_pred):
        plt.figure(figsize=(12, 6))
        
        # Puntos reales
        plt.scatter(X[y_true==0, 0], X[y_true==0, 1], c='red', alpha=0.3, label='Real: No Marcha', s=20)
        plt.scatter(X[y_true==1, 0], X[y_true==1, 1], c='blue', alpha=0.3, label='Real: Marcha', s=20)
        
        # Errores del modelo (Circulos negros alrededor)
        errors = y_true != y_pred
        if errors.any():
            plt.scatter(X[errors, 0], X[errors, 1], facecolors='none', edgecolors='black', s=80, label='Error ML')

        plt.axvline(LOWER_FREQ, c='k', ls='--')
        plt.axvline(UPPER_FREQ, c='k', ls='--', label='Limites Heurísticos')
        
        plt.title("Espacio de Características: Frecuencia vs Potencia")
        plt.xlabel("Frecuencia Dominante (Hz)")
        plt.ylabel("Potencia Espectral (Energía)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / "robust_comparison.png")
        logger.info("Gráfica guardada en RESULTADOS_ROBUSTOS")

if __name__ == "__main__":
    RobustComparator().run_comparison()