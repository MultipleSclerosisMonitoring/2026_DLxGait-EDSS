# -*- coding: utf-8 -*-
"""
Este módulo implementa un pipeline comparativo para la detección de eventos de
marcha a partir de señales inerciales. Evalúa la eficacia de un umbral
frecuencial heurístico estático frente a un modelo Random Forest validado
mediante validación cruzada estratificada (K-Fold).
"""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, Optional
from pydantic import BaseModel, Field

# CONFIGURAR REGISTRO LOGS
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineConfig(BaseModel):
    """
    Configuración estructurada para el pipeline de extracción.
    """
    input_dir: Path
    output_dir: Path = Field(default=Path("resultados"))
    fs: int = Field(default=100, gt=0)
    window_sec: float = Field(default=4.0, gt=0.0)
    lower_freq: float = Field(default=0.5, ge=0.0)
    upper_freq: float = Field(default=3.5, gt=0.0)
    seed: int = Field(default=42, ge=0)

class RobustComparator:
    """
    Clase para procesar señales y comparar clasificadores.
    """
    
    def __init__(self, config: PipelineConfig) -> None:
        """
        Inicializa el comparador validando la configuración.

        :param config: Parámetros del pipeline.
        :type config: PipelineConfig
        :return: Nada.
        :rtype: None
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def load_and_process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga archivos y extrae atributos frecuenciales.

        :return: Características y etiquetas extraídas.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        x_features = []
        y_labels = []
        
        # BUSCAR ARCHIVOS CSV
        files = list(self.config.input_dir.glob("*.csv"))
        logger.info(f"Procesando {len(files)} archivos.")

        for f in files:
            # ASIGNAR ETIQUETA CLASE
            fname = f.name.lower()
            if "walking" in fname and "not_walking" not in fname and "rest" not in fname:
                label = 1
            else:
                label = 0

            try:
                # LEER DATOS CSV
                df = pd.read_csv(f)
                sig = self._preprocess_signal(df)
                
                if sig is None: 
                    continue

                # EXTRAER CARACTERISTICAS VENTANA
                features = self._extract_windows(sig)
                
                if len(features) > 0:
                    x_features.append(features)
                    y_labels.extend([label] * len(features))
                    
            except Exception as e:
                logger.warning(f"Error procesando {fname}: {e}")

        if not x_features:
            return np.array([]), np.array([])

        return np.concatenate(x_features), np.array(y_labels)

    def _preprocess_signal(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Filtra y normaliza la señal inercial.

        :param df: Datos de los sensores.
        :type df: pd.DataFrame
        :return: Señal interpolada o nula.
        :rtype: Optional[np.ndarray]
        """
        # FILTRAR PIE IZQUIERDO
        if "Foot" in df.columns and "Left" in df["Foot"].values:
            df = df[df["Foot"] == "Left"]
        
        # MAGNITUD DE ACELERACION
        if "modA" in df.columns:
            val = df["modA"].values
        elif {"Ax", "Ay", "Az"}.issubset(df.columns):
            val = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)
        else:
            return None
            
        # VERIFICAR COLUMNA TIEMPO
        if "_time" not in df.columns: 
            return None
            
        # INTERPOLAR SEÑAL UNIFORME
        df["_time"] = pd.to_datetime(df["_time"], format='mixed', utc=True)
        t_sec = (df["_time"] - df["_time"].iloc[0]).dt.total_seconds().values
        t_new = np.arange(0, t_sec[-1], 1 / self.config.fs)
        val_interp = np.interp(t_new, t_sec, val)
        
        return val_interp - np.mean(val_interp)

    def _extract_windows(self, signal_arr: np.ndarray) -> np.ndarray:
        """
        Divide la señal y extrae frecuencias.

        :param signal_arr: Señal inercial preprocesada.
        :type signal_arr: np.ndarray
        :return: Matriz de características espectrales.
        :rtype: np.ndarray
        """
        win_samp = int(self.config.window_sec * self.config.fs)
        feats = []
        
        # ITERAR POR VENTANAS
        for i in range(0, len(signal_arr) - win_samp, win_samp):
            segment = signal_arr[i:i + win_samp]
            
            # APLICAR TRANSFORMADA WELCH
            freqs, psd = signal.welch(segment, fs=self.config.fs, nperseg=min(len(segment), 256))
            dom_freq = freqs[np.argmax(psd)]
            total_power = np.sum(psd)
            
            feats.append([dom_freq, total_power])
            
        return np.array(feats)

    def run_comparison(self) -> None:
        """
        Ejecuta el pipeline de comparación.

        :return: Nada.
        :rtype: None
        """
        # CARGAR DATOS PROCESADOS
        x_data, y_data = self.load_and_process_data()
        if len(x_data) == 0:
            logger.error("Dataset vacío. Finalizando.")
            return

        logger.info(f"Dataset generado: {x_data.shape} muestras.")

        # VALIDACION CRUZADA ESTRATIFICADA
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.seed)
        rf = RandomForestClassifier(n_estimators=100, random_state=self.config.seed)
        
        # EVALUAR RANDOM FOREST
        scores_rf = cross_val_score(rf, x_data, y_data, cv=cv, scoring='accuracy')
        y_pred_cv = cross_val_predict(rf, x_data, y_data, cv=cv)

        # EVALUAR REGLA HEURISTICA
        y_pred_heur = ((x_data[:, 0] >= self.config.lower_freq) & (x_data[:, 0] <= self.config.upper_freq)).astype(int)
        acc_heur = accuracy_score(y_data, y_pred_heur)

        # MOSTRAR RESULTADOS METRICAS
        print("\n" + "="*50)
        print(" RESULTADOS ROBUSTOS (VALIDACION CRUZADA)")
        print("="*50)
        print(f"Heurística (Global):      {acc_heur:.2%}")
        print(f"Random Forest (CV Media): {scores_rf.mean():.2%} (+/- {scores_rf.std()*2:.2%})")
        print("-" * 50)
        print("Matriz de Confusión (Random Forest):")
        print(confusion_matrix(y_data, y_pred_cv))
        print("\nReporte de Clasificación:")
        print(classification_report(y_data, y_pred_cv, labels=[0, 1], target_names=["No Marcha (0)", "Marcha (1)"], zero_division=0))

        # GUARDAR GRAFICO DISPERSION
        self._plot_results(x_data, y_data, y_pred_cv)

    def _plot_results(self, x_data: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Genera gráfico bidimensional de características.

        :param x_data: Matriz de características.
        :type x_data: np.ndarray
        :param y_true: Etiquetas reales.
        :type y_true: np.ndarray
        :param y_pred: Etiquetas predichas.
        :type y_pred: np.ndarray
        :return: Nada.
        :rtype: None
        """
        plt.figure()
        
        # GRAFICAR CLASES REALES
        plt.scatter(x_data[y_true==0, 0], x_data[y_true==0, 1], label='Real: No Marcha', s=20)
        plt.scatter(x_data[y_true==1, 0], x_data[y_true==1, 1], label='Real: Marcha', s=20)
        
        # RESALTAR ERRORES MODELO
        errors = y_true != y_pred
        if errors.any():
            plt.scatter(x_data[errors, 0], x_data[errors, 1], marker='x', color='black', label='Error ML')

        # DIBUJAR LIMITES HEURISTICOS
        plt.axvline(self.config.lower_freq, color='black', linestyle='--')
        plt.axvline(self.config.upper_freq, color='black', linestyle='--', label='Limites Heurísticos')
        
        plt.xlabel("Frecuencia Dominante (Hz)")
        plt.ylabel("Potencia Espectral")
        plt.legend()
        
        # GUARDAR FIGURA LOCAL
        out_path = self.config.output_dir / "robust_comparison.png"
        plt.savefig(out_path)
        logger.info(f"Gráfica guardada en: {out_path}")

def main() -> None:
    """
    Punto de entrada principal del script.

    :return: Nada.
    :rtype: None
    """
    # ANALIZAR ARGUMENTOS CLI
    parser = argparse.ArgumentParser(description="Comparador Heurístico vs ML para análisis de marcha.")
    parser.add_argument("--input", type=str, required=True, help="Ruta al directorio de archivos CSV.")
    parser.add_argument("--output", type=str, default="RESULTADOS_ROBUSTOS", help="Ruta de guardado.")
    parser.add_argument("--fs", type=int, default=100, help="Frecuencia de muestreo (Hz).")
    parser.add_argument("--window", type=float, default=4.0, help="Tamaño de la ventana (segundos).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    
    args = parser.parse_args()

    # VALIDAR MEDIANTE PYDANTIC
    config = PipelineConfig(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        fs=args.fs,
        window_sec=args.window,
        seed=args.seed
    )
    
    # INICIAR EJECUCION COMPARADOR
    comparator = RobustComparator(config)
    comparator.run_comparison()

if __name__ == "__main__":
    main()
