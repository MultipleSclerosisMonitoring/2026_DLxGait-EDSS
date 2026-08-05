# -*- coding: utf-8 -*-
"""
Este módulo implementa un pipeline comparativo para la detección de eventos de
marcha a partir de señales inerciales. Evalúa la eficacia de un umbral
frecuencial heurístico estático frente a un modelo Random Forest, bajo dos
esquemas de validación (StratifiedKFold convencional y Leave-One-Patient-Out),
e incluye una visualización temporal de ground truth vs. predicciones sobre
un paciente y rango de tiempo específicos.
"""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy import signal
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
)
from typing import Tuple, Optional, List, Dict
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_TIMEZONE = "Europe/Madrid"


def extraer_paciente(stem: str) -> str:
    """
    Extrae el CodeID del paciente desde el nombre de archivo (sin extension),
    manejando correctamente el caso "not_walking", que contiene "walking"
    como substring y rompería un split ingenuo por "_walking" primero.

    :param stem: Nombre de archivo sin extension (Path.stem).
    :type stem: str
    :return: Identificador del paciente (CodeID).
    :rtype: str
    """
    if "_not_walking" in stem:
        return stem.split("_not_walking")[0]
    return stem.split("_walking")[0]


def hora_local_a_utc(fecha_str: str) -> pd.Timestamp:
    """
    Interpreta una fecha/hora como hora local (Europe/Madrid) y la
    convierte a UTC, consistente con el manejo de zona horaria del
    resto del pipeline (A01, extract_data_csv.py).

    :param fecha_str: Fecha/hora en formato "YYYY-MM-DD HH:MM:SS", interpretada como hora local.
    :type fecha_str: str
    :return: Timestamp en UTC.
    :rtype: pd.Timestamp
    """
    naive = pd.Timestamp(fecha_str)
    return naive.tz_localize(ZoneInfo(LOCAL_TIMEZONE)).tz_convert("UTC")


class PipelineConfig(BaseModel):
    """Configuración estructurada para el pipeline de comparación."""
    input_dir: Path
    output_dir: Path = Field(default=Path("resultados"))
    fs: int = Field(default=100, gt=0)
    window_sec: float = Field(default=4.0, gt=0.0)
    lower_freq: float = Field(default=1.0, ge=0.0)   
    upper_freq: float = Field(default=2.5, gt=0.0)   
    seed: int = Field(default=42, ge=0)
    paciente_visualizar: Optional[str] = Field(default=None)
    rango_inicio: Optional[str] = Field(default=None)
    rango_fin: Optional[str] = Field(default=None)
    tramos_ground_truth: Optional[List[Dict]] = Field(default=None)


class RobustComparator:
    """Clase para procesar señales, comparar clasificadores y visualizar resultados."""

    def __init__(self, config: PipelineConfig) -> None:
        """
        Inicializa el comparador validando la configuración.

        :param config: Parámetros del pipeline.
        :type config: PipelineConfig
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================== CARGA Y PREPROCESAMIENTO ====================

    def load_and_process_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga archivos, extrae features y el grupo (paciente) de cada
        ventana, extraído del nombre de archivo (convención:
        {CodeID}_{walking|not_walking}_{idx}.csv).

        :return: Features, etiquetas y grupos (paciente) por ventana.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        x_features, y_labels, groups = [], [], []
        files = list(self.config.input_dir.glob("*.csv"))
        logger.info(f"Procesando {len(files)} archivos.")

        for f in files:
            fname = f.name.lower()
            label = 1 if ("walking" in fname and "not_walking" not in fname and "rest" not in fname) else 0
            paciente = extraer_paciente(f.stem)

            try:
                df = pd.read_csv(f)
                sig, _ = self._preprocess_signal(df)
                if sig is None:
                    continue

                features = self._extract_windows(sig)
                if len(features) > 0:
                    x_features.append(features)
                    y_labels.extend([label] * len(features))
                    groups.extend([paciente] * len(features))

            except Exception as e:
                logger.warning(f"Error procesando {fname}: {e}")

        if not x_features:
            return np.array([]), np.array([]), np.array([])

        return np.concatenate(x_features), np.array(y_labels), np.array(groups)

    def _preprocess_signal(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[pd.Series]]:
        """
        Filtra y normaliza la señal inercial, devolviendo también las
        marcas de tiempo absolutas de cada muestra interpolada.

        :param df: Datos de los sensores.
        :type df: pd.DataFrame
        :return: Señal centrada y marcas de tiempo, o (None, None).
        :rtype: Tuple[Optional[np.ndarray], Optional[pd.Series]]
        """
        if "Foot" in df.columns and "Left" in df["Foot"].values:
            df = df[df["Foot"] == "Left"].copy()

        if "modA" in df.columns:
            val = df["modA"].values
        elif {"Ax", "Ay", "Az"}.issubset(df.columns):
            val = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)
        else:
            return None, None

        if "_time" not in df.columns:
            return None, None

        df["_time"] = pd.to_datetime(df["_time"], format='mixed', utc=True)
        t_sec = (df["_time"] - df["_time"].iloc[0]).dt.total_seconds().values
        t_new = np.arange(0, t_sec[-1], 1 / self.config.fs)
        val_interp = np.interp(t_new, t_sec, val)
        tiempos_abs = df["_time"].iloc[0] + pd.to_timedelta(t_new, unit="s")

        return val_interp - np.mean(val_interp), tiempos_abs

    def _extract_windows(self, signal_arr: np.ndarray) -> np.ndarray:
        """
        Divide la señal en ventanas y extrae características espectrales.

        :param signal_arr: Señal inercial preprocesada.
        :type signal_arr: np.ndarray
        :return: Matriz de características (frecuencia dominante, potencia total).
        :rtype: np.ndarray
        """
        win_samp = int(self.config.window_sec * self.config.fs)
        feats = []
        for i in range(0, len(signal_arr) - win_samp, win_samp):
            segment = signal_arr[i:i + win_samp]
            freqs, psd = signal.welch(segment, fs=self.config.fs, nperseg=min(len(segment), 256))
            feats.append([freqs[np.argmax(psd)], np.sum(psd)])
        return np.array(feats)

    def _extract_windows_con_tiempo(self, signal_arr: np.ndarray, tiempos_abs) -> Tuple[np.ndarray, list]:
        """
        Igual que _extract_windows, pero además devuelve la marca de
        tiempo central de cada ventana (para la visualización temporal).

        :param signal_arr: Señal inercial preprocesada.
        :type signal_arr: np.ndarray
        :param tiempos_abs: Marcas de tiempo absolutas de la señal.
        :type tiempos_abs: pd.Series
        :return: Features y lista de tiempos centrales por ventana.
        :rtype: Tuple[np.ndarray, list]
        """
        win_samp = int(self.config.window_sec * self.config.fs)
        feats, t_centro = [], []
        for i in range(0, len(signal_arr) - win_samp, win_samp):
            segment = signal_arr[i:i + win_samp]
            freqs, psd = signal.welch(segment, fs=self.config.fs, nperseg=min(len(segment), 256))
            feats.append([freqs[np.argmax(psd)], np.sum(psd)])
            t_centro.append(tiempos_abs[i + win_samp // 2])
        return np.array(feats), t_centro

    # ==================== COMPARACION PRINCIPAL ====================

    def run_comparison(self) -> None:
        """
        Ejecuta la comparación completa: heurística, RF sin agrupar
        (StratifiedKFold) y RF bajo LeaveOneGroupOut por paciente,
        reportando métricas equivalentes para ambos métodos.
        """
        x_data, y_data, groups = self.load_and_process_data()
        if len(x_data) == 0:
            logger.error("Dataset vacío. Finalizando.")
            return

        n_pacientes = len(np.unique(groups))
        logger.info(f"Dataset generado: {x_data.shape} muestras, {n_pacientes} pacientes.")

        # ----- HEURISTICA (metricas completas) -----
        y_pred_heur = ((x_data[:, 0] >= self.config.lower_freq) & (x_data[:, 0] <= self.config.upper_freq)).astype(int)
        acc_heur = accuracy_score(y_data, y_pred_heur)
        auc_heur = roc_auc_score(y_data, y_pred_heur)
        prauc_heur = average_precision_score(y_data, y_pred_heur)
        balacc_heur = balanced_accuracy_score(y_data, y_pred_heur)
        mcc_heur = matthews_corrcoef(y_data, y_pred_heur)

        # ----- RF SIN AGRUPAR (StratifiedKFold, para contraste) -----
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.seed)
        rf_sin_agrupar = RandomForestClassifier(n_estimators=100, random_state=self.config.seed)
        scores_rf_sin_agrupar = cross_val_score(rf_sin_agrupar, x_data, y_data, cv=cv, scoring='accuracy')

        # ----- RF LOPO (agrupado por paciente, sin fuga) -----
        logo = LeaveOneGroupOut()
        y_true_lopo, y_pred_lopo, y_prob_lopo = [], [], []
        fold_aucs = []

        for train_idx, test_idx in logo.split(x_data, y_data, groups):
            y_train_fold = y_data[train_idx]
            if len(np.unique(y_train_fold)) < 2:
                continue

            rf_fold = RandomForestClassifier(n_estimators=100, random_state=self.config.seed)
            rf_fold.fit(x_data[train_idx], y_train_fold)

            y_test_fold = y_data[test_idx]
            y_pred_fold = rf_fold.predict(x_data[test_idx])
            y_prob_fold = rf_fold.predict_proba(x_data[test_idx])[:, 1]

            y_true_lopo.extend(y_test_fold)
            y_pred_lopo.extend(y_pred_fold)
            y_prob_lopo.extend(y_prob_fold)

            if len(np.unique(y_test_fold)) > 1:
                fold_aucs.append(roc_auc_score(y_test_fold, y_prob_fold))

        y_true_lopo = np.array(y_true_lopo)
        y_pred_lopo = np.array(y_pred_lopo)
        y_prob_lopo = np.array(y_prob_lopo)

        acc_lopo = accuracy_score(y_true_lopo, y_pred_lopo)
        auc_lopo = roc_auc_score(y_true_lopo, y_prob_lopo)
        prauc_lopo = average_precision_score(y_true_lopo, y_prob_lopo)
        balacc_lopo = balanced_accuracy_score(y_true_lopo, y_pred_lopo)
        mcc_lopo = matthews_corrcoef(y_true_lopo, y_pred_lopo)

        # ----- REPORTE -----
        print("\n" + "=" * 60)
        print(" COMPARACION: HEURISTICA vs RF (SIN AGRUPAR) vs RF (LOPO)")
        print("=" * 60)
        print(f"{'Metrica':<25}{'Heuristica':>15}{'RF LOPO':>15}")
        print(f"{'Accuracy':<25}{acc_heur:>15.2%}{acc_lopo:>15.2%}")
        print(f"{'AUC':<25}{auc_heur:>15.4f}{auc_lopo:>15.4f}")
        print(f"{'PR-AUC':<25}{prauc_heur:>15.4f}{prauc_lopo:>15.4f}")
        print(f"{'Balanced Accuracy':<25}{balacc_heur:>15.4f}{balacc_lopo:>15.4f}")
        print(f"{'MCC':<25}{mcc_heur:>15.4f}{mcc_lopo:>15.4f}")
        print("-" * 60)
        print(f"RF sin agrupar (CV Media): {scores_rf_sin_agrupar.mean():.2%} (+/- {scores_rf_sin_agrupar.std()*2:.2%})")
        if fold_aucs:
            print(f"RF LOPO (AUC medio por fold): {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs)*2:.4f})")
        else:
            print("RF LOPO (AUC medio por fold): No calculable (ningun fold con ambas clases en test)")
        print("-" * 60)
        print("Matriz de Confusión (RF, LOPO):")
        print(confusion_matrix(y_true_lopo, y_pred_lopo))
        print("\nReporte de Clasificación (RF, LOPO):")
        print(classification_report(y_true_lopo, y_pred_lopo, labels=[0, 1],
                                     target_names=["No Marcha (0)", "Marcha (1)"], zero_division=0))

        self._plot_comparacion(acc_heur, scores_rf_sin_agrupar.mean(), acc_lopo, auc_lopo,
                                y_true_lopo, y_pred_lopo)

        # ----- VISUALIZACION TEMPORAL (opcional, requiere config completa) -----
        if self.config.paciente_visualizar and self.config.rango_inicio and self.config.rango_fin:
            self._plot_timeline_ground_truth_coloreado()

    def _plot_comparacion(self, acc_heur, acc_sin_agrupar, acc_lopo, auc_lopo,
                           y_true_lopo, y_pred_lopo) -> None:
        """
        Genera panel comparativo: barras de accuracy y matriz de
        confusión del RF bajo LOPO.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        metodos = ["Heuristica", "RF sin agrupar\n(con posible fuga)", "RF LOPO\n(sin fuga)"]
        valores = [acc_heur, acc_sin_agrupar, acc_lopo]
        colores = ["salmon", "khaki", "seagreen"]

        axes[0].bar(metodos, valores, color=colores, edgecolor="black")
        axes[0].axhline(0.5, color="gray", linestyle=":", label="Azar (50%)")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Accuracy: efecto de agrupar por paciente")
        axes[0].legend()
        for i, v in enumerate(valores):
            axes[0].text(i, v + 0.03, f"{v:.1%}", ha="center", fontweight="bold")

        cm = confusion_matrix(y_true_lopo, y_pred_lopo)
        axes[1].imshow(cm, cmap="Greens")
        axes[1].set_title(f"Matriz de Confusion - RF LOPO\n(AUC: {auc_lopo:.3f})")
        axes[1].set_xlabel("Prediccion")
        axes[1].set_ylabel("Real")
        axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["No Marcha", "Marcha"])
        axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["No Marcha", "Marcha"])
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)

        plt.tight_layout()
        out_path = self.config.output_dir / "comparativa_lopo_vs_sin_agrupar.png"
        plt.savefig(out_path, dpi=200)
        logger.info(f"Grafica comparativa guardada en: {out_path}")

    # ==================== VISUALIZACION TEMPORAL CON GROUND TRUTH COLOREADO ====================

    def _plot_timeline_ground_truth_coloreado(self) -> None:
        """
        Entrena un RF excluyendo al paciente a visualizar (para evitar
        fuga), y grafica sobre un rango de tiempo especifico (hora local
        Europe/Madrid, convertida a UTC) la comparacion entre ground
        truth (fondo coloreado por tramo real), prediccion heuristica
        y prediccion del RF.
        """
        paciente = self.config.paciente_visualizar
        rango_inicio = hora_local_a_utc(self.config.rango_inicio)
        rango_fin = hora_local_a_utc(self.config.rango_fin)
        tramos = self.config.tramos_ground_truth or []

        logger.info(f"Generando timeline con ground truth coloreado para: {paciente}")
        logger.info(f"Rango solicitado (local): {self.config.rango_inicio} - {self.config.rango_fin}")
        logger.info(f"Rango convertido (UTC): {rango_inicio} - {rango_fin}")

        # ENTRENAR RF EXCLUYENDO AL PACIENTE VISUALIZADO
        x_train, y_train = [], []
        for f in self.config.input_dir.glob("*.csv"):
            paciente_f = extraer_paciente(f.stem)
            if paciente_f == paciente:
                continue
            fname = f.name.lower()
            label = 1 if ("walking" in fname and "not_walking" not in fname and "rest" not in fname) else 0
            df = pd.read_csv(f)
            sig, _ = self._preprocess_signal(df)
            if sig is None:
                continue
            feats = self._extract_windows(sig)
            if len(feats) > 0:
                x_train.append(feats)
                y_train.extend([label] * len(feats))

        x_train = np.concatenate(x_train)
        y_train = np.array(y_train)
        rf = RandomForestClassifier(n_estimators=100, random_state=self.config.seed)
        rf.fit(x_train, y_train)
        logger.info(f"RF entrenado con {len(x_train)} ventanas de otros pacientes.")

        # PROCESAR SOLO EL RANGO SOLICITADO DEL PACIENTE
        archivos_paciente = sorted(self.config.input_dir.glob(f"{paciente}_*.csv"))
        resultados = []

        for f in archivos_paciente:
            df = pd.read_csv(f)
            sig, tiempos_abs = self._preprocess_signal(df)
            if sig is None:
                continue

            if tiempos_abs[-1] < rango_inicio or tiempos_abs[0] > rango_fin:
                continue

            feats, t_centro = self._extract_windows_con_tiempo(sig, tiempos_abs)
            if len(feats) == 0:
                continue

            pred_rf = rf.predict(feats)
            pred_heur = ((feats[:, 0] >= self.config.lower_freq) & (feats[:, 0] <= self.config.upper_freq)).astype(int)

            for t, p_rf, p_heur in zip(t_centro, pred_rf, pred_heur):
                if rango_inicio <= t <= rango_fin:
                    resultados.append({"tiempo": t, "pred_rf": p_rf, "pred_heur": p_heur})

        if not resultados:
            logger.warning(f"No se encontraron ventanas para {paciente} en el rango solicitado.")
            return

        df_resultados = pd.DataFrame(resultados).sort_values("tiempo")
        logger.info(f"Ventanas dentro del rango solicitado: {len(df_resultados)}")

        # GRAFICAR
        fig, ax = plt.subplots(figsize=(14, 5))

        for tramo in tramos:
            inicio = hora_local_a_utc(tramo["inicio"])
            fin = hora_local_a_utc(tramo["fin"])
            if tramo["mov_type"] == 1:
                ax.axvspan(inicio, fin, color="gold", alpha=0.4)

        ax.plot(df_resultados["tiempo"], df_resultados["pred_heur"],
                label="Prediccion Heuristica", color="red", linewidth=1, linestyle="--", alpha=0.8, marker='x', markersize=6)
        ax.plot(df_resultados["tiempo"], df_resultados["pred_rf"],
                label="Prediccion RF (LOPO)", color="green", linewidth=1.5, linestyle="-", alpha=0.9, marker='o', markersize=6)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No Marcha", "Marcha"])
        ax.set_xlabel("Tiempo")
        ax.set_xlim(rango_inicio, rango_fin)
        ax.set_title(f"Comparacion temporal: Ground Truth (fondo) vs. Heuristica vs. RF (LOPO)\nPaciente: {paciente}")

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="gold", alpha=0.4, label="Marcha (Ground Truth)"))
        ax.legend(handles=handles, loc="lower right")

        plt.xticks(rotation=30)
        plt.tight_layout()

        salida = self.config.output_dir / f"timeline_{paciente}_rango.png"
        plt.savefig(salida, dpi=200)
        logger.info(f"Timeline guardado en: {salida}")


def main() -> None:
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(description="Comparador Heurística vs ML, con y sin agrupamiento LOPO.")
    parser.add_argument("--input", type=str, required=True, help="Ruta al directorio de archivos CSV.")
    parser.add_argument("--output", type=str, default="RESULTADOS_ROBUSTOS", help="Ruta de guardado.")
    parser.add_argument("--fs", type=int, default=100, help="Frecuencia de muestreo (Hz).")
    parser.add_argument("--window", type=float, default=4.0, help="Tamaño de la ventana (segundos).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--paciente-timeline", type=str, default=None,
                         help="CodeID del paciente a visualizar en la comparacion temporal (opcional).")

    args = parser.parse_args()

    config = PipelineConfig(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        fs=args.fs,
        window_sec=args.window,
        seed=args.seed,
        paciente_visualizar=args.paciente_timeline
    )

    comparator = RobustComparator(config)
    comparator.run_comparison()


if __name__ == "__main__":
    main()
