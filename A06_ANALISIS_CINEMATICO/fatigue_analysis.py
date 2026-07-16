# -*- coding: utf-8 -*-
"""
Análisis de fatiga motora mediante regresión lineal.
Calcula la pendiente de degradación (biomarcador) utilizando datos sintéticos.
sucede al kinematic engine
"""

# IMPORTAR LIBRERIAS
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel, Field
from typing import Tuple, Dict

# CONFIGURAR PARAMETROS
class FatigueConfig(BaseModel):
    """
    Configuracion para analisis de fatiga.
    """
    file_path: str = Field(..., description="Ruta al archivo CSV")
    target_feature: str = Field(default="Gait_Speed_ms", description="Variable a analizar")

# DEFINIR CLASE ANALISIS
class FatigueAnalyzer:
    """
    Analizador longitudinal de fatiga motora.
    """
    def __init__(self, config: FatigueConfig) -> None:
        """
        Inicializa el analizador.

        :param config: Parametros de configuracion.
        :type config: FatigueConfig
        :return: Nada.
        :rtype: None
        """
        self.config = config
        self.model = LinearRegression()

    # CARGAR DATOS CSV
    def load_data(self) -> pd.DataFrame:
        """
        Carga datos desde archivo.

        :return: DataFrame con datos.
        :rtype: pd.DataFrame
        """
        return pd.read_csv(self.config.file_path)

    # CALCULAR REGRESION LINEAL
    def compute_regression(self, y_values: np.ndarray) -> Tuple[float, float]:
        """
        Calcula pendiente de degradacion.

        :param y_values: Valores de la variable.
        :type y_values: np.ndarray
        :return: Pendiente e interseccion.
        :rtype: Tuple[float, float]
        """
        x_values = np.arange(len(y_values)).reshape(-1, 1)
        self.model.fit(x_values, y_values)
        
        slope = float(self.model.coef_[0])
        intercept = float(self.model.intercept_)
        
        return slope, intercept

   # EJECUTAR PIPELINE COMPLETO
    def run_analysis(self) -> Dict[str, float]:
        """
        Ejecuta analisis de fatiga y grafica.

        :return: Diccionario con biomarcadores.
        :rtype: Dict[str, float]
        """
        df = self.load_data()
        y = df[self.config.target_feature].dropna().values
        
        # ESCUDO ANTI-CRASH: MINIMO 2 PUNTOS PARA REGRESION
        if len(y) < 2:
            print("⚠️ FATIGA: Datos insuficientes para calcular degradacion.")
            return {"slope": 0.0, "intercept": 0.0}
            
        slope, intercept = self.compute_regression(y)
        
        # RETORNAR DICCIONARIO BIOMARCADORES
        return {"slope": slope, "intercept": intercept}
