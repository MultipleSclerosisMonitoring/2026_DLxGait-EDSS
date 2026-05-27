# -*- coding: utf-8 -*-
"""
Análisis de fatiga motora mediante regresión lineal.
Calcula la pendiente de degradación (biomarcador) utilizando datos sintéticos.
"""

# IMPORTAR LIBRERIAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        
        slope, intercept = self.compute_regression(y)

        # GRAFICAR RESULTADOS BASICOS
        plt.figure()
        plt.scatter(np.arange(len(y)), y)
        plt.plot(np.arange(len(y)), intercept + slope * np.arange(len(y)))
        plt.title(f"Fatiga: {self.config.target_feature}")
        plt.xlabel("Pasos")
        plt.ylabel(self.config.target_feature)
        plt.show()

        return {"slope": slope, "intercept": intercept}

# EJECUTAR SCRIPT PRINCIPAL
def main() -> None:
    """
    Punto de entrada principal.
    """
    # CREAR DATOS EJEMPLO
    np.random.seed(42)
    sample_data = pd.DataFrame({
        "Stride_Length_m": np.linspace(1.2, 0.9, 100) + np.random.normal(0, 0.05, 100),
        "Gait_Speed_ms": np.linspace(1.1, 0.8, 100) + np.random.normal(0, 0.05, 100),
        "MTC_m": np.linspace(0.05, 0.02, 100) + np.random.normal(0, 0.005, 100)
    })
    sample_data.to_csv("resultados_ejemplo.csv", index=False)

    # CONFIGURAR PARAMETROS
    config = FatigueConfig(
        file_path="resultados_ejemplo.csv", 
        target_feature="Gait_Speed_ms"
    )
    
    analyzer = FatigueAnalyzer(config)
    results = analyzer.run_analysis()

    print("BIOMARCADORES DE FATIGA")
    print(f"Variable: {config.target_feature}")
    print(f"Pendiente (m): {results['slope']:.6f}")

if __name__ == "__main__":
    main()
