# -*- coding: utf-8 -*-

"""
Suite de Pruebas Unitarias (Unit Tests) para el TFM.
Verifica la integridad geométrica de los tensores y las transformaciones
matemáticas clave del preprocesamiento y la extracción de características.
"""

import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# IMPORTAR MODULOS TFM
from A03_PREPROCESAMIENTO.LIMPIEZA import (
    GaitDataArchiver,
    PreprocessConfig
)

# INICIO DE PRUEBAS

@pytest.fixture
def config_prueba(tmp_path):
    """Fixture que provee una configuración simulada para las pruebas."""
    # Creamos una carpeta real temporal para que Pydantic la valide
    input_dir = tmp_path / "dummy_in"
    input_dir.mkdir()
    
    return PreprocessConfig(
        input_path=input_dir,
        output_path=tmp_path / "dummy_out",
        excel_path=tmp_path / "dummy.xlsx",
        fixed_length=100,
        step_size=25
    )

@patch("A03_PREPROCESAMIENTO.LIMPIEZA.pd.read_excel")
def test_fix_length_resampling(mock_read_excel, config_prueba):
    """
    Prueba Unitaria 1: Verifica que el remuestreo (resampling) ajusta
    correctamente un tensor corto a la longitud fija requerida (100 puntos).
    """
    mock_read_excel.return_value = MagicMock()
    archiver = GaitDataArchiver(config_prueba)
    
    # GENERAR DATOS SIMULADOS
    datos_cortos = np.random.rand(85, 290)
    datos_ajustados = archiver._fix_length(datos_cortos)
    
    # VALIDAR DIMENSIONES
    assert datos_ajustados.shape == (100, 290), f"ERROR: Esperaba (100, 290), obtuvo {datos_ajustados.shape}"
    assert not np.isnan(datos_ajustados).any(), "ERROR: El resampling introdujo valores nulos (NaN)."

def test_fft_features_extraction():
    """
    Prueba Unitaria 2: Verifica que la Transformada de Fourier Rápida (rfft)
    devuelve las dimensiones de frecuencias correctas y normalizadas.
    """
    # GENERAR BATCH SIMULADO
    batch_datos = np.random.rand(32, 100, 290)
    
    # EJECUTAR EXTRACCION FFT
    datos_fft = MotorInferenciaMixta._get_fft(None, batch_datos)
    
    # VALIDAR TEOREMA NYQUIST
    assert datos_fft.shape == (32, 51, 290), f"ERROR: Dimensión FFT incorrecta. Esperaba (32, 51, 290), obtuvo {datos_fft.shape}"
    assert datos_fft.dtype == np.float32, "ERROR: El tipo de dato debe ser float32."
    assert np.all(datos_fft >= 0), "ERROR: Las magnitudes del espectro no pueden ser negativas."

if __name__ == "__main__":
    # EJECUTAR CON SPYDER
    pytest.main(["-v", __file__])