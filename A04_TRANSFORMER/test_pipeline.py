# -*- coding: utf-8 -*-

"""
Suite de Pruebas Unitarias (Unit Tests) para el TFM.
"""

# IMPORTAR DEPENDENCIAS
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# IMPORTAR MODULOS LOCALES
from A03_PREPROCESAMIENTO.LIMPIEZA import GaitDataArchiver, PreprocessConfig
from A04_TRANSFORMER.AA_TRANSFORMER_V1 import FFTProcessor

# CONFIGURAR FIXTURES
@pytest.fixture
def config_prueba(tmp_path):
    input_dir = tmp_path / "dummy_in"
    input_dir.mkdir()
    excel_file = tmp_path / "dummy.xlsx"
    excel_file.touch()
    
    return PreprocessConfig(
        input_path=input_dir,
        output_path=tmp_path / "dummy_out",
        excel_path=excel_file,
        fixed_length=100,
        step_size=25
    )

# PROBAR REMUESTREO
@patch("A03_PREPROCESAMIENTO.LIMPIEZA.pd.read_excel")
def test_fix_length_resampling(mock_read_excel, config_prueba):
    mock_read_excel.return_value = MagicMock()
    archiver = GaitDataArchiver(config_prueba)
    datos_cortos = np.random.rand(85, 290)
    
    datos_ajustados = archiver._fix_length(datos_cortos)
    
    assert datos_ajustados.shape == (100, 290)
    assert not np.isnan(datos_ajustados).any()

# PROBAR EXTRACCION FFT
def test_fft_features_extraction():
    batch_datos = np.random.rand(32, 100, 290).astype(np.float32)
    
    datos_fft = FFTProcessor.get_fft_features(batch_datos)
    
    assert datos_fft.shape == (32, 51, 290)
    assert datos_fft.dtype == np.float32
    assert np.all(datos_fft >= 0)

# EJECUTAR TESTS
if __name__ == "__main__":
    pytest.main(["-v", __file__])
