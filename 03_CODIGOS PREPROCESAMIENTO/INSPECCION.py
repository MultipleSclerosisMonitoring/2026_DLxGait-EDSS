# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 2026
@author: jairi
"""

import h5py
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from pydantic import BaseModel, FilePath, DirectoryPath
from typing import Dict, Any

# CONFIG

class VisualizerConfig(BaseModel):
    """CONFIGURACION DE RUTAS PARA ANALISIS ESTADISTICO"""
    h5_file: FilePath
    parquet_dir: DirectoryPath

class GaitSummarizer:
    """GENERA RESUMENES COMPACTOS DEL DATASET PARA EL TRANSFORMER"""

    def __init__(self, config: VisualizerConfig):
        """
        INICIALIZA EL RESUMIDOR
        
        :param config: CONFIGURACION DE RUTAS
        """
        self.config = config

    def print_dataset_summary(self) -> None:
        """ANALIZA EL HDF5 Y MUESTRA SOLO CIFRAS CLAVE"""
        total_chunks = 0
        labels = []
        shapes = []
        patients = set()

        print(f"\n# ANALIZANDO: {self.config.h5_file.name}")
        
        with h5py.File(self.config.h5_file, "r") as hf:
            # RECORRER JERARQUIA SIN IMPRIMIR TODO
            for patient_group in hf.keys():
                patients.add(patient_group)
                for chunk_name in hf[patient_group].keys():
                    group = hf[patient_group][chunk_name]
                    # BUSCAR DATASETS DENTRO DEL CHUNK (Left/Right)
                    for side in group.keys():
                        ds = group[side]
                        total_chunks += 1
                        labels.append(ds.attrs.get("label"))
                        shapes.append(ds.shape)

        # PROCESAR ESTADISTICAS
        label_counts = Counter(labels)
        unique_shapes = set(shapes)

        # RESUMEN
        print("-" * 40)
        print(f"  PACIENTES TOTALES : {len(patients)}")
        print(f"  EJEMPLOS (CHUNKS) : {total_chunks}")
        print("-" * 40)
        print(f"  CLASE 1 (MARCHA)  : {label_counts.get(1, 0)}")
        print(f"  CLASE 0 (NO MARCHA): {label_counts.get(0, 0)}")
        print("-" * 40)
        print(f"  FORMA DEL TENSOR  : {unique_shapes}")
        print("-" * 40)

    def discover_features(self) -> None:
        """RESUMEN TECNICO DE LOS 290 SENSORES"""
        try:
            sample_file = next(self.config.parquet_dir.glob("*.parquet"))
            df = pd.read_parquet(sample_file)
            nombres = df.index.astype(str).tolist()
            
            prefijos = [n.split('-')[0] if '-' in n else n.split('_')[0] for n in nombres]
            conteo = Counter(prefijos)

            print("\n# COMPOSICION DEL VECTOR DE ENTRADA (290):")
            for tipo, cantidad in conteo.items():
                print(f"  - {tipo}: {cantidad} canales")
                
        except Exception as e:
            print(f"# ERROR SENSORES: {e}")

# EJECUCION 

if __name__ == "__main__":
    H5_PATH = Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\CODIGOS PREPROCESAMIENTO\DATASET_LISTO\dataset_jerarquico.hdf5")
    PARQUET_PATH = Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\CODIGOS EXTRACCION\DATOS_PARQUET_CRUDOS\PARQUETS")

    cfg = VisualizerConfig(h5_file=H5_PATH, parquet_dir=PARQUET_PATH)
    resumen = GaitSummarizer(cfg)

    resumen.print_dataset_summary()
    resumen.discover_features()