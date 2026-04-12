# -*- coding: utf-8 -*-
"""
Módulo de limpieza y preprocesamiento de datos biomecánicos.
Convierte archivos Parquet en un archivo HDF5 balanceado, corrigiendo 
la fuga de datos mediante el mapeo de identidades (Patient ID) y aplicando
remuestreo (resampling) y balanceo aleatorio.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, DirectoryPath
from scipy import signal
import random 

# CONFIGURACION
class PreprocessConfig(BaseModel):
    """
    Configuración de rutas y parámetros para el preprocesamiento de la señal.
    """
    input_path: DirectoryPath
    output_path: Path
    excel_path: Path 
    fixed_length: int = 100
    step_size: int = 75  
    min_records: int = 10

class GaitDataArchiver:
    """
    Clase para generar un archivo HDF5 balanceado con resampling.
    Corrección de Data Leakage (Patient ID) y balanceo aleatorio.
    """

    def __init__(self, config: PreprocessConfig):
        """
        Inicializa el archivador con la configuración y genera el mapa de pacientes.

        :param config: Objeto de configuración con las rutas y tamaños de ventana.
        :type config: PreprocessConfig
        """
        self.config = config
        self.output_file = config.output_path / "dataset_jerarquico.hdf5"
        self._load_patient_map()

    def _load_patient_map(self):
        """
        Lee el archivo Excel de solicitudes y mapea los identificadores 
        de segmento al ID real del paciente para evitar fuga de datos.
        """
        df = pd.read_excel(self.config.excel_path, sheet_name=0)
        df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)
        # Asegurar que los IDs son strings limpios
        self.patient_map = {f"segment_{i:03d}": str(ref).strip() for i, ref in enumerate(df["reference"])}

    def run_pipeline(self) -> None:
        """
        Ejecuta el flujo completo de procesamiento: lectura de archivos,
        extracción de ventanas, balanceo aleatorio de clases y escritura en HDF5.
        """
        # 1 LISTAR ARCHIVOS
        files = list(self.config.input_path.glob("*.parquet"))
        
        # 2 COLECTORES TEMPORALES
        storage = {0: [], 1: []}

        for f in files:
            label = 1 if "-1.parquet" in f.name else 0
            chunks = self._generate_chunks_metadata(f)
            storage[label].extend(chunks)

        limit = min(len(storage[0]), len(storage[1]))
        
        if limit == 0:
            print("# ERROR: NO HAY DATOS SUFICIENTES PARA BALANCEAR")
            return

        # 3 BALANCEO ALEATORIO (PUNTO 10 DEL PROFESOR)
        # En lugar de recortar secuencialmente, mezclamos aleatoriamente antes de recortar
        random.seed(42) # Semilla para reproducibilidad
        random.shuffle(storage[0])
        random.shuffle(storage[1])

        # 4 GUARDAR EN HDF5 
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output_file, "w") as hf:
            print(f"# LIMITE CALCULADO PARA BALANCEO: {limit} muestras por clase")
            
            for label in [0, 1]:
                # Ahora coge los primeros 'limit', pero como ya están mezclados, es una muestra representativa
                for data, path_str, lbl in storage[label][:limit]:
                    ds = hf.create_dataset(path_str, data=data, compression="gzip")
                    ds.attrs["label"] = lbl
        
        print(f"# ARCHIVO BALANCEADO Y PROTEGIDO: {self.output_file}")
        print(f"# TOTAL VENTANAS: {limit * 2}")

    def _generate_chunks_metadata(self, file_path: Path) -> List[Tuple[np.ndarray, str, int]]:
        """
        Genera metadatos y divide la señal en ventanas para un archivo especifico,
        preservando la identidad real del paciente.

        :param file_path: Ruta del archivo parquet a procesar.
        :type file_path: Path
        :return: Lista de tuplas que contienen el tensor de datos, la ruta interna HDF5 y la etiqueta.
        :rtype: List[Tuple[np.ndarray, str, int]]
        """
        try:
            df = pd.read_parquet(file_path).T.dropna()
            data_values = df.values
            num_records = len(data_values)
            
            # Extraer información del nombre del archivo (ej. segment_000_tensor-1.parquet)
            parts = file_path.stem.split("_")
            seg_name = f"{parts[0]}_{parts[1]}" # "segment_000"
            foot_info = parts[2].split("-") # ["tensor", "1"]
            label = int(foot_info[1])
            
            # RECUPERAR IDENTIDAD REAL DEL PACIENTE (PUNTO 3 DEL PROFESOR)
            paciente_id = self.patient_map.get(seg_name, "PACIENTE_DESCONOCIDO")

            chunks_list = []

            # SEÑAL CORTA: RESAMPLING (INTERPOLACION)
            if num_records < self.config.fixed_length:
                data = self._fix_length(data_values)
                # SE GUARDA BAJO EL NOMBRE DEL PACIENTE REAL
                path = f"{paciente_id}/{seg_name}_CH_000/Both"
                chunks_list.append((data, path, label))
                return chunks_list

            # VENTANA DESLIZANTE
            target = self.config.fixed_length
            step = self.config.step_size
            idx = 0

            for start in range(0, num_records - target + 1, step):
                chunk_data = data_values[start : start + target, :]
                # SE GUARDA BAJO EL NOMBRE DEL PACIENTE REAL
                path = f"{paciente_id}/{seg_name}_CH_{idx:03d}/Both"
                chunks_list.append((chunk_data, path, label))
                idx += 1

            return chunks_list

        except Exception as e:
            print(f"# ERROR PROCESANDO {file_path.name}: {e}")
            return []

    def _fix_length(self, data: np.ndarray) -> np.ndarray:
        """
        Ajusta la longitud del tensor mediante interpolación (resampling)
        utilizando la transformada de Fourier.

        :param data: Tensor bidimensional con la señal original.
        :type data: np.ndarray
        :return: Tensor bidimensional ajustado a la longitud configurada.
        :rtype: np.ndarray
        """
        target = self.config.fixed_length
        return signal.resample(data, target, axis=0)

# EJECUTAR
if __name__ == "__main__":
    # RUTAS (Puedes ajustarlas a tu entorno)
    INPUT = r"C:\Users\jairi\OneDrive\Escritorio\TFM\01_EXTRACCION DE DATOS\resultados"
    OUTPUT = Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\DATASET_LISTONUEVO")
    EXCEL_REF = Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\01_EXTRACCION DE DATOS\solicitud.xlsx")

    cfg = PreprocessConfig(input_path=INPUT, output_path=OUTPUT, excel_path=EXCEL_REF)
    archiver = GaitDataArchiver(cfg)
    archiver.run_pipeline()