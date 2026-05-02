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
import logging
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict
from pydantic import BaseModel, DirectoryPath, Field
from scipy import signal

# CONFIGURAR LOGGING CENTRAL
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# EXCEPCIONES PERSONALIZADAS
class PreprocessingError(Exception):
    """Excepción para errores de preprocesamiento."""
    pass

# CONFIGURACION PYDANTIC
class PreprocessConfig(BaseModel):
    """
    Configuración de rutas y parámetros para el preprocesamiento de la señal.
    """
    input_path: DirectoryPath
    output_path: Path
    excel_path: Path
    fixed_length: int = Field(default=100, gt=0)
    step_size: int = Field(default=75, gt=0)
    min_records: int = Field(default=10, gt=0)
    random_seed: int = Field(default=42, ge=0)

class GaitDataArchiver:
    """
    Clase para generar un archivo HDF5 balanceado con resampling.
    Corrección de Data Leakage (Patient ID) y balanceo aleatorio.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """
        Inicializa el archivador con la configuración y genera el mapa.

        :param config: Objeto de configuración de preprocesamiento.
        :type config: PreprocessConfig
        """
        self.config = config
        self.output_file: Path = config.output_path / "dataset_jerarquico.hdf5"
        self.patient_map: Dict[str, str] = {}
        self._load_patient_map()

    def _load_patient_map(self) -> None:
        """
        Lee archivo Excel y mapea identificadores reales.
        """
        try:
            df = pd.read_excel(self.config.excel_path, sheet_name=0)
            df.columns = [str(col).lower() for col in df.columns]
            self.patient_map = {
                f"segment_{i:03d}": str(ref).strip() 
                for i, ref in enumerate(df["reference"])
            }
            logger.info("MAPA PACIENTES CARGADO")
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"ERROR CARGANDO EXCEL: {e}")
            raise PreprocessingError("Fallo al cargar mapa de pacientes.") from e

    def run_pipeline(self) -> None:
        """
        Ejecuta flujo de procesamiento, balanceo aleatorio y guardado.
        """
        # LISTAR ARCHIVOS PARQUET
        files: List[Path] = list(self.config.input_path.glob("*.parquet"))
        if not files:
            logger.warning("NO HAY ARCHIVOS PARQUET")
            return

        # CREAR COLECTORES TIPADOS
        storage: Dict[int, List[Tuple[np.ndarray, str, int]]] = {0: [], 1: []}

        for f in files:
            label: int = 1 if "-1.parquet" in f.name else 0
            chunks: List[Tuple[np.ndarray, str, int]] = self._generate_chunks_metadata(f)
            storage[label].extend(chunks)

        limit: int = min(len(storage[0]), len(storage[1]))
        
        if limit == 0:
            logger.error("SIN DATOS PARA BALANCEAR")
            return

        # APLICAR BALANCEO ALEATORIO
        random.seed(self.config.random_seed)
        random.shuffle(storage[0])
        random.shuffle(storage[1])

        # GUARDAR ARCHIVO HDF5
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        try:
            with h5py.File(self.output_file, "w") as hf:
                logger.info(f"LIMITE BALANCEO: {limit}")
                
                for label in [0, 1]:
                    for data, path_str, lbl in storage[label][:limit]:
                        ds = hf.create_dataset(path_str, data=data, compression="gzip")
                        ds.attrs["label"] = lbl
                        
            logger.info(f"ARCHIVO FINAL CREADO: {self.output_file.name}")
        except OSError as e:
            logger.error(f"ERROR GUARDANDO HDF5: {e}")
            raise PreprocessingError("Fallo escritura de disco.") from e

    def _generate_chunks_metadata(self, file_path: Path) -> List[Tuple[np.ndarray, str, int]]:
        """
        Genera metadatos y divide señal preservando identidad.

        :param file_path: Ruta del archivo parquet a procesar.
        :type file_path: Path
        :return: Lista de tensores, rutas internas y etiquetas.
        :rtype: List[Tuple[np.ndarray, str, int]]
        """
        try:
            df = pd.read_parquet(file_path).T.dropna()
            data_values: np.ndarray = df.values
            num_records: int = len(data_values)
            
            parts: List[str] = file_path.stem.split("_")
            seg_name: str = f"{parts[0]}_{parts[1]}"
            foot_info: List[str] = parts[2].split("-")
            label: int = int(foot_info[1])
            
            paciente_id: str = self.patient_map.get(seg_name, "PACIENTE_DESCONOCIDO")
            chunks_list: List[Tuple[np.ndarray, str, int]] = []

            # APLICAR INTERPOLACION CORTA
            if num_records < self.config.fixed_length:
                data: np.ndarray = self._fix_length(data_values)
                path: str = f"{paciente_id}/{seg_name}_CH_000/Both"
                chunks_list.append((data, path, label))
                return chunks_list

            # APLICAR VENTANA DESLIZANTE
            target: int = self.config.fixed_length
            step: int = self.config.step_size
            idx: int = 0

            for start in range(0, num_records - target + 1, step):
                chunk_data: np.ndarray = data_values[start : start + target, :]
                path = f"{paciente_id}/{seg_name}_CH_{idx:03d}/Both"
                chunks_list.append((chunk_data, path, label))
                idx += 1

            return chunks_list

        except (OSError, pd.errors.EmptyDataError, ValueError) as e:
            logger.warning(f"ARCHIVO IGNORADO {file_path.name}: {e}")
            return []

    def _fix_length(self, data: np.ndarray) -> np.ndarray:
        """
        Ajusta longitud del tensor mediante interpolación.

        :param data: Tensor bidimensional original.
        :type data: np.ndarray
        :return: Tensor ajustado.
        :rtype: np.ndarray
        """
        target: int = self.config.fixed_length
        return signal.resample(data, target, axis=0)

def main() -> None:
    """
    Punto de entrada con CLI.
    """
    parser = argparse.ArgumentParser(description="Preprocesamiento Parquet a HDF5")
    parser.add_argument("--input", type=Path, required=True, help="Carpeta Parquets")
    parser.add_argument("--output", type=Path, required=True, help="Carpeta Salida")
    parser.add_argument("--excel", type=Path, required=True, help="Excel Solicitud")
    args = parser.parse_args()

    # INICIAR PIPELINE PRINCIPAL
    cfg = PreprocessConfig(
        input_path=args.input,
        output_path=args.output,
        excel_path=args.excel
    )
    
    archiver = GaitDataArchiver(cfg)
    archiver.run_pipeline()

if __name__ == "__main__":
    main()
    archiver = GaitDataArchiver(cfg)
    archiver.run_pipeline()
