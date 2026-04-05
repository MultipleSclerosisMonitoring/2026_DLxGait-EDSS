import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, DirectoryPath
from scipy import signal # AGREGADO PARA INTERPOLACION

# CONFIGURACION
class PreprocessConfig(BaseModel):
    input_path: DirectoryPath
    output_path: Path
    fixed_length: int = 100
    step_size: int = 75  # SALTO PARA SOLAPAMIENTO DEL 25%
    min_records: int = 10

class GaitDataArchiver:
    """
    Clase para generar un archivo HDF5 balanceado con resampling.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.output_file = config.output_path / "dataset_jerarquico.hdf5"

    def run_pipeline(self) -> None:
        """
        Ejecuta el flujo completo de procesamiento.
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
            print("# ERROR NO DATOS")
            return

        # 3 GUARDAR EN HDF5 
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output_file, "w") as hf:
            print(f"# LIMITE {limit}")
            
            for label in [0, 1]:
                # BALANCEO DE CLASES
                for data, path_str, lbl in storage[label][:limit]:
                    ds = hf.create_dataset(path_str, data=data, compression="gzip")
                    ds.attrs["label"] = lbl
        
        print(f"# ARCHIVO BALANCEADO {self.output_file}")
        print(f"# TOTAL {limit * 2}")

    def _generate_chunks_metadata(self, file_path: Path) -> List[Tuple[np.ndarray, str, int]]:
        """
        Genera metadatos y ventanas para un archivo especifico.

        :param file_path: Ruta del archivo parquet.
        :return: Lista de tuplas (data, path, label).
        """
        try:
            df = pd.read_parquet(file_path).T.dropna()
            data_values = df.values
            num_records = len(data_values)
            
            parts = file_path.stem.split("_")
            seg_id = parts[1]
            foot_info = parts[2].split("-")
            pie = foot_info[0]
            label = int(foot_info[1])

            chunks_list = []

            # SEÑAL CORTA: RESAMPLING (INTERPOLACION)
            if num_records < self.config.fixed_length:
                data = self._fix_length(data_values)
                path = f"REF_PACIENTE/SEG_{seg_id}_CH_000/{pie}"
                chunks_list.append((data, path, label))
                return chunks_list

            # VENTANA DESLIZANTE
            target = self.config.fixed_length
            step = self.config.step_size
            idx = 0

            for start in range(0, num_records - target + 1, step):
                chunk_data = data_values[start : start + target, :]
                path = f"REF_PACIENTE/SEG_{seg_id}_CH_{idx:03d}/{pie}"
                chunks_list.append((chunk_data, path, label))
                idx += 1

            return chunks_list

        except Exception as e:
            print(f"# ERROR PROCESANDO {file_path.name}: {e}")
            return []

    def _fix_length(self, data: np.ndarray) -> np.ndarray:
        """
        Ajusta la longitud mediante interpolacion (resampling).

        :param data: Array original.
        :return: Array interpolado a longitud fija.
        """
        target = self.config.fixed_length
        # INTERPOLACION MEDIANTE FOURIER (SCIPY)
        return signal.resample(data, target, axis=0)

# EJECUTAR
if __name__ == "__main__":
    INPUT = r"C:\Users\jairi\OneDrive\Escritorio\TFM\01_EXTRACCION DE DATOS\resultadosprueba"
    OUTPUT = Path(r"C:\Users\jairi\OneDrive\Escritorio\TFM\DATASET_LISTO2")

    cfg = PreprocessConfig(input_path=INPUT, output_path=OUTPUT)
    archiver = GaitDataArchiver(cfg)
    archiver.run_pipeline()