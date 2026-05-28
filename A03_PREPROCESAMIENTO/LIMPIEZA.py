# -*- coding: utf-8 -*-
"""
Módulo de limpieza y preprocesamiento de datos biomecánicos.
Convierte archivos Parquet en HDF5 balanceado naturalmente.
Incluye auditoría de reproducibilidad MLOps y tipado estricto.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import logging
import argparse
import random
import sys
import json
import hashlib
import subprocess
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, DirectoryPath, Field

# CONFIGURAR LOGGING CENTRAL
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PreprocessingError(Exception):
    """Excepción para errores de preprocesamiento."""
    pass

# ============================================================
# AUDITORIA Y REPRODUCIBILIDAD (MLOps)
# ============================================================

class ExperimentAuditor:
    """
    Auditor de reproducibilidad científica y determinismo.
    """
    def __init__(self, output_dir: Path) -> None:
        """
        Inicializa gestor de metadatos.

        :param output_dir: Directorio de exportación.
        :type output_dir: Path
        """
        self.output_dir = output_dir

    @staticmethod
    def enforce_determinism(seed: int = 42) -> None:
        """
        Fija semillas pseudoaleatorias globales.

        :param seed: Valor de anclaje pseudoaleatorio.
        :type seed: int
        :return: Nada.
        :rtype: None
        """
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"DETERMINISMO FORZADO: SEED {seed}")

    def snapshot_experiment(self, config: BaseModel) -> Dict[str, Any]:
        """
        Genera registro inmutable del preprocesamiento.

        :param config: Configuración validada Pydantic.
        :type config: BaseModel
        :return: Diccionario de auditoría.
        :rtype: Dict[str, Any]
        """
        # SERIALIZAR CONFIGURACION
        params_dict = config.model_dump()
        serialized = json.dumps(params_dict, sort_keys=True, default=str)
        
        # GENERAR HASH UNICO
        exp_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        # LEER REPOSITORIO GIT
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            git_commit = "untracked"

        # EMPAQUETAR METADATA
        metadata = {
            "experiment_hash": exp_hash,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "effective_params": params_dict
        }

        # GUARDAR JSON
        self.output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.output_dir / f"prep_metadata_{exp_hash[:8]}.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=4, ensure_ascii=False, default=str)

        logger.info(f"SNAPSHOT GUARDADO: {meta_path.name}")
        return metadata

# ============================================================
# CONFIGURACION PYDANTIC
# ============================================================

class PreprocessConfig(BaseModel):
    """
    Parámetros inmutables de preprocesamiento y ventanas.
    """
    input_path: DirectoryPath
    output_path: Path
    excel_path: Path
    fixed_length: int = Field(default=100, gt=0)
    step_size: int = Field(default=10, gt=0)
    min_records: int = Field(default=10, gt=0)
    random_seed: int = Field(default=42, ge=0)

# ============================================================
# LOGICA DE EMPAQUETADO HDF5
# ============================================================

class GaitDataArchiver:
    """
    Empaquetador de datos en formato HDF5 jerárquico.
    """
    def __init__(self, config: PreprocessConfig) -> None:
        """
        Inicializa motor de almacenamiento.

        :param config: Parametros de empaquetado.
        :type config: PreprocessConfig
        """
        self.config = config
        self.output_file: Path = config.output_path / "dataset_jerarquico.hdf5"
        self.patient_map: Dict[str, str] = {}
        self._load_patient_map()

    def _load_patient_map(self) -> None:
        """
        Lee archivo Excel y mapea identificadores de pacientes.

        :return: Nada. Actualiza diccionario interno.
        :rtype: None
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
            logger.error(f"ERROR LEYENDO EXCEL: {e}")
            raise PreprocessingError("Mapa fallido.") from e

    def run_pipeline(self) -> None:
        """
        Ejecuta flujo de empaquetado y mezcla aleatoria en HDF5.

        :return: Nada. Guarda archivo en disco.
        :rtype: None
        """
        # LISTAR ARCHIVOS PARQUET
        files: List[Path] = list(self.config.input_path.glob("*.parquet"))
        if not files:
            logger.warning("PARQUETS NO ENCONTRADOS")
            return

        # COLECTOR DE DATOS
        storage: List[Tuple[np.ndarray, str, int]] = []

        for f in files:
            chunks: List[Tuple[np.ndarray, str, int]] = self._generate_chunks_metadata(f)
            storage.extend(chunks)
        
        if not storage:
            logger.error("NO HAY DATOS")
            return

        # MEZCLA ALEATORIA DATOS
        random.shuffle(storage)

        # GUARDAR HDF5 DISCO
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        try:
            with h5py.File(self.output_file, "w") as hf:
                for data, path_str, lbl in storage:
                    ds = hf.create_dataset(path_str, data=data, compression="gzip")
                    ds.attrs["label"] = lbl
                        
            logger.info(f"HDF5 CREADO: {len(storage)} muestras")
        except OSError as e:
            logger.error(f"ERROR ESCRITURA DISCO: {e}")
            raise PreprocessingError("Fallo guardado.") from e

    def _generate_chunks_metadata(self, file_path: Path) -> List[Tuple[np.ndarray, str, int]]:
        """
        Aplica ventana deslizante y extrae metadatos.

        :param file_path: Ruta archivo parquet individual.
        :type file_path: Path
        :return: Lista de tuplas con tensores, rutas jerárquicas y etiquetas.
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

            # PADDING PARA SEÑALES CORTAS
            if num_records < self.config.fixed_length:
                data: np.ndarray = self._fix_length(data_values)
                path: str = f"{paciente_id}/{seg_name}_CH_000/Both"
                chunks_list.append((data, path, label))
                return chunks_list

            # VENTANA DESLIZANTE CON OVERLAP
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
            logger.warning(f"OMITIENDO ARCHIVO {file_path.name}: {e}")
            return []

    def _fix_length(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza padding replicando el valor del último estado.

        :param data: Tensor original incompleto.
        :type data: np.ndarray
        :return: Tensor extendido a longitud fija objetivo.
        :rtype: np.ndarray
        """
        target: int = self.config.fixed_length
        current_len = data.shape[0]
        
        if current_len >= target:
            return data[:target, :]
            
        # PADDING VALOR FINAL
        pad_size = target - current_len
        pad_values = np.tile(data[-1, :], (pad_size, 1))
        return np.vstack((data, pad_values))

# ============================================================
# FLUJO PRINCIPAL
# ============================================================

def main() -> None:
    """
    Punto de entrada de ejecución del preprocesamiento.
    """
    parser = argparse.ArgumentParser(description="Pipeline Parquet a HDF5")
    parser.add_argument("--input", type=Path, required=True, help="Ruta de lectura Parquet")
    parser.add_argument("--output", type=Path, required=True, help="Ruta destino HDF5")
    parser.add_argument("--excel", type=Path, required=True, help="Ruta metadatos Excel")
    args = parser.parse_args()

    try:
        # VALIDAR CONFIGURACION
        cfg = PreprocessConfig(
            input_path=args.input,
            output_path=args.output,
            excel_path=args.excel
        )
        
        # FIJAR DETERMINISMO GLOBALES
        ExperimentAuditor.enforce_determinism(cfg.random_seed)

        # INICIAR AUDITORIA Y SNAPSHOT
        auditor = ExperimentAuditor(cfg.output_path)
        auditor.snapshot_experiment(cfg)
        
        # EJECUTAR PIPELINE
        archiver = GaitDataArchiver(cfg)
        archiver.run_pipeline()
        
    except Exception as e:
        logger.critical(f"ERROR IRRECUPERABLE DETECTADO: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    main()
    archiver = GaitDataArchiver(args)
    archiver.run_pipeline()
