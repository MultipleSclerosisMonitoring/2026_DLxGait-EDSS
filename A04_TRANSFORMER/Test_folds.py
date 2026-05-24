# -*- coding: utf-8 -*-
"""
test de folds para analizar resultados 
"""

# -*- coding: utf-8 -*-

from pathlib import Path
import h5py
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

DATASET_PATH = Path(
    r"C:\Users\jairi\OneDrive\Escritorio\TFM\DATASET_LISTO\dataset_jerarquico.hdf5"
)

x_list = []
groups = []
labels = []

# CARGAR DATOS
with h5py.File(DATASET_PATH, "r") as hf:

    for patient in hf.keys():

        for seg_chunk in hf[patient].keys():

            for pie in hf[patient][seg_chunk].keys():

                dataset = hf[patient][seg_chunk][pie]

                x_list.append(dataset[:])

                labels.append(dataset.attrs["label"])

                groups.append(
                    f"{patient}_{seg_chunk.split('_CH_')[0]}"
                )

x_all = np.array(x_list)
y_all = np.array(labels)
groups_all = np.array(groups)

# REPRODUCIR KFOLD EXACTAMENTE
sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=13
)

print("\n========= ANALISIS FOLDS =========\n")

for fold, (train_idx, val_idx) in enumerate(
    sgkf.split(x_all, y_all, groups_all),
    1
):

    val_groups = np.unique(groups_all[val_idx])

    print(f"\nFOLD {fold}")
    print(f"NUM PACIENTES VALIDACION: {len(val_groups)}")

    for g in val_groups:
        print(g)