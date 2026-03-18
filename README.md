# EVALUACIÓN DE MODELO DE MARCHA (FFT)

Este repositorio contiene los pesos y el código para ejecutar la inferencia del modelo basado en el dominio de la frecuencia (FFT) para la detección de marcha.

## REQUISITOS
Archivos `modelo_frecuencia.pth` y `scaler_gait.joblib` en el mismo directorio que el script.

`pip install torch numpy joblib scipy pydantic h5py scikit-learn`

## SCRIPT DE EVALUACIÓN
Código para cargar el modelo preentrenado, aplicar el preprocesamiento y generar métricas de evaluación (Accuracy, Matriz de Confusión) sobre nuevos datos etiquetados en formato HDF5.

```python
import torch
import torch.nn as nn
import numpy as np
import joblib
import h5py
from scipy.fft import rfft
from pydantic import BaseModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# CONFIGURAR RUTAS LOCALES
class ModelConfig(BaseModel):
    ruta_modelo: str = "modelo_frecuencia.pth"
    ruta_scaler: str = "scaler_gait.joblib"
    input_dim: int = 51 * 290

# DEFINIR MODELO FFT
class FFTModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

# CLASE DE INFERENCIA
class GaitPredictor:
    """Clase para evaluar modelo FFT."""
    
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(cfg.ruta_scaler)
        self.model = FFTModel(cfg.input_dim).to(self.device)
        self.model.load_state_dict(torch.load(cfg.ruta_modelo, map_location=self.device))
        self.model.eval()

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """
        Genera predicciones sobre datos.
        
        :param x_raw: Array numpy (N, 100, 290).
        :return: Array predicciones (0 o 1).
        """
        # ESCALAR DATOS
        n, t, d = x_raw.shape
        x_flat = x_raw.reshape(-1, d)
        x_scaled = self.scaler.transform(x_flat).reshape(n, t, d)
        
        # APLICAR TRANSFORMADA
        x_f = (np.abs(rfft(x_scaled, axis=1)) / t).astype(np.float32)
        x_tensor = torch.from_numpy(x_f).to(self.device)

        # PREDECIR CLASES
        with torch.no_grad():
            logits = self.model(x_tensor)
            return logits.argmax(dim=1).cpu().numpy()

# EJECUTAR SCRIPT
if __name__ == "__main__":
    # INICIALIZAR CONFIGURACION
    cfg = ModelConfig()
    predictor = GaitPredictor(cfg)
    
    # CARGAR DATOS HDF5
    ruta_h5 = "dataset_jerarquico.hdf5"
    x_list, y_list = [], []
    
    with h5py.File(ruta_h5, "r") as hf:
        for p in hf.keys():
            for s in hf[p].keys():
                for pie in hf[p][s].keys():
                    ds = hf[p][s][pie]
                    x_list.append(ds[:])
                    y_list.append(ds.attrs["label"])
                    
    # CONVERTIR A NUMPY
    X_test = np.array(x_list).astype(np.float32)
    y_true = np.array(y_list).astype(np.int64)
    
    # GENERAR PREDICCIONES
    y_pred = predictor.predict(X_test)
    
    # IMPRIMIR METRICAS FINALES
    print("\n# RESULTADOS DE EVALUACION")
    print("-" * 30)
    print(f"ACCURACY TOTAL: {accuracy_score(y_true, y_pred):.4f}\n")
    print("# MATRIZ DE CONFUSION")
    print(confusion_matrix(y_true, y_pred))
    print("\n# REPORTE DETALLADO")
    print(classification_report(y_true, y_pred, target_names=["REPOSO", "MARCHA"]))
