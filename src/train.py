
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

def entrenar_modelo():
    # Configuración de rutas (Basadas en tu estructura real de carpetas)
    PATH_TRAINING = '/content/drive/MyDrive/MLOPS trabajo final/data/training'
    PATH_MODELS = '/content/drive/MyDrive/MLOPS trabajo final/models'
    
    print("--- Iniciando Proceso de Entrenamiento Modular ---")

    # 1. Cargar el Final Training Dataset (Archivos ya verificados en tu Drive)
    try:
        X_train = pd.read_csv(f"{PATH_TRAINING}/X_train_scaled.csv")
        y_train = pd.read_csv(f"{PATH_TRAINING}/y_train.csv").values.ravel()
        print(f"✅ Datos cargados: {X_train.shape[0]} muestras para entrenamiento.")
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontraron los archivos en {PATH_TRAINING}. Verifique la ruta.")
        return

    # 2. Implementar la lógica de entrenamiento (Champion Model)
    # Usamos Random Forest con parámetros para evitar sobreajuste (Regularización)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Entrenando Random Forest Regressor...")
    model.fit(X_train, y_train)

    # 3. Serialización del modelo (Model Serialization)
    # Requisito MLOps: Guardar el modelo para uso futuro
    os.makedirs(PATH_MODELS, exist_ok=True)
    ruta_modelo_final = f"{PATH_MODELS}/mejor_modelo_volumen.pkl"
    
    joblib.dump(model, ruta_modelo_final)
    
    print(f"✅ ÉXITO: Modelo serializado y guardado en: {ruta_modelo_final}")

if __name__ == "__main__":
    entrenar_modelo()
