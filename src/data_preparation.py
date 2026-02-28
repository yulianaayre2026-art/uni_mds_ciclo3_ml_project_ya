
import pandas as pd
import numpy as np
import os

def ejecutar_preparacion():
    # Ruta de origen (Raw)
    ruta_raw = "/content/drive/MyDrive/MLOPS trabajo final/data/raw/sunat_2022_2025_raw.csv"
    
    # 1. Cargar y Procesar columnas (Usando la lógica de df_final)
    df_final = pd.read_csv(ruta_raw)
    df_proc = df_final.copy()

    # --- TIEMPO LINEAL (FECHA int64 -> componentes) ---
    df_proc['FECHA_DT'] = pd.to_datetime(df_proc['FECHA'].astype(str), format='%Y%m%d')
    df_proc['ANIO'] = df_proc['FECHA_DT'].dt.year
    df_proc['MES'] = df_proc['FECHA_DT'].dt.month
    df_proc['SEMANA'] = df_proc['FECHA_DT'].dt.isocalendar().week.astype(int)

    # --- MEJORA: Transformación Binaria para Orgánico ---
    col_tipo = 'TIPO DE PRODUCTO (ORGANICO CONVENCIONAL)'
    df_proc['ES_ORGANICO'] = df_proc[col_tipo].str.contains('ORGANICO', case=False, na=False).astype(int)

    # --- MEJORA: Label Encoding para categorías ---
    cat_cols = ['ADUA_DESC', 'PAIS_DESC', 'VARIEDAD']
    for col in cat_cols:
        df_proc[f'{col}_CODE'] = df_proc[col].astype('category').cat.codes

    # 2. Selección de variables finales
    features = [
        'ANIO', 'MES', 'SEMANA',
        'ADUA_DESC_CODE',
        'PAIS_DESC_CODE',
        'VARIEDAD_CODE',
        'ES_ORGANICO',     
        'FOB_DOLPOL_KG'
    ]

    # 3. Crear dataset consolidado
    df_transformado = df_proc[features + ['PESO_NETO']].copy()

    # Guardar en la carpeta PROCESSED como definiste
    ruta_transformada = "/content/drive/MyDrive/MLOPS trabajo final/data/processed/dataset_transformado.csv"
    
    # Asegurar que la carpeta processed existe antes de guardar
    os.makedirs(os.path.dirname(ruta_transformada), exist_ok=True)
    
    df_transformado.to_csv(ruta_transformada, index=False)
    
    print(f"✅ Módulo src/data_preparation.py: Dataset guardado en {ruta_transformada}")
    
    # 4. Retornar X e y para consistencia (opcional si se llama desde el script)
    X = df_transformado[features]
    y = df_transformado['PESO_NETO']
    return X, y

if __name__ == "__main__":
    ejecutar_preparacion()
