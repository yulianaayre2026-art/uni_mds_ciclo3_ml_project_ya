
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import datetime
import os

app = Flask(__name__)

# 1. Configuración de rutas de artefactos
BASE_PATH = '/content/drive/MyDrive/MLOPS trabajo final/models/'
MODEL_PATH = os.path.join(BASE_PATH, 'mejor_modelo_volumen.pkl')
SCALER_PATH = os.path.join(BASE_PATH, 'scaler_blueberry.pkl')
MAPPINGS_PATH = os.path.join(BASE_PATH, 'mappings.pkl')

# 2. Carga de modelos y diccionarios al inicio
print("Cargando artefactos del modelo...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    mappings = joblib.load(MAPPINGS_PATH)
    print("✅ Artefactos cargados correctamente.")
except Exception as e:
    print(f"❌ Error al cargar artefactos: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recepción de datos JSON
        data = request.get_json()
        
        # --- TRANSFORMACIONES EN TIEMPO REAL ---
        
        # A. Procesar Fecha (FECHA int -> ANIO, MES, SEMANA)
        fecha_dt = pd.to_datetime(str(data['FECHA']), format='%Y%m%d')
        anio = fecha_dt.year
        mes = fecha_dt.month
        semana = int(fecha_dt.isocalendar()[1])

        # B. Procesar Orgánico (Lógica de texto)
        tipo_str = str(data.get('TIPO DE PRODUCTO (ORGANICO CONVENCIONAL)', ''))
        es_organico = 1 if 'ORGANICO' in tipo_str.upper() else 0

        # C. Mapear Categorías usando mappings.pkl
        # Si el valor enviado no existe, se asigna -1 (valor por defecto)
        adua_code = mappings['ADUA_DESC'].get(data['ADUA_DESC'], -1)
        pais_code = mappings['PAIS_DESC'].get(data['PAIS_DESC'], -1)
        var_code = mappings['VARIEDAD'].get(data['VARIEDAD'], -1)

        # 3. Construir el DataFrame de entrada (orden exacto del entrenamiento)
        input_data = pd.DataFrame([[
            anio, mes, semana, 
            adua_code, pais_code, var_code, 
            es_organico, 
            data['FOB_DOLPOL_KG']
        ]], columns=['ANIO', 'MES', 'SEMANA', 'ADUA_DESC_CODE', 'PAIS_DESC_CODE', 'VARIEDAD_CODE', 'ES_ORGANICO', 'FOB_DOLPOL_KG'])

        # 4. Escalar e Inferir
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        # 5. Respuesta
        return jsonify({
            'status': 'success',
            'prediccion': {
                'PESO_NETO_KG': round(float(prediction[0]), 2)
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Lanzar en puerto 5000
    app.run(host='0.0.0.0', port=5000)
