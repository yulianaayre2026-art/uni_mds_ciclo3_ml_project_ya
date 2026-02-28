# 🫐 Proyecto MLOps: Inteligencia Predictiva para la Exportación de Arándanos (Perú 2026)

Este proyecto implementa un ciclo de vida completo de MLOps para predecir el volumen de exportación de arándanos peruanos con alta granularidad, optimizando la logística por variedad y terminal.

## 🎯 A) Definición del Problema y Adquisición de Datos
### Caso de Uso: Optimización de Slots Portuarios
**Problema:** La falta de proyecciones precisas por Variedad y Puerto causa saturación en Paita y subutilización en Salaverry.
**Meta:** Predecir el `PESO_NETO_KG` basado en [Fecha + Puerto + Variedad].

### Adquisición de Datos
* **Fuente:** SUNAT (Superintendencia Nacional de Aduanas).
* **Dataset:** `sunat_2022_2025_mar.xlsx`
* **Periodo:** 2022 - Marzo 2025.

| Variable | Tipo | Función |
| :--- | :--- | :--- |
| `ADUA_DESC` | Categórica | Puerto de salida |
| `VARIEDAD` | Categórica | Tipo genético |
| `FECHA` | Temporal | Estacionalidad |
| `PESO_NETO_KG` | Numérica | **Target** |

## 🛠️ B-C) Preparación y Entrenamiento
* **EDA:** Imputación de "SIN VARIEDAD" y codificación de categorías (`mappings.pkl`).
* **Modelo:** Regresor optimizado guardado en `models/mejor_modelo_volumen.pkl`.

## 🚀 D-E) Model Serving y Resultados
API construida con **Flask** (`src/serving.py`). Escenarios validados para 2026:

| Escenario | Puerto | Variedad | Precio | Resultado (Predicción) |
| :--- | :--- | :--- | :--- | :--- |
| **Prueba Puntual** | PAITA | VENTURA | $9.45 | **10,114.75 KG** |
| **Semana Completa** | SALAVERRY | SIN VARIEDAD | $5.00 | **42,042.77 KG** |

---
*Desarrollado para el Trabajo Final de MLOps - 2026*
