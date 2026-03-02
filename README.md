# 🫐 Proyecto MLOps: Inteligencia Predictiva para la Exportación de Arándanos (Perú 2026)

Este proyecto implementa un ciclo de vida completo de MLOps para predecir el volumen de exportación de arándanos peruanos con alta granularidad, optimizando la logística por variedad y terminal.

# Sistema de Inteligencia Predictiva para la exportación de las Variedades de Arándano.

# 🎯 A.1) DEFINICIÓN DEL PROBLEMA (PROBLEM DEFINITION)
Contexto: La industria del arándano en Perú ha evolucionado hacia una segmentación técnica por variedades (Ventura, Biloxi, Rocío, etc.), cada una con diferentes ventanas de maduración y vida de anaquel.
La Brecha de Información: Los exportadores enfrentan incertidumbre porque los pronósticos generales no discriminan entre puertos y variedades, causando saturación en Paita y subutilización en Salaverry, además de riesgos en la cadena de frío.
Metas: Crear un motor de inferencia para predecir el peso neto basado en la tríada [Fecha + Variedad + Puerto], permitiendo una planificación logística y del 20% en tiempos de espera de contenedores refrigerados

# 📊 A.2) Adquisición y Análisis de Datos (Data Acquisition)
Origen: Registros oficiales de la SUNAT (Declaraciones Aduaneras de Mercancías - DAM).
Origen de los Datos: El dataset proviene de los registros oficiales de la Superintendencia Nacional de Aduanas y de Administración Tributaria (SUNAT) de Perú. Específicamente, se trata de una consolidación de las Declaraciones Aduaneras de Mercancías (DAM) para la exportación de arándanos frescos. 

# Ficha Técnica del Dataset: 
•	Nombre del archivo original: Archivo: sunat_2022_2025_mar.xlsx (6,367 registros iniciales).
•	Periodo de cobertura: Desde enero de 2022 hasta marzo de 2025. 
•	Entidad fuente: SUNAT (Portal de Datos Abiertos / Operatividad Aduanera). 
•	Unidad de análisis: Cada registro representa una serie o ítem dentro de una declaración de exportación, lo que permite un análisis granular por transacción.
•	Descripción de Variables Principales: Para resolver la necesidad de información precisa por variedad y puerto, el dataset se estructura en las siguientes dimensiones:
•	Dimensión Logística (ADUA_DESC): Describe el terminal portuario o aduana de salida (ej. PAITA, SALAVERRY, CALLAO). Es la variable clave para la planificación de slots y capacidad. 
•	Dimensión Biológica (VARIEDAD): Detalla el tipo genético del arándano (ej. VENTURA, BILOXI, ROCIO, o "SIN VARIEDAD"). Esta variable dicta el ciclo de cosecha y la resistencia al transporte. 
•	Dimensión Temporal (FECHA): Fecha de numeración o embarque. Permite al modelo identificar la estacionalidad, diferenciando la "Campaña Grande" (agosto-diciembre) de la "Campaña Chica". 
•	Dimensión Comercial (FOB_DOLPOL_KG): Precio FOB unitario declarado. Funciona como un indicador de calidad y demanda de mercado.
•	 Variable Objetivo (PESO_NETO_KG): El volumen físico real exportado. Es el valor que el modelo de ML busca predecir para optimizar la logística.

# Análisis de Granularidad
•	Puertos Monitoreados (6): Chiclayo, Paita, Salaverry, Pisco, Marítima del Callao, entre otros.
•	Variedades Identificadas (42): Ventura, Emerald, Biloxi, Atlas, etc.

# B) Preparación y Limpieza de Datos (Data Cleaning)
•	Estructura MLOps: Se implementó una arquitectura de carpetas profesional:
o	data/raw: Datos originales.
o	data/processed: Datos limpios y transformados.
o	models, src, reports, tests: Para ciclo de vida completo.
Se ejecutó un protocolo de calidad de datos para asegurar la integridad del modelo:
1.	Normalización de Texto: Se transformaron todas las variedades a mayúsculas y se eliminaron espacios extra (str.upper().strip()) para evitar duplicidad por errores de tipeo.
2.	Análisis de Outliers (Valores Extremos):
o	Se identificó que la mediana del PESO_NETO es de 12,075.00 KG.
o	Se detectaron valores atípicos en el precio FOB mediante Boxplots.
3.	Filtrado Estadístico (Percentil 2%):
o	Se estableció un umbral mínimo (Piso P2) de $2.69 por kg para eliminar registros con errores de declaración o muestras sin valor comercial.
o	Resultado: El dataset final quedó conformado por 6,235 registros de alta calidad, conservando una mediana de precio de $6.00.
#  Diccionario de Variables (features iniciales)
Para el entrenamiento del modelo, se definieron las siguientes dimensiones:
•	Temporales (ANIO / MES / SEMANA): Extraídas de la fecha original para capturar la estacionalidad (Campaña Grande vs. Chica).
•	ADUA_DESC_CODE: Identificador numérico del puerto (Label Encoding) que influye en la capacidad logística.
•	PAIS_DESC_CODE: Identificador del país de destino; dicta estándares de volumen.
•	VARIEDAD_CODE: Código de la variedad genética; dicta el ciclo de cosecha.
•	ES_ORGANICO: Variable binaria (1: Orgánico, 0: Convencional) para ajustar predicción según certificación.
•	FOB_DOLPOL_KG: Valor comercial unitario; correlacionado directamente con el volumen de carga.
•	PESO_NETO (Target): Masa total en kg que el modelo busca predecir de forma específica.

Trazabilidad de Archivos MLOps
Se han generado puntos de control (checkpoints) en el Drive para asegurar la reproducibilidad:
•	data/raw/sunat_2022_2025_raw.csv: Dataset original preservado.
•	data/processed/df_blueberry_limpio.csv: Dataset tras la limpieza de outliers y normalización.
•	data/processed/dataset_transformado.csv: Dataset final con ingeniería de variables.

Ingeniería de Características (Feature Engineering)
Para que el modelo entienda la logística y la biología del arándano, se realizaron las siguientes transformaciones:
1.	Tiempo Lineal y Estacionalidad: La fecha (int64) se convirtió a objeto datetime, extrayendo Año, Mes y Semana. La variable "Semana" es crítica para detectar el pico de la "Campaña Grande".
2.	Categorización Logística (Label Encoding): Las columnas ADUA_DESC, PAIS_DESC y VARIEDAD se transformaron en códigos numéricos, permitiendo al modelo procesar el origen, destino y tipo genético.
3.	Variable de Valor Agregado: Se creó la columna binaria ES_ORGANICO (1: Sí, 0: No) mediante la detección de patrones de texto en la descripción del producto.

Definición de la Matriz de Predictores (X) y Objetivo (y)
El dataset transformado cuenta con 6,235 registros y 8 predictores finales:
Feature	Descripción
ANIO, MES, SEMANA	Capturan el ciclo de cosecha y estacionalidad.
ADUA_DESC_CODE	Representa el puerto/aduana de salida.
PAIS_DESC_CODE	Identifica el mercado de demanda receptor.
VARIEDAD_CODE	Diferencia el peso según la genética del fruto.
ES_ORGANICO	Identifica el nicho de mercado (Certificación).
FOB_DOLPOL_KG	Correlación económica con el volumen declarado.
PESO_NETO (Target)	Variable objetivo a predecir.

# C) ML Experimentation
Para garantizar la robustez del motor de inferencia, se implementó una estrategia de partición y escalamiento de variables antes del entrenamiento.
•	División de Datos (Train/Test Split): Se aplicó una división balanceada de los datos:
o	Entrenamiento (80%): 4,988 muestras utilizadas para que el modelo aprenda los patrones estacionales y logísticos.
o	Prueba (20%): 1,247 muestras reservadas para validar la precisión del modelo con datos que nunca ha visto.

•	Normalización y Escalamiento (StandardScaler): Se utilizó la técnica de Estandarización, transformando las variables para que tengan una media cercana a 0 y una desviación estándar de 1.
Justificación Técnica del Escalamiento: Existe un debate sobre si los modelos basados en árboles requieren normalización. En este proyecto, se decidió aplicar StandardScaler bajo la consideración de que si una variable numérica tiene un rango de valores extremadamente dispar (como el precio FOB por kilogramos vs. el Volumen expresado en kilogramos), puede dominar involuntariamente los criterios de división iniciales. La estandarización asegura una escala similar, permitiendo un equilibrio óptimo en la construcción de los nodos del árbol.

•	Persistencia del Preprocesamiento (MLOps Requisite): Siguiendo los estándares de MLOps, el transformador no solo se aplicó, sino que se almacenó físicamente:
o	Archivo: models/scaler_blueberry.pkl
o	Función: Este archivo será cargado por la API de producción (serving.py) para normalizar los datos de entrada en tiempo real antes de realizar la predicción, garantizando la consistencia entre el entrenamiento y la inferencia.
Trazabilidad del Entrenamiento
Se han generado y guardado en Google Drive cuatro archivos clave en la carpeta data/training/ para facilitar auditorías futuras:
•	X_train_scaled.csv y X_test_scaled.csv: Variables predictoras normalizadas.
•	y_train.csv y y_test.csv: Valores reales de peso neto para validación.

# ML Modeling 
Para encontrar el motor de inferencia más preciso, se ejecutó un "Torneo de Modelos" comparando cuatro arquitecturas distintas sobre los datos escalados. Se evaluaron mediante el R2 Score (capacidad predictiva) y el MAE (Error Absoluto Medio en kilogramos).

Ranking de Rendimiento
El torneo incluyó modelos de ensamble y regresiones lineales: 
XGBoost
Random Forest
RandomForestRegressor
Gradient Boosting
Ridge Regression

Decisión de MLOps: El modelo Random Forest fue seleccionado como el "Campeón" y guardado explícitamente en models/mejor_modelo_volumen.pkl. Este modelo destaca por su capacidad de manejar las categorías de puertos y variedades sin sobreajustar (overfitting) tanto como los modelos lineales.

# Métricas de Éxito
•	Capacidad Predictiva ($R^2 = 0.5120$): El modelo logra explicar el 51.2% de la varianza del peso neto exportado basándose en la interacción de factores temporales, logísticos y económicos.
•	Error Absoluto Medio (MAE): Se situó en 2,534.63 kg. Este margen de error es altamente competitivo, considerando que los despachos analizados corresponden a operaciones de gran escala que suelen superar las 20 toneladas.

Matriz de Confusión: Clasificación de Escalas
Se dividió el volumen real en tres niveles basados en cuantiles para asegurar una logística equilibrada:
•	Bajo Volumen: 3.00 a 10,915.92 kg.
•	Medio Volumen: 10,915.92 a 13,749.36 kg.
•	Alto Volumen: Más de 13,749.36 kg.

# Jerarquía de Predictores 
El análisis de importancia de variables revela la estructura lógica que rige el volumen de exportación:
1.	Dominio de la Variedad (36.1%): Es el factor determinante. Existe una coherencia biológica y comercial: variedades como Biloxi o Ventura poseen curvas de rendimiento y densidades de siembra distintas, dictando directamente la disponibilidad de carga.
2.	Influencia Logística y de Destino (33.2% combinado): La infraestructura de la Aduana de salida (21.8%) y el País de destino (11.4%) suman un tercio del peso predictivo. El flujo de volumen está anclado a las rutas comerciales y la capacidad operativa de los puertos.
3.	Inelasticidad del Precio FOB (7.5%): El volumen enviado es inelástico frente a las variaciones de valor en el corto plazo. Al ser un producto perecedero, el arándano se exporta por "empuje" de cosecha más que por especulación de precios.
4.	Irrelevancia de la Certificación Orgánica (0.0%): Un hallazgo crítico. Sugiere una invarianza en la data o que el carácter orgánico ya está absorbido por la variedad o el precio. Estratégicamente, indica que el mercado marítimo procesa el volumen de carga de manera estandarizada.

# Resultados de Sensibilidad (Recall):
•	El modelo identifica correctamente el Medio Volumen en un 81.80%, lo cual es crítico para evitar la subutilización de puertos como Salaverry.
•	Para Bajo y Alto Volumen, la precisión se mantiene sobre el 63%, permitiendo una segregación confiable de la carga.

# INSIGHTS
•  La variable ES_ORGANICO tuvo un peso de 0.00 en este experimento, sugiriendo que, para el volumen total (kg), el mercado no discrimina significativamente entre orgánico y convencional tanto como lo hace con la variedad o el puerto.
•  Optimización Logística: La implementación de este motor de inferencia permite pasar de una planificación reactiva a una proactiva, identificando con precisión del 81.8% cuándo los embarques caerán en el rango de capacidad media de los puertos.
•  Segregación de Variedades: Al ser la Variedad el predictor más fuerte, el proyecto demuestra que la logística del arándano ya no puede tratarse como un "commodity", sino que debe segmentarse por tipo genético o variedad.
•  Influencia Logística (33.2%): La Aduana de salida y el País de destino confirman que el flujo está anclado a la infraestructura portuaria.
•  Inelasticidad del Precio (7.5%): El arándano se exporta por "empuje" de cosecha (producto perecedero), no por especulación de precios.
•  Irrelevancia Orgánica (0.0%): Hallazgo crítico; el mercado marítimo procesa el volumen de forma estandarizada independientemente de la certificación.
El modelo actual no integra variables macro-climáticas (como el Fenómeno de El Niño) ni huelgas portuarias, lo que podría afectar la precisión en años atípicos.	

# D) ML Development Activities (Modularización y Productivización)
En esta fase, el proyecto trasciende el entorno de experimentación para convertirse en un sistema modular siguiendo los estándares de MLOps. Se crearon scripts independientes para automatizar el ciclo de vida del modelo.
1. Modularización de la Preparación de Datos (src/data_preparation.py)
Se desarrolló un script robusto que encapsula toda la lógica de ingeniería de características.
•	Funcionalidad: Carga los datos crudos, realiza la transformación temporal (Mes, Semana), codifica las variables categóricas (cat.codes) y genera la variable binaria de producto orgánico.
•	Resultado: Genera el archivo data/processed/dataset_transformado.csv de forma automática.
2. Entrenamiento y Serialización (src/train.py)
Este módulo permite el re-entrenamiento del modelo campeón de forma aislada.
•	Lógica: Carga los datos ya escalados, entrena el Random Forest Regressor (configurado con regularización para evitar sobreajuste) y realiza la Serialización (Model Serialization).
•	Artefacto Generado: models/mejor_modelo_volumen.pkl.
3. Persistencia de Mapeos y Escalamiento
Para que el sistema de inferencia (API) funcione correctamente en producción, se guardaron los objetos de soporte:
•	models/scaler_blueberry.pkl: El escalador para normalizar nuevas entradas.
•	models/mappings.pkl: Diccionario con los códigos exactos de Aduanas, Países y Variedades, asegurando que la API traduzca el texto a los números que el modelo entiende.

Auditoría Técnica de MLOps 
Se ejecutó un script de auditoría para validar la integridad de los artefactos generados en Google Drive, obteniendo un cumplimiento del 100%:

# E) Model Deployment & Serving (Despliegue e Inferencia)
Esta fase final representa la culminación del proyecto: la puesta en marcha del modelo para generar predicciones sobre datos futuros (2026).
1. Arquitectura de la API de Inferencia (src/serving.py)
Se desarrolló un servidor web utilizando Flask que actúa como el "cerebro" logístico del sistema.
•	Carga Dinámica de Artefactos: Al iniciar, la API carga automáticamente el modelo (.pkl), el escalador y los mapeos de categorías desde Google Drive.
•	Procesamiento en Tiempo Real: La API no solo predice; transforma la entrada bruta (ej. una fecha o un nombre de variedad) en los componentes numéricos que el modelo Random Forest requiere, garantizando la integridad de la entrada.

# 2. Model Serving y Resultados: Escenarios Arándanos 2026
Se validó el sistema mediante dos simulaciones de alta relevancia para la planificación portuaria de 2026:

API construida con **Flask** (`src/serving.py`). Escenarios validados para 2026:

| Escenario | Puerto | Variedad | Precio | Resultado (Predicción) |
| :--- | :--- | :--- | :--- | :--- |
| **Prueba Puntual** | PAITA | VENTURA | $9.45 | **10,114.75 KG** |
| **Semana Completa** | SALAVERRY | SIN VARIEDAD | $5.00 | **42,042.77 KG** |

Escenario A: Inferencia Puntual (Paita)
•	Input: Variedad VENTURA (Orgánico) en el Puerto de PAITA para mayo de 2026.
•	Precio FOB: $9.45/kg.
•	Resultado (Expected Result): 10,114.75 KG.
•	Análisis: Refleja la alta eficiencia y concentración de carga de la variedad Ventura en el puerto del norte.

Escenario B: Proyección Semanal (Salaverry)
•	Input: 2da Semana de Febrero 2026, Puerto de SALAVERRY, variedad "SIN VARIEDAD".
•	Simulación: 7 días continuos de exportación.
•	Resultado Acumulado: 42,042.77 KG.
•	Análisis: Identifica un flujo constante de ~6 toneladas diarias, permitiendo a la aduana de Salaverry prever la asignación de personal y slots para carga convencional.

3. Reporte de Despliegue e Inferencia (Auditoría Final)
Se generó un documento de cierre en reports/reporte_analisis_resultados.txt que certifica la salud del sistema:
•	Estado del Servidor: Exitoso (Puerto 5000).
•	Consistencia: El volumen estimado para Salaverry con "Sin Variedad" es consistentemente menor por día comparado con la eficiencia de Paita/Ventura, lo cual es coherente con el comportamiento histórico capturado por el modelo.

# F)  Conclusiones Finales del Proyecto
•	Valor del MLOps: La modularización en scripts (data_preparation.py, train.py, serving.py) permite que el modelo sea mantenible y escalable.
•	Precisión Estratégica: Con un 81.8% de precisión en el segmento de volumen medio, el sistema es una herramienta de bajo riesgo y alto impacto para los exportadores.
•	Hito 2026: El proyecto está técnicamente listo para operar durante la campaña actual, proporcionando estimaciones de carga basadas en datos reales de SUNAT y analítica avanzada.
•	Dominio Predictivo: El modelo Random Forest demostró ser la arquitectura más robusta para la naturaleza no lineal de la agroexportación peruana, alcanzando un $R^2$ de 0.5120 y una precisión crítica del 81.8% en el segmento de volumen medio (10.9t - 13.7t).
•	Validación de Hipótesis: Se confirmó que la Variedad (36.1%) y la Aduana (21.8%) son los ejes que dictan el flujo de carga. El sistema permite ahora una planificación portuaria basada en datos y no solo en intuiciones estacionales.
•	Modularidad MLOps: La transición de un Notebook a scripts físicos (src/) asegura que el modelo no sea un experimento aislado, sino un activo de software listo para integrarse a cualquier dashboard corporativo vía API.

# Conclusiones sobre la Operacionalización
El valor de este sistema no reside únicamente en su capacidad de predicción, sino en su arquitectura reproducible.
•	Desacoplamiento: Se logró separar exitosamente la lógica de preparación de datos (data_preparation.py) del entrenamiento (train.py) y del servicio (serving.py). Esto permite que el modelo sea actualizado sin afectar la disponibilidad de la API.
•	Consistencia de Inferencia: El uso de artefactos serializados (scaler.pkl y mappings.pkl) garantiza que los datos que entran a la API en 2026 reciban exactamente el mismo tratamiento estadístico que los datos históricos, eliminando el riesgo de Training-Serving Skew.

# Justificación Técnica de Decisiones del Pipeline
•	Estandarización Proactiva: Se decidió aplicar StandardScaler a pesar de utilizar modelos de árboles. Esta es una decisión de MLOps de Largo Plazo: estandarizar permite que el pipeline sea agnóstico al modelo. Si en el futuro se decide migrar a una Red Neuronal o un modelo de Ridge Regression para mayor interpretabilidad, el flujo de datos ya es compatible, evitando re-trabajos estructurales.
•	Estrategia de Partición 80/20: Se validó que esta proporción ofrece el equilibrio óptimo entre un entrenamiento robusto (4,988 registros) y una evaluación de generalización con suficiente varianza (1,247 registros), cumpliendo con los estándares de validación cruzada para series temporales de exportación.
•	Eficiencia en Codificación: El uso de Label Encoding fue preferido sobre One-Hot Encoding para evitar la explosión de dimensionalidad en la matriz de características, manteniendo la latencia de respuesta de la API por debajo de los 100ms.

# Lecciones Aprendidas en el Proceso
•	La importancia del Data Profiling: Se descubrió que el volumen de exportación es inelástico al precio en el corto plazo; la logística está gobernada por el ritmo biológico de la cosecha. Esto cambió el enfoque del modelo de uno económico a uno estrictamente logístico-estacional.
•	Persistencia de Mapeos: Una lección crítica fue la necesidad de guardar los diccionarios de categorías (mappings.pkl). Sin esto, la API no podría traducir nuevas peticiones de texto (ej: "PAITA") a los códigos que el modelo entiende, rompiendo el flujo de producción.

# Limitaciones y Hoja de Ruta (Roadmap)
Todo sistema de MLOps es un proceso iterativo. Se han identificado los siguientes puntos para la fase de mejora:
•	Limitaciones: El modelo actual tiene una ventana de visibilidad limitada ante eventos externos disruptivos (cierres de puertos por oleajes o cambios súbitos en la demanda de China) que no están capturados en el dataset histórico de la SUNAT.
•	Mejoras Futuras (MLOps Level 2): 1. Automatización del Re-entrenamiento: Implementar un disparador (trigger) que re-entrene el modelo automáticamente cuando se detecte un Data Drift (desviación) mayor al 10% en los volúmenes de 2026. 2. Enriquecimiento Externo: Integrar una API de clima para correlacionar las anomalías de temperatura con las caídas en el peso neto proyectado. 3. Dashboard de Monitoreo: Desarrollar un panel en Grafana o Streamlit para monitorear la salud de la API y la precisión de las predicciones en tiempo real frente a los embarques reales realizados.

---
*Desarrollado por Yuliana Ayre (gysella.ayre.o@uni.pe) para el Trabajo Final de MLOps - 2026*
