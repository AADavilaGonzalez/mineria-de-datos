# Resumen de la práctica 5

Esta práctica construye un modelo de **regresión lineal** para predecir el
total de alquileres (`cnt`). Utiliza como variables predictoras la temporada,
el año, la hora, el tipo de día, la situación meteorológica, la temperatura,
la humedad y la velocidad del viento.

El dataset se divide aleatoriamente en un conjunto de entrenamiento y otro de
prueba. El modelo aprende los coeficientes usando los datos de entrenamiento y
después genera predicciones para los registros de prueba.

El programa calcula dos métricas:

- **R²**, que indica qué proporción de la variabilidad del objetivo explica el
  modelo.
- **RMSE**, que mide el error promedio de las predicciones dando mayor peso a
  los errores grandes.

También genera tres gráficas: valores observados frente a predichos, residuos
del modelo y coeficientes de las variables. Estas imágenes ayudan a evaluar
tanto el ajuste como la influencia relativa de cada predictor.
