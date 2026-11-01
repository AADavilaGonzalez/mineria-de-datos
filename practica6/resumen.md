# Resumen de la práctica 6

Esta práctica implementa un modelo de clasificación utilizando el algoritmo
**K-Nearest Neighbors (KNN)**. El objetivo es predecir la situación
meteorológica (`weathersit`) a partir de variables temporales y ambientales.

Antes de entrenar el modelo, las variables predictoras se estandarizan con
`StandardScaler`, ya que KNN utiliza distancias y las variables deben estar en
escalas comparables. Los datos se dividen en entrenamiento y prueba mediante
una separación estratificada para conservar la proporción de cada clase.

El modelo utiliza cinco vecinos cercanos y se evalúa con:

- Exactitud (`accuracy`).
- Precisión (`precision`).
- Recall.
- F1-score.
- Reporte detallado por clase.

Finalmente, se generan una matriz de confusión y una gráfica con las métricas
globales. La matriz permite identificar qué situaciones meteorológicas se
clasifican correctamente y cuáles tienden a confundirse.
