# Resumen de la práctica 8

Esta práctica implementa un modelo básico de **series de tiempo** para
predecir el total de alquileres por hora (`cnt`). A diferencia de una división
aleatoria, los datos se ordenan cronológicamente y se utiliza el primer 80 %
para entrenamiento y el último 20 % para prueba.

El modelo utiliza variables contextuales como la hora, el tipo de día, la
situación meteorológica, la temperatura, la humedad y la velocidad del viento.
También se crean variables específicas para series de tiempo:

- Seno y coseno de la hora, para representar el ciclo diario.
- Alquileres de la hora anterior.
- Alquileres de la misma hora del día anterior.
- Alquileres de la misma hora de la semana anterior.
- Promedio móvil de los alquileres de las 24 horas anteriores.

Con estas variables se entrena una regresión lineal sin utilizar información
futura. El modelo se evalúa con MAE, RMSE y R², y se genera una gráfica que
compara los valores reales con los valores pronosticados en el conjunto de
prueba.
