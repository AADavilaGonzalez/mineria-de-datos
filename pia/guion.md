# Guión sugerido para el video (máximo 8 minutos)

## Introducción

"Hola, buenos días. Soy Aldo Dávila y este es el video correspondiente al PIA
para la materia de Mineria de Datos. El dataset que elegí es de un sistema de
bikesharing popular en Estados Unidos llamado Capital Bikeshare que le da
servicio una gran parte de el area metropolitana de Washington D.C. Los datos
son correponden a un periodo de tiempo del 2011 a 2012. No son completamente
actuales pero sirven para propositos educativos."

## Problema de negocio

“Una empresa de bicicletas necesita anticipar cuántas rentas tendrá con la
mayor precisión posible. El objetivo de este análisis es pronosticar la demanda
y evaluar si el error restante contiene patrones previsibles.”

## Qué muestran los datos

“El dataset contiene 17,379 registros dividios por hora. La demanda
promedio no se distribuye de forma uniforme. Las horas con mayor actividad son
17:00 y 18:00, y también aparece un pico importante a las 8:00. Esto indica que
la hora y el tipo de día son señales relevantes para planear la operación.”

Mostrar `graficas/06_demanda_por_hora.png`.

## Modelo de series de tiempo

“El primer modelo responde cuántas rentas podemos esperar. Se entrenó respetando
el orden temporal y usando el historial de la hora anterior, del día anterior,
de la semana anterior y un promedio móvil de 24 horas. En el conjunto de prueba
obtuvo un R² de 0.953, un MAE de 29.83 y un RMSE de
47.71. El modelo no solo describe el pasado: genera un pronóstico
para apoyar decisiones antes de que ocurra el periodo.”

Mostrar `graficas/01_pronostico_temporal.png`.

## Clasificación del error

“Como prueba adicional, calculamos el residual de cada pronóstico:
demanda real menos demanda pronosticada. Usando el MAE obtenido fuera de
muestra, clasificamos el error como subestimado, adecuado o sobreestimado.
Esta clasificación no pretende crear una acción de negocio; sirve para medir
si el error restante contiene patrones previsibles.”

Mostrar `graficas/02_distribucion_residual.png`.

## HistGradientBoostingClassifier

“El clasificador obtuvo una exactitud de 67.91% y un F1 macro de 48.75%.
Sin embargo, una regla que siempre predice la clase mayoritaria alcanzó
73.16% de exactitud. Por ello, el clasificador no supera una estrategia
trivial y no aporta una mejora útil al pronóstico.”

Mostrar `graficas/03_matriz_hist_gradient_boosting.png` y
`graficas/04_importancia_hist_gradient_boosting.png`.

## Conclusión del experimento

“La clasificación del error demuestra que gran parte del residuo no sigue
patrones adicionales suficientemente fuertes. El modelo base conserva mejor
rendimiento y simplicidad. Por tanto, el resultado principal del proyecto es
el pronóstico puntual, respaldado por MAE, RMSE y R².”

Mostrar `graficas/05_tipos_error_observados.png`.

## Conclusión y límites

“El hallazgo central es que la demanda puede anticiparse a partir de sus
patrones temporales y del contexto ambiental. Esto convierte el análisis en
una herramienta de decisión, no solo en una colección de gráficas. Para
calcular el beneficio económico real todavía harían falta datos de precios,
promociones, inventario y costos. Por ello, el siguiente paso sería probar la
política de descuentos con un experimento controlado.”
