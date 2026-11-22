# Mockup del PIA: pronóstico preciso de Bike Sharing

## Pregunta de negocio

¿Qué tan exactamente puede anticiparse la demanda de bicicletas a partir de
patrones temporales, climáticos y ambientales?

## Hallazgo principal

La demanda no es constante: se concentra en horarios específicos y depende
del contexto temporal y ambiental. La mayor demanda promedio ocurre a las
17:00 y 18:00, mientras que los periodos de menor actividad se concentran en
la madrugada y en condiciones menos favorables.

El modelo temporal explica **95.29%** de la variabilidad observada en el
conjunto de prueba, con un error absoluto medio de **29.83 rentas por
hora**. Esto permite estimar la cantidad esperada antes de tomar una decisión.

Como prueba adicional, `HistGradientBoostingClassifier` intenta clasificar el
error fuera de muestra en tres categorías: subestimado, adecuado y
sobreestimado. Su exactitud fue **67.91%** y su F1 macro fue **48.75%**.
Esto es inferior al baseline dummy de **73.16%** de exactitud. Por ello, la
clasificación no debe utilizarse como modelo principal: confirma que gran parte
del error no tiene patrones previsibles adicionales.

## Diseño de los modelos

### Serie de tiempo

- Entrenamiento: primer 80% de los registros utilizables en orden cronológico.
- Prueba: último 20%, sin mezclar el futuro con el pasado.
- Variables: hora, calendario, clima, rezagos de 1, 24 y 168 horas y promedio
  móvil de las 24 horas anteriores.
- Métricas: MAE, RMSE y R².

### Clasificación del error

- Residual: `cnt real - cnt pronosticado`.
- Subestimado: residual mayor que el MAE fuera de muestra.
- Adecuado: residual dentro del intervalo ±MAE.
- Sobreestimado: residual menor que el negativo del MAE.
- Variables: contexto temporal, clima y pronóstico puntual.
- Métricas: exactitud, F1 macro, matriz de confusión y baseline dummy.

El clasificador se evalúa sin fuga de datos: el umbral MAE se obtiene con una
validación interna y el modelo de clasificación se entrena con residuales
históricos fuera de muestra.

## Conclusión técnica

El modelo base es suficiente para el objetivo de predicción. La segunda etapa
no mejora el resultado ni supera una regla trivial que siempre predice la
categoría más frecuente. Esto apoya la conclusión de que gran parte del error
restante es ruido o variabilidad no capturada por las variables disponibles.
Por tanto, conviene conservar `HistGradientBoostingRegressor` como modelo
principal.

## Evidencia para las diapositivas

1. `graficas/06_demanda_por_hora.png`: demuestra los picos de demanda.
2. `graficas/01_pronostico_temporal.png`: compara valores reales y pronosticados.
3. `graficas/02_distribucion_residual.png`: muestra la distribución del residual.
4. `graficas/03_matriz_hist_gradient_boosting.png`: evalúa la clasificación del error.
5. `graficas/04_importancia_hist_gradient_boosting.png`: identifica los factores clave.

## Limitaciones

- La clasificación utiliza categorías basadas en el MAE fuera de muestra; otro
  umbral podría producir una distribución distinta.
- La exactitud no debe interpretarse como probabilidad calibrada de error.
- El modelo no incluye eventos especiales ni variables geográficas por estación.
