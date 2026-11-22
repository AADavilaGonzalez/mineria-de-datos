# Documentación de los modelos

## 1. Serie de tiempo

La serie de tiempo utiliza `HistGradientBoostingRegressor` para pronosticar la
demanda respetando el orden cronológico. Este modelo captura relaciones no
lineales entre los rezagos, la hora y las condiciones ambientales, por lo que
representa mejor los picos y valles que la regresión lineal inicial.

### Variable objetivo

- `cnt`: total de rentas por hora.

### Variables contextuales utilizadas

- `hr`
- `workingday`
- `weathersit`
- `temp`
- `hum`
- `windspeed`

### Variables temporales derivadas

- `hour_sin`
- `hour_cos`
- `cnt_lag_1`: rentas de la hora anterior.
- `cnt_lag_24`: rentas de la misma hora del día anterior.
- `cnt_lag_168`: rentas de la misma hora de la semana anterior.
- `cnt_rolling_24`: promedio de las 24 horas anteriores.

Las variables cíclicas representan el comportamiento periódico de la hora.
Los rezagos y el promedio móvil incorporan la dependencia entre la demanda
actual y la demanda observada anteriormente.

### Variables omitidas

- `instant`: identificador del registro.
- `dteday`: se utiliza para ordenar cronológicamente, pero no como predictor directo.
- `season`
- `yr`
- `mnth`
- `holiday`
- `weekday`
- `atemp`
- `casual`
- `registered`

`casual` y `registered` se excluyen porque forman parte de `cnt`:
```text cnt = casual + registered ```

Usarlas como predictores produciría fuga de información. El valor actual de
`cnt` tampoco se utiliza directamente; solo se emplean sus valores pasados
mediante los rezagos y el promedio móvil.

### Entrenamiento y evaluación

Los registros se mantienen en orden cronológico:

- 80% inicial: entrenamiento.
- 20% final: prueba.

No se mezclan aleatoriamente los registros, porque eso permitiría que el
modelo aprendiera información del futuro.

La regresión lineal inicial obtuvo MAE de 59.15, RMSE de 88.03 y R² de 0.8396
en una división 80/20. El modelo actualizado obtuvo:

- **MAE:** 29.83 rentas por hora.
- **RMSE:** 47.71 rentas por hora.
- **R²:** 0.9529.

La implementación final se encuentra en `pia/main.py`, especialmente en:

- `TIME_CONTEXT` y `LAGS`: líneas 42–57.
- `build_time_data`: líneas 75–83.
- División cronológica y entrenamiento: líneas 102–119.
- `HistGradientBoostingRegressor`: líneas 109–115.

## 2. HistGradientBoostingClassifier

HistGradientBoostingClassifier clasifica el **error del pronóstico** en tres
categorías: subestimado, adecuado y sobreestimado.

### Variable objetivo

Se calcula el residual:

```text
residual = cnt_real - cnt_pronosticado
```

Las clases se definen con el MAE obtenido en una validación interna:

- **Subestimado:** residual mayor que MAE.
- **Adecuado:** residual entre -MAE y MAE.
- **Sobreestimado:** residual menor que -MAE.

### Variables utilizadas

- `hr`
- `workingday`
- `weathersit`
- `temp`
- `hum`
- `windspeed`
- `hour_sin`
- `hour_cos`
- rezagos de 1, 24 y 168 horas
- promedio móvil de 24 horas
- `forecast`: pronóstico puntual del modelo de series de tiempo.

Estas variables describen el horario, calendario y condiciones ambientales
disponibles antes del periodo y el valor pronosticado.

### Variables omitidas

- `instant`: identificador del registro.
- `dteday`: se utiliza para ordenar los datos, no como predictor directo.
- `cnt`: se utiliza únicamente fuera de muestra para calcular el residual.
- `casual`
- `registered`
- Variables que directamente revelan `cnt` real.

### Entrenamiento y evaluación

HistGradientBoostingClassifier utiliza hasta 250 iteraciones, división
cronológica 80/20 y ponderación de clases. Se evaluó con:

- Exactitud: **67.91%**.
- F1 macro: **48.75%**.
- Accuracy de baseline dummy: **73.16%**.
- Matriz de confusión por tipo de error.

La implementación se encuentra en `pia/main.py`, en `train_models`,
`residual_labels`, `HistGradientBoostingClassifier` y la validación interna
fuera de muestra.

## 3. Diferencia entre ambos modelos

| Modelo | Pregunta que responde | Salida |
|---|---|---|
| Serie de tiempo | ¿Cuántas rentas habrá? | Cantidad numérica pronosticada |
| HistGradientBoostingClassifier | ¿Cómo se equivoca el pronóstico? | Subestimado, adecuado o sobreestimado |

La serie de tiempo permite estimar la cantidad esperada. HistGradientBoostingClassifier
permite verificar si el residuo contiene patrones adicionales. En esta
validación no supera al modelo base ni al dummy baseline, por lo que el modelo
de series de tiempo permanece como solución principal.
