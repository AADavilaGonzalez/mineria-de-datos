# Mockup del PIA

Este directorio contiene un prototipo reproducible para la presentación final.
El proyecto responde la pregunta:

> ¿Qué tan exactamente puede anticiparse la demanda de bicicletas y cómo
> puede evaluarse el error restante del pronóstico?

## Generar el mockup

Desde la raíz del repositorio:

```bash
python pia/main.py
```

El script lee `dataset.csv` y genera:
- `graficas/`: imágenes listas para insertar en diapositivas.
- `metricas.csv`: métricas resumidas de ambos modelos.
- `reporte_clasificacion.txt`: métricas detalladas del clasificador.

## Orden sugerido de las diapositivas

1. Problema de negocio.
2. Datos y hallazgo inicial: demanda por hora.
3. Pronóstico de series de tiempo.
4. Distribución del error y bandas ±MAE.
5. Clasificación del residual y matriz de confusión.
6. Comparación contra el baseline dummy.
7. Variables más importantes del clasificador.
8. Conclusión: conservar el modelo base.
