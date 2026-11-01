# Resumen de la práctica 7

Esta práctica aplica el algoritmo de agrupamiento **K-Means** para descubrir
perfiles similares dentro de los registros del dataset, sin utilizar una
variable objetivo específica.

El programa prepara las variables para el agrupamiento. Estandariza las
variables numéricas, transforma el total de alquileres con `log1p` y convierte
la hora, el mes y el día de la semana en variables cíclicas mediante seno y
coseno. Esto permite representar correctamente la naturaleza circular de
estas variables.

Se prueban diferentes cantidades de clusters, desde 2 hasta 8. Para comparar
las alternativas se calculan:

- La inercia, que mide la compactación de los grupos.
- El coeficiente silhouette, que evalúa qué tan separados están los clusters.

Se selecciona la cantidad de clusters con el mejor silhouette score. Después,
los grupos se visualizan mediante PCA, matrices de dispersión y perfiles
promedio. Estas gráficas ayudan a interpretar las características ambientales,
temporales y de demanda asociadas con cada cluster.
