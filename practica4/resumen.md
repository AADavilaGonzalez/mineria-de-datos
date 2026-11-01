# Resumen de la práctica 4

Esta práctica aplica pruebas estadísticas para determinar si existen
diferencias significativas en el total de alquileres (`cnt`) entre diferentes
grupos.

Primero se utiliza un **ANOVA de una vía** para comparar las medias de
alquileres correspondientes a las cuatro temporadas: primavera, verano, otoño
e invierno. La prueba produce un estadístico `F` y un valor `p`.

Después se aplica una **prueba t de Welch** para comparar los alquileres de
2011 y 2012. La versión de Welch es apropiada cuando no se desea asumir que
ambos grupos tienen exactamente la misma varianza.

En ambos análisis se utiliza un nivel de significancia de `0.05`. Si el valor
`p` es menor que este límite, el programa reporta que existe una diferencia
estadísticamente significativa entre los grupos. Además, muestra el tamaño y
la media de cada grupo para complementar la interpretación.
