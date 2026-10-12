# Pruebas realizadas al componente (Actividad 6)

Entorno: Python 3.12, scikit-learn 1.8.0. Comandos: `python src/clustering_estaciones.py` y
`python -m pytest src/test_clustering.py -v`. Salida completa en `resultados/salida_consola.txt`.

## 1. Selección del número de grupos k (K-Means, 32 estaciones)

| k | Inercia | Silhouette | Davies-Bouldin | Estabilidad (ARI, 10 semillas) |
|---|---|---|---|---|
| 2 | 341,29 | **0,4322** | 0,9023 | 1,000 |
| 3 | 236,50 | 0,4217 | 0,8676 | 0,673 |
| **4** | 184,82 | 0,3435 | 0,9924 | 0,733 |
| 5 | 144,69 | 0,3659 | 0,7499 | 0,729 |
| 6 | 115,42 | 0,3428 | 0,8886 | 0,812 |
| 7 | 95,20 | 0,3425 | 0,7876 | 0,789 |
| 8 | 85,17 | 0,3258 | 0,7828 | 0,739 |

(Silhouette: más alto es mejor; Davies-Bouldin: más bajo es mejor; estabilidad: coincidencia entre ejecuciones con distinta inicialización.)

**Decisión sobre k.** El silhouette más alto es k = 2, pero solo separa estaciones de "origen" (pico AM) de las de
"destino" (pico PM) y mezcla los corredores mixtos con los centros. Se eligió **k = 4** por criterio de aplicación: es
el menor k que separa los tres patrones horarios (pico AM, pico PM y mixto). Es un compromiso: la calidad
estadística de k = 4 es moderada (silhouette 0,34).

## 2. Resultado con k = 4

| Grupo | Estaciones | Volumen promedio (val./hora) |
|---|---|---|
| Destino / centro (pico PM) | 7: Av. Jiménez, Calle 19, Calle 26, Calle 34, Las Aguas, Museo del Oro, Universidades | 2.316 |
| Corredor mixto (picos AM y PM) | 8: Calle 45, 57, 63, 72, 76, 100, 127 y Héroes | 1.586 |
| Origen (pico AM) – volumen alto | 11: Portal 80, Eldorado, Norte, Sur, Tunal, Usme, Alcalá, Calle 142, Marly, Mazurén, Ricaurte | 1.874 |
| Origen (pico AM) – volumen bajo | 6: Av. Chile, Banderas, Portal Américas, Portal Suba, Prado, Toberín | 1.232 |

## 3. Comparación de algoritmos y validación

| Métrica | K-Means | Jerárquico (Ward) |
|---|---|---|
| Silhouette | 0,3435 | 0,3124 |
| Davies-Bouldin | 0,9924 | 1,0380 |

- **Acuerdo entre algoritmos (ARI):** 0,8171. Los dos métodos producen agrupaciones muy parecidas.
- **Estabilidad de K-Means con k = 4:** ARI medio 0,733 y mínimo 0,399 entre 10 inicializaciones simples; con `n_init = 20` (usado en el modelo final) el resultado es reproducible.
- **Validación externa (posible solo por ser datos sintéticos):** ARI = 0,6063 frente a los arquetipos simulados. Los grupos **centro** (7 de 7) y **corredor** (8 de 8) se recuperan sin error. La división del grupo AM en "volumen alto/bajo" **no coincide** con portal vs. residencial (los portales quedan repartidos 6 y 2, los residenciales 5 y 4): esa división es débil y no debe interpretarse como una diferencia real de tipo de estación.
- **PCA:** las dos primeras componentes explican 61,3 % y 28,3 % de la varianza (89,6 % en total), por lo que la gráfica en 2D representa bien los datos.

## 4. Pruebas automatizadas (`src/test_clustering.py`): 10/10 aprobadas

| Prueba | Verifica |
|---|---|
| test_dataset_sin_nulos_y_rangos | Sin nulos, horas 5–22, validaciones ≥ 0, tipos de día |
| test_una_fila_por_estacion_sin_nulos | 32 estaciones únicas, sin nulos en las características |
| test_forma_horaria_suma_uno | La participación horaria de cada estación suma 1 |
| test_kmeans_reproducible | Misma semilla → mismos grupos |
| test_calidad_minima_silhouette | Silhouette > 0,30 |
| test_todos_los_grupos_tienen_al_menos_dos_estaciones | 4 grupos, 32 estaciones, ninguno con menos de 2 |
| test_grupos_recuperan_patron_pm_y_mixto | Museo del Oro con Las Aguas; Calle 72 con Calle 100; ambos grupos distintos |
| test_pico_am_en_portal_y_pico_pm_en_centro | Portal Norte pico AM en 6–8 h; Museo del Oro pico PM en 17–19 h |
| test_funcion_grupo_de_estacion | `grupo_de_estacion("Portal Norte")` devuelve un grupo de origen |
| test_estacion_desconocida | Lanza ValueError |

Evidencias en `resultados/`: `seleccion_k.png`, `grupos_pca.png`, `perfiles_por_grupo.png`, `dendrograma.png`, `grupos_estaciones.csv`, `metricas.json`.

## 5. Uso en el proyecto de rutas
`grupo_de_estacion(estacion)` entrega el perfil de cada estación. Aplicaciones: (a) usar el grupo como variable adicional del modelo supervisado de congestión, y (b) asignar a una estación nueva, sin historial, la penalización típica de su grupo en A*.
