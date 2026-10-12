# Sistema Inteligente para Encontrar la Mejor Ruta en TransMilenio
## Actividad 6: Aprendizaje no supervisado (agrupamiento de estaciones por patrón de demanda)

Se agrupan 32 estaciones según la forma de su perfil horario de demanda usando K-Means y clustering
jerárquico (capítulo 16, técnicas de agrupamiento).

```
data/afluencia_horaria.csv        dataset sintético (16.128 filas)
data/arquetipos_ocultos.csv       arquetipo simulado, solo para validar
src/generar_dataset.py            genera el dataset
src/clustering_estaciones.py      características, K-Means, jerárquico, métricas, gráficas
src/test_clustering.py            pruebas automatizadas
docs/descripcion_datos.md         descripción de los datos
docs/pruebas.md                   pruebas y resultados
resultados/                       grupos, métricas y gráficas
```

## Ejecución (desde cualquier carpeta)
```bash
pip install -r requirements.txt
python src/generar_dataset.py
python src/clustering_estaciones.py > resultados/salida_consola.txt
python -m pytest src/test_clustering.py -v
```

## Uso en el proyecto
```python
from clustering_estaciones import grupo_de_estacion
grupo, nombre = grupo_de_estacion("Portal Norte")   # (1, 'Origen (pico AM) - volumen alto')
```

Entregables: PDF con enlaces al repositorio y al video (≤ 10 min). Agregar al tutor como colaborador.
