# Descripción de los datos (Actividad 6 – Aprendizaje no supervisado)

## 1. Origen
No se contó con un archivo público con afluencia **por estación, día y hora** alineado con la red del proyecto,
así que se construyó un **dataset sintético de muestra** (`src/generar_dataset.py`, semilla 42, reproducible).
Los **nombres de las 32 estaciones son reales** (portales, Caracas, Autonorte, Eje Ambiental, Calle 80, etc.);
las **cifras de validaciones son simuladas**. El arquetipo con el que se simuló cada estación se guarda aparte
(`data/arquetipos_ocultos.csv`) solo para evaluar el resultado al final; **el modelo no lo usa**. La asignación de
arquetipo a cada estación es un **supuesto del equipo** y no un dato oficial.

Fuentes reales para reemplazar la muestra manteniendo las mismas columnas:
- Datos abiertos de TransMilenio / Datos Abiertos Bogotá: validaciones de acceso por estación y franja horaria.
- GTFS de TransMilenio: estaciones, rutas y coordenadas.

## 2. Archivo `data/afluencia_horaria.csv` (16.128 filas)
32 estaciones × 28 días (4 semanas desde el 2 de marzo de 2026) × 18 horas (5:00 a 22:00).

| Variable | Tipo | Valores | Uso |
|---|---|---|---|
| estacion | categórica | 32 estaciones | Identificador (unidad a agrupar) |
| fecha | fecha | 2026-03-02 a 2026-03-29 | Agregación |
| tipo_dia | categórica | laboral / fin_de_semana | Agregación |
| hora | entera | 5 a 22 | Agregación |
| validaciones | entera ≥ 0 | validaciones en esa hora | Medida base |

## 3. Lógica de simulación
Cuatro arquetipos con distinto patrón (picos centrados en las 7:00 y las 18:00 en días laborales; perfil plano
más bajo en fin de semana), más variación propia de cada estación (volumen ±18 %, picos ±15 %), variación diaria
(±7 %) y ruido por hora (±10 %):

| Arquetipo | Estaciones | Patrón simulado |
|---|---|---|
| portal | 8 portales | Alto volumen, pico AM fuerte, pico PM débil |
| centro | 7 (Museo del Oro, Las Aguas, Av. Jiménez, Calle 19/26/34, Universidades) | Pico PM fuerte, pico AM débil |
| corredor | 8 (Calle 45 a 127 y Héroes) | Picos AM y PM similares |
| residencial | 9 (Toberín, Mazurén, Alcalá, Prado, Calle 142, Marly, Banderas, Av. Chile, Ricaurte) | Volumen bajo, pico AM |

## 4. Ingeniería de características (una fila por estación, 22 variables)
Se usan solo días laborales, salvo el indicador de fin de semana:

| Variable | Descripción |
|---|---|
| h5 … h22 (18) | Participación de cada hora en la demanda diaria (la **forma** del perfil, sin el volumen) |
| pico_am | Suma de la participación de 6, 7 y 8 h |
| pico_pm | Suma de la participación de 17, 18 y 19 h |
| ratio_fin_semana | Demanda promedio de fin de semana / día laboral |
| log_volumen | Logaritmo de la demanda promedio por hora |

Luego se **estandarizan** (media 0, desviación 1), porque K-Means y el clustering jerárquico usan distancias y las
variables tienen escalas distintas.

## 5. Calidad
Sin nulos ni valores negativos (verificado en las pruebas). No hay etiquetas: es aprendizaje **no supervisado**.

## 6. Limitaciones
Al ser sintéticos, los resultados demuestran que el procedimiento funciona, **no el comportamiento real** de la red.
Además todos los arquetipos comparten las horas pico (7 y 18 h) porque así se simuló, por lo que la hora pico exacta
no distingue grupos; lo que los distingue es la proporción de demanda en cada pico y el volumen.
