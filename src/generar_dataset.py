"""Genera un dataset SINTÉTICO de afluencia horaria por estación de TransMilenio.

Los nombres de estaciones son reales; las cifras de validaciones son simuladas.
Cada estación se simula según un arquetipo (portal, centro, corredor mixto,
residencial). El arquetipo se guarda SOLO en data/arquetipos_ocultos.csv para
poder evaluar el clustering a posteriori; el modelo NO lo usa.

Salida: data/afluencia_horaria.csv  (estacion, fecha, tipo_dia, hora, validaciones)
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(42)

# estación -> arquetipo (supuesto del equipo para la simulación)
ESTACIONES = {
    "Portal Norte": "portal", "Portal 80": "portal", "Portal Suba": "portal",
    "Portal Eldorado": "portal", "Portal Américas": "portal", "Portal Sur": "portal",
    "Portal Tunal": "portal", "Portal Usme": "portal",
    "Museo del Oro": "centro", "Las Aguas": "centro", "Av. Jiménez": "centro",
    "Calle 19": "centro", "Calle 26": "centro", "Calle 34": "centro", "Universidades": "centro",
    "Calle 45": "corredor", "Calle 57": "corredor", "Calle 63": "corredor",
    "Calle 72": "corredor", "Calle 76": "corredor", "Calle 100": "corredor", "Calle 127": "corredor",
    "Toberín": "residencial", "Mazurén": "residencial", "Alcalá": "residencial",
    "Prado": "residencial", "Calle 142": "residencial", "Marly": "residencial",
    "Banderas": "residencial", "Av. Chile": "residencial", "Ricaurte": "residencial",
    "Héroes": "corredor",
}

# (volumen base por hora, amplitud pico AM, amplitud pico PM, factor fin de semana)
PARAMS = {
    "portal":      (2600, 2.3, 0.7, 0.70),
    "centro":      (2100, 0.8, 2.1, 0.55),
    "corredor":    (1700, 1.5, 1.5, 0.60),
    "residencial": (950,  1.9, 0.9, 0.75),
}

HORAS = np.arange(5, 23)
FECHAS = pd.date_range("2026-03-02", periods=28, freq="D")   # 4 semanas


def perfil(h, am, pm, laboral):
    if laboral:
        return 0.55 + am * np.exp(-((h - 7) ** 2) / 2.2) + pm * np.exp(-((h - 18) ** 2) / 2.8)
    return 0.45 + 0.55 * np.exp(-((h - 13) ** 2) / 14.0)


filas = []
for est, arq in ESTACIONES.items():
    vol, am, pm, wk = PARAMS[arq]
    # variación propia de cada estación (para que no sean copias exactas)
    vol *= rng.normal(1, 0.18); am *= rng.normal(1, 0.15); pm *= rng.normal(1, 0.15)
    wk *= rng.normal(1, 0.08)
    for f in FECHAS:
        laboral = f.dayofweek < 5
        dia = rng.normal(1, 0.07)                      # variación diaria
        for h in HORAS:
            v = vol * perfil(h, am, pm, laboral) * (1 if laboral else wk) * dia
            v *= rng.normal(1, 0.10)
            filas.append((est, f.date().isoformat(), "laboral" if laboral else "fin_de_semana",
                          int(h), max(int(round(v)), 0)))

df = pd.DataFrame(filas, columns=["estacion", "fecha", "tipo_dia", "hora", "validaciones"])
(RAIZ / "data").mkdir(exist_ok=True)
df.to_csv(RAIZ / "data" / "afluencia_horaria.csv", index=False)
pd.DataFrame({"estacion": list(ESTACIONES), "arquetipo": list(ESTACIONES.values())}).to_csv(
    RAIZ / "data" / "arquetipos_ocultos.csv", index=False)
print(f"Dataset: {len(df)} filas, {df['estacion'].nunique()} estaciones, {df['fecha'].nunique()} días")
