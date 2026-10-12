"""Pruebas del componente de clustering. Ejecutar desde cualquier carpeta:
    python -m pytest src/test_clustering.py -v
(requiere haber ejecutado generar_dataset.py y clustering_estaciones.py)"""
import os, sys
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(__file__))
from clustering_estaciones import (cargar, construir_caracteristicas, grupo_de_estacion,
                                   K_ELEGIDO, RAIZ)


@pytest.fixture(scope="module")
def datos():
    X, lab = construir_caracteristicas(cargar())
    return X, StandardScaler().fit_transform(X)


def test_dataset_sin_nulos_y_rangos():
    df = cargar()
    assert df.isna().sum().sum() == 0
    assert df["hora"].between(5, 22).all()
    assert (df["validaciones"] >= 0).all()
    assert set(df["tipo_dia"]) == {"laboral", "fin_de_semana"}


def test_una_fila_por_estacion_sin_nulos(datos):
    X, _ = datos
    assert X.index.is_unique and X.shape[0] == 32
    assert not X.isna().any().any()


def test_forma_horaria_suma_uno(datos):
    X, _ = datos
    horas = [c for c in X.columns if c.startswith("h")]
    assert np.allclose(X[horas].sum(axis=1), 1.0)


def test_kmeans_reproducible(datos):
    _, Z = datos
    a = KMeans(K_ELEGIDO, n_init=20, random_state=42).fit(Z).labels_
    b = KMeans(K_ELEGIDO, n_init=20, random_state=42).fit(Z).labels_
    assert adjusted_rand_score(a, b) == 1.0


def test_calidad_minima_silhouette(datos):
    _, Z = datos
    lab = KMeans(K_ELEGIDO, n_init=20, random_state=42).fit(Z).labels_
    assert silhouette_score(Z, lab) > 0.30


def test_todos_los_grupos_tienen_al_menos_dos_estaciones():
    g = pd.read_csv(RAIZ / "resultados" / "grupos_estaciones.csv")
    assert g["grupo"].nunique() == K_ELEGIDO
    assert g["grupo"].value_counts().min() >= 2
    assert len(g) == 32


def test_grupos_recuperan_patron_pm_y_mixto():
    # Centros (pico PM) y corredores (mixto) deben quedar cada uno en su propio grupo
    g = pd.read_csv(RAIZ / "resultados" / "grupos_estaciones.csv").set_index("estacion")
    assert g.loc["Museo del Oro", "nombre_grupo"] == g.loc["Las Aguas", "nombre_grupo"]
    assert g.loc["Calle 72", "nombre_grupo"] == g.loc["Calle 100", "nombre_grupo"]
    assert g.loc["Museo del Oro", "nombre_grupo"] != g.loc["Calle 72", "nombre_grupo"]


def test_pico_am_en_portal_y_pico_pm_en_centro():
    g = pd.read_csv(RAIZ / "resultados" / "grupos_estaciones.csv").set_index("estacion")
    assert g.loc["Portal Norte", "hora_pico_am"] in (6, 7, 8)
    assert g.loc["Museo del Oro", "hora_pico_pm"] in (17, 18, 19)


def test_funcion_grupo_de_estacion():
    gid, nombre = grupo_de_estacion("Portal Norte")
    assert isinstance(gid, int) and "Origen" in nombre


def test_estacion_desconocida():
    with pytest.raises(ValueError):
        grupo_de_estacion("Estacion Inventada")
