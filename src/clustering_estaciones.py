"""Aprendizaje NO supervisado: agrupa las estaciones de TransMilenio según su
patrón de demanda (K-Means y clustering jerárquico aglomerativo).

Uso:  python src/clustering_estaciones.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
SEMILLA = 42
# k elegido por criterio de negocio (ver docs/descripcion_datos.md y docs/pruebas.md):
# silhouette favorece k=2 (solo separa origen/destino); k=4 es el menor k que separa
# los tres patrones horarios (pico AM, pico PM, mixto) y distingue volumen dentro del grupo AM.
K_ELEGIDO = 4
HORAS = list(range(5, 23))


def cargar():
    return pd.read_csv(RAIZ / "data" / "afluencia_horaria.csv")


def construir_caracteristicas(df):
    """Una fila por estación: forma del perfil horario laboral + 4 indicadores."""
    lab = df[df["tipo_dia"] == "laboral"].groupby(["estacion", "hora"])["validaciones"].mean().unstack()
    lab = lab[HORAS]
    forma = lab.div(lab.sum(axis=1), axis=0)             # participación por hora (forma, sin volumen)
    forma.columns = [f"h{h}" for h in HORAS]
    fin = df[df["tipo_dia"] == "fin_de_semana"].groupby("estacion")["validaciones"].mean()
    prom_lab = lab.mean(axis=1)
    X = forma.copy()
    X["pico_am"] = forma[["h6", "h7", "h8"]].sum(axis=1)
    X["pico_pm"] = forma[["h17", "h18", "h19"]].sum(axis=1)
    X["ratio_fin_semana"] = fin / prom_lab
    X["log_volumen"] = np.log(prom_lab)
    return X, lab


def seleccionar_k(Z, ks=range(2, 9)):
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(Z)
        aris = [adjusted_rand_score(km.labels_, KMeans(n_clusters=k, n_init=1, random_state=s).fit(Z).labels_)
                for s in range(100, 110)]
        filas.append({"k": k, "inercia": round(km.inertia_, 2), "estabilidad_ARI": round(float(np.mean(aris)), 3),
                      "silhouette": round(silhouette_score(Z, km.labels_), 4),
                      "davies_bouldin": round(davies_bouldin_score(Z, km.labels_), 4)})
    return filas


def nombrar_grupos(X, etiquetas):
    """Nombre interpretable de cada grupo según su pico AM vs PM y volumen."""
    resumen = X.assign(grupo=etiquetas).groupby("grupo")[["pico_am", "pico_pm", "log_volumen"]].mean()
    vol_med = X["log_volumen"].median()
    nombres = {}
    for g, r in resumen.iterrows():
        dif = r["pico_am"] - r["pico_pm"]
        if dif > 0.07:
            n = "Origen (pico AM)"
        elif dif < -0.04:
            n = "Destino / centro (pico PM)"
        else:
            n = "Corredor mixto (picos AM y PM)"
        nombres[int(g)] = n
    # nombres únicos: si dos grupos comparten nombre, se distinguen por volumen
    repetidos = pd.Series(nombres).value_counts()
    for n_rep in repetidos[repetidos > 1].index:
        gs = sorted([g for g, n in nombres.items() if n == n_rep], key=lambda g: resumen.loc[g, "log_volumen"])
        for i, g in enumerate(gs):
            nombres[g] = f"{n_rep} - volumen {'bajo' if i == 0 else 'alto'}" if len(gs) == 2 else f"{n_rep} #{i+1}"
    return nombres


def ejecutar():
    (RAIZ / "resultados").mkdir(exist_ok=True)
    df = cargar()
    X, lab = construir_caracteristicas(df)
    Z = StandardScaler().fit_transform(X)
    res = {"n_estaciones": int(X.shape[0]), "n_caracteristicas": int(X.shape[1])}

    # 1) Selección de k
    tabla = seleccionar_k(Z)
    res["seleccion_k"] = tabla
    res["k_optimo_silhouette"] = int(max(tabla, key=lambda t: t["silhouette"])["k"])
    k = K_ELEGIDO
    res["k_elegido"] = int(k)

    # 2) K-Means final
    km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(Z)
    res["kmeans"] = {"silhouette": round(silhouette_score(Z, km.labels_), 4),
                     "davies_bouldin": round(davies_bouldin_score(Z, km.labels_), 4),
                     "tamanos": {int(g): int(c) for g, c in zip(*np.unique(km.labels_, return_counts=True))}}

    # 3) Jerárquico (Ward) con el mismo k
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z)
    res["jerarquico"] = {"silhouette": round(silhouette_score(Z, ag.labels_), 4),
                         "davies_bouldin": round(davies_bouldin_score(Z, ag.labels_), 4)}
    res["acuerdo_kmeans_vs_jerarquico_ARI"] = round(adjusted_rand_score(km.labels_, ag.labels_), 4)

    # 4) Estabilidad: K-Means con 10 semillas distintas vs. el resultado base
    aris = [adjusted_rand_score(km.labels_, KMeans(n_clusters=k, n_init=1, random_state=s).fit(Z).labels_)
            for s in range(100, 110)]
    res["estabilidad_ARI_media"] = round(float(np.mean(aris)), 4)
    res["estabilidad_ARI_min"] = round(float(np.min(aris)), 4)

    # 5) Validación externa (solo posible por ser datos sintéticos)
    arq = pd.read_csv(RAIZ / "data" / "arquetipos_ocultos.csv").set_index("estacion").loc[X.index, "arquetipo"]
    res["ARI_vs_arquetipo_simulado"] = round(adjusted_rand_score(arq, km.labels_), 4)
    res["cruce_grupo_arquetipo"] = pd.crosstab(pd.Series(km.labels_, index=X.index, name="grupo"), arq).to_dict()

    # 6) Nombres y tabla de salida
    nombres = nombrar_grupos(X, km.labels_)
    salida = pd.DataFrame({"estacion": X.index, "grupo": km.labels_,
                           "nombre_grupo": [nombres[g] for g in km.labels_],
                           "hora_pico_am": lab[[h for h in range(5, 12)]].idxmax(axis=1).values,
                           "hora_pico_pm": lab[[h for h in range(14, 23)]].idxmax(axis=1).values,
                           "volumen_prom_hora": lab.mean(axis=1).round(0).astype(int).values})
    salida.sort_values(["grupo", "estacion"]).to_csv(RAIZ / "resultados" / "grupos_estaciones.csv", index=False)
    res["nombres_grupos"] = {int(g): n for g, n in nombres.items()}
    res["miembros"] = {nombres[g]: sorted(X.index[km.labels_ == g]) for g in sorted(nombres)}

    # 7) Gráficas
    ks = [t["k"] for t in tabla]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ks, [t["inercia"] for t in tabla], "o-"); ax[0].set_title("Método del codo"); ax[0].set_xlabel("k")
    ax[1].plot(ks, [t["silhouette"] for t in tabla], "o-", color="tab:green")
    ax[1].axvline(k, ls="--", color="gray", label=f"k elegido = {k}"); ax[1].legend(); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
    plt.tight_layout(); plt.savefig(RAIZ / "resultados" / "seleccion_k.png", dpi=130); plt.close()

    pca = PCA(n_components=2, random_state=SEMILLA).fit(Z)
    P = pca.transform(Z)
    res["pca_varianza_explicada"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    plt.figure(figsize=(8, 6))
    for g in sorted(nombres):
        m = km.labels_ == g
        plt.scatter(P[m, 0], P[m, 1], label=nombres[g], s=60)
    for i, e in enumerate(X.index):
        plt.annotate(e, P[i], fontsize=6, alpha=0.7)
    plt.legend(fontsize=8); plt.title("Estaciones agrupadas (proyección PCA)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    plt.tight_layout(); plt.savefig(RAIZ / "resultados" / "grupos_pca.png", dpi=130); plt.close()

    plt.figure(figsize=(9, 4.5))
    forma = lab.div(lab.sum(axis=1), axis=0)
    for g in sorted(nombres):
        plt.plot(HORAS, forma[km.labels_ == g].mean().values, "o-", label=nombres[g])
    plt.xlabel("Hora del día"); plt.ylabel("Participación de la demanda diaria")
    plt.title("Perfil horario promedio por grupo (día laboral)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RAIZ / "resultados" / "perfiles_por_grupo.png", dpi=130); plt.close()

    plt.figure(figsize=(10, 5))
    dendrogram(linkage(Z, "ward"), labels=list(X.index), leaf_font_size=7, color_threshold=None)
    plt.title("Dendrograma (Ward)"); plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig(RAIZ / "resultados" / "dendrograma.png", dpi=130); plt.close()

    with open(RAIZ / "resultados" / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return res, salida


def grupo_de_estacion(estacion):
    """Devuelve (grupo, nombre) de una estación, leyendo resultados/grupos_estaciones.csv."""
    g = pd.read_csv(RAIZ / "resultados" / "grupos_estaciones.csv").set_index("estacion")
    if estacion not in g.index:
        raise ValueError(f"Estación desconocida: {estacion}")
    return int(g.loc[estacion, "grupo"]), g.loc[estacion, "nombre_grupo"]


if __name__ == "__main__":
    r, s = ejecutar()
    print("k óptimo por silhouette:", r["k_optimo_silhouette"], "| k elegido:", r["k_elegido"])
    print(pd.DataFrame(r["seleccion_k"]).to_string(index=False))
    print("K-Means:", r["kmeans"]); print("Jerárquico:", r["jerarquico"])
    print("ARI kmeans vs jerárquico:", r["acuerdo_kmeans_vs_jerarquico_ARI"])
    print("Estabilidad ARI media/min:", r["estabilidad_ARI_media"], r["estabilidad_ARI_min"])
    print("ARI vs arquetipo simulado:", r["ARI_vs_arquetipo_simulado"])
    print("PCA varianza:", r["pca_varianza_explicada"])
    for n, m in r["miembros"].items():
        print(f"\n{n} ({len(m)}): {', '.join(m)}")
    print("\nCruce grupo x arquetipo:", r["cruce_grupo_arquetipo"])
