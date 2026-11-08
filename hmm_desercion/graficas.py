"""Figuras a 300 dpi con títulos, ejes y leyendas en español."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

from hmm_desercion import metricas  # pylint: disable=wrong-import-position
from hmm_desercion.experimentos import METRICAS  # pylint: disable=wrong-import-position
from hmm_desercion.modelo import ESTADOS, OBSERVACIONES  # pylint: disable=wrong-import-position

DPI = 300
COLORES = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
GRIS = "#52514e"
# E3 discontinua: comparte A y semillas con E1, así que sus curvas de estados coinciden.
ESTILOS = ("-", "-", "--")
NOMBRES_ESCENARIO = {"E1": "E1 Nominal", "E2": "E2 Intervención temprana",
                     "E3": "E3 Observación ruidosa"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": GRIS,
    "lines.linewidth": 2, "figure.dpi": 100, "savefig.bbox": "tight",
})


def _guardar(figura, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=DPI)
    plt.close(figura)
    return ruta


def secuencia_ejemplo(estados, observaciones, inferidos, ruta: Path) -> Path:
    """Estado real, actividad observada y estado inferido por Viterbi."""
    semanas = np.arange(1, len(estados) + 1)
    figura, (arriba, abajo) = plt.subplots(2, 1, figsize=(8.5, 4.2), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 2]})
    arriba.step(semanas, estados, where="mid", color=COLORES[0], label="Estado real")
    arriba.plot(semanas, inferidos, "o--", color=COLORES[1], markersize=6, linewidth=1.2,
                label="Estado inferido (Viterbi)")
    arriba.set_yticks(range(3), ESTADOS)
    arriba.set_ylim(2.4, -0.4)
    arriba.set_ylabel("Estado oculto")
    arriba.set_title(f"Ejemplo de un estudiante: coincidencia de Viterbi "
                     f"{np.mean(estados == inferidos):.0%} de las semanas")
    arriba.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    abajo.plot(semanas, observaciones, "s", color=COLORES[2], markersize=7,
               label="Actividad observada")
    abajo.set_yticks(range(3), OBSERVACIONES)
    abajo.set_ylim(2.5, -0.5)
    abajo.set_ylabel("Actividad semanal")
    abajo.set_xlabel("Semana")
    abajo.set_xticks(semanas)
    abajo.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return _guardar(figura, ruta)


def evolucion_estados(proporciones: pd.DataFrame, ruta: Path) -> Path:
    """Proporción semanal en cada estado por escenario, con IC 95 % y curva teórica."""
    claves = list(dict.fromkeys(proporciones["escenario"]))
    figura, ejes = plt.subplots(1, len(claves), figsize=(10, 3.4), sharey=True)
    for eje, clave in zip(np.atleast_1d(ejes), claves):
        sub = proporciones[proporciones["escenario"] == clave]
        for estado, nombre in enumerate(ESTADOS):
            datos = sub[sub["estado"] == estado]
            eje.fill_between(datos["semana"], datos["ic_inferior"], datos["ic_superior"],
                             color=COLORES[estado], alpha=0.25, linewidth=0)
            eje.plot(datos["semana"], datos["media"], color=COLORES[estado], label=nombre)
            eje.plot(datos["semana"], datos["teorica"], ":", color=GRIS, linewidth=1)
        eje.set_title(NOMBRES_ESCENARIO.get(clave, clave))
        eje.set_xlabel("Semana")
        eje.set_ylim(0, 1)
    np.atleast_1d(ejes)[0].set_ylabel("Proporción de estudiantes")
    manejadores, etiquetas = np.atleast_1d(ejes)[0].get_legend_handles_labels()
    manejadores.append(plt.Line2D([], [], linestyle=":", color=GRIS, linewidth=1))
    etiquetas.append("Teórica $\\pi A^{t-1}$")
    figura.legend(manejadores, etiquetas, loc="lower center", ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, -0.08))
    figura.suptitle("Evolución semanal de la proporción en cada estado "
                    "(media de las réplicas e IC 95 %)", y=1.02)
    return _guardar(figura, ruta)


def _limitar_desde(eje, replicas, columna, claves, desde: int) -> None:
    """Fija el eje y con los IC desde ``desde`` réplicas (los primeros son muy anchos)."""
    bajos, altos = [], []
    for clave in claves:
        valores = replicas.loc[replicas["escenario"] == clave, columna].dropna().to_numpy()
        curva = metricas.convergencia(valores)
        bajos.append(np.nanmin(curva["ic_inferior"][desde - 1:]))
        altos.append(np.nanmax(curva["ic_superior"][desde - 1:]))
    margen = 0.15 * (max(altos) - min(bajos))
    eje.set_ylim(min(bajos) - margen, max(altos) + margen)


def convergencia_replicas(replicas: pd.DataFrame, columnas, ruta: Path) -> Path:
    """Media acumulada de cada métrica e IC 95 % al aumentar las réplicas."""
    claves = list(dict.fromkeys(replicas["escenario"]))
    filas = int(np.ceil(len(columnas) / 3))
    figura, ejes = plt.subplots(filas, 3, figsize=(10, 3.0 * filas))
    for eje, columna in zip(ejes.ravel(), columnas):
        for indice, clave in enumerate(claves):
            valores = replicas.loc[replicas["escenario"] == clave, columna].dropna()
            curva = metricas.convergencia(valores.to_numpy())
            eje.fill_between(curva["n"], curva["ic_inferior"], curva["ic_superior"],
                             color=COLORES[indice], alpha=0.2, linewidth=0)
            eje.plot(curva["n"], curva["media"], color=COLORES[indice],
                     linestyle=ESTILOS[indice % len(ESTILOS)], linewidth=1.5,
                     label=NOMBRES_ESCENARIO.get(clave, clave))
        _limitar_desde(eje, replicas, columna, claves, desde=10)
        eje.set_title(METRICAS[columna], fontsize=9)
        eje.set_xlabel("Número de réplicas")
    for eje in ejes.ravel()[len(columnas):]:
        eje.set_visible(False)
    manejadores, etiquetas = ejes.ravel()[0].get_legend_handles_labels()
    figura.legend(manejadores, etiquetas, loc="lower center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, -0.03))
    figura.suptitle("Convergencia de las métricas al aumentar las réplicas "
                    "(media acumulada e IC 95 %)")
    figura.tight_layout(rect=(0, 0.03, 1, 1))
    return _guardar(figura, ruta)


def comparacion_escenarios(resumen: pd.DataFrame, columnas, ruta: Path) -> Path:
    """Media por escenario (punto) con barra de error del IC 95 %.

    Se usan puntos y no barras porque las escalas no parten de cero (la
    log-verosimilitud es negativa) y una barra truncada exagera diferencias.
    """
    filas = int(np.ceil(len(columnas) / 3))
    figura, ejes = plt.subplots(filas, 3, figsize=(10, 3.0 * filas))
    for eje, columna in zip(ejes.ravel(), columnas):
        sub = resumen[resumen["metrica"] == columna].reset_index(drop=True)
        posiciones = np.arange(len(sub))
        for i, fila in sub.iterrows():
            error = [[fila["media"] - fila["ic_inferior"]], [fila["ic_superior"] - fila["media"]]]
            eje.errorbar(i, fila["media"], yerr=error, fmt="o", markersize=8, capsize=5,
                         color=COLORES[i], ecolor=COLORES[i], elinewidth=1.5)
            eje.annotate(f"{fila['media']:.3f}", (i, fila["media"]), xytext=(9, 0),
                         textcoords="offset points", ha="left", va="center", fontsize=7)
        eje.set_xticks(posiciones, sub["escenario"])
        eje.set_xlim(-0.5, len(sub) - 0.3)
        rango = (sub["ic_superior"].max() - sub["ic_inferior"].min()) or 1.0
        inferior = sub["ic_inferior"].min() - 0.25 * rango
        if sub["ic_inferior"].min() >= 0:
            inferior = max(0.0, inferior)
        eje.set_ylim(inferior, sub["ic_superior"].max() + 0.25 * rango)
        eje.set_title(METRICAS[columna], fontsize=9)
        eje.grid(axis="x", visible=False)
    for eje in ejes.ravel()[len(columnas):]:
        eje.set_visible(False)
    figura.suptitle("Comparación de escenarios: media e IC 95 % "
                    "(E1 Nominal, E2 Intervención, E3 Ruidosa)")
    figura.tight_layout()
    return _guardar(figura, ruta)


def _mapa(eje, matriz, titulo, filas, columnas, *, mostrar_filas=True):
    imagen = eje.imshow(matriz, cmap="Blues", vmin=0, vmax=1)
    for (i, j), valor in np.ndenumerate(matriz):
        eje.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8,
                 color="white" if valor > 0.55 else "#0b0b0b")
    eje.set_xticks(range(len(columnas)), columnas, fontsize=7)
    eje.set_yticks(range(len(filas)), filas if mostrar_filas else [], fontsize=7)
    eje.set_title(titulo, fontsize=9)
    eje.grid(False)
    return imagen


def matrices(reales, aprendidas: dict, ruta: Path) -> Path:
    """Mapas de calor de A y B reales y de las aprendidas por Baum-Welch.

    ``reales`` es ``(A, B)`` y ``aprendidas`` es ``{etiqueta: (A, B)}``.
    """
    columnas = 1 + len(aprendidas)
    figura, ejes = plt.subplots(2, columnas, figsize=(3.3 * columnas, 6.2))
    paneles = [("real", reales)] + list(aprendidas.items())
    imagen = None
    for col, (etiqueta, (a, b)) in enumerate(paneles):
        sufijo = "real" if etiqueta == "real" else f"aprendida ({etiqueta})"
        _mapa(ejes[0, col], a, f"A {sufijo}", ESTADOS, ESTADOS, mostrar_filas=col == 0)
        imagen = _mapa(ejes[1, col], b, f"B {sufijo}", ESTADOS, OBSERVACIONES,
                       mostrar_filas=col == 0)
    for eje in ejes[0]:
        eje.set_xlabel("Estado siguiente", fontsize=8)
    for eje in ejes[1]:
        eje.set_xlabel("Actividad observada", fontsize=8)
    ejes[0, 0].set_ylabel("Estado actual", fontsize=8)
    ejes[1, 0].set_ylabel("Estado oculto", fontsize=8)
    figura.colorbar(imagen, ax=ejes, shrink=0.6, label="Probabilidad")
    figura.suptitle("Matrices de transición (A) y emisión (B): reales y aprendidas "
                    "por Baum-Welch (media de las réplicas, E1)")
    return _guardar(figura, ruta)


def escalado_baum_welch(resumen: pd.DataFrame, ruta: Path) -> Path:
    """Error medio de A y B frente al tamaño de la cohorte, con IC 95 %."""
    figura, eje = plt.subplots(figsize=(6, 3.6))
    for indice, (columna, nombre) in enumerate([("error_bw_a", "Matriz A (transición)"),
                                                ("error_bw_b", "Matriz B (emisión)")]):
        sub = resumen[resumen["metrica"] == columna].sort_values("estudiantes")
        error = np.vstack([sub["media"] - sub["ic_inferior"], sub["ic_superior"] - sub["media"]])
        eje.errorbar(sub["estudiantes"], sub["media"], yerr=error, marker="o", markersize=7,
                     capsize=4, color=COLORES[indice], label=nombre)
    eje.set_xscale("log")
    eje.set_xticks(sorted(resumen["estudiantes"].unique()),
                   [str(v) for v in sorted(resumen["estudiantes"].unique())])
    eje.minorticks_off()
    eje.set_ylim(bottom=0)
    eje.set_xlabel("Estudiantes por cohorte (escala logarítmica)")
    eje.set_ylabel("Error absoluto medio")
    eje.set_title("Error de Baum-Welch según el tamaño de la cohorte (E1, media e IC 95 %)")
    eje.legend(frameon=False)
    return _guardar(figura, ruta)


def tiempos(tabla: pd.DataFrame, ruta: Path) -> Path:
    """Tiempo de simulación con bucles y vectorizada (escala log-log)."""
    figura, eje = plt.subplots(figsize=(6, 3.6))
    eje.plot(tabla["estudiantes"], tabla["segundos_bucles"], "o-", color=COLORES[1],
             markersize=7, label="Con bucles (un estudiante a la vez)")
    eje.plot(tabla["estudiantes"], tabla["segundos_vectorizada"], "s-", color=COLORES[0],
             markersize=7, label="Vectorizada (NumPy)")
    for _, fila in tabla.iterrows():
        eje.annotate(f"×{fila['aceleracion']:.0f}", (fila["estudiantes"],
                     fila["segundos_vectorizada"]), xytext=(0, -14),
                     textcoords="offset points", ha="center", fontsize=7, color=GRIS)
    eje.set_xscale("log")
    eje.set_yscale("log")
    eje.set_ylim(tabla["segundos_vectorizada"].min() / 4, tabla["segundos_bucles"].max() * 2)
    eje.set_xlabel("Estudiantes simulados (16 semanas)")
    eje.set_ylabel("Tiempo (segundos)")
    eje.set_title("Tiempo de simulación: bucles frente a vectorización")
    eje.legend(frameon=False)
    return _guardar(figura, ruta)


def sensibilidad(resumen: pd.DataFrame, ruta: Path) -> Path:
    """Tasa de deserción y exactitud por configuración, con IC 95 %."""
    paneles = [("tasa_desercion_s16", "Tasa de deserción (semana 16)"),
               ("exactitud_viterbi", "Exactitud de Viterbi"),
               ("exactitud_viterbi_nominal", "Exactitud con Viterbi nominal")]
    configuraciones = list(dict.fromkeys(resumen["configuracion"]))
    figura, ejes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
    posiciones = np.arange(len(configuraciones))[::-1]
    for indice, (eje, (columna, titulo)) in enumerate(zip(ejes, paneles)):
        sub = resumen[resumen["metrica"] == columna].set_index("configuracion")
        sub = sub.loc[configuraciones]
        error = np.vstack([sub["media"] - sub["ic_inferior"], sub["ic_superior"] - sub["media"]])
        eje.errorbar(sub["media"], posiciones, xerr=error, fmt="o", markersize=6,
                     capsize=3, color=COLORES[indice])
        eje.axvline(sub.loc["Nominal", "media"], color=GRIS, linestyle=":", linewidth=1)
        eje.set_title(titulo, fontsize=9)
        eje.set_yticks(posiciones, configuraciones, fontsize=7)
    figura.suptitle("Sensibilidad a pi, A y B (media e IC 95 %; la línea punteada es el "
                    "valor nominal)")
    figura.tight_layout()
    return _guardar(figura, ruta)


def _efecto_por_entrada(eje, entradas: pd.DataFrame) -> None:
    entradas = entradas.assign(etiqueta=[
        f"{e.replace(' -> ', ' → ')} ({v:.2f})"
        for e, v in zip(entradas["entrada"], entradas["valor_nominal"])])
    orden = (entradas.assign(magnitud=entradas["cambio_desercion_simulada"].abs())
             .groupby("etiqueta", sort=False)["magnitud"].max().sort_values().index)
    posiciones = {etiqueta: k for k, etiqueta in enumerate(orden)}
    alto = 0.36
    for desplazamiento, signo, color, nombre in ((alto / 2, 1.0, COLORES[0], "+0.05"),
                                                 (-alto / 2, -1.0, COLORES[1], "−0.05")):
        sub = entradas[entradas["delta"] * signo > 0]
        y = np.array([posiciones[e] for e in sub["etiqueta"]]) + desplazamiento
        eje.barh(y, sub["cambio_desercion_simulada"], height=alto * 0.9, color=color,
                 label=f"Entrada {nombre} y fila renormalizada")
        eje.errorbar(sub["cambio_desercion_simulada"], y,
                     xerr=np.vstack([sub["desercion_simulada"] - sub["ic_inferior"],
                                     sub["ic_superior"] - sub["desercion_simulada"]]),
                     fmt="none", ecolor=GRIS, capsize=2, linewidth=1)
    eje.axvline(0, color=GRIS, linewidth=1)
    eje.set_yticks(range(len(orden)), orden, fontsize=7)
    eje.set_xlabel("Cambio en la tasa de deserción (semana 16) frente al nominal")
    eje.set_title("Una entrada de A a la vez (media e IC 95 %)", fontsize=9)
    eje.legend(loc="lower right")
    eje.grid(axis="y", visible=False)
    eje.set_axisbelow(True)


def _participacion_filas(eje, atribucion: pd.DataFrame) -> None:
    total = atribucion["cambio_total"].var(ddof=1)
    participacion = [atribucion[f"cambio_solo_{e}"].var(ddof=1) / total for e in ESTADOS][::-1]
    eje.barh([f"Fila {e}" for e in ESTADOS][::-1], participacion, color=COLORES[3], height=0.5)
    for k, valor in enumerate(participacion):
        eje.text(valor + 0.01, k, f"{100 * valor:.0f} %", va="center", fontsize=8)
    eje.set_xlim(0, max(participacion) * 1.25)
    eje.tick_params(axis="y", labelsize=8)
    eje.set_axisbelow(True)
    eje.set_xlabel("Varianza explicada por la fila sola")
    eje.set_title(f"Perturbaciones aleatorias de A (n = {len(atribucion)})", fontsize=9)
    eje.grid(axis="y", visible=False)


def sensibilidad_entradas(entradas: pd.DataFrame, atribucion: pd.DataFrame,
                          ruta: Path) -> Path:
    """Cambio en la deserción al perturbar cada entrada de A y peso de cada fila."""
    figura, (izquierda, derecha) = plt.subplots(
        1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    _efecto_por_entrada(izquierda, entradas)
    _participacion_filas(derecha, atribucion)
    figura.suptitle("¿Qué entrada de A pesa más en la deserción?")
    figura.tight_layout()
    return _guardar(figura, ruta)
