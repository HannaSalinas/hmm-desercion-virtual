"""Ejecuta todos los experimentos: ``python -m hmm_desercion [--rapido]``.

Escribe en ``resultados/``: CSV con datos crudos y resúmenes, figuras a 300 dpi
y ``reporte.txt`` con todos los números y las versiones del entorno.
"""

from __future__ import annotations

import argparse
import platform
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from hmm_desercion import SEMILLA, experimentos as ex, graficas
from hmm_desercion.escenarios import escenarios_principales
from hmm_desercion.modelo import DESCONECTADO, ESTADOS
from hmm_desercion.tiempos import comparar_tiempos

COLUMNAS_ESCENARIO = ["tasa_desercion_s16", "semana_primera_desconexion",
                      "prop_nunca_desconecta", "exactitud_viterbi", "log_verosimilitud",
                      "error_bw_a", "error_bw_b", "error_bw"]
COLUMNAS_FIGURAS = ["tasa_desercion_s16", "semana_primera_desconexion",
                    "prop_nunca_desconecta", "exactitud_viterbi", "log_verosimilitud",
                    "error_bw"]
COLUMNAS_SENSIBILIDAD = ["tasa_desercion_s16", "semana_primera_desconexion",
                         "prop_nunca_desconecta", "exactitud_viterbi",
                         "exactitud_viterbi_nominal", "log_verosimilitud"]
LIBRERIAS = ("numpy", "scipy", "pandas", "matplotlib", "hmmlearn", "pytest", "pylint")


def _versiones() -> list[str]:
    lineas = [f"Python {platform.python_version()} ({platform.system()} {platform.machine()})"]
    lineas += [f"{nombre} {metadata.version(nombre)}" for nombre in LIBRERIAS]
    return lineas


def _matriz_texto(matriz: np.ndarray) -> str:
    return "\n".join("    [" + ", ".join(f"{v:.4f}" for v in fila) + "]" for fila in matriz)


def _tabla_texto(resumen: pd.DataFrame, grupo: str) -> str:
    columnas = [grupo, "metrica", "n", "media", "desviacion", "minimo", "maximo",
                "ic_inferior", "ic_superior"]
    return resumen[columnas].to_string(index=False, float_format=lambda v: f"{v:.4f}")


def _elegir_ejemplo(cohortes: ex.Cohortes) -> int:
    """Primer estudiante de la réplica 0 que pasa por los tres estados."""
    estados = cohortes.estados[0]
    for indice, fila in enumerate(estados):
        if len(np.unique(fila)) == 3:
            return indice
    return 0


def _matrices_a_csv(matrices: dict, ruta: Path) -> None:
    """Guarda ``{origen: (A, B)}`` en formato largo."""
    filas = []
    for origen, pares in matrices.items():
        for nombre, matriz in zip(("A", "B"), pares):
            for (i, j), valor in np.ndenumerate(matriz):
                filas.append({"matriz": nombre, "origen": origen, "fila": i, "columna": j,
                              "valor": valor})
    pd.DataFrame(filas).to_csv(ruta, index=False)


def _matrices_desde_csv(ruta: Path) -> dict:
    tabla = pd.read_csv(ruta, dtype={"origen": str})
    resultado = {}
    for origen in dict.fromkeys(tabla["origen"]):
        pares = []
        for nombre in ("A", "B"):
            sub = tabla[(tabla["origen"] == origen) & (tabla["matriz"] == nombre)]
            matriz = np.zeros((sub["fila"].max() + 1, sub["columna"].max() + 1))
            matriz[sub["fila"], sub["columna"]] = sub["valor"]
            pares.append(matriz)
        resultado[origen] = tuple(pares)
    return resultado


def generar_figuras(datos: Path, figuras: Path) -> None:
    """Construye todas las figuras leyendo solo los CSV de ``datos``."""
    secuencia = pd.read_csv(datos / "secuencia_ejemplo.csv")
    graficas.secuencia_ejemplo(secuencia["estado_real"].to_numpy(),
                               secuencia["actividad"].to_numpy(),
                               secuencia["estado_viterbi"].to_numpy(),
                               figuras / "fig1_secuencia_ejemplo.png")
    graficas.evolucion_estados(pd.read_csv(datos / "proporcion_semanal.csv"),
                               figuras / "fig2_evolucion_estados.png")
    graficas.convergencia_replicas(pd.read_csv(datos / "replicas_escenarios.csv"),
                                   COLUMNAS_FIGURAS, figuras / "fig3_convergencia.png")
    graficas.comparacion_escenarios(pd.read_csv(datos / "resumen_escenarios.csv"),
                                    COLUMNAS_FIGURAS, figuras / "fig4_comparacion_escenarios.png")
    matrices = _matrices_desde_csv(datos / "matrices_baum_welch.csv")
    reales = matrices.pop("real")
    graficas.matrices(reales, {f"{k} est.": v for k, v in matrices.items()},
                      figuras / "fig5_matrices.png")
    graficas.escalado_baum_welch(pd.read_csv(datos / "resumen_baum_welch_tamano.csv"),
                                 figuras / "fig6_baum_welch_tamano.png")
    graficas.tiempos(pd.read_csv(datos / "tiempos.csv"), figuras / "fig7_tiempos.png")
    graficas.sensibilidad(pd.read_csv(datos / "resumen_sensibilidad.csv"),
                          figuras / "fig8_sensibilidad.png")
    graficas.sensibilidad_entradas(pd.read_csv(datos / "sensibilidad_entradas.csv"),
                                   pd.read_csv(datos / "atribucion_filas.csv"),
                                   figuras / "fig9_sensibilidad_entradas.png")


def configurar(rapido: bool) -> dict:
    """Tamaños del experimento completo o de una corrida rápida de prueba."""
    if rapido:
        return {"replicas": 20, "replicas_bw": 4, "tamanos_bw": (100, 300),
                "perturbaciones": 3, "replicas_perturbacion": 5,
                "tamanos_tiempo": (20, 50), "repeticiones": 1,
                "longitudes": (16, 100, 1000)}
    return {"replicas": ex.N_REPLICAS, "replicas_bw": ex.N_REPLICAS_BW,
            "tamanos_bw": ex.TAMANOS_BW, "perturbaciones": 30, "replicas_perturbacion": 50,
            "tamanos_tiempo": (100, 300, 1000, 3000, 10000), "repeticiones": 3,
            "longitudes": (16, 50, 100, 200, 300, 400, 500, 750, 1000, 5000)}


def ejecutar(salida: Path, rapido: bool = False) -> Path:  # pylint: disable=too-many-locals
    """Corre todo y devuelve la ruta del reporte."""
    cfg = configurar(rapido)
    datos, figuras = salida / "datos", salida / "figuras"
    datos.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()

    print("1/6 Escenarios E1, E2 y E3 ...", flush=True)
    resultados = [ex.ejecutar_escenario(e, cfg["replicas"], cfg["replicas_bw"])
                  for e in escenarios_principales()]
    replicas = pd.concat([r.replicas for r in resultados], ignore_index=True)
    resumen = ex.resumir(replicas, "escenario", COLUMNAS_ESCENARIO)
    proporciones = ex.proporciones_semanales(resultados)
    replicas.to_csv(datos / "replicas_escenarios.csv", index=False)
    resumen.to_csv(datos / "resumen_escenarios.csv", index=False)
    proporciones.to_csv(datos / "proporcion_semanal.csv", index=False)

    print("2/6 Baum-Welch por tamaño de cohorte ...", flush=True)
    e1 = resultados[0].escenario
    escalado, medias_bw = ex.escalado_baum_welch(e1, cfg["tamanos_bw"], cfg["replicas_bw"])
    resumen_escalado = ex.resumir(escalado, "estudiantes",
                                  ["error_bw_a", "error_bw_b", "error_bw", "iteraciones_bw"])
    escalado.to_csv(datos / "baum_welch_tamano.csv", index=False)
    resumen_escalado.to_csv(datos / "resumen_baum_welch_tamano.csv", index=False)

    print("3/6 Sensibilidad ...", flush=True)
    sens = ex.sensibilidad_determinista(cfg["replicas"])
    resumen_sens = ex.resumir(sens, "configuracion", COLUMNAS_SENSIBILIDAD)
    sens_aleatoria = ex.sensibilidad_aleatoria(cfg["perturbaciones"], 0.05,
                                               cfg["replicas_perturbacion"])
    sens.to_csv(datos / "sensibilidad_replicas.csv", index=False)
    resumen_sens.to_csv(datos / "resumen_sensibilidad.csv", index=False)
    sens_aleatoria.to_csv(datos / "sensibilidad_aleatoria.csv", index=False)
    entradas = ex.sensibilidad_por_entrada(0.05, cfg["replicas"])
    atribucion = ex.atribucion_por_fila(sens_aleatoria)
    entradas.to_csv(datos / "sensibilidad_entradas.csv", index=False)
    atribucion.to_csv(datos / "atribucion_filas.csv", index=False)

    print("4/6 Viterbi de la plantilla y tiempos ...", flush=True)
    modelo_e1 = e1.modelo()
    plantilla = ex.viterbi_frente_a_plantilla(modelo_e1, cfg["longitudes"])
    plantilla.to_csv(datos / "viterbi_plantilla.csv", index=False)
    tabla_tiempos = pd.DataFrame(comparar_tiempos(modelo_e1, cfg["tamanos_tiempo"],
                                                  ex.N_SEMANAS, SEMILLA, cfg["repeticiones"]))
    tabla_tiempos.to_csv(datos / "tiempos.csv", index=False)

    cohortes_e1 = resultados[0].cohortes
    ejemplo = _elegir_ejemplo(cohortes_e1)
    pd.DataFrame({"semana": np.arange(1, ex.N_SEMANAS + 1),
                  "estado_real": cohortes_e1.estados[0, ejemplo],
                  "actividad": cohortes_e1.observaciones[0, ejemplo],
                  "estado_viterbi": cohortes_e1.inferidos[0, ejemplo]}
                 ).to_csv(datos / "secuencia_ejemplo.csv", index=False)
    tamanos = cfg["tamanos_bw"]
    _matrices_a_csv({"real": (e1.a, e1.b), str(tamanos[0]): medias_bw[tamanos[0]],
                     str(tamanos[-1]): medias_bw[tamanos[-1]]},
                    datos / "matrices_baum_welch.csv")

    print("5/6 Figuras ...", flush=True)
    generar_figuras(datos, figuras)

    print("6/6 Reporte ...", flush=True)
    contexto = {"cfg": cfg, "resultados": resultados, "resumen": resumen,
                "proporciones": proporciones, "resumen_escalado": resumen_escalado,
                "medias_bw": medias_bw, "resumen_sens": resumen_sens,
                "sens_aleatoria": sens_aleatoria, "entradas": entradas,
                "atribucion": atribucion, "plantilla": plantilla,
                "tiempos": tabla_tiempos, "ejemplo": ejemplo,
                "duracion": time.perf_counter() - inicio}
    ruta = salida / "reporte.txt"
    ruta.write_text(construir_reporte(contexto), encoding="utf-8")
    print(f"Listo en {contexto['duracion']:.0f} s: {ruta}", flush=True)
    return ruta


def _seccion(titulo: str) -> list[str]:
    return ["", "=" * 78, titulo, "=" * 78]


def _seccion_entradas(ctx: dict) -> list[str]:
    """Efecto de cada entrada de A y atribución por fila de las perturbaciones aleatorias."""
    entradas = ctx["entradas"]
    lineas = _seccion("Sensibilidad por entrada de A (+/-0.05 en una entrada y fila "
                      f"renormalizada; {ctx['cfg']['replicas']} réplicas)")
    lineas.append(f"Deserción teórica nominal: {ex.desercion_teorica(ex.A_NOMINAL):.4f}")
    columnas = ["entrada", "valor_nominal", "delta", "valor_nuevo", "cambio_efectivo",
                "cambio_relativo", "desercion_teorica", "desercion_simulada", "ic_inferior",
                "ic_superior", "cambio_desercion_simulada", "efecto_por_unidad"]
    orden = entradas.assign(m=entradas["cambio_desercion_teorica"].abs()).sort_values(
        "m", ascending=False)
    lineas.append(orden[columnas].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    aleatoria = ctx["sens_aleatoria"]
    sub = aleatoria[aleatoria["matriz"] == "A"]
    atribucion = ctx["atribucion"]
    total = atribucion["cambio_total"].var(ddof=1)
    lineas += _seccion("Atribución por fila de las perturbaciones aleatorias de A (teórica)")
    lineas.append("Diferencia máxima |simulada - teórica| en la deserción: "
                  f"{(sub['tasa_desercion_s16'] - sub['desercion_teorica']).abs().max():.4f}")
    for estado in ESTADOS:
        columna = atribucion[f"cambio_solo_{estado}"]
        lineas.append(f"Solo fila {estado}: varianza explicada {columna.var(ddof=1) / total:.4f}; "
                      f"|cambio| medio {columna.abs().mean():.4f}")
    cambios = sub[[f"m_{i}{j}" for i in range(3) for j in range(3)]] - ex.A_NOMINAL.ravel()
    correlacion = cambios.corrwith(sub["tasa_desercion_s16"])
    lineas.append("Correlación entre el cambio de cada entrada y la deserción simulada:")
    lineas += [f"    {ESTADOS[int(c[2])]} -> {ESTADOS[int(c[3])]}: {v:+.4f}"
               for c, v in correlacion.items()]
    return lineas


def construir_reporte(ctx: dict) -> str:  # pylint: disable=too-many-locals
    """Texto plano con todos los números del experimento."""
    cfg = ctx["cfg"]
    lineas = ["REPORTE DE RESULTADOS: HMM PARA EL RIESGO DE DESERCIÓN EN EDUCACIÓN VIRTUAL",
              "Parámetros supuestos e ilustrativos (no estimados de datos reales)."]
    lineas += _seccion("Entorno")
    lineas += _versiones()
    lineas += [f"Duración total: {ctx['duracion']:.1f} s"]

    lineas += _seccion("Diseño")
    lineas += [f"Semilla base: {SEMILLA} (SeedSequence -> flujos -> semillas por réplica)",
               f"Réplicas por escenario: {cfg['replicas']}; estudiantes por cohorte: "
               f"{ex.N_ESTUDIANTES}; semanas: {ex.N_SEMANAS}",
               f"Baum-Welch: {cfg['replicas_bw']} réplicas; pi fijo; se aprenden A y B; "
               "tol = 1e-4 en la log-verosimilitud total; máximo 1000 iteraciones",
               "IC 95 %: t de Student con n-1 grados de libertad"]

    for res in ctx["resultados"]:
        esc = res.escenario
        modelo = esc.modelo()
        lineas += _seccion(f"{esc.clave} {esc.nombre}: {esc.descripcion}")
        lineas += ["pi = " + str(np.round(esc.pi, 4).tolist()), "A =", _matriz_texto(esc.a),
                   "B =", _matriz_texto(esc.b)]
        estacionaria = modelo.distribucion_estacionaria()
        teorica_16 = modelo.distribucion_por_semana(ex.N_SEMANAS)[-1]
        lineas += ["Distribución estacionaria de A: " + ", ".join(
            f"{n} {v:.4f}" for n, v in zip(ESTADOS, estacionaria)),
            f"Proporción teórica en Desconectado en la semana 16 (pi A^15): "
            f"{teorica_16[DESCONECTADO]:.4f}",
            "A aprendida (media de réplicas):", _matriz_texto(res.a_aprendidas.mean(axis=0)),
            "B aprendida (media de réplicas):", _matriz_texto(res.b_aprendidas.mean(axis=0))]

    lineas += _seccion("Resumen por escenario")
    lineas.append(_tabla_texto(ctx["resumen"], "escenario"))

    semana16 = ctx["proporciones"][ctx["proporciones"]["semana"] == ex.N_SEMANAS]
    lineas += _seccion("Proporción por estado en la semana 16 (simulada frente a teórica)")
    lineas.append(semana16[["escenario", "estado", "media", "ic_inferior", "ic_superior",
                            "teorica"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    lineas += _seccion("Baum-Welch según el tamaño de la cohorte (E1)")
    lineas.append(_tabla_texto(ctx["resumen_escalado"], "estudiantes"))
    for tamano, (a, b) in ctx["medias_bw"].items():
        lineas += [f"A aprendida media, {tamano} estudiantes:", _matriz_texto(a),
                   f"B aprendida media, {tamano} estudiantes:", _matriz_texto(b)]

    lineas += _seccion("Sensibilidad determinista (P3)")
    lineas.append(_tabla_texto(ctx["resumen_sens"], "configuracion"))
    aleatoria = ctx["sens_aleatoria"]
    lineas += _seccion(f"Sensibilidad a perturbaciones aleatorias (+/-0.05, "
                       f"{cfg['perturbaciones']} por matriz, "
                       f"{cfg['replicas_perturbacion']} réplicas cada una)")
    columnas = ["cambio_max_absoluto", "tasa_desercion_s16", "exactitud_viterbi",
                "exactitud_viterbi_nominal", "log_verosimilitud"]
    lineas.append(aleatoria.groupby("matriz")[columnas].agg(["mean", "std", "min", "max"])
                  .T.to_string(float_format=lambda v: f"{v:.4f}"))

    lineas += _seccion_entradas(ctx)

    lineas += _seccion("Viterbi de la plantilla (sin logaritmos) frente al logarítmico")
    lineas.append(ctx["plantilla"].to_string(index=False))

    lineas += _seccion("Tiempos de simulación: bucles frente a vectorizada (mejor de "
                       f"{cfg['repeticiones']})")
    lineas.append(ctx["tiempos"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    lineas += _seccion("Figura 1")
    lineas.append(f"Estudiante {ctx['ejemplo']} de la réplica 0 de E1 (primero que pasa "
                  "por los tres estados).")
    return "\n".join(lineas) + "\n"


def main() -> None:
    """Punto de entrada de la línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rapido", action="store_true",
                        help="corrida pequeña para comprobar que todo funciona")
    parser.add_argument("--solo-figuras", action="store_true",
                        help="regenera las figuras desde los CSV existentes")
    parser.add_argument("--salida", type=Path, default=Path("resultados"))
    argumentos = parser.parse_args()
    if argumentos.solo_figuras:
        generar_figuras(argumentos.salida / "datos", argumentos.salida / "figuras")
    else:
        ejecutar(argumentos.salida, argumentos.rapido)


if __name__ == "__main__":
    main()
