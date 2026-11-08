"""Diseño experimental: escenarios, Baum-Welch por tamaño de cohorte y sensibilidad.

Semillas. La semilla base es ``SEMILLA = 2026``; ``np.random.default_rng(2026)``
se construye sobre ``SeedSequence(2026)``. De ella se derivan flujos
independientes (uno por experimento) y, dentro de cada flujo, una semilla por
réplica. Todos los escenarios usan las mismas semillas por réplica (números
aleatorios comunes): las diferencias entre escenarios se deben a los
parámetros y no al azar de la muestra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hmm_desercion import SEMILLA, metricas
from hmm_desercion.baum_welch import baum_welch, error_absoluto_medio
from hmm_desercion.escenarios import (A_NOMINAL, B_NOMINAL, PI_NOMINAL, Escenario,
                                      perturbar_aleatoria, perturbar_diagonal,
                                      perturbar_entrada)
from hmm_desercion.guia_base import viterbi_sin_logaritmos
from hmm_desercion.modelo import DESCONECTADO, ESTADOS, HMM

N_REPLICAS = 200
N_ESTUDIANTES = 300
N_SEMANAS = 16
N_REPLICAS_BW = 50
TAMANOS_BW = (300, 1000, 3000)

FLUJO_ESCENARIOS = 0
FLUJO_ESCALADO_BW = 1
FLUJO_PERTURBACIONES = 2
FLUJO_PLANTILLA = 3

METRICAS = {
    "tasa_desercion_s16": "Tasa de deserción (semana 16)",
    "semana_primera_desconexion": "Semana media de primera desconexión",
    "prop_nunca_desconecta": "Proporción que nunca se desconecta",
    "exactitud_viterbi": "Exactitud de Viterbi",
    "log_verosimilitud": "Log-verosimilitud media por estudiante",
    "error_bw_a": "Error absoluto medio de A (Baum-Welch)",
    "error_bw_b": "Error absoluto medio de B (Baum-Welch)",
    "error_bw": "Error absoluto medio de A y B (Baum-Welch)",
}


def semillas(flujo: int, n_replicas: int) -> list[np.random.SeedSequence]:
    """Semillas por réplica del flujo ``flujo``, derivadas de ``SEMILLA``."""
    return np.random.SeedSequence(SEMILLA, spawn_key=(flujo,)).spawn(n_replicas)


def generadores(flujo: int, n_replicas: int) -> list[np.random.Generator]:
    """Un ``Generator`` nuevo por réplica (siempre los mismos para un flujo)."""
    return [np.random.default_rng(s) for s in semillas(flujo, n_replicas)]


@dataclass
class Cohortes:
    """Resultado de simular y analizar R cohortes con un mismo modelo."""

    estados: np.ndarray
    observaciones: np.ndarray
    inferidos: np.ndarray
    replicas: pd.DataFrame


def analizar_cohortes(modelo: HMM, n_replicas: int = N_REPLICAS, *,
                      n_estudiantes: int = N_ESTUDIANTES, n_semanas: int = N_SEMANAS,
                      flujo: int = FLUJO_ESCENARIOS,
                      decodificador: HMM | None = None) -> Cohortes:
    """Simula R cohortes y calcula las métricas por réplica.

    ``decodificador`` es el modelo con que se infiere el estado (Viterbi) y se
    calcula la log-verosimilitud; por defecto es el mismo que generó los datos.
    """
    decodificador = modelo if decodificador is None else decodificador
    estados, obs = modelo.simular(n_estudiantes, n_semanas, generadores(flujo, n_replicas))
    inferidos, _ = decodificador.viterbi(obs)
    semana_media, nunca = metricas.primera_desconexion(estados)
    replicas = pd.DataFrame({
        "replica": np.arange(n_replicas),
        "tasa_desercion_s16": metricas.tasa_desercion(estados),
        "semana_primera_desconexion": semana_media,
        "prop_nunca_desconecta": nunca,
        "exactitud_viterbi": metricas.exactitud(estados, inferidos),
        "log_verosimilitud": decodificador.log_verosimilitud(obs).mean(axis=-1),
    })
    return Cohortes(estados, obs, inferidos, replicas)


def ajustar_baum_welch(observaciones, a_real, b_real, pi) -> tuple[pd.DataFrame, list]:
    """Baum-Welch sobre cada cohorte de ``observaciones`` (R, N, T)."""
    filas, ajustes = [], []
    for replica, obs in enumerate(observaciones):
        ajuste = baum_welch(obs, pi)
        ajustes.append(ajuste)
        error_a = error_absoluto_medio(ajuste.a, a_real)
        error_b = error_absoluto_medio(ajuste.b, b_real)
        filas.append({"replica": replica, "error_bw_a": error_a, "error_bw_b": error_b,
                      "error_bw": (error_a + error_b) / 2,
                      "iteraciones_bw": ajuste.iteraciones,
                      "convergio_bw": ajuste.convergio})
    return pd.DataFrame(filas), ajustes


@dataclass
class ResultadoEscenario:
    """Métricas por réplica, proporciones semanales y matrices aprendidas."""

    escenario: Escenario
    cohortes: Cohortes
    replicas: pd.DataFrame
    a_aprendidas: np.ndarray
    b_aprendidas: np.ndarray


def ejecutar_escenario(escenario: Escenario, n_replicas: int = N_REPLICAS,
                       n_replicas_bw: int = N_REPLICAS_BW,
                       n_estudiantes: int = N_ESTUDIANTES) -> ResultadoEscenario:
    """Todas las métricas de un escenario; Baum-Welch en las primeras réplicas."""
    modelo = escenario.modelo()
    cohortes = analizar_cohortes(modelo, n_replicas, n_estudiantes=n_estudiantes)
    bw, ajustes = ajustar_baum_welch(cohortes.observaciones[:n_replicas_bw],
                                     escenario.a, escenario.b, escenario.pi)
    replicas = cohortes.replicas.merge(bw, on="replica", how="left")
    replicas.insert(0, "escenario", escenario.clave)
    return ResultadoEscenario(escenario, cohortes, replicas,
                              np.array([a.a for a in ajustes]),
                              np.array([a.b for a in ajustes]))


def escalado_baum_welch(escenario: Escenario, tamanos=TAMANOS_BW,
                        n_replicas: int = N_REPLICAS_BW) -> tuple[pd.DataFrame, dict]:
    """Error de Baum-Welch al crecer la cohorte.

    En cada réplica se simula la cohorte más grande y las menores son sus
    primeros estudiantes (cohortes anidadas), así el efecto observado es el
    del tamaño y no el de muestras distintas.
    """
    modelo = escenario.modelo()
    _, obs = modelo.simular(max(tamanos), N_SEMANAS, generadores(FLUJO_ESCALADO_BW, n_replicas))
    tablas, medias = [], {}
    for tamano in tamanos:
        tabla, ajustes = ajustar_baum_welch(obs[:, :tamano], escenario.a, escenario.b,
                                            escenario.pi)
        tabla.insert(0, "estudiantes", tamano)
        tablas.append(tabla)
        medias[tamano] = (np.mean([a.a for a in ajustes], axis=0),
                          np.mean([a.b for a in ajustes], axis=0))
    return pd.concat(tablas, ignore_index=True), medias


def configuraciones_sensibilidad() -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """Variaciones deterministas de pi y perturbaciones de la diagonal de A y B."""
    return [
        ("Nominal", "Nominal", PI_NOMINAL, A_NOMINAL, B_NOMINAL),
        ("pi", "pi optimista [0.85, 0.12, 0.03]",
         np.array([0.85, 0.12, 0.03]), A_NOMINAL, B_NOMINAL),
        ("pi", "pi pesimista [0.50, 0.35, 0.15]",
         np.array([0.50, 0.35, 0.15]), A_NOMINAL, B_NOMINAL),
        ("pi", "pi uniforme [1/3, 1/3, 1/3]", np.full(3, 1 / 3), A_NOMINAL, B_NOMINAL),
        ("A", "diag(A) + 0.05", PI_NOMINAL, perturbar_diagonal(A_NOMINAL, 0.05), B_NOMINAL),
        ("A", "diag(A) - 0.05", PI_NOMINAL, perturbar_diagonal(A_NOMINAL, -0.05), B_NOMINAL),
        ("B", "diag(B) + 0.05", PI_NOMINAL, A_NOMINAL, perturbar_diagonal(B_NOMINAL, 0.05)),
        ("B", "diag(B) - 0.05", PI_NOMINAL, A_NOMINAL, perturbar_diagonal(B_NOMINAL, -0.05)),
    ]


def sensibilidad_determinista(n_replicas: int = N_REPLICAS) -> pd.DataFrame:
    """Métricas por réplica para cada configuración determinista.

    Además de la exactitud con el modelo correcto, se mide la exactitud cuando
    Viterbi usa los parámetros nominales sobre datos generados con los
    parámetros perturbados (modelo mal especificado).
    """
    nominal = HMM(PI_NOMINAL, A_NOMINAL, B_NOMINAL)
    tablas = []
    for grupo, nombre, pi, a, b in configuraciones_sensibilidad():
        modelo = HMM(pi, a, b)
        cohortes = analizar_cohortes(modelo, n_replicas)
        inferidos_nominal, _ = nominal.viterbi(cohortes.observaciones)
        tabla = cohortes.replicas.copy()
        tabla["exactitud_viterbi_nominal"] = metricas.exactitud(cohortes.estados,
                                                                inferidos_nominal)
        tabla.insert(0, "configuracion", nombre)
        tabla.insert(0, "grupo", grupo)
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True)


def desercion_teorica(a, pi=PI_NOMINAL, n_semanas: int = N_SEMANAS) -> float:
    """Proporción esperada en Desconectado en la última semana: (pi A^(T-1))_D."""
    return float((pi @ np.linalg.matrix_power(a, n_semanas - 1))[DESCONECTADO])


def matrices_perturbadas(matriz, n_perturbaciones: int, escala: float) -> list[np.ndarray]:
    """Perturbaciones aleatorias reproducibles de ``matriz`` (flujo de perturbaciones)."""
    gen = np.random.default_rng(semillas(FLUJO_PERTURBACIONES, 1)[0])
    return [perturbar_aleatoria(matriz, escala, gen) for _ in range(n_perturbaciones)]


def _columnas_matriz(matriz: np.ndarray) -> dict:
    return {f"m_{i}{j}": float(v) for (i, j), v in np.ndenumerate(matriz)}


def sensibilidad_aleatoria(n_perturbaciones: int = 30, escala: float = 0.05,
                           n_replicas: int = 50) -> pd.DataFrame:
    """Perturbaciones aleatorias pequeñas de A o de B (ruido uniforme +/- escala).

    Cada fila es una perturbación con la media de sus ``n_replicas`` réplicas,
    la matriz perturbada (columnas ``m_ij``) y la deserción teórica.
    """
    nominal = HMM(PI_NOMINAL, A_NOMINAL, B_NOMINAL)
    filas = []
    for matriz, base in (("A", A_NOMINAL), ("B", B_NOMINAL)):
        for indice, perturbada in enumerate(matrices_perturbadas(base, n_perturbaciones,
                                                                 escala)):
            a, b = (perturbada, B_NOMINAL) if matriz == "A" else (A_NOMINAL, perturbada)
            modelo = HMM(PI_NOMINAL, a, b)
            cohortes = analizar_cohortes(modelo, n_replicas)
            inferidos_nominal, _ = nominal.viterbi(cohortes.observaciones)
            medias = cohortes.replicas.drop(columns="replica").mean()
            filas.append({"matriz": matriz, "perturbacion": indice,
                          "cambio_max_absoluto": float(np.max(np.abs(perturbada - base))),
                          **medias.to_dict(),
                          "exactitud_viterbi_nominal": float(metricas.exactitud(
                              cohortes.estados, inferidos_nominal).mean()),
                          "desercion_teorica": desercion_teorica(a),
                          **_columnas_matriz(perturbada)})
    return pd.DataFrame(filas)


def sensibilidad_por_entrada(delta: float = 0.05,
                             n_replicas: int = N_REPLICAS) -> pd.DataFrame:
    """Efecto en la deserción de perturbar una sola entrada de A en +/- ``delta``.

    Cada entrada se perturba con ``perturbar_entrada`` (suma y renormalización
    de la fila) y se simula con las mismas semillas que los escenarios.
    """
    base = desercion_teorica(A_NOMINAL)
    filas = []
    for (i, j), valor in np.ndenumerate(A_NOMINAL):
        for signo in (1.0, -1.0):
            a = perturbar_entrada(A_NOMINAL, i, j, signo * delta)
            tasa = analizar_cohortes(HMM(PI_NOMINAL, a, B_NOMINAL),
                                     n_replicas).replicas["tasa_desercion_s16"]
            resumen = metricas.resumen(tasa.to_numpy())
            cambio = a[i, j] - valor
            teorica = desercion_teorica(a)
            filas.append({"fila": i, "columna": j,
                          "entrada": f"{ESTADOS[i]} -> {ESTADOS[j]}",
                          "valor_nominal": valor, "delta": signo * delta,
                          "valor_nuevo": a[i, j], "cambio_efectivo": cambio,
                          "cambio_relativo": cambio / valor,
                          "desercion_teorica": teorica,
                          "cambio_desercion_teorica": teorica - base,
                          "efecto_por_unidad": (teorica - base) / cambio,
                          "desercion_simulada": resumen["media"],
                          "ic_inferior": resumen["ic_inferior"],
                          "ic_superior": resumen["ic_superior"],
                          "cambio_desercion_simulada": resumen["media"] - base})
    return pd.DataFrame(filas)


def atribucion_por_fila(aleatoria: pd.DataFrame) -> pd.DataFrame:
    """Cambio teórico de la deserción si solo se aplica la perturbación de una fila de A.

    Usa las matrices guardadas en ``sensibilidad_aleatoria`` (``matriz == "A"``).
    """
    base = desercion_teorica(A_NOMINAL)
    filas = []
    for _, registro in aleatoria[aleatoria["matriz"] == "A"].iterrows():
        perturbada = np.array([[registro[f"m_{i}{j}"] for j in range(3)] for i in range(3)])
        fila = {"perturbacion": int(registro["perturbacion"]),
                "cambio_total": desercion_teorica(perturbada) - base}
        for k, estado in enumerate(ESTADOS):
            parcial = A_NOMINAL.copy()
            parcial[k] = perturbada[k]
            fila[f"cambio_solo_{estado}"] = desercion_teorica(parcial) - base
        filas.append(fila)
    return pd.DataFrame(filas)


def viterbi_frente_a_plantilla(modelo: HMM, longitudes) -> pd.DataFrame:
    """Compara el Viterbi de la plantilla con el logarítmico al crecer T."""
    filas = []
    gen = np.random.default_rng(semillas(FLUJO_PLANTILLA, 1)[0])
    _, obs = modelo.simular(1, max(longitudes), gen)
    for longitud in longitudes:
        tramo = obs[0, :longitud]
        ruta, log_prob = modelo.viterbi(tramo)
        ruta_p, prob_p = viterbi_sin_logaritmos(tramo, modelo.a, modelo.b, modelo.pi)
        filas.append({"semanas": longitud, "log_prob_logaritmico": float(log_prob),
                      "prob_plantilla": prob_p,
                      "subdesborda": prob_p == 0.0,
                      "coincidencia_ruta": float(np.mean(ruta == ruta_p))})
    return pd.DataFrame(filas)


def resumir(tabla: pd.DataFrame, grupo: str, columnas) -> pd.DataFrame:
    """Resumen (media, desviación, mín., máx., IC 95 %) por grupo y métrica."""
    filas = []
    for clave, sub in tabla.groupby(grupo, sort=False):
        for columna in columnas:
            if columna in sub:
                filas.append({grupo: clave, "metrica": columna,
                              **metricas.resumen(sub[columna].to_numpy(dtype=float))})
    return pd.DataFrame(filas)


def proporciones_semanales(resultados: list[ResultadoEscenario]) -> pd.DataFrame:
    """Proporción por semana y estado con IC 95 % y el valor teórico pi A^(t-1)."""
    filas = []
    for res in resultados:
        prop = metricas.proporcion_por_semana(res.cohortes.estados)
        teorica = res.escenario.modelo().distribucion_por_semana(prop.shape[1])
        for semana in range(prop.shape[1]):
            for estado in range(prop.shape[2]):
                filas.append({"escenario": res.escenario.clave, "semana": semana + 1,
                              "estado": estado, "teorica": teorica[semana, estado],
                              **metricas.resumen(prop[:, semana, estado])})
    return pd.DataFrame(filas)
