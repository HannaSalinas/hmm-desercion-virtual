"""Métricas por réplica y resumen estadístico con intervalos de confianza.

Convenciones: los arreglos de estados tienen forma ``(R, N, T)``
(réplicas, estudiantes, semanas) y las semanas se reportan desde 1.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from hmm_desercion.modelo import DESCONECTADO


def tasa_desercion(estados: np.ndarray, semana: int = -1) -> np.ndarray:
    """Proporción de estudiantes en Desconectado en la semana dada, por réplica."""
    return (estados[..., semana] == DESCONECTADO).mean(axis=-1)


def primera_desconexion(estados: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Semana media de la primera desconexión y proporción que nunca se desconecta.

    La media se calcula solo entre quienes llegan a Desconectado (datos
    censurados aparte). Si en una réplica nadie se desconecta, la media es
    ``nan``. Devuelve dos arreglos con una entrada por réplica.
    """
    desconectado = estados == DESCONECTADO
    alguna_vez = desconectado.any(axis=-1)
    semana = np.argmax(desconectado, axis=-1).astype(float) + 1.0
    semana[~alguna_vez] = np.nan
    con_desconexion = alguna_vez.sum(axis=-1)
    suma = np.nansum(semana, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        media = np.where(con_desconexion > 0, suma / np.maximum(con_desconexion, 1), np.nan)
    return media, 1.0 - alguna_vez.mean(axis=-1)


def exactitud(estados_reales: np.ndarray, estados_inferidos: np.ndarray) -> np.ndarray:
    """Fracción de semanas-estudiante bien clasificadas, por réplica."""
    if estados_reales.shape != estados_inferidos.shape:
        raise ValueError("Los arreglos de estados deben tener la misma forma")
    return (estados_reales == estados_inferidos).mean(axis=(-2, -1))


def proporcion_por_semana(estados: np.ndarray, n_estados: int = 3) -> np.ndarray:
    """Proporción de estudiantes en cada estado por semana; forma ``(R, T, n)``."""
    return np.stack([(estados == i).mean(axis=-2) for i in range(n_estados)], axis=-1)


def resumen(valores, nivel: float = 0.95) -> dict[str, float]:
    """Media, desviación estándar muestral, mínimo, máximo e IC con t de Student.

    Se ignoran los ``nan``. Con menos de dos valores la desviación y el
    intervalo quedan en ``nan``.
    """
    datos = np.asarray(valores, dtype=float)
    datos = datos[~np.isnan(datos)]
    n = datos.size
    if n == 0:
        nan = float("nan")
        return {"n": 0, "media": nan, "desviacion": nan, "minimo": nan,
                "maximo": nan, "ic_inferior": nan, "ic_superior": nan}
    media = float(datos.mean())
    if n > 1:
        desviacion = float(datos.std(ddof=1))
        margen = float(stats.t.ppf(0.5 + nivel / 2, n - 1) * desviacion / np.sqrt(n))
    else:
        desviacion = margen = float("nan")
    return {"n": n, "media": media, "desviacion": desviacion,
            "minimo": float(datos.min()), "maximo": float(datos.max()),
            "ic_inferior": media - margen, "ic_superior": media + margen}


def convergencia(valores, nivel: float = 0.95) -> dict[str, np.ndarray]:
    """Media acumulada e IC al usar las primeras 1, 2, ..., R réplicas."""
    datos = np.asarray(valores, dtype=float)
    n = np.arange(1, datos.size + 1)
    media = np.cumsum(datos) / n
    cuadrados = np.cumsum(datos ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        varianza = (cuadrados - n * media ** 2) / (n - 1)
        varianza = np.clip(varianza, 0.0, None)
        cuantil = stats.t.ppf(0.5 + nivel / 2, np.maximum(n - 1, 1))
        margen = np.where(n > 1, cuantil * np.sqrt(varianza / n), np.nan)
    return {"n": n, "media": media, "ic_inferior": media - margen,
            "ic_superior": media + margen}
