"""Comparación de tiempos: simulación con bucles frente a vectorizada."""

from __future__ import annotations

import time

import numpy as np

from hmm_desercion.modelo import HMM


def _cronometrar(funcion, repeticiones: int) -> float:
    """Mejor tiempo (en segundos) de varias ejecuciones."""
    mejor = float("inf")
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion()
        mejor = min(mejor, time.perf_counter() - inicio)
    return mejor


def comparar_tiempos(modelo: HMM, tamanos, n_semanas: int, semilla: int,
                     repeticiones: int = 3) -> list[dict[str, float]]:
    """Mide ambas versiones para cada número de estudiantes en ``tamanos``."""
    filas = []
    for n_estudiantes in tamanos:
        t_bucles = _cronometrar(
            lambda n=n_estudiantes: modelo.simular_con_bucles(
                n, n_semanas, np.random.default_rng(semilla)), repeticiones)
        t_vector = _cronometrar(
            lambda n=n_estudiantes: modelo.simular(
                n, n_semanas, np.random.default_rng(semilla)), repeticiones)
        filas.append({"estudiantes": int(n_estudiantes), "semanas": n_semanas,
                      "segundos_bucles": t_bucles, "segundos_vectorizada": t_vector,
                      "aceleracion": t_bucles / t_vector})
    return filas
