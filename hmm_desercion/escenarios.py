"""Parámetros nominales, escenarios experimentales y perturbaciones.

Todos los valores son supuestos e ilustrativos: no se estimaron con datos
reales de estudiantes. Su justificación cualitativa está en el artículo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hmm_desercion.modelo import HMM, validar_estocastica

PI_NOMINAL = np.array([0.70, 0.25, 0.05])
A_NOMINAL = np.array([[0.85, 0.12, 0.03],
                      [0.30, 0.55, 0.15],
                      [0.05, 0.10, 0.85]])
B_NOMINAL = np.array([[0.70, 0.25, 0.05],
                      [0.20, 0.50, 0.30],
                      [0.03, 0.17, 0.80]])

# E2: la tutoría actúa sobre la fila "En riesgo": sube la recuperación
# (0.30 -> 0.45) y baja el paso a Desconectado (0.15 -> 0.08).
A_INTERVENCION = np.array([[0.85, 0.12, 0.03],
                           [0.45, 0.47, 0.08],
                           [0.05, 0.10, 0.85]])

# E3: emisiones más difusas; la actividad semanal discrimina menos.
B_RUIDOSA = np.array([[0.50, 0.30, 0.20],
                      [0.30, 0.40, 0.30],
                      [0.15, 0.30, 0.55]])


@dataclass(frozen=True)
class Escenario:
    """Un conjunto de parámetros con nombre y descripción."""

    clave: str
    nombre: str
    descripcion: str
    pi: np.ndarray
    a: np.ndarray
    b: np.ndarray

    def modelo(self) -> HMM:
        """Construye y valida el HMM del escenario."""
        return HMM(self.pi, self.a, self.b)


def escenarios_principales() -> list[Escenario]:
    """E1 nominal, E2 intervención temprana y E3 observación ruidosa."""
    return [
        Escenario("E1", "Nominal", "Parámetros nominales de partida",
                  PI_NOMINAL, A_NOMINAL, B_NOMINAL),
        Escenario("E2", "Intervención temprana",
                  "Tutoría: En riesgo->Comprometido 0.45 y En riesgo->Desconectado 0.08",
                  PI_NOMINAL, A_INTERVENCION, B_NOMINAL),
        Escenario("E3", "Observación ruidosa",
                  "B difusa: la actividad informa menos sobre el estado real",
                  PI_NOMINAL, A_NOMINAL, B_RUIDOSA),
    ]


def perturbar_diagonal(matriz, delta: float) -> np.ndarray:
    """Suma ``delta`` a la diagonal y reparte ``-delta`` fuera de ella.

    La compensación es proporcional a los valores fuera de la diagonal, así
    cada fila sigue sumando 1. ``delta > 0`` hace los estados más
    persistentes (en ``A``) o las emisiones más informativas (en ``B``).
    """
    matriz = validar_estocastica(matriz, "matriz")
    if matriz.ndim != 2 or matriz.shape[0] > matriz.shape[1]:
        raise ValueError("Se espera una matriz con al menos tantas columnas como filas")
    resultado = matriz.copy()
    for i in range(matriz.shape[0]):
        fuera = np.delete(np.arange(matriz.shape[1]), i)
        masa_fuera = matriz[i, fuera].sum()
        nueva_diagonal = matriz[i, i] + delta
        if not 0.0 < nueva_diagonal < 1.0 or masa_fuera <= 0:
            raise ValueError(f"delta={delta} deja la fila {i} fuera de [0, 1]")
        resultado[i, i] = nueva_diagonal
        resultado[i, fuera] = matriz[i, fuera] * (1.0 - nueva_diagonal) / masa_fuera
    return validar_estocastica(resultado, "matriz perturbada")


def perturbar_aleatoria(matriz, escala: float, generador: np.random.Generator,
                        minimo: float = 1e-3) -> np.ndarray:
    """Ruido uniforme en ``[-escala, escala]`` en cada entrada y renormalización."""
    matriz = validar_estocastica(matriz, "matriz")
    ruido = generador.uniform(-escala, escala, size=matriz.shape)
    perturbada = np.clip(matriz + ruido, minimo, None)
    perturbada /= perturbada.sum(axis=-1, keepdims=True)
    return validar_estocastica(perturbada, "matriz perturbada")


def perturbar_entrada(matriz, fila: int, columna: int, delta: float,
                      minimo: float = 1e-3) -> np.ndarray:
    """Suma ``delta`` a una sola entrada y renormaliza su fila.

    Es el mismo mecanismo de ``perturbar_aleatoria`` aplicado a una entrada:
    el valor se recorta en ``minimo`` y la fila se divide por su suma, así que
    el cambio efectivo de la entrada es menor que ``delta``.
    """
    matriz = validar_estocastica(matriz, "matriz")
    perturbada = matriz.copy()
    perturbada[fila, columna] = max(perturbada[fila, columna] + delta, minimo)
    perturbada[fila] /= perturbada[fila].sum()
    return validar_estocastica(perturbada, "matriz perturbada")
