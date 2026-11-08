"""Modelo Oculto de Markov (HMM) discreto para el riesgo de deserción.

La clase :class:`HMM` reúne lo que el proyecto necesita del modelo:

* validación de que ``pi``, ``a`` (transición) y ``b`` (emisión) son estocásticas;
* simulación vectorizada de muchas réplicas y estudiantes a la vez;
* algoritmos Forward, Backward y Viterbi en espacio logarítmico;
* distribución estacionaria de la cadena oculta.

Notación (Rabiner, 1989): ``pi[i] = P(q_1 = i)``,
``a[i, j] = P(q_{t+1} = j | q_t = i)`` y ``b[i, k] = P(o_t = k | q_t = i)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp

ESTADOS = ("Comprometido", "En riesgo", "Desconectado")
OBSERVACIONES = ("Alta", "Media", "Baja")
DESCONECTADO = 2

TOLERANCIA = 1e-8


def validar_estocastica(matriz, nombre: str) -> np.ndarray:
    """Comprueba que un vector o matriz sea estocástico por filas.

    Lanza ``ValueError`` si hay valores negativos, valores no finitos o
    filas que no suman 1. Devuelve una copia en ``float64``.
    """
    arreglo = np.array(matriz, dtype=float)
    if arreglo.ndim not in (1, 2) or arreglo.size == 0:
        raise ValueError(f"{nombre} debe ser un vector o una matriz no vacía")
    if not np.all(np.isfinite(arreglo)):
        raise ValueError(f"{nombre} contiene valores no finitos")
    if np.any(arreglo < 0):
        raise ValueError(f"{nombre} contiene probabilidades negativas")
    sumas = arreglo.sum(axis=-1)
    if not np.allclose(sumas, 1.0, atol=TOLERANCIA):
        raise ValueError(f"Las filas de {nombre} deben sumar 1; suman {np.round(sumas, 6)}")
    return arreglo


def _log_seguro(arreglo: np.ndarray) -> np.ndarray:
    """Logaritmo natural que devuelve ``-inf`` en los ceros sin advertencias."""
    with np.errstate(divide="ignore"):
        return np.log(arreglo)


def _acumulada(probabilidades: np.ndarray) -> np.ndarray:
    """Suma acumulada por filas con el último valor fijado en 1 exacto."""
    acumulada = np.cumsum(probabilidades, axis=-1)
    acumulada[..., -1] = 1.0
    return acumulada


class HMM:
    """HMM discreto con ``n`` estados ocultos y ``m`` símbolos observables."""

    def __init__(self, pi, a, b, estados: Sequence[str] = ESTADOS,
                 observaciones: Sequence[str] = OBSERVACIONES) -> None:
        self.pi = validar_estocastica(pi, "pi")
        self.a = validar_estocastica(a, "A")
        self.b = validar_estocastica(b, "B")
        n_estados = self.pi.shape[0]
        if self.pi.ndim != 1:
            raise ValueError("pi debe ser un vector")
        if self.a.shape != (n_estados, n_estados):
            raise ValueError(f"A debe ser {n_estados}x{n_estados}; es {self.a.shape}")
        if self.b.ndim != 2 or self.b.shape[0] != n_estados:
            raise ValueError(f"B debe tener {n_estados} filas; tiene forma {self.b.shape}")
        if len(estados) != n_estados or len(observaciones) != self.b.shape[1]:
            raise ValueError("Los nombres no coinciden con las dimensiones del modelo")
        self.estados = tuple(estados)
        self.observaciones = tuple(observaciones)
        self.log_pi = _log_seguro(self.pi)
        self.log_a = _log_seguro(self.a)
        self.log_b = _log_seguro(self.b)

    @property
    def n_estados(self) -> int:
        """Número de estados ocultos."""
        return self.pi.shape[0]

    @property
    def n_simbolos(self) -> int:
        """Número de símbolos observables."""
        return self.b.shape[1]

    def __repr__(self) -> str:
        return f"HMM(n_estados={self.n_estados}, n_simbolos={self.n_simbolos})"

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    def simular(self, n_estudiantes: int, n_semanas: int,
                generadores) -> tuple[np.ndarray, np.ndarray]:
        """Simula cohortes completas sin bucles por estudiante.

        ``generadores`` puede ser un ``np.random.Generator`` (una sola
        réplica) o una lista de generadores, uno por réplica. En el segundo
        caso cada réplica toma sus números aleatorios de su propio generador,
        de modo que la réplica ``r`` es idéntica aunque cambie el número total
        de réplicas. Después, las transiciones se calculan para todas las
        réplicas y estudiantes a la vez; el único bucle recorre las semanas.

        Devuelve ``(estados, observaciones)`` con forma
        ``(n_estudiantes, n_semanas)`` para un generador o
        ``(n_replicas, n_estudiantes, n_semanas)`` para una lista.
        """
        if n_estudiantes < 1 or n_semanas < 1:
            raise ValueError("n_estudiantes y n_semanas deben ser positivos")
        una_replica = isinstance(generadores, np.random.Generator)
        lista = [generadores] if una_replica else list(generadores)
        if not lista:
            raise ValueError("Se necesita al menos un generador")
        forma = (2, n_estudiantes, n_semanas)
        uniformes = np.stack([gen.random(forma) for gen in lista], axis=1)
        estados = self._estados_desde_uniformes(uniformes[0])
        observaciones = (uniformes[1][..., None] >= _acumulada(self.b)[estados]).sum(axis=-1)
        observaciones = np.minimum(observaciones, self.n_simbolos - 1).astype(np.int8)
        if una_replica:
            return estados[0], observaciones[0]
        return estados, observaciones

    def _estados_desde_uniformes(self, uniformes: np.ndarray) -> np.ndarray:
        """Transforma uniformes ``(..., T)`` en estados por la inversa de la acumulada."""
        acum_a = _acumulada(self.a)
        estados = np.empty(uniformes.shape, dtype=np.int8)
        estados[..., 0] = (uniformes[..., 0, None] >= _acumulada(self.pi)).sum(axis=-1)
        for semana in range(1, uniformes.shape[-1]):
            limites = acum_a[estados[..., semana - 1]]
            estados[..., semana] = (uniformes[..., semana, None] >= limites).sum(axis=-1)
        return np.minimum(estados, self.n_estados - 1, out=estados)

    def simular_con_bucles(self, n_estudiantes: int, n_semanas: int,
                           generador: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Versión de referencia con un bucle por estudiante y por semana.

        Sigue el esquema del notebook de la guía base (``choice`` paso a paso)
        y solo se usa para comparar tiempos con :meth:`simular`.
        """
        estados = np.empty((n_estudiantes, n_semanas), dtype=np.int8)
        observaciones = np.empty((n_estudiantes, n_semanas), dtype=np.int8)
        indices_estado = np.arange(self.n_estados)
        indices_obs = np.arange(self.n_simbolos)
        for estudiante in range(n_estudiantes):
            actual = generador.choice(indices_estado, p=self.pi)
            for semana in range(n_semanas):
                if semana > 0:
                    actual = generador.choice(indices_estado, p=self.a[actual])
                estados[estudiante, semana] = actual
                observaciones[estudiante, semana] = generador.choice(indices_obs, p=self.b[actual])
        return estados, observaciones

    # ------------------------------------------------------------------
    # Inferencia en espacio logarítmico
    # ------------------------------------------------------------------
    def _log_emision(self, obs: np.ndarray) -> np.ndarray:
        """``log b[i, o_t]`` con forma ``(..., T, n_estados)``."""
        obs = np.asarray(obs)
        if obs.ndim == 0 or obs.shape[-1] == 0:
            raise ValueError("La secuencia de observaciones está vacía")
        if obs.min() < 0 or obs.max() >= self.n_simbolos:
            raise ValueError("Hay observaciones fuera del alfabeto del modelo")
        return self.log_b.T[obs.astype(np.intp)]

    def forward(self, obs) -> tuple[np.ndarray, np.ndarray]:
        """Algoritmo Forward en logaritmos.

        Acepta una secuencia ``(T,)`` o un lote ``(..., T)``. Devuelve
        ``(log_alfa, log_verosimilitud)``, donde ``log_alfa`` tiene forma
        ``(..., T, n_estados)`` y ``log_verosimilitud = log P(O | modelo)``.
        """
        log_e = self._log_emision(obs)
        log_alfa = np.empty_like(log_e)
        log_alfa[..., 0, :] = self.log_pi + log_e[..., 0, :]
        for t in range(1, log_e.shape[-2]):
            previo = log_alfa[..., t - 1, :, None] + self.log_a
            log_alfa[..., t, :] = logsumexp(previo, axis=-2) + log_e[..., t, :]
        return log_alfa, logsumexp(log_alfa[..., -1, :], axis=-1)

    def log_verosimilitud(self, obs) -> np.ndarray:
        """``log P(O | modelo)`` para una secuencia o un lote."""
        return self.forward(obs)[1]

    def backward(self, obs) -> np.ndarray:
        """Algoritmo Backward en logaritmos; forma ``(..., T, n_estados)``."""
        log_e = self._log_emision(obs)
        log_beta = np.zeros_like(log_e)
        for t in range(log_e.shape[-2] - 2, -1, -1):
            siguiente = self.log_a + (log_e[..., t + 1, :] + log_beta[..., t + 1, :])[..., None, :]
            log_beta[..., t, :] = logsumexp(siguiente, axis=-1)
        return log_beta

    def viterbi(self, obs) -> tuple[np.ndarray, np.ndarray]:
        """Algoritmo de Viterbi en logaritmos.

        Acepta una secuencia ``(T,)`` o un lote ``(..., T)``. Devuelve
        ``(ruta, log_prob)``: la secuencia de estados más probable y
        ``log P(O, ruta | modelo)``. Los empates se resuelven hacia el
        estado de menor índice, igual que ``np.argmax``.
        """
        log_e = self._log_emision(obs)
        n_pasos = log_e.shape[-2]
        punteros = np.zeros(log_e.shape, dtype=np.intp)
        delta = self.log_pi + log_e[..., 0, :]
        for t in range(1, n_pasos):
            candidatos = delta[..., :, None] + self.log_a
            punteros[..., t, :] = np.argmax(candidatos, axis=-2)
            delta = np.max(candidatos, axis=-2) + log_e[..., t, :]

        ruta = np.empty(log_e.shape[:-1], dtype=np.intp)
        ruta[..., -1] = np.argmax(delta, axis=-1)
        for t in range(n_pasos - 2, -1, -1):
            siguiente = ruta[..., t + 1, None]
            ruta[..., t] = np.take_along_axis(punteros[..., t + 1, :], siguiente, axis=-1)[..., 0]
        return ruta, np.max(delta, axis=-1)

    # ------------------------------------------------------------------
    # Propiedades de la cadena oculta
    # ------------------------------------------------------------------
    def distribucion_estacionaria(self) -> np.ndarray:
        """Resuelve ``mu A = mu`` con ``sum(mu) = 1``."""
        n_estados = self.n_estados
        sistema = np.vstack([self.a.T - np.eye(n_estados), np.ones(n_estados)])
        lado_derecho = np.zeros(n_estados + 1)
        lado_derecho[-1] = 1.0
        mu, *_ = np.linalg.lstsq(sistema, lado_derecho, rcond=None)
        mu = np.clip(mu, 0.0, None)
        return mu / mu.sum()

    def distribucion_por_semana(self, n_semanas: int) -> np.ndarray:
        """Distribución teórica de estados ``pi A^(t-1)``; forma ``(T, n)``."""
        distribucion = np.empty((n_semanas, self.n_estados))
        distribucion[0] = self.pi
        for semana in range(1, n_semanas):
            distribucion[semana] = distribucion[semana - 1] @ self.a
        return distribucion
