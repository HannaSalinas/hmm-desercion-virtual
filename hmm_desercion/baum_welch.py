"""Estimación de parámetros con Baum-Welch (EM) sobre secuencias simuladas.

Decisión de diseño: ``pi`` queda fijo en su valor real y solo se aprenden
``A`` y ``B``. Con 16 semanas por estudiante cada secuencia aporta una única
observación de la semana inicial, y lo que interesa evaluar es si la dinámica
(``A``) y la relación estado-actividad (``B``) se recuperan desde la
actividad observada.

El punto de partida es genérico y no usa los valores reales: diagonal
dominante en ``A`` y una ``B`` que solo conserva el orden "Comprometido
emite más Alta, Desconectado emite más Baja". Ese orden evita el cambio de
etiquetas entre estados, un problema conocido de EM en mezclas y HMM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hmm_desercion.modelo import HMM

A_INICIAL = np.array([[0.60, 0.20, 0.20],
                      [0.20, 0.60, 0.20],
                      [0.20, 0.20, 0.60]])
B_INICIAL = np.array([[0.50, 0.30, 0.20],
                      [0.30, 0.40, 0.30],
                      [0.20, 0.30, 0.50]])


@dataclass
class ResultadoBaumWelch:
    """Parámetros aprendidos y traza de la log-verosimilitud por iteración."""

    a: np.ndarray
    b: np.ndarray
    pi: np.ndarray
    historial: list[float] = field(default_factory=list)
    convergio: bool = False

    @property
    def iteraciones(self) -> int:
        """Número de iteraciones EM realizadas."""
        return len(self.historial)

    def modelo(self) -> HMM:
        """Construye un :class:`HMM` con los parámetros aprendidos."""
        return HMM(self.pi, self.a, self.b)


def forward_backward_escalado(modelo: HMM, obs: np.ndarray):
    """Forward-Backward con factores de escala (Rabiner, 1989, sec. V-A).

    En cada semana ``alfa`` se normaliza para que sume 1 y el factor ``c_t``
    se guarda; ``log P(O) = sum_t log c_t``. Así no hay subdesbordamiento y
    todo son productos de matrices, mucho más rápidos que ``logsumexp``.
    Devuelve ``(alfa, beta, c)`` con formas ``(M, T, n)``, ``(M, T, n)`` y
    ``(M, T)``.
    """
    emision = modelo.b.T[obs.astype(np.intp)]
    n_pasos = obs.shape[-1]
    alfa = np.empty_like(emision)
    escala = np.empty(emision.shape[:-1])
    actual = modelo.pi * emision[:, 0]
    for t in range(n_pasos):
        if t > 0:
            actual = (alfa[:, t - 1] @ modelo.a) * emision[:, t]
        escala[:, t] = actual.sum(axis=-1)
        alfa[:, t] = actual / escala[:, t, None]
    beta = np.ones_like(emision)
    for t in range(n_pasos - 2, -1, -1):
        beta[:, t] = ((emision[:, t + 1] * beta[:, t + 1]) @ modelo.a.T) / escala[:, t + 1, None]
    return alfa, beta, escala


def _paso_em(modelo: HMM, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Una iteración E+M para un lote de secuencias ``(M, T)``."""
    alfa, beta, escala = forward_backward_escalado(modelo, obs)
    gamma = alfa * beta

    # sum_{m,t} xi_t(i, j) = a_ij * sum alfa_t(i) b_j(o_{t+1}) beta_{t+1}(j) / c_{t+1}
    emision = modelo.b.T[obs[:, 1:].astype(np.intp)]
    siguiente = emision * beta[:, 1:] / escala[:, 1:, None]
    nueva_a = modelo.a * np.einsum("mti,mtj->ij", alfa[:, :-1], siguiente)
    nueva_a /= nueva_a.sum(axis=1, keepdims=True)
    log_ver = float(np.log(escala).sum())
    return nueva_a, _reestimar_emision(gamma, obs, modelo.n_simbolos), log_ver


def _reestimar_emision(gamma: np.ndarray, obs: np.ndarray, n_simbolos: int) -> np.ndarray:
    """``b[i, k]`` = tiempo esperado en ``i`` emitiendo ``k`` / tiempo esperado en ``i``."""
    gamma_plana = gamma.reshape(-1, gamma.shape[-1])
    simbolos = obs.reshape(-1)
    conteos = np.stack([gamma_plana[simbolos == k].sum(axis=0) for k in range(n_simbolos)],
                       axis=1)
    return conteos / gamma_plana.sum(axis=0)[:, None]


def baum_welch(obs, pi, *, a_inicial=None, b_inicial=None, max_iter: int = 1000,
               tol: float = 1e-4) -> ResultadoBaumWelch:
    """Ajusta ``A`` y ``B`` por máxima verosimilitud con ``pi`` fijo.

    ``obs`` es un arreglo ``(M, T)`` de secuencias independientes. Se
    detiene cuando la mejora de la log-verosimilitud total es menor que
    ``tol`` o al llegar a ``max_iter``. ``historial[i]`` es la
    log-verosimilitud de los parámetros al comienzo de la iteración ``i``.
    """
    obs = np.atleast_2d(np.asarray(obs))
    a = A_INICIAL if a_inicial is None else a_inicial
    b = B_INICIAL if b_inicial is None else b_inicial
    modelo = HMM(pi, a, b)
    historial: list[float] = []
    convergio = False
    for _ in range(max_iter):
        nueva_a, nueva_b, log_ver = _paso_em(modelo, obs)
        if historial and log_ver - historial[-1] < tol:
            historial.append(log_ver)
            convergio = True
            break
        historial.append(log_ver)
        modelo = HMM(modelo.pi, nueva_a, nueva_b)
    return ResultadoBaumWelch(a=modelo.a, b=modelo.b, pi=modelo.pi,
                              historial=historial, convergio=convergio)


def error_absoluto_medio(estimada, real) -> float:
    """Error absoluto medio entre dos matrices de probabilidad."""
    return float(np.mean(np.abs(np.asarray(estimada) - np.asarray(real))))
