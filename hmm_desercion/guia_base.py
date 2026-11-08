"""Viterbi tal como aparece en la plantilla del artículo (guía base).

Se conserva solo para documentar la falla: multiplica probabilidades sin
logaritmos, así que ``delta`` se hace 0 por subdesbordamiento en secuencias
largas y la ruta reconstruida deja de tener sentido. La versión corregida es
:meth:`hmm_desercion.modelo.HMM.viterbi`.
"""

from __future__ import annotations

import numpy as np


def viterbi_sin_logaritmos(obs, a, b, pi) -> tuple[np.ndarray, float]:
    """Transcripción del pseudocódigo de la plantilla (sin cambios de lógica)."""
    obs = np.asarray(obs)
    a, b, pi = np.asarray(a), np.asarray(b), np.asarray(pi)
    n_estados = a.shape[0]
    n_pasos = len(obs)
    delta = np.zeros((n_pasos, n_estados))
    psi = np.zeros((n_pasos, n_estados), dtype=int)
    delta[0, :] = pi * b[:, obs[0]]
    for t in range(1, n_pasos):
        for j in range(n_estados):
            temp = delta[t - 1, :] * a[:, j]
            delta[t, j] = np.max(temp) * b[j, obs[t]]
            psi[t, j] = np.argmax(temp)
    ruta = np.zeros(n_pasos, dtype=int)
    ruta[n_pasos - 1] = np.argmax(delta[n_pasos - 1, :])
    for t in range(n_pasos - 2, -1, -1):
        ruta[t] = psi[t + 1, ruta[t + 1]]
    return ruta, float(np.max(delta[n_pasos - 1, :]))
