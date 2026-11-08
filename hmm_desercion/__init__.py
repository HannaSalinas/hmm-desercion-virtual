"""HMM para detectar el riesgo de deserción en educación virtual."""

from hmm_desercion.modelo import ESTADOS, OBSERVACIONES, HMM, validar_estocastica

SEMILLA = 2026

__all__ = ["ESTADOS", "OBSERVACIONES", "HMM", "SEMILLA", "validar_estocastica"]
