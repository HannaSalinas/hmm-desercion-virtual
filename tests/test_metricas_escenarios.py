"""Pruebas de métricas, resumen estadístico, escenarios, perturbaciones y tiempos."""

import numpy as np
import pytest
from scipy import stats

from hmm_desercion import metricas
from hmm_desercion.escenarios import (A_NOMINAL, B_NOMINAL, escenarios_principales,
                                      perturbar_aleatoria, perturbar_diagonal,
                                      perturbar_entrada)
from hmm_desercion.tiempos import comparar_tiempos

# Dos réplicas, tres estudiantes, cuatro semanas.
ESTADOS = np.array([
    [[0, 0, 1, 2], [0, 1, 1, 1], [2, 2, 0, 0]],
    [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
])


def test_tasa_desercion():
    """Proporción en Desconectado en la última semana."""
    assert np.allclose(metricas.tasa_desercion(ESTADOS), [1 / 3, 0.0])


def test_primera_desconexion_y_censura():
    """Media entre quienes se desconectan; la réplica sin desconexiones da nan."""
    media, nunca = metricas.primera_desconexion(ESTADOS)
    assert media[0] == pytest.approx((4 + 1) / 2)
    assert np.isnan(media[1])
    assert np.allclose(nunca, [1 / 3, 1.0])


def test_exactitud():
    """Fracción de coincidencias por réplica."""
    inferidos = ESTADOS.copy()
    inferidos[0, 0, 0] = 1
    assert np.allclose(metricas.exactitud(ESTADOS, inferidos), [11 / 12, 1.0])
    with pytest.raises(ValueError):
        metricas.exactitud(ESTADOS, ESTADOS[0])


def test_proporcion_por_semana():
    """Las proporciones por semana suman 1."""
    prop = metricas.proporcion_por_semana(ESTADOS)
    assert prop.shape == (2, 4, 3)
    assert np.allclose(prop.sum(-1), 1.0)
    assert np.allclose(prop[0, 0], [2 / 3, 0, 1 / 3])


def test_resumen_con_intervalo_t():
    """El IC usa la t de Student con n-1 grados de libertad."""
    valores = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    res = metricas.resumen(valores)
    margen = stats.t.ppf(0.975, 3) * np.std([1, 2, 3, 4], ddof=1) / 2
    assert res["n"] == 4 and res["media"] == pytest.approx(2.5)
    assert res["minimo"] == 1.0 and res["maximo"] == 4.0
    assert res["ic_superior"] - res["media"] == pytest.approx(margen)


def test_resumen_casos_borde():
    """Sin datos todo es nan; con un dato no hay desviación ni IC."""
    assert np.isnan(metricas.resumen([np.nan])["media"])
    uno = metricas.resumen([0.5])
    assert uno["media"] == 0.5 and np.isnan(uno["desviacion"])


def test_convergencia_coincide_con_resumen():
    """El último punto de la curva de convergencia es el resumen completo."""
    valores = np.random.default_rng(0).normal(size=50)
    curva = metricas.convergencia(valores)
    res = metricas.resumen(valores)
    assert curva["media"][-1] == pytest.approx(res["media"])
    assert curva["ic_superior"][-1] == pytest.approx(res["ic_superior"])
    assert np.isnan(curva["ic_inferior"][0])


def test_escenarios_principales_validos():
    """E1, E2 y E3 construyen modelos válidos y E2 solo cambia la fila En riesgo."""
    escenarios = escenarios_principales()
    assert [e.clave for e in escenarios] == ["E1", "E2", "E3"]
    for escenario in escenarios:
        escenario.modelo()
    e1, e2, e3 = escenarios
    assert np.array_equal(e1.a[[0, 2]], e2.a[[0, 2]])
    assert e2.a[1, 0] > e1.a[1, 0] and e2.a[1, 2] < e1.a[1, 2]
    assert np.all(np.diag(e3.b) < np.diag(e1.b))


@pytest.mark.parametrize("delta", [-0.05, 0.05])
def test_perturbar_diagonal(delta):
    """La diagonal cambia en delta y las filas siguen sumando 1."""
    for matriz in (A_NOMINAL, B_NOMINAL):
        nueva = perturbar_diagonal(matriz, delta)
        assert np.allclose(np.diag(nueva), np.diag(matriz) + delta)
        assert np.allclose(nueva.sum(1), 1.0)


def test_perturbar_diagonal_fuera_de_rango():
    """Un delta que saca la diagonal de (0, 1) se rechaza."""
    with pytest.raises(ValueError):
        perturbar_diagonal(A_NOMINAL, 0.2)


def test_perturbar_aleatoria():
    """La perturbación aleatoria es pequeña, reproducible y estocástica."""
    uno = perturbar_aleatoria(A_NOMINAL, 0.05, np.random.default_rng(1))
    dos = perturbar_aleatoria(A_NOMINAL, 0.05, np.random.default_rng(1))
    assert np.array_equal(uno, dos)
    assert np.allclose(uno.sum(1), 1.0) and np.all(uno > 0)
    assert np.max(np.abs(uno - A_NOMINAL)) < 0.1


def test_comparar_tiempos():
    """Devuelve una fila por tamaño con tiempos positivos."""
    modelo = escenarios_principales()[0].modelo()
    filas = comparar_tiempos(modelo, [10, 20], 16, semilla=2026, repeticiones=1)
    assert [f["estudiantes"] for f in filas] == [10, 20]
    assert all(f["segundos_bucles"] > 0 and f["aceleracion"] > 0 for f in filas)


def test_perturbar_entrada():
    """Solo cambia la fila elegida, sigue estocástica y respeta el mínimo."""
    nueva = perturbar_entrada(A_NOMINAL, 0, 2, 0.05)
    assert np.allclose(nueva.sum(1), 1) and np.array_equal(nueva[1:], A_NOMINAL[1:])
    assert nueva[0, 2] > A_NOMINAL[0, 2]
    recortada = perturbar_entrada(A_NOMINAL, 0, 2, -0.05)
    assert recortada[0, 2] == pytest.approx(1e-3 / (1 - 0.029))
