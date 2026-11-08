"""Pruebas de la clase HMM: validación, simulación, Forward, Viterbi y estacionaria."""

import itertools

import numpy as np
import pytest
from hmmlearn import hmm as hmmlearn_hmm

from hmm_desercion.escenarios import A_NOMINAL, B_NOMINAL, PI_NOMINAL
from hmm_desercion.guia_base import viterbi_sin_logaritmos
from hmm_desercion.modelo import HMM, validar_estocastica


@pytest.fixture(name="modelo")
def fixture_modelo():
    """Modelo con los parámetros nominales."""
    return HMM(PI_NOMINAL, A_NOMINAL, B_NOMINAL)


def _hmmlearn(modelo):
    """Modelo equivalente en hmmlearn, solo para verificar."""
    ref = hmmlearn_hmm.CategoricalHMM(n_components=3, n_features=3, init_params="")
    ref.startprob_ = modelo.pi
    ref.transmat_ = modelo.a
    ref.emissionprob_ = modelo.b
    return ref


def _fuerza_bruta(modelo, obs):
    """Enumera todas las rutas: log P(O) y la mejor ruta con su log P(O, ruta)."""
    total, mejor, mejor_ruta = 0.0, -np.inf, None
    for ruta in itertools.product(range(modelo.n_estados), repeat=len(obs)):
        prob = modelo.pi[ruta[0]] * modelo.b[ruta[0], obs[0]]
        for t in range(1, len(obs)):
            prob *= modelo.a[ruta[t - 1], ruta[t]] * modelo.b[ruta[t], obs[t]]
        total += prob
        if prob > mejor:
            mejor, mejor_ruta = prob, ruta
    return np.log(total), np.array(mejor_ruta), np.log(mejor)


# ---------------------------------------------------------------- validación
def test_parametros_nominales_son_estocasticos():
    """Cada fila de pi, A y B suma 1."""
    for matriz in (PI_NOMINAL, A_NOMINAL, B_NOMINAL):
        assert np.allclose(np.asarray(matriz).sum(axis=-1), 1.0)


@pytest.mark.parametrize("matriz", [
    [[0.5, 0.6], [0.5, 0.5]],
    [[1.1, -0.1], [0.5, 0.5]],
    [[np.nan, 1.0], [0.5, 0.5]],
    [],
])
def test_validacion_rechaza_matrices_invalidas(matriz):
    """Filas que no suman 1, negativos, nan o vacío lanzan ValueError."""
    with pytest.raises(ValueError):
        validar_estocastica(matriz, "M")


def test_validacion_de_dimensiones():
    """A debe ser cuadrada y B debe tener una fila por estado."""
    with pytest.raises(ValueError):
        HMM(PI_NOMINAL, A_NOMINAL[:2, :2] / A_NOMINAL[:2, :2].sum(1, keepdims=True), B_NOMINAL)
    with pytest.raises(ValueError):
        HMM(PI_NOMINAL, A_NOMINAL, B_NOMINAL[:2])


# ---------------------------------------------------------------- simulación
def test_simulacion_reproducible_con_semilla(modelo):
    """La misma semilla produce exactamente las mismas secuencias."""
    uno = modelo.simular(50, 16, np.random.default_rng(2026))
    dos = modelo.simular(50, 16, np.random.default_rng(2026))
    assert np.array_equal(uno[0], dos[0]) and np.array_equal(uno[1], dos[1])


def test_replica_no_depende_del_numero_de_replicas(modelo):
    """Con semillas derivadas, la réplica r es igual con 3 o con 10 réplicas."""
    pocas = modelo.simular(20, 16, np.random.default_rng(2026).spawn(3))
    muchas = modelo.simular(20, 16, np.random.default_rng(2026).spawn(10))
    assert pocas[0].shape == (3, 20, 16)
    assert np.array_equal(pocas[0], muchas[0][:3])
    assert np.array_equal(pocas[1], muchas[1][:3])


def test_simulacion_recupera_frecuencias(modelo):
    """Con muchos estudiantes, las frecuencias empíricas se acercan a pi, A y B."""
    estados, obs = modelo.simular(40_000, 16, np.random.default_rng(1))
    assert estados.min() >= 0 and estados.max() <= 2 and obs.max() <= 2
    pi_emp = np.bincount(estados[:, 0], minlength=3) / estados.shape[0]
    assert np.allclose(pi_emp, modelo.pi, atol=0.01)
    origen, destino = estados[:, :-1].ravel(), estados[:, 1:].ravel()
    conteo_a = np.zeros((3, 3))
    np.add.at(conteo_a, (origen, destino), 1)
    assert np.allclose(conteo_a / conteo_a.sum(1, keepdims=True), modelo.a, atol=0.01)
    conteo_b = np.zeros((3, 3))
    np.add.at(conteo_b, (estados.ravel(), obs.ravel()), 1)
    assert np.allclose(conteo_b / conteo_b.sum(1, keepdims=True), modelo.b, atol=0.01)


def test_simulacion_con_bucles_tiene_misma_distribucion(modelo):
    """La versión con bucles genera la misma distribución de estados."""
    estados, _ = modelo.simular_con_bucles(3000, 16, np.random.default_rng(3))
    teorica = modelo.distribucion_por_semana(16)[-1]
    empirica = np.bincount(estados[:, -1], minlength=3) / estados.shape[0]
    assert np.allclose(empirica, teorica, atol=0.03)


def test_simulacion_rechaza_tamanos_invalidos(modelo):
    """Cero estudiantes o una lista vacía de generadores no son válidos."""
    with pytest.raises(ValueError):
        modelo.simular(0, 16, np.random.default_rng(0))
    with pytest.raises(ValueError):
        modelo.simular(5, 16, [])


# ---------------------------------------------------------------- forward
def test_forward_coincide_con_fuerza_bruta(modelo):
    """Para T=6 el Forward en logaritmos iguala la suma sobre las 3^6 rutas."""
    obs = np.array([0, 1, 2, 2, 1, 0])
    esperado, _, _ = _fuerza_bruta(modelo, obs)
    assert modelo.log_verosimilitud(obs) == pytest.approx(esperado, abs=1e-10)


def test_forward_coincide_con_hmmlearn(modelo):
    """Mismo log P(O) que hmmlearn para una secuencia de 16 semanas."""
    _, obs = modelo.simular(1, 16, np.random.default_rng(5))
    esperado = _hmmlearn(modelo).score(obs.reshape(-1, 1))
    assert modelo.log_verosimilitud(obs[0]) == pytest.approx(esperado, abs=1e-9)


def test_forward_en_lote_igual_que_individual(modelo):
    """El cálculo en lote da lo mismo que secuencia por secuencia."""
    _, obs = modelo.simular(25, 16, np.random.default_rng(6))
    lote = modelo.log_verosimilitud(obs)
    individual = np.array([modelo.log_verosimilitud(o) for o in obs])
    assert np.allclose(lote, individual)


def test_forward_y_backward_son_consistentes(modelo):
    """sum_i alfa_t(i) beta_t(i) = P(O) para todo t."""
    obs = np.array([0, 0, 1, 2, 2, 2, 1, 0])
    log_alfa, log_ver = modelo.forward(obs)
    log_beta = modelo.backward(obs)
    por_t = np.logaddexp.reduce(log_alfa + log_beta, axis=-1)
    assert np.allclose(por_t, log_ver)


def test_forward_no_falla_con_secuencias_largas(modelo):
    """Con T=10000 la log-verosimilitud es finita."""
    _, obs = modelo.simular(1, 10_000, np.random.default_rng(7))
    assert np.isfinite(modelo.log_verosimilitud(obs[0]))


def test_observaciones_fuera_de_rango(modelo):
    """Símbolos fuera del alfabeto o secuencias vacías se rechazan."""
    with pytest.raises(ValueError):
        modelo.forward(np.array([0, 3]))
    with pytest.raises(ValueError):
        modelo.viterbi(np.array([], dtype=int))


# ---------------------------------------------------------------- viterbi
def test_viterbi_coincide_con_fuerza_bruta(modelo):
    """La ruta y su probabilidad son las de la enumeración exhaustiva."""
    obs = np.array([0, 1, 2, 2, 1, 0])
    _, ruta_esperada, log_esperado = _fuerza_bruta(modelo, obs)
    ruta, log_prob = modelo.viterbi(obs)
    assert np.array_equal(ruta, ruta_esperada)
    assert log_prob == pytest.approx(log_esperado, abs=1e-10)


def test_viterbi_coincide_con_hmmlearn(modelo):
    """Misma ruta y log-probabilidad que hmmlearn en 200 estudiantes."""
    _, obs = modelo.simular(200, 16, np.random.default_rng(8))
    rutas, log_probs = modelo.viterbi(obs)
    referencia = _hmmlearn(modelo)
    for i in (0, 57, 199):
        log_ref, ruta_ref = referencia.decode(obs[i].reshape(-1, 1), algorithm="viterbi")
        assert log_probs[i] == pytest.approx(log_ref, abs=1e-9)
        assert np.array_equal(rutas[i], ruta_ref)


def test_viterbi_en_lote_con_replicas(modelo):
    """Acepta lotes (R, N, T) y devuelve rutas con la misma forma."""
    _, obs = modelo.simular(10, 16, np.random.default_rng(0).spawn(4))
    rutas, log_probs = modelo.viterbi(obs)
    assert rutas.shape == obs.shape and log_probs.shape == (4, 10)
    assert np.array_equal(rutas[2, 3], modelo.viterbi(obs[2, 3])[0])


def test_viterbi_largo_frente_a_la_plantilla(modelo):
    """Con T=10000 el Viterbi de la plantilla subdesborda y el logarítmico no."""
    _, obs = modelo.simular(1, 10_000, np.random.default_rng(9))
    ruta, log_prob = modelo.viterbi(obs[0])
    _, prob_plantilla = viterbi_sin_logaritmos(obs[0], modelo.a, modelo.b, modelo.pi)
    assert np.isfinite(log_prob) and ruta.shape == (10_000,)
    assert prob_plantilla == 0.0


def test_viterbi_corto_igual_a_la_plantilla(modelo):
    """En secuencias cortas ambas versiones coinciden."""
    obs = np.array([0, 1, 1, 2, 2, 0, 1, 2])
    ruta, log_prob = modelo.viterbi(obs)
    ruta_p, prob_p = viterbi_sin_logaritmos(obs, modelo.a, modelo.b, modelo.pi)
    assert np.array_equal(ruta, ruta_p)
    assert log_prob == pytest.approx(np.log(prob_p))


# ---------------------------------------------------------------- estacionaria
def test_distribucion_estacionaria(modelo):
    """mu A = mu, suma 1 y coincide con pi A^t para t grande."""
    mu = modelo.distribucion_estacionaria()
    assert mu.sum() == pytest.approx(1.0)
    assert np.allclose(mu @ modelo.a, mu)
    assert np.allclose(modelo.distribucion_por_semana(500)[-1], mu, atol=1e-8)
