"""Pruebas de Baum-Welch: monotonía, recuperación y verificación con hmmlearn."""

import warnings

import numpy as np
import pytest
from hmmlearn import hmm as hmmlearn_hmm

from hmm_desercion.baum_welch import (A_INICIAL, B_INICIAL, baum_welch,
                                      error_absoluto_medio, forward_backward_escalado)
from hmm_desercion.escenarios import A_NOMINAL, B_NOMINAL, PI_NOMINAL
from hmm_desercion.modelo import HMM


@pytest.fixture(name="observaciones", scope="module")
def fixture_observaciones():
    """3000 estudiantes de 16 semanas con los parámetros nominales."""
    modelo = HMM(PI_NOMINAL, A_NOMINAL, B_NOMINAL)
    return modelo.simular(3000, 16, np.random.default_rng(11))[1]


def test_escalado_coincide_con_forward_logaritmico(observaciones):
    """El Forward-Backward escalado da el mismo log P(O) y las mismas gamma."""
    modelo = HMM(PI_NOMINAL, A_NOMINAL, B_NOMINAL)
    datos = observaciones[:50]
    alfa, beta, escala = forward_backward_escalado(modelo, datos)
    log_alfa, log_ver = modelo.forward(datos)
    log_beta = modelo.backward(datos)
    assert np.allclose(np.log(escala).sum(axis=1), log_ver)
    assert np.allclose(alfa * beta, np.exp(log_alfa + log_beta - log_ver[:, None, None]))


def test_log_verosimilitud_no_decrece(observaciones):
    """EM nunca empeora la log-verosimilitud."""
    resultado = baum_welch(observaciones[:300], PI_NOMINAL, max_iter=60)
    assert np.all(np.diff(resultado.historial) >= -1e-8)


def test_parametros_aprendidos_son_estocasticos(observaciones):
    """A y B aprendidas siguen siendo estocásticas y pi no cambia."""
    resultado = baum_welch(observaciones[:300], PI_NOMINAL, max_iter=60)
    assert np.allclose(resultado.a.sum(1), 1.0) and np.allclose(resultado.b.sum(1), 1.0)
    assert np.array_equal(resultado.pi, PI_NOMINAL)
    assert isinstance(resultado.modelo(), HMM)


def test_recupera_parametros_con_muchos_datos(observaciones):
    """Con 3000 secuencias el error absoluto medio es pequeño."""
    resultado = baum_welch(observaciones, PI_NOMINAL)
    assert resultado.convergio
    assert error_absoluto_medio(resultado.a, A_NOMINAL) < 0.03
    assert error_absoluto_medio(resultado.b, B_NOMINAL) < 0.03


def test_coincide_con_hmmlearn(observaciones):
    """Mismas A y B que hmmlearn con igual inicio, pi fijo y 20 iteraciones."""
    datos = observaciones[:500]
    propio = baum_welch(datos, PI_NOMINAL, max_iter=20, tol=-np.inf)
    ref = hmmlearn_hmm.CategoricalHMM(n_components=3, n_features=3, n_iter=20,
                                      tol=0.0, params="te", init_params="")
    ref.startprob_ = PI_NOMINAL.copy()
    ref.transmat_ = A_INICIAL.copy()
    ref.emissionprob_ = B_INICIAL.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref.fit(datos.reshape(-1, 1), lengths=[datos.shape[1]] * datos.shape[0])
    assert np.allclose(propio.a, ref.transmat_, atol=1e-8)
    assert np.allclose(propio.b, ref.emissionprob_, atol=1e-8)


def test_error_absoluto_medio():
    """El error es cero entre matrices iguales y positivo si difieren."""
    assert error_absoluto_medio(A_NOMINAL, A_NOMINAL) == 0.0
    assert error_absoluto_medio(A_INICIAL, A_NOMINAL) > 0.0
