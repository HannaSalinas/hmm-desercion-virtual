"""Pruebas del diseño experimental y de la tubería de resultados (tamaños pequeños)."""

import numpy as np
import pandas as pd

from hmm_desercion import __main__ as principal
from hmm_desercion import experimentos as ex
from hmm_desercion.escenarios import escenarios_principales


def test_semillas_reproducibles_e_independientes():
    """Mismo flujo, mismas semillas; flujos distintos, números distintos."""
    uno = ex.generadores(0, 3)[1].random(5)
    dos = ex.generadores(0, 3)[1].random(5)
    otro = ex.generadores(1, 3)[1].random(5)
    assert np.array_equal(uno, dos) and not np.array_equal(uno, otro)


def test_numeros_aleatorios_comunes_entre_escenarios():
    """E1 y E3 comparten A y pi, así que con las mismas semillas tienen los mismos estados."""
    e1, _, e3 = escenarios_principales()
    c1 = ex.analizar_cohortes(e1.modelo(), 3, n_estudiantes=40)
    c3 = ex.analizar_cohortes(e3.modelo(), 3, n_estudiantes=40)
    assert np.array_equal(c1.estados, c3.estados)
    assert c1.replicas["exactitud_viterbi"].mean() > c3.replicas["exactitud_viterbi"].mean()


def test_ejecutar_escenario_columnas():
    """Las réplicas sin Baum-Welch quedan con nan en las columnas de error."""
    res = ex.ejecutar_escenario(escenarios_principales()[1], n_replicas=4, n_replicas_bw=2,
                                n_estudiantes=60)
    assert len(res.replicas) == 4 and res.a_aprendidas.shape == (2, 3, 3)
    assert res.replicas["error_bw"].notna().sum() == 2
    assert set(ex.METRICAS) <= set(res.replicas.columns)


def test_escalado_cohortes_anidadas():
    """Una fila por tamaño y réplica, con matrices medias estocásticas."""
    tabla, medias = ex.escalado_baum_welch(escenarios_principales()[0], (50, 100), 2)
    assert list(tabla["estudiantes"]) == [50, 50, 100, 100]
    for a, b in medias.values():
        assert np.allclose(a.sum(1), 1) and np.allclose(b.sum(1), 1)


def test_sensibilidad_y_plantilla():
    """Sensibilidad determinista, aleatoria y comparación con la plantilla."""
    sens = ex.sensibilidad_determinista(n_replicas=2)
    assert sens["configuracion"].nunique() == len(ex.configuraciones_sensibilidad())
    nominal = sens[sens["configuracion"] == "Nominal"]
    assert np.allclose(nominal["exactitud_viterbi"], nominal["exactitud_viterbi_nominal"])
    aleatoria = ex.sensibilidad_aleatoria(n_perturbaciones=2, n_replicas=2)
    assert len(aleatoria) == 4 and (aleatoria["cambio_max_absoluto"] > 0).all()
    plantilla = ex.viterbi_frente_a_plantilla(escenarios_principales()[0].modelo(), (16, 2000))
    assert not plantilla["subdesborda"].iloc[0] and plantilla["subdesborda"].iloc[1]
    assert plantilla["coincidencia_ruta"].iloc[0] == 1.0


def test_resumir_y_proporciones():
    """El resumen tiene una fila por grupo y métrica; las proporciones suman 1."""
    res = ex.ejecutar_escenario(escenarios_principales()[0], n_replicas=3, n_replicas_bw=1,
                                n_estudiantes=30)
    resumen = ex.resumir(res.replicas, "escenario", ["tasa_desercion_s16", "no_existe"])
    assert len(resumen) == 1
    prop = ex.proporciones_semanales([res])
    assert np.allclose(prop.groupby("semana")["media"].sum(), 1.0)


def test_matrices_csv_ida_y_vuelta(tmp_path):
    """Las matrices guardadas en CSV se recuperan iguales."""
    a, b = np.eye(3), np.full((3, 3), 1 / 3)
    ruta = tmp_path / "m.csv"
    principal._matrices_a_csv({"real": (a, b)}, ruta)  # pylint: disable=protected-access
    leidas = principal._matrices_desde_csv(ruta)  # pylint: disable=protected-access
    assert np.allclose(leidas["real"][0], a) and np.allclose(leidas["real"][1], b)
    assert isinstance(pd.read_csv(ruta), pd.DataFrame)


def test_sensibilidad_por_entrada_y_atribucion():
    """Una fila por entrada y signo; la teoría coincide con la simulación."""
    entradas = ex.sensibilidad_por_entrada(0.05, n_replicas=20)
    assert len(entradas) == 18
    assert np.all(np.sign(entradas["cambio_efectivo"]) == np.sign(entradas["delta"]))
    assert np.allclose(entradas["desercion_simulada"], entradas["desercion_teorica"], atol=0.02)
    aleatoria = ex.sensibilidad_aleatoria(n_perturbaciones=3, n_replicas=2)
    atribucion = ex.atribucion_por_fila(aleatoria)
    assert len(atribucion) == 3
    sub = aleatoria[aleatoria["matriz"] == "A"]
    assert np.allclose(atribucion["cambio_total"],
                       sub["desercion_teorica"] - ex.desercion_teorica(ex.A_NOMINAL))
