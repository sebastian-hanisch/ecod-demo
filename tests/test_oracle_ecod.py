"""Unabhängiges Orakel für ECOD und die kopierten Bausteine: alle fünf Varianten gegen die Definition (empirische Verteilungsfunktionen per Schleife, Schiefe aus scipy, die PyOD-Formel mit Vorzeichen der Schiefe
als Algebra), die Fisher-Schwelle gegen die Gamma-Verteilung aus scipy, LOF gegen scikit-learn, der Isolation Forest gegen scikit-learn (Pfadlängen-Konstante exakt, Werte statistisch)."""

import numpy as np
import pytest
from scipy.stats import gamma, skew

import ecod_algorithm as ecod
import ecod_isolation_forest as isf
import ecod_lof as lof
from ecod_evaluation import fisher_threshold


def _definition(X, variant):
    n, p = X.shape
    Fl = np.array([[sum(X[b, j] <= X[a, j] for b in range(n)) / n for j in range(p)] for a in range(n)])
    Fr = np.array([[sum(X[b, j] >= X[a, j] for b in range(n)) / n for j in range(p)] for a in range(n)])
    Ul, Ur = -np.log(Fl), -np.log(Fr)
    sk = np.array([skew(X[:, j]) if X[:, j].std() > 1e-12 else 0.0 for j in range(p)])
    auto = np.where(sk < 0, Ul, Ur)
    if variant == "paper":
        return np.maximum(np.maximum(Ul.sum(1), Ur.sum(1)), auto.sum(1))
    if variant == "auto":
        return auto.sum(1)
    if variant == "both":
        return np.maximum(Ul, Ur).sum(1)
    if variant == "twosided":
        return -np.log(np.minimum(1.0, 2.0 * np.minimum(Fl, Fr))).sum(1)
    s = np.sign(sk)                                                                      # PyOD: U_skew = U_l * -sign(s - 1) + U_r * sign(s + 1), dann Maximum je Merkmal
    return np.maximum(np.maximum(Ul, Ur), Ul * -np.sign(s - 1) + Ur * np.sign(s + 1)).sum(1)


def test_all_variants_equal_the_definition_also_with_ties_and_constant_features():
    rng = np.random.default_rng(11)
    for i in range(60):
        n, p = int(rng.integers(2, 30)), int(rng.integers(1, 5))
        kind = i % 4
        X = [rng.standard_normal((n, p)), rng.integers(0, 4, (n, p)).astype(float), rng.exponential(size=(n, p)), np.round(rng.standard_normal((n, p)), 1)][kind]
        if kind == 3:
            X[:, 0] = 1.0
        for variant in ecod.VARIANTS:
            assert np.allclose(ecod.score(X, variant), _definition(X, variant), rtol=1e-10, atol=1e-10), (variant, i)


def test_pyod_skew_term_is_both_tails_only_at_exactly_zero_skew_hand_instance():
    X = np.array([[-1.0], [0.0], [1.0]])                       # symmetrisch: Schiefe genau 0 -> PyOD nimmt U_l + U_r als Schiefe-Term, das Maximum je Merkmal bleibt max(U_l, U_r, U_l + U_r)
    u_l, u_r = ecod.contributions(X)
    assert ecod.skewness(X)[0] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(ecod.score(X, "pyod"), (u_l + u_r)[:, 0])


@pytest.mark.parametrize("p", [1, 2, 5, 12, 30, 52])
@pytest.mark.parametrize("q", [0.9, 0.95, 0.975, 0.99, 0.999])
def test_fisher_threshold_is_the_gamma_quantile(p, q):
    assert fisher_threshold(p, q) == pytest.approx(gamma.ppf(q, p), rel=1e-6)


def test_lof_equals_scikit_learn_including_new_points():
    neighbors = pytest.importorskip("sklearn.neighbors")
    rng = np.random.default_rng(7)
    for _ in range(40):
        n, p = int(rng.integers(6, 60)), int(rng.integers(1, 4))
        X = rng.standard_normal((n, p)) * rng.uniform(0.5, 3, p)
        X[: max(1, n // 10)] += rng.uniform(3, 8)
        k = lof.clamp_k(int(rng.integers(1, n)), n)
        fit = lof.fit_lof(X, k)
        sk = neighbors.LocalOutlierFactor(n_neighbors=k, algorithm="brute").fit(X)
        assert np.allclose(fit.lof, -sk.negative_outlier_factor_, rtol=1e-8)
        Q = rng.standard_normal((6, p)) * 3
        skn = neighbors.LocalOutlierFactor(n_neighbors=k, algorithm="brute", novelty=True).fit(X)
        assert np.allclose(lof.score_new(X, fit, Q), -skn.score_samples(Q), rtol=1e-8)


def test_isolation_forest_matches_scikit_learn_statistically_and_the_path_constant_exactly():
    ensemble = pytest.importorskip("sklearn.ensemble")
    iforest = pytest.importorskip("sklearn.ensemble._iforest")
    n_values = np.array([0, 1, 2, 3, 4, 5, 10, 50, 256, 1000])
    assert np.allclose(isf.c_factor(n_values), iforest._average_path_length(n_values), atol=1e-12)
    rng = np.random.default_rng(5)
    for i, psi in enumerate((8, 64)):
        X = rng.standard_normal((150, 3)) * [1, 2, 0.5]
        X[:8] += 5
        ours = isf.score(isf.fit_forest(X, 300, psi, seed=i), X)
        theirs = -ensemble.IsolationForest(n_estimators=300, max_samples=psi, random_state=i).fit(X).score_samples(X)
        assert abs(float(np.mean(ours - theirs))) < 0.01 and float(np.sqrt(np.mean((ours - theirs) ** 2))) < 0.02                 # Monte-Carlo-Rauschen beider Wälder bei 300 Bäumen: ~0.008
