"""ECOD: Handinstanzen (Schwanzwahrscheinlichkeiten mit Bindungen, Beiträge, drei Summen), eine Referenzimplementierung mit expliziten Schleifen, Kreuzprüfung gegen PyOD (falls installiert),
Rang-Invarianz, Fisher-Verteilung der zweiseitigen Variante unter Unabhängigkeit; die kopierten Komponenten der Vorgänger (LOF, Isolation Forest, χ², MCD)."""

import numpy as np
import pytest
from scipy.stats import chi2, gamma, kstest, skew

import ecod_algorithm as ecod
import ecod_ee_algorithm as alg
import ecod_isolation_forest as isf
import ecod_lof as lof
from ecod_evaluation import fisher_threshold


def _reference(X, variant):
    """Wert je Tour mit expliziten Schleifen (keine Vektorisierung, kein searchsorted)."""
    n, p = X.shape
    out = np.zeros(n)
    gam = [skew(X[:, j]) for j in range(p)]
    for i in range(n):
        left, right, auto, two = 0.0, 0.0, 0.0, 0.0
        for j in range(p):
            fl = sum(1 for v in X[:, j] if v <= X[i, j]) / n
            fr = sum(1 for v in X[:, j] if v >= X[i, j]) / n
            left += -np.log(fl)
            right += -np.log(fr)
            auto += -np.log(fl) if gam[j] < 0 else -np.log(fr)
            two += -np.log(min(1.0, 2 * min(fl, fr)))
        out[i] = {"paper": max(left, right, auto), "auto": auto, "twosided": two}[variant]
    return out


# --- Handinstanzen ------------------------------------------------------------------------------------------------------------------------


def test_tail_probabilities_hand_instance_with_ties():
    X = np.array([[1.0], [2.0], [2.0], [3.0], [10.0]])
    left, right = ecod.tail_probabilities(X)
    assert np.allclose(left[:, 0], [0.2, 0.6, 0.6, 0.8, 1.0])                 # #{x_i <= x} / n, Bindungen zählen mit
    assert np.allclose(right[:, 0], [1.0, 0.8, 0.8, 0.4, 0.2])                # #{x_i >= x} / n
    u_l, u_r = ecod.contributions(X)
    assert np.allclose(u_l[:, 0], -np.log([0.2, 0.6, 0.6, 0.8, 1.0])) and np.allclose(u_r[:, 0], -np.log([1.0, 0.8, 0.8, 0.4, 0.2]))
    assert (left >= 1 / 5 - 1e-12).all() and (right <= 1.0).all() and np.isfinite(u_l).all() and np.isfinite(u_r).all()      # nie unendlich: kleinster Wert 1 / n


def test_two_feature_hand_instance_for_the_three_sums():
    """Vier Touren, zwei Merkmale, ohne Bindungen: von Hand gerechnet."""
    X = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [10.0, 1.0]])
    # Merkmal 1: links 1/4, 2/4, 3/4, 1 ; rechts 1, 3/4, 2/4, 1/4.  Merkmal 2: links 1, 3/4, 2/4, 1/4 ; rechts 1/4, 2/4, 3/4, 1
    O_l = -np.log([1 / 4 * 1, 2 / 4 * 3 / 4, 3 / 4 * 2 / 4, 1 * 1 / 4])
    O_r = -np.log([1 * 1 / 4, 3 / 4 * 2 / 4, 2 / 4 * 3 / 4, 1 / 4 * 1])
    assert np.allclose(O_l, O_r)                                              # symmetrisch angelegt
    assert ecod.skewness(X)[0] > 0 and ecod.skewness(X)[1] == pytest.approx(0.0)
    s = ecod.score(X, "paper")
    assert np.allclose(s, O_l) and s[0] == pytest.approx(np.log(4)) and s[3] == pytest.approx(np.log(4))
    assert np.allclose(ecod.score(X, "twosided"), -np.log(np.minimum(1, 2 * np.minimum([1 / 4, 2 / 4, 3 / 4, 1], [1, 3 / 4, 2 / 4, 1 / 4]))) - np.log(np.minimum(1, 2 * np.minimum([1, 3 / 4, 2 / 4, 1 / 4], [1 / 4, 2 / 4, 3 / 4, 1]))))


def test_skew_direction_selects_the_tail_for_auto():
    rng = np.random.default_rng(0)
    right_skewed = np.exp(rng.standard_normal((200, 1)))
    left_skewed = -np.exp(rng.standard_normal((200, 1)))
    u_l, u_r = ecod.contributions(right_skewed)
    assert ecod.skewness(right_skewed)[0] > 0 and np.allclose(ecod.score(right_skewed, "auto"), u_r.sum(axis=1))
    u_l, u_r = ecod.contributions(left_skewed)
    assert ecod.skewness(left_skewed)[0] < 0 and np.allclose(ecod.score(left_skewed, "auto"), u_l.sum(axis=1))
    assert ecod.skewness(np.ones((5, 1)))[0] == 0.0                            # konstantes Merkmal


@pytest.mark.parametrize("variant", ["paper", "auto", "twosided"])
def test_vectorised_score_equals_the_explicit_loop_reference(variant):
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.standard_normal((40, 3)), np.round(rng.standard_normal((20, 3)) * 2) / 2])              # mit Bindungen
    assert np.allclose(ecod.score(X, variant), _reference(X, variant), atol=1e-12)


def test_per_feature_contribution_sums_to_the_score_for_every_variant():
    rng = np.random.default_rng(2)
    X = np.exp(rng.standard_normal((60, 5)))
    for v in ecod.VARIANTS:
        assert np.allclose(ecod.per_feature_contribution(X, v).sum(axis=1), ecod.score(X, v), atol=1e-12), v


def test_pyod_variant_equals_pyod_if_installed():
    pyod_ecod = pytest.importorskip("pyod.models.ecod")
    rng = np.random.default_rng(3)
    X = np.concatenate([rng.standard_normal((150, 6)) * [1, 2, 3, 1, 1, 5], 6 + rng.standard_normal((10, 6))])
    X[:, 2] = np.exp(X[:, 2] / 3)
    model = pyod_ecod.ECOD().fit(X)
    assert np.allclose(ecod.score(X, "pyod"), model.decision_scores_, atol=1e-9)


# --- Eigenschaften -------------------------------------------------------------------------------------------------------------------------


def test_rank_invariance_under_monotone_transformations_per_feature():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((100, 4))
    Y = X.copy()
    Y[:, 0] = np.exp(3 * X[:, 0])
    Y[:, 1] = 1000 * X[:, 1] + 7
    Y[:, 2] = X[:, 2] ** 3
    Y[:, 3] = np.arctan(X[:, 3])
    for v in ("twosided",):
        assert np.allclose(ecod.score(X, v), ecod.score(Y, v))                    # nur die Ränge zählen (Schiefe fällt in der zweiseitigen Variante heraus)
    assert np.allclose(ecod.tail_probabilities(X)[0], ecod.tail_probabilities(Y)[0]) and np.allclose(ecod.tail_probabilities(X)[1], ecod.tail_probabilities(Y)[1])
    a = alg.fit_classical(X)
    b = alg.fit_classical(Y)
    assert not np.allclose(a.d2, b.d2)                                            # Mahalanobis-Abstände dagegen nicht


def test_ecod_is_deterministic_and_row_permutation_equivariant():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((80, 5))
    s = ecod.score(X)
    perm = rng.permutation(80)
    assert np.array_equal(s, ecod.score(X)) and np.allclose(ecod.score(X[perm]), s[perm])


def test_marginal_extremes_get_the_maximum_contribution_log_n():
    X = np.arange(50.0)[:, None] * np.ones((1, 3))
    assert ecod.score(X, "twosided")[0] == pytest.approx(3 * np.log(25))                                                  # -log(2 / 50) je Merkmal
    left, right = ecod.tail_probabilities(X)
    assert left[0, 0] == pytest.approx(1 / 50) and right[-1, 0] == pytest.approx(1 / 50)
    assert ecod.score(X, "paper").max() == pytest.approx(3 * np.log(50))


def test_twosided_score_under_independence_is_gamma_and_the_fisher_threshold_is_calibrated():
    """Unter Unabhängigkeit und stetigen Randverteilungen: Summe der zweiseitigen Beiträge ~ Gamma(p, 1) (Fisher-Methode); das Quantil trifft den Sollwert."""
    rng = np.random.default_rng(6)
    n, p = 4000, 5
    s = ecod.score(rng.standard_normal((n, p)), "twosided")
    assert abs(s.mean() - p) < 0.15 and abs(s.var() - p) < 0.6
    assert kstest(s, gamma(a=p).cdf).pvalue > 0.001
    thr = fisher_threshold(p, 0.975)
    assert thr == pytest.approx(chi2.ppf(0.975, 2 * p) / 2, rel=1e-6)
    assert abs((s > thr).mean() - 0.025) < 0.012
    # die Original-Variante (Maximum dreier Summen) hat die Verteilung verschoben: mehr als der Sollwert
    assert (ecod.score(rng.standard_normal((n, p)), "paper") > thr).mean() > 0.03


def test_fisher_threshold_grows_with_the_number_of_features_and_the_quantile():
    assert fisher_threshold(2, 0.975) < fisher_threshold(12, 0.975) < fisher_threshold(30, 0.975)
    assert fisher_threshold(12, 0.9) < fisher_threshold(12, 0.975) < fisher_threshold(12, 0.999)


def test_ecod_finds_scattered_outliers_and_misses_a_pure_correlation_break():
    rng = np.random.default_rng(7)
    base = rng.standard_normal((300, 1))
    normal = base + 0.3 * rng.standard_normal((300, 4))
    # Streuung in alle Richtungen: fällt in mindestens einem Rand auf
    out = 6 * rng.choice([-1.0, 1.0], size=(15, 4)) + rng.standard_normal((15, 4))
    X = np.concatenate([normal, out])
    y = np.arange(315) >= 300
    from ecod_evaluation import roc_auc
    assert roc_auc(ecod.score(X, "twosided"), y) > 0.97 and roc_auc(ecod.score(X), y) > 0.9
    # Korrelationsbruch: dieselben Randverteilungen, unabhängig gezogen
    broken = np.stack([normal[rng.integers(0, 300, 15), j] for j in range(4)], axis=1)
    X2 = np.concatenate([normal, broken])
    assert roc_auc(ecod.score(X2), y) < 0.7 and roc_auc(alg.fit_classical(X2).d2, y) > 0.9


# --- Kopierte Komponenten der Vorgänger -----------------------------------------------------------------------------------------------


def test_copied_components_still_behave_like_their_predecessors():
    rng = np.random.default_rng(8)
    X = np.concatenate([rng.standard_normal((200, 3)), 7 + rng.standard_normal((10, 3))])
    y = np.arange(210) >= 200
    f = lof.fit_lof(X, 20)
    assert f.lof[y].min() > 1.5 and np.median(f.lof[~y]) < 1.2
    forest = isf.fit_forest(X, 100, 128, 0)
    sc = isf.score_from_paths(isf.path_lengths(forest, X), forest.psi)
    assert sc[y].min() > sc[~y].mean() and (sc >= 0).all() and (sc <= 1).all()
    r = alg.fit_mcd(X, 0.75, 0)
    assert r.d2[y].min() > alg.threshold(r, 0.975)
    assert alg.chi2_ppf(0.975, 12) == pytest.approx(chi2.ppf(0.975, 12), rel=1e-6)
