"""Szenario (identisch zu den Vorgänger-Demos, dazu der Korrelationsbruch), Kennzahlen, Analyse und Schwellen für vier Detektoren, Varianten, Tabellen, Urteil."""

import numpy as np
import pytest

import ecod_algorithm as ecod
import ecod_constants as C
import ecod_ee_algorithm as alg
import ecod_evaluation as ev
import ecod_isolation_forest as isf
import ecod_lof as lof

# Zeilensummen der ersten acht Zeilen der PCA-Demo (dieselben normalen Zeilen wie in den Vorgänger-Demos): permutationsinvariant, eingefroren
PCA_ROW_SUMS = [39967.27507413389, 42083.657538741994, 40172.02304566072, 44766.540605465496, 45034.44402326185, 55757.45917619934, 50727.55911151846, 43923.05872324106]


# --- Szenario ------------------------------------------------------------------------------------------------------------------------


def test_normal_rows_equal_the_pca_and_the_predecessor_rows():
    ds = ev.make_dataset(contamination=1)
    assert not ds.anomaly[:8].any() and ds.X.shape == (300, 12)
    assert np.allclose(ds.X[:8].sum(axis=1), PCA_ROW_SUMS, rtol=1e-12)


def test_default_dataset_is_frozen_at_the_predecessor_values():
    ds = ev.make_dataset()
    assert float(ds.X.sum()) == pytest.approx(13396385.215115668, rel=1e-12) and int(ds.anomaly.sum()) == 30 and ds.X[0, 0] == pytest.approx(19701.54827513291, rel=1e-12)


def test_noise_features_are_appended_and_change_nothing_else():
    base = ev.make_dataset()
    for n_noise in (1, 10, 40):
        ds = ev.make_dataset(n_noise=n_noise)
        assert ds.X.shape == (300, 12 + n_noise) and np.array_equal(ds.X[:, :12], base.X) and np.array_equal(ds.anomaly, base.anomaly)
        assert len(ds.names) == 12 + n_noise and ds.names[-1] == f"Rauschmerkmal {n_noise}" and ds.n_noise == n_noise
    assert np.array_equal(ev.make_dataset(p=30, n_noise=5).X[:, :30], ev.make_dataset(p=30).X)


@pytest.mark.parametrize("n,pct,expected", [(300, 10, 30), (300, 1, 3), (20, 1, 1), (30, 10, 3), (600, 45, 270)])
def test_contamination_is_exact_with_at_least_one_anomaly(n, pct, expected):
    assert int(ev.make_dataset(n=n, contamination=pct).anomaly.sum()) == expected


def test_gap_needs_two_modes_and_kinds_have_their_geometry():
    assert ev.make_dataset(kind="gap").kind == "scattered"
    gap = ev.make_dataset(n_modes=2, kind="gap")
    assert np.abs(gap.z[gap.anomaly]).max() < 1.2
    clu = ev.make_dataset(kind="cluster", contamination=20)
    assert np.linalg.norm(clu.z[clu.anomaly].mean(axis=0) - 6.0 * np.array([np.cos(C.CLUSTER_ANGLE), np.sin(C.CLUSTER_ANGLE)])) < 0.15


def test_decorrelated_anomalies_keep_the_marginals_and_lose_the_dependence():
    base = ev.make_dataset()
    ds = ev.make_dataset(kind="decorrelated", contamination=20)
    assert ds.kind == "decorrelated" and int(ds.anomaly.sum()) == 60 and ds.X.shape == base.X.shape
    normal = ds.X[~ds.anomaly]
    anom = ds.X[ds.anomaly]
    for j in range(ds.p):
        assert set(np.round(anom[:, j], 9)) <= set(np.round(normal[:, j], 9))                          # jede Spalte stammt aus den Normalen
    assert np.array_equal(normal, ev.make_dataset(kind="scattered", contamination=20).X[~ev.make_dataset(kind="scattered", contamination=20).anomaly])
    ds = ev.make_dataset(kind="decorrelated", contamination=40, n=600)
    c_norm = np.corrcoef(ds.X[~ds.anomaly].T)
    c_anom = np.corrcoef(ds.X[ds.anomaly].T)
    off = ~np.eye(ds.p, dtype=bool)
    assert np.abs(c_norm[off]).mean() > 0.4 and np.abs(c_anom[off]).mean() < 0.15                       # Abhängigkeit zerstört
    for j in range(ds.p):                                                                                 # gleiche Randverteilung: Quantile der Anomalien im Bereich der Normalen
        q_n, q_a = np.quantile(ds.X[~ds.anomaly][:, j], [0.05, 0.5, 0.95]), np.quantile(ds.X[ds.anomaly][:, j], [0.05, 0.5, 0.95])
        assert np.all(np.abs(q_n - q_a) < 0.35 * (q_n[2] - q_n[0]))


def test_decorrelated_is_drawn_last_so_other_kinds_and_defaults_stay_bit_identical():
    a = ev.make_dataset(seed=11, kind="scattered")
    b = ev.make_dataset(seed=11, kind="decorrelated")
    assert np.array_equal(a.X[~a.anomaly], b.X[~b.anomaly]) and np.array_equal(a.anomaly, b.anomaly)
    assert not np.array_equal(a.X[a.anomaly], b.X[b.anomaly])


def test_dataset_is_deterministic():
    assert np.array_equal(ev.make_dataset(seed=3, n_noise=4, kind="decorrelated").X, ev.make_dataset(seed=3, n_noise=4, kind="decorrelated").X)


# --- Kennzahlen: Handinstanzen ------------------------------------------------------------------------------------------------------------


def test_roc_auc_and_average_precision_hand_instances():
    assert ev.roc_auc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert ev.roc_auc(np.array([1, 1, 1, 1.0]), np.array([0, 0, 1, 1], bool)) == 0.5
    assert ev.roc_auc(np.array([1, 4, 2, 3.0]), np.array([0, 1, 1, 0], bool)) == pytest.approx(0.75)
    assert np.isnan(ev.roc_auc(np.array([1.0, 2.0]), np.array([False, False])))
    rng = np.random.default_rng(0)
    s, y = rng.standard_normal(200), rng.random(200) < 0.3
    assert ev.roc_auc(s, y) == pytest.approx(np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]]))
    assert ev.average_precision(np.array([4, 3, 2, 1.0]), np.array([1, 0, 1, 0], bool)) == pytest.approx((1 + 2 / 3) / 2)


def test_roc_curve_and_flag_metrics():
    rng = np.random.default_rng(1)
    s, y = rng.standard_normal(300), rng.random(300) < 0.2
    fpr, tpr = ev.roc_curve(s, y)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0.0, 0.0, 1.0, 1.0) and np.trapezoid(tpr, fpr) == pytest.approx(ev.roc_auc(s, y), abs=1e-9)
    m = ev.flag_metrics(np.array([1, 1, 0, 0, 1, 0], bool), np.array([1, 0, 1, 0, 0, 0], bool))
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == 0.5 and m["false_alarm"] == pytest.approx(0.5) and m["f1"] == pytest.approx(0.4)


def test_lof_k_is_twenty_but_at_most_half_the_tours():
    assert [ev.lof_k(n) for n in (8, 20, 30, 100, 300, 600)] == [4, 10, 15, 20, 20, 20]


def test_fisher_threshold_is_the_gamma_quantile():
    from scipy.stats import gamma
    for p, q in ((2, 0.975), (12, 0.975), (30, 0.99)):
        assert ev.fisher_threshold(p, q) == pytest.approx(gamma(a=p).ppf(q), rel=1e-6)


# --- Analyse und Schwellen ----------------------------------------------------------------------------------------------------------------


def _params(**kw):
    p = {**ev.DEFAULT_DATA, **kw}
    return (p["n"], p["p"], p["n_noise"], p["n_modes"], p["curvature"], p["noise"], p["contamination"], p["kind"], p["strength"], 7)


def test_analysis_fields_and_consistency():
    a = ev.analyse_for(_params())
    assert set(a.scores) == set(ev.DETECTORS) == set(a.flags) == set(a.values) == set(a.oracle_f1) == {"ecod", "lof", "iforest", "robust"}
    assert a.p_total == 12 and a.contributions.shape == (300, 12) and a.forest_if.psi == 256 and a.robust.h == 156
    assert np.allclose(a.values["ecod"], ecod.score(a.ds.X, "paper")) and np.allclose(a.contributions.sum(axis=1), a.values["ecod"])
    assert np.allclose(a.values["lof"], lof.fit_lof(ev.standardise(a.ds.X), 20).lof) and np.allclose(a.values["iforest"], isf.score_from_paths(isf.path_lengths(a.forest_if, a.ds.X), a.forest_if.psi))
    for d in ev.DETECTORS:
        assert a.scores[d]["n_flagged"] == int(a.flags[d].sum()) and 0 <= a.oracle_f1[d] <= 1
    assert a.scores["ecod"]["threshold"] == pytest.approx(ev.fisher_threshold(12, 0.975)) and (a.flags["ecod"] == (a.values["ecod"] > a.scores["ecod"]["threshold"])).all()
    assert a.scores["lof"]["threshold"] == 1.5 and a.scores["iforest"]["threshold"] == 0.5 and a.scores["robust"]["threshold"] == pytest.approx(alg.threshold(a.robust, 0.975))
    assert a.seconds["ecod"] > 0 and a.seconds["lof"] > 0 and a.seconds["iforest"] > 0


def test_share_threshold_flags_exactly_the_assumed_share_for_all_four_detectors():
    for share in (2, 10, 40):
        a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=share))
        for d in ev.DETECTORS:
            assert int(a.flags[d].sum()) == max(1, round(share / 100 * 300))
            assert a.flags[d][np.argsort(-a.values[d])[:3]].all()
    a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=10))
    assert all(a.scores[d]["f1"] == pytest.approx(a.oracle_f1[d]) for d in ev.DETECTORS)                    # 10 % = wahrer Anteil


def test_ecod_quantile_and_chi2_quantile_move_only_their_own_detector():
    base = ev.analyse_for(_params())
    e = ev.analyse_for(_params(), ev.Settings(ecod_quantile=0.999))
    assert e.scores["ecod"]["false_alarm"] < base.scores["ecod"]["false_alarm"] and e.scores["ecod"]["auc"] == base.scores["ecod"]["auc"]
    assert all(e.scores[d] == base.scores[d] for d in ("lof", "iforest", "robust"))
    q = ev.analyse_for(_params(), ev.Settings(quantile=0.999))
    assert q.scores["ecod"] == base.scores["ecod"] and q.scores["robust"]["false_alarm"] <= base.scores["robust"]["false_alarm"] and q.scores["robust"]["auc"] == base.scores["robust"]["auc"]


def test_variant_changes_only_ecod():
    a = ev.analyse_for(_params())
    b = ev.analyse_for(_params(), ev.Settings(variant="twosided"))
    assert not np.allclose(a.values["ecod"], b.values["ecod"]) and b.scores["ecod"]["false_alarm"] < a.scores["ecod"]["false_alarm"]
    assert all(a.scores[d] == b.scores[d] for d in ("lof", "iforest", "robust"))
    assert np.allclose(b.contributions.sum(axis=1), b.values["ecod"])


def test_forest_seed_is_separate_from_the_data_seed_and_ecod_and_lof_are_deterministic():
    a, b = ev.analyse_for(_params()), ev.analyse_for(_params(), ev.Settings(start=5))
    assert np.array_equal(a.ds.X, b.ds.X) and a.values["iforest"].tolist() != b.values["iforest"].tolist()
    assert np.array_equal(a.values["ecod"], b.values["ecod"]) and np.array_equal(a.values["lof"], b.values["lof"])


def test_noise_features_are_used_and_few_tours_many_features_work():
    b = ev.analyse_for(_params(n_noise=10))
    assert b.ds.X.shape[1] == 22 and b.p_total == 22 and b.scores["ecod"]["threshold"] == pytest.approx(ev.fisher_threshold(22, 0.975))
    c = ev.analyse_for(_params(n=20, p=30))
    assert c.scores["ecod"]["auc"] > 0.95 and c.forest_if.psi == 20


def test_sweep_rows_labels_and_the_variant_sweep():
    rows = ev.sweep("contamination", values=(5, 10))
    assert [r["x"] for r in rows] == [5, 10]
    for r in rows:
        for d in ev.DETECTORS:
            assert r[f"{d}_auc_min"] <= r[f"{d}_auc"] <= r[f"{d}_auc_max"] and r[f"{d}_auc_std"] >= 0 and f"{d}_oracle_f1" in r and f"{d}_seconds" in r
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    v = ev.sweep("variant")
    assert [r["x"] for r in v] == list(ev.VARIANTS) and v[0]["lof_auc"] == v[1]["lof_auc"] == v[2]["lof_auc"] and v[0]["ecod_auc"] != v[2]["ecod_auc"]
    q = ev.sweep("ecod_quantile", values=(0.9, 0.999))
    assert q[0]["ecod_false_alarm"] > q[1]["ecod_false_alarm"] and q[0]["ecod_auc"] == q[1]["ecod_auc"] and q[0]["robust_recall"] == q[1]["robust_recall"]
    n = ev.sweep("n", values=(20,))
    assert n[0]["lof_auc"] > 0.9                                                                        # k = 20 würde bei n = 20 klemmen; der Vergleich begrenzt k auf n / 2


def test_settings_defaults_agree_with_the_constants():
    s = ev.Settings()
    assert (s.variant, s.threshold_kind, s.ecod_quantile, s.quantile, s.share) == (C.DEFAULT_VARIANT, C.DEFAULT_THRESHOLD_KIND, C.DEFAULT_ECOD_QUANTILE, C.DEFAULT_QUANTILE, C.DEFAULT_SHARE)
    assert set(ev.DEFAULT_DATA) == set(ev.DATA_KEYS) and C.SWEEP_SEEDS == tuple(range(100000, 100005)) and set(C.VARIANTS) == set(ev.VARIANTS)


# --- Experimente: Form der Tabellen ----------------------------------------------------------------------------------------------------------


def test_scenario_variants_calibration_cost_and_threshold_tables_have_their_documented_shape():
    sc = ev.scenario_table(scenarios=ev.SCENARIOS[:2])
    assert [r["scenario"] for r in sc] == [s[0] for s in ev.SCENARIOS[:2]] and all(f"{d}_auc" in sc[0] for d in ev.DETECTORS)
    assert len(ev.SCENARIOS) == 9
    vt = ev.variants_table()
    assert [r["scenario"] for r in vt] == [s[0] for s in ev.SCENARIOS] and set(vt[0]) == {"scenario", *ev.VARIANTS} and set(vt[0]["paper"]) == {"auc", "oracle_f1"}
    cal = ev.calibration_table()
    assert [r["p"] for r in cal] == [2, 5, 12, 20, 30] and all({"paper", "twosided", "paper_indep", "twosided_indep", "nominal"} <= set(r) for r in cal) and cal[0]["nominal"] == pytest.approx(0.025)
    ct = ev.cost_table()["times"]
    assert [(t["n"], t["p"]) for t in ct] == [(100, 12), (300, 12), (600, 12), (100, 52), (300, 52), (600, 52)]
    tt = ev.threshold_table(ev.Settings())
    assert [r["x"] for r in tt["cutoff"]] == list(ev.SWEEP_VALUES["ecod_quantile"]) and [r["factor"] for r in tt["wrong_share"]] == [0.5, 1.0, 2.0] and [r["x"] for r in tt["wrong_share"]] == [5, 10, 20]


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------


def _verdict(**kw):
    settings = ev.Settings(**{k: kw.pop(k) for k in ("variant", "threshold_kind", "share", "ecod_quantile") if k in kw})
    return ev.verdict(ev.analyse_for(_params(**kw), settings))


def test_verdict_codes_for_the_presets_and_edge_cases():
    assert _verdict()[:2] == ("warning", "threshold_off")
    assert _verdict(variant="twosided")[:2] == ("success", "comparable")
    assert _verdict(kind="cluster", contamination=30)[:2] == ("success", "ecod_wins")
    assert _verdict(kind="decorrelated")[:2] == ("warning", "blind")
    assert _verdict(n_modes=2, kind="gap")[:2] == ("warning", "gap")
    assert _verdict(contamination=45)[1] in ("comparable", "threshold_off")
    assert _verdict(n_noise=40)[1] in ("comparable", "others_win", "threshold_off")


def test_verdict_data_carries_the_numbers_the_messages_use():
    kind, code, data = _verdict(kind="decorrelated")
    assert code == "blind"
    for key in ("ecod_auc", "lof_auc", "iforest_auc", "robust_auc", "best_other_auc", "ecod_recall", "ecod_false_alarm", "ecod_f1", "ecod_threshold", "lof_f1", "iforest_f1", "robust_f1", "oracle_ecod",
                "contamination", "n_anomalies", "n", "p", "kind", "variant"):
        assert key in data
    assert data["p"] == 12 and data["n_anomalies"] == 30 and data["contamination"] == pytest.approx(10.0)


def test_analysis_time_stays_small():
    a = ev.analyse(ev.make_dataset(n=600, p=30, n_noise=40, contamination=45))
    assert a.seconds["ecod"] < 0.5 and a.seconds["lof"] < 1.5 and a.seconds["iforest"] < 4.0 and a.seconds["robust"] < 10.0
