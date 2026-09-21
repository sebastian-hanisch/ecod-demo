"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Datensätze belegt (Mittel; Toleranz ±0.02 = Rundung auf zwei Stellen plus Luft).
Positive UND negative Aussagen: wo ECOD gegen LOF, Isolation Forest oder die Wurzel verliert, steht das hier ebenso als Test wie dort, wo es gewinnt."""

from functools import lru_cache

import numpy as np
import pytest

import ecod_constants as C
import ecod_evaluation as ev

TOL = 0.02
STANDARD = ev.Settings()
TWOSIDED = ev.Settings(variant="twosided")


@lru_cache(maxsize=None)
def _runs(items, settings):
    return tuple(ev._analyse_seed(s, settings, dict(items)) for s in C.SWEEP_SEEDS)


def runs(settings=STANDARD, **kw):
    return _runs(tuple(sorted(kw.items())), settings)


def m(det, key, settings=STANDARD, **kw):
    return float(np.nanmean([a.scores[det][key] for a in runs(settings, **kw)]))


def oracle(det, settings=STANDARD, **kw):
    return float(np.mean([a.oracle_f1[det] for a in runs(settings, **kw)]))


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


@lru_cache(maxsize=None)
def variants():
    return {r["scenario"]: r for r in ev.variants_table()}


@lru_cache(maxsize=None)
def calibration():
    return {r["p"]: r for r in ev.calibration_table(0.975)}


# --- Seitenleiste: Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil -------------------------------------------------


@pytest.mark.parametrize("n,auc,f1,fa,lof_f1,if_f1", [(20, 0.99, 0.75, 0.08, 0.96, 0.60), (30, 0.99, 0.59, 0.14, 0.94, 0.67), (50, 0.98, 0.60, 0.15, 0.88, 0.78), (100, 0.99, 0.61, 0.15, 0.86, 0.89), (200, 0.99, 0.58, 0.16, 0.91, 0.96)])
def test_tours_sweep(n, auc, f1, fa, lof_f1, if_f1):
    near(m("ecod", "auc", n=n), auc, 0.01)
    near(m("ecod", "f1", n=n), f1)
    near(m("ecod", "false_alarm", n=n), fa, 0.015)
    near(m("lof", "f1", n=n), lof_f1)
    near(m("iforest", "f1", n=n), if_f1)
    near(m("lof", "auc", n=n), 1.00, 0.01)
    near(m("iforest", "auc", n=n), 1.00, 0.01)


@pytest.mark.parametrize("p,auc,fa", [(2, 0.98, 0.003), (5, 0.97, 0.06), (8, 0.99, 0.13), (12, 0.99, 0.20), (20, 0.99, 0.19), (30, 0.99, 0.22)])
def test_features_sweep_the_ranking_holds_but_the_fisher_threshold_drifts(p, auc, fa):
    near(m("ecod", "auc", p=p), auc, 0.012)
    near(m("ecod", "false_alarm", p=p), fa, 0.015)


@pytest.mark.parametrize("nn,auc,ecod_recall,lof_recall,if_recall", [(0, 0.99, 0.99, 1.00, 0.99), (5, 0.98, 0.99, 0.93, 0.95), (10, 0.98, 0.96, 0.71, 0.91), (20, 0.97, 0.91, 0.18, 0.71), (30, 0.96, 0.88, 0.03, 0.58), (40, 0.95, 0.84, 0.00, 0.39)])
def test_noise_features_hurt_ecod_less_than_lof_at_the_thresholds(nn, auc, ecod_recall, lof_recall, if_recall):
    near(m("ecod", "auc", n_noise=nn), auc, 0.012)
    near(m("ecod", "recall", n_noise=nn), ecod_recall)
    near(m("lof", "recall", n_noise=nn), lof_recall)
    near(m("iforest", "recall", n_noise=nn), if_recall)
    near(m("lof", "auc", n_noise=nn), 0.99 if nn else 1.0, 0.012)
    near(m("iforest", "auc", n_noise=nn), 0.99 if nn else 1.0, 0.012)


def test_noise_features_hurt_the_robust_estimate_in_the_ranking():
    near(m("robust", "auc", n_noise=0), 1.00, 0.01)
    near(m("robust", "auc", n_noise=40), 0.88, 0.015)


@pytest.mark.parametrize("modes,robust_auc", [(1, 1.00), (2, 0.95), (3, 0.85)])
def test_modes_sweep(modes, robust_auc):
    near(m("ecod", "auc", n_modes=modes), 0.99, 0.012)
    near(m("robust", "auc", n_modes=modes), robust_auc, 0.015)


@pytest.mark.parametrize("curv,f1,fa,lof_f1", [(0.0, 0.52, 0.20, 0.94), (0.25, 0.62, 0.13, 0.90), (0.5, 0.75, 0.07, 0.83), (0.75, 0.85, 0.04, 0.78), (1.0, 0.88, 0.029, 0.74)])
def test_curvature_improves_the_fisher_threshold_and_hurts_lof(curv, f1, fa, lof_f1):
    near(m("ecod", "auc", curvature=curv), 0.99, 0.012)
    near(m("ecod", "f1", curvature=curv), f1)
    near(m("ecod", "false_alarm", curvature=curv), fa, 0.015)
    near(m("lof", "f1", curvature=curv), lof_f1)


def test_curvature_decorrelates_the_kpis():
    def mean_abs_corr(curv):
        ds = ev.make_dataset(curvature=curv, seed=C.SWEEP_SEEDS[0])
        c = np.corrcoef(ds.X[~ds.anomaly].T)
        return float(np.abs(c[~np.eye(12, dtype=bool)]).mean())
    near(mean_abs_corr(0.0), 0.51, 0.04)
    near(mean_abs_corr(1.0), 0.31, 0.04)


@pytest.mark.parametrize("noise,f1,fa", [(0.0, 0.51, 0.21), (0.25, 0.52, 0.20), (0.5, 0.57, 0.16), (1.0, 0.70, 0.09)])
def test_measurement_noise_decorrelates_and_lowers_false_alarms(noise, f1, fa):
    near(m("ecod", "auc", noise=noise), 0.99, 0.012)
    near(m("ecod", "f1", noise=noise), f1)
    near(m("ecod", "false_alarm", noise=noise), fa, 0.015)


@pytest.mark.parametrize("c,ecod_auc,lof_auc,f1,fa", [(2, 0.99, 1.00, 0.13, 0.27), (5, 0.99, 1.00, 0.29, 0.26), (10, 0.99, 1.00, 0.52, 0.20), (20, 0.99, 0.99, 0.83, 0.09), (30, 0.98, 0.90, 0.88, 0.035),
                                              (40, 0.98, 0.72, 0.85, 0.004), (45, 0.98, 0.64, 0.84, 0.004)])
def test_contamination_ecod_is_insensitive_to_many_scattered_anomalies(c, ecod_auc, lof_auc, f1, fa):
    near(m("ecod", "auc", contamination=c), ecod_auc, 0.012)
    near(m("lof", "auc", contamination=c), lof_auc)
    near(m("ecod", "f1", contamination=c), f1)
    near(m("ecod", "false_alarm", contamination=c), fa, 0.015)


def test_contamination_robust_estimate_breaks_down():
    near(m("robust", "auc", contamination=2), 1.00, 0.01)
    near(m("robust", "auc", contamination=45), 0.87, 0.015)


# --- Art der Anomalien --------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kw,ecod,iforest,lof,robust", [(dict(kind="cluster"), 0.99, 0.95, 0.35, 1.00), (dict(kind="cluster", contamination=30), 0.86, 0.70, 0.47, 0.49),
                                                         (dict(n_modes=2, kind="gap"), 0.08, 0.54, 0.53, 0.40), (dict(kind="decorrelated"), 0.40, 0.80, 0.99, 1.00)])
def test_kind_help_numbers(kw, ecod, iforest, lof, robust):
    near(m("ecod", "auc", **kw), ecod, 0.03)
    near(m("iforest", "auc", **kw), iforest)
    near(m("lof", "auc", **kw), lof)
    near(m("robust", "auc", **kw), robust)


@pytest.mark.parametrize("nm,c", [(2, 10), (3, 10)])
def test_gap_is_below_chance_for_ecod_and_nobody_finds_it(nm, c):
    for det in ev.DETECTORS:
        assert m(det, "auc", n_modes=nm, kind="gap", contamination=c) < 0.6
    assert m("ecod", "auc", n_modes=nm, kind="gap") < 0.1


def test_the_gap_others_range():
    values = [m(d, "auc", n_modes=nm, kind="gap") for d in ("lof", "iforest", "robust") for nm in (2, 3)]
    assert 0.25 <= min(values) and max(values) <= 0.56


def test_correlation_break_with_thirty_features():
    near(m("ecod", "auc", kind="decorrelated", p=30), 0.47, 0.03)
    near(m("lof", "auc", kind="decorrelated", p=30), 1.00, 0.02)
    near(m("iforest", "auc", kind="decorrelated", p=30), 0.83)
    near(m("robust", "auc", kind="decorrelated", p=30), 1.00, 0.02)


# --- ECOD-Regler: Variante, Fisher-Quantil, angenommener Anteil ------------------------------------------------------------------------------------


def test_variant_help_numbers():
    near(m("ecod", "auc"), 0.99, 0.012)
    near(m("ecod", "false_alarm"), 0.20, 0.015)
    near(m("ecod", "f1"), 0.52)
    near(m("ecod", "auc", TWOSIDED), 1.00, 0.012)
    near(m("ecod", "false_alarm", TWOSIDED), 0.045, 0.01)
    near(m("ecod", "f1", TWOSIDED), 0.83)
    near(m("ecod", "auc", ev.Settings(variant="auto")), 0.69, 0.03)
    near(oracle("ecod", TWOSIDED), 0.89)
    near(oracle("ecod"), 0.83)


@pytest.mark.parametrize("q,f1,fa,recall", [(0.9, 0.37, 0.39, 1.00), (0.95, 0.45, 0.28, 0.99), (0.975, 0.52, 0.20, 0.99), (0.99, 0.62, 0.13, 0.98), (0.999, 0.81, 0.03, 0.88)])
def test_fisher_quantile_help_numbers(q, f1, fa, recall):
    s = ev.Settings(ecod_quantile=q)
    near(m("ecod", "f1", s), f1)
    near(m("ecod", "false_alarm", s), fa, 0.015)
    near(m("ecod", "recall", s), recall)


@pytest.mark.parametrize("q,f1", [(0.9, 0.67), (0.95, 0.77), (0.975, 0.84), (0.99, 0.89), (0.999, 0.90)])
def test_chi2_quantile_help_numbers(q, f1):
    near(m("robust", "f1", ev.Settings(quantile=q)), f1)


@pytest.mark.parametrize("share,ecod,lof_if,robust", [(2, 0.33, 0.33, 0.33), (5, 0.66, 0.67, 0.67), (10, 0.83, 0.97, 0.90), (20, 0.65, 0.67, 0.66), (40, 0.40, 0.40, 0.40)])
def test_assumed_share_costs_all_detectors_about_the_same(share, ecod, lof_if, robust):
    s = ev.Settings(threshold_kind="share", share=share)
    near(m("ecod", "f1", s), ecod)
    near(m("lof", "f1", s), lof_if)
    near(m("iforest", "f1", s), lof_if)
    near(m("robust", "f1", s), robust)


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_standard_preset_numbers():
    near(m("ecod", "auc"), 0.99, 0.012)
    near(m("ecod", "f1"), 0.52)
    near(m("lof", "f1"), 0.94)
    near(m("iforest", "f1"), 0.96)
    near(m("robust", "f1"), 0.84)
    near(m("ecod", "false_alarm"), 0.20, 0.015)


def test_many_anomalies_and_noise_presets():
    near(m("ecod", "f1", contamination=45), 0.84)
    near(m("iforest", "auc", contamination=45), 1.00, 0.01)
    near(m("iforest", "f1", contamination=45), 0.81)
    near(m("ecod", "recall", n_noise=40), 0.84)
    near(m("ecod", "f1", n_noise=40), 0.66)
    near(m("iforest", "recall", n_noise=40), 0.39)
    near(m("lof", "recall", n_noise=40), 0.0, 0.02)
    near(m("ecod", "recall", kind="cluster", contamination=30), 0.86)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario,ecod_auc", [("Standardfall", 0.99), ("dichte Gruppe 10 %", 0.99), ("dichte Gruppe 30 %", 0.86), ("Lücke, 2 Betriebsarten", 0.08), ("Lücke, 3 Betriebsarten", 0.04),
                                               ("Korrelationsbruch", 0.40), ("Korrelationsbruch, p = 30", 0.47), ("45 % verstreut", 0.98), ("40 Rauschmerkmale", 0.95)])
def test_scenario_table_ecod_auc(scenario, ecod_auc):
    kw = dict(dict(ev.SCENARIOS)[scenario])
    near(m("ecod", "auc", **kw), ecod_auc, 0.03)


def test_scenario_standard_case_known_share_f1s():
    near(oracle("ecod"), 0.83)
    near(oracle("lof"), 0.97)
    near(oracle("iforest"), 0.97)


@pytest.mark.parametrize("scenario,paper,twosided,auto", [("Standardfall", 0.99, 1.00, 0.69), ("dichte Gruppe 10 %", 0.99, 0.98, 1.00), ("dichte Gruppe 30 %", 0.86, 0.84, 0.99), ("45 % verstreut", 0.98, 1.00, 0.67)])
def test_variants_table(scenario, paper, twosided, auto):
    v = variants()[scenario]
    near(v["paper"]["auc"], paper, 0.012)
    near(v["twosided"]["auc"], twosided, 0.012)
    near(v["auto"]["auc"], auto, 0.03)


def test_variants_no_variant_finds_the_correlation_break_or_the_gap():
    for scenario in ("Korrelationsbruch", "Korrelationsbruch, p = 30"):
        assert max(variants()[scenario][v]["auc"] for v in ev.VARIANTS) <= 0.58
    for scenario in ("Lücke, 2 Betriebsarten", "Lücke, 3 Betriebsarten"):
        assert max(variants()[scenario][v]["auc"] for v in ev.VARIANTS) <= 0.47


@pytest.mark.parametrize("p,two_corr,paper_corr", [(2, 0.019, 0.018), (5, 0.065, 0.14), (12, 0.12, 0.29), (20, 0.18, 0.28), (30, 0.21, 0.32)])
def test_calibration_with_the_correlated_kpis(p, two_corr, paper_corr):
    near(calibration()[p]["twosided"], two_corr, 0.02)
    near(calibration()[p]["paper"], paper_corr, 0.02)


def test_calibration_under_independence_twosided_holds_and_paper_doubles():
    for p, row in calibration().items():
        assert 0.010 <= row["twosided_indep"] <= 0.022 and 0.04 <= row["paper_indep"] <= 0.065 and row["nominal"] == pytest.approx(0.025)


def test_mean_absolute_correlation_grows_with_the_feature_count():
    def mean_abs_corr(p):
        ds = ev.make_dataset(p=p, seed=C.SWEEP_SEEDS[0])
        c = np.corrcoef(ds.X[~ds.anomaly].T)
        return float(np.abs(c[~np.eye(p, dtype=bool)]).mean())
    near(mean_abs_corr(2), 0.28, 0.06)
    near(mean_abs_corr(30), 0.58, 0.05)


@pytest.mark.parametrize("nn,paper_auc,two_auc,two_f1,two_recall", [(0, 0.99, 1.00, 0.83, 0.99), (40, 0.95, 0.97, 0.74, 0.68)])
def test_noise_features_paper_and_twosided(nn, paper_auc, two_auc, two_f1, two_recall):
    near(m("ecod", "auc", n_noise=nn), paper_auc, 0.012)
    near(m("ecod", "auc", TWOSIDED, n_noise=nn), two_auc, 0.012)
    near(m("ecod", "f1", TWOSIDED, n_noise=nn), two_f1)
    near(m("ecod", "recall", TWOSIDED, n_noise=nn), two_recall, 0.03)
    if nn:
        near(m("ecod", "false_alarm", TWOSIDED, n_noise=nn), 0.015, 0.01)
        near(m("lof", "f1", n_noise=nn), 0.0, 0.02)
        near(m("iforest", "f1", n_noise=nn), 0.56)
        near(np.sqrt(nn), 6.3, 0.1)
        near(np.log(300), 5.7, 0.05)


@pytest.mark.parametrize("c,auc", [(2, 1.00), (5, 1.00), (10, 0.99), (20, 0.95), (30, 0.86), (40, 0.71), (45, 0.62)])
def test_dense_group_sweep_ecod_needs_no_k_and_no_covariance(c, auc):
    near(m("ecod", "auc", kind="cluster", n_modes=1, contamination=c), auc, 0.03)
    assert 0.17 <= m("ecod", "false_alarm", kind="cluster", n_modes=1, contamination=c) <= 0.40


def test_dense_group_others_at_large_shares():
    near(m("iforest", "auc", kind="cluster", contamination=40), 0.40)
    near(m("iforest", "auc", kind="cluster", contamination=45), 0.27)
    near(m("lof", "auc", kind="cluster", contamination=40), 0.49)
    near(m("lof", "auc", kind="cluster", contamination=45), 0.50)
    near(m("robust", "auc", kind="cluster", contamination=40), 0.39)
    near(m("robust", "auc", kind="cluster", contamination=45), 0.35)
    for c in (40, 45):
        assert m("ecod", "auc", kind="cluster", n_modes=1, contamination=c) > max(m(d, "auc", kind="cluster", contamination=c) for d in ("lof", "iforest", "robust"))


def test_threshold_table_wrong_share_costs_ecod_the_same_as_the_others():
    tt = ev.threshold_table(STANDARD)
    by_factor = {r["factor"]: r for r in tt["wrong_share"]}
    near(by_factor[1.0]["ecod_f1"], 0.83)
    near(by_factor[0.5]["ecod_f1"], 0.66)
    near(by_factor[2.0]["ecod_f1"], 0.65)
    near(by_factor[1.0]["lof_f1"], 0.97)
    near(by_factor[0.5]["lof_f1"], 0.67)


def test_cost_table_orders_of_magnitude():
    times = {(t["n"], t["p"]): t for t in ev.cost_table()["times"]}
    for t in times.values():
        assert t["ecod"] < 0.02 and t["iforest"] > 10 * t["ecod"]                                    # ECOD höchstens einige ms, der Isolation Forest deutlich mehr als 10-mal
    assert times[(600, 12)]["lof"] > 3 * times[(600, 12)]["ecod"] and times[(600, 52)]["lof"] > 1.5 * times[(600, 52)]["ecod"]     # LOF wächst quadratisch mit n
    assert times[(100, 52)]["lof"] < times[(100, 52)]["ecod"]                                         # ... bei kleinem n und vielen Merkmalen ist er schneller


# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------


def test_limits_table_numbers():
    near(m("ecod", "auc", kind="decorrelated"), 0.40, 0.03)
    near(m("lof", "auc", kind="decorrelated"), 0.99, 0.012)
    near(m("robust", "auc", kind="decorrelated"), 1.00, 0.012)
    near(m("iforest", "auc", kind="decorrelated"), 0.80)
    near(m("ecod", "false_alarm", TWOSIDED), 0.045, 0.01)
    near(oracle("ecod"), 0.83)
    near(oracle("lof"), 0.97)
    near(oracle("iforest"), 0.97)
