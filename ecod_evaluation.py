"""Auswertung der ECOD-Demo: Kennzahlen der Anomalie-Erkennung (AUC, mittlere Präzision, Precision/Recall/F1, Fehlalarmrate; aus der Wurzel-Demo übernommen), Analyse einer Aufnahme für ECOD, den LOF, den
Isolation Forest und die robuste Schätzung der Wurzel, Sweeps, Experimente auf Abruf (Szenarien, Varianten, Kalibrierung der Fisher-Schwelle, Kosten) und Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import ecod_algorithm as ecod
import ecod_lof as lof
import ecod_isolation_forest as isf
import ecod_ee_algorithm as alg
import ecod_constants as C
import ecod_scenario as sc


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------


def roc_auc(score, positive):
    """Fläche unter der ROC-Kurve über die Rangsumme (Mann-Whitney), Bindungen zählen halb. NaN, wenn eine Klasse fehlt."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = np.asarray(score)[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score, positive):
    """Mittlere Präzision (Fläche unter der Precision-Recall-Kurve als Summe über die Treffer). NaN ohne Anomalien."""
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    precision_at = np.cumsum(hits) / (np.arange(len(hits)) + 1.0)
    return float(precision_at[hits].sum() / positive.sum())


def flag_metrics(flagged, positive):
    """Precision, Recall, F1 und Fehlalarmrate (Anteil der Normalen, die markiert werden). Ohne Anomalien: Recall/F1 NaN; ohne Markierung: Precision 1 (nichts falsch)."""
    flagged, positive = np.asarray(flagged, bool), np.asarray(positive, bool)
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if n_pos == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if n_pos and (precision + recall) > 0 else (float("nan") if not n_pos else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": fp / n_neg if n_neg else float("nan"), "n_flagged": int(flagged.sum())}


def roc_curve(score, positive):
    """ROC-Kurve: (Fehlalarmrate, Trefferquote) für alle Schwellen, von (0, 0) bis (1, 1)."""
    positive = np.asarray(positive, bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    tpr = np.concatenate([[0.0], np.cumsum(hits) / max(hits.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~hits) / max((~hits).sum(), 1)])
    return fpr, tpr


# --- Analyse einer Aufnahme ---------------------------------------------------------------------------------------------------------


def standardise(X):
    """Kennzahlen auf Mittelwert 0 und Streuung 1 (für den LOF des Vergleichs; ECOD und der Isolation Forest brauchen das nicht)."""
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


def fisher_threshold(p, quantile):
    """Schwelle für einen Wert, der unter Unabhängigkeit Gamma(p, 1) ist (Summe von p Exp(1)-Beiträgen): das Quantil, über chi²(2p) / 2 (Fisher-Methode)."""
    return float(alg.chi2_ppf(quantile, 2 * int(p)) / 2.0)


VARIANTS = ("paper", "twosided", "auto")
VARIANT_LABELS = {"paper": "Paper: Maximum der drei Summen (Original)", "twosided": "zweiseitig mit p-Werten (Fisher)", "auto": "nur nach Schiefe (O_auto)"}


def ecod_score(X, variant):
    return ecod.score(X, variant)


@dataclass(frozen=True)
class Settings:
    variant: str = C.DEFAULT_VARIANT                    # Kombination der Merkmale: paper (Original), twosided, both, auto
    threshold_kind: str = C.DEFAULT_THRESHOLD_KIND      # "standard": Fisher-Quantil (ECOD), LOF 1.5, Score 0.5 (Isolation Forest), chi²-Quantil (robust); "share": der erwartete Anteil für alle vier
    ecod_quantile: float = C.DEFAULT_ECOD_QUANTILE      # Quantil der Gamma(p, 1)-Verteilung als Schwelle für ECOD
    quantile: float = C.DEFAULT_QUANTILE                # chi²-Quantil der Wurzel
    share: int = C.DEFAULT_SHARE                        # erwarteter Anteil der Anomalien [%]
    start: int = 0                                      # Seed des Isolation Forest und der MCD-Starts (ECOD ist deterministisch)


DATA_KEYS = ("n", "p", "n_noise", "n_modes", "curvature", "noise", "contamination", "kind", "strength")
DEFAULT_DATA = dict(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                    contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH)
DETECTORS = ("ecod", "lof", "iforest", "robust")
DETECTOR_NAMES = {"ecod": "ECOD", "lof": "LOF", "iforest": "Isolation Forest", "robust": "robust (MCD)"}
METRICS = ("auc", "ap", "precision", "recall", "f1", "false_alarm")
IF_TREES, IF_PSI, IF_CUTOFF, LOF_CUTOFF, LOF_K = C.DEFAULT_TREES, C.DEFAULT_PSI, C.DEFAULT_CUTOFF_IF, C.DEFAULT_CUTOFF_LOF, C.DEFAULT_LOF_K


def lof_k(n):
    """k des LOF im Vergleich: 20, höchstens n / 2 (k nahe n macht LOF unbrauchbar)."""
    return int(max(3, min(LOF_K, n // 2)))


def make_dataset(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                 contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH, seed=C.DEFAULT_SEED):
    return sc.generate_dataset(n, p, n_modes, curvature, noise, contamination, kind, strength, seed, n_noise)


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    params: tuple                 # (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)
    p_total: int                  # Zahl der Merkmale einschließlich Rauschmerkmale
    contributions: np.ndarray     # [n, p_total] Beitrag jedes Merkmals zum ECOD-Wert (für die Darstellung)
    forest_if: isf.Forest
    values: dict                  # Detektor -> Anomalie-Wert je Tour (ECOD: Summe der negativen Log-Schwanzwahrscheinlichkeiten, LOF, Isolation Forest: Score, robust: quadrierter Mahalanobis-Abstand)
    classical: alg.Fit
    robust: alg.Fit
    scores: dict                  # Detektor -> Kennzahlen (auc, ap, precision, recall, f1, false_alarm, n_flagged, threshold)
    flags: dict                   # Detektor -> markierte Touren bei der gewählten Schwelle
    oracle_f1: dict               # Detektor (ecod, lof, iforest, robust) -> F1, wenn der wahre Anteil bekannt wäre (die k größten Werte)
    seconds: dict


_ROOT_CACHE = {}


def _root_fits(X, key, start):
    """Klassische und robuste Schätzung der Wurzel (unabhängig von den ECOD-Reglern: werden je Aufnahme nur einmal gerechnet)."""
    cache_key = (key, start)
    if key is not None and cache_key in _ROOT_CACHE:
        return _ROOT_CACHE[cache_key]
    t0 = time.perf_counter()
    classical = alg.fit_classical(X)
    t1 = time.perf_counter()
    robust = alg.fit_mcd(X, C.DEFAULT_SUPPORT, start, reweight=C.DEFAULT_REWEIGHT)
    out = (classical, robust, t1 - t0, time.perf_counter() - t1)
    if key is not None:
        if len(_ROOT_CACHE) > 600:
            _ROOT_CACHE.clear()
        _ROOT_CACHE[cache_key] = out
    return out


def _thresholds(values, settings, robust, p_total):
    """Markierung und Schwellenwert je Detektor: Standard = Fisher-Quantil (ECOD), LOF 1.5, Score 0.5 (Isolation Forest), chi²-Quantil (robust), sonst die k größten Werte mit dem angenommenen Anteil."""
    flags, thr = {}, {}
    if settings.threshold_kind == "share":
        for d in DETECTORS:
            flags[d] = isf.flag_top(values[d], settings.share / 100.0)
            thr[d] = float(np.sort(values[d])[::-1][int(flags[d].sum()) - 1])
    else:
        thr["ecod"] = fisher_threshold(p_total, settings.ecod_quantile)
        thr["lof"] = LOF_CUTOFF
        thr["iforest"] = IF_CUTOFF
        thr["robust"] = alg.threshold(robust, settings.quantile)
        for d in DETECTORS:
            flags[d] = values[d] > thr[d]
    return flags, thr


def analyse(ds, settings=Settings(), params=None, root_key=None):
    secs = {}
    X = ds.X
    p_total = X.shape[1]
    t0 = time.perf_counter()
    ecod_values = ecod_score(X, settings.variant)
    secs["ecod"] = time.perf_counter() - t0
    contributions = ecod.per_feature_contribution(X, settings.variant)
    t0 = time.perf_counter()
    lof_values = lof.fit_lof(standardise(X), lof_k(len(X))).lof
    secs["lof"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    forest_if = isf.fit_forest(X, IF_TREES, min(IF_PSI, len(X)), settings.start)
    paths_if = isf.path_lengths(forest_if, X)
    secs["iforest"] = time.perf_counter() - t0
    classical, robust, secs["classical"], secs["robust"] = _root_fits(X, root_key, settings.start)
    values = {"ecod": ecod_values, "lof": lof_values, "iforest": isf.score_from_paths(paths_if, forest_if.psi), "robust": robust.d2}
    flags, thr = _thresholds(values, settings, robust, p_total)
    scores = {}
    for d in DETECTORS:
        m = flag_metrics(flags[d], ds.anomaly)
        m.update(auc=roc_auc(values[d], ds.anomaly), ap=average_precision(values[d], ds.anomaly), threshold=thr[d])
        scores[d] = m
    oracle = {d: flag_metrics(isf.flag_top(values[d], ds.anomaly.mean()), ds.anomaly)["f1"] for d in DETECTORS}
    return Analysis(ds, settings, params, p_total, contributions, forest_if, values, classical, robust, scores, flags, oracle, secs)


def analyse_for(params, settings=Settings()):
    """`params` = (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)."""
    return analyse(make_dataset(*params), settings, params, root_key=params)


# --- Sweeps und Experimente -----------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n": (20, 30, 50, 100, 200, 400, 600),
    "p": (2, 5, 8, 12, 20, 30),
    "n_noise": (0, 5, 10, 20, 30, 40),
    "n_modes": (1, 2, 3),
    "curvature": (0.0, 0.25, 0.5, 0.75, 1.0),
    "noise": (0.0, 0.25, 0.5, 0.75, 1.0),
    "contamination": (2, 5, 10, 20, 30, 40, 45),
    "strength": (3.0, 4.0, 6.0, 9.0, 12.0),
    "ecod_quantile": (0.9, 0.95, 0.975, 0.99, 0.999),
    "quantile": (0.9, 0.95, 0.975, 0.99, 0.999),
    "share": (2, 5, 10, 20, 40),
    "variant": VARIANTS,
}
SWEEP_LABELS = {"n": "Anzahl Touren", "p": "Anzahl Merkmale", "n_noise": "Anzahl Rauschmerkmale", "n_modes": "Anzahl Betriebsarten", "curvature": "Krümmung des Normalbereichs", "noise": "Rauschen",
                "contamination": "Anteil der Anomalien [%]", "strength": "Abstand der Anomalien (Faktor-σ)", "ecod_quantile": "Fisher-Quantil der Schwelle (ECOD)", "quantile": "chi²-Quantil der Schwelle (robust)",
                "share": "angenommener Anteil der Anomalien [%]", "variant": "Variante von ECOD"}
SETTING_PARAMETERS = ("variant", "ecod_quantile", "quantile", "share")


def _record(a):
    out = {f"{d}_{k}": a.scores[d][k] for d in DETECTORS for k in METRICS}
    out["n_anomalies"] = float(a.ds.anomaly.sum())
    for d in DETECTORS:
        out[f"{d}_oracle_f1"] = a.oracle_f1[d]
        out[f"{d}_seconds"] = a.seconds[d]
    return out


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def _analyse_seed(seed, settings, kw):
    data = {**DEFAULT_DATA, **kw}
    ds = make_dataset(seed=seed, **data)
    return analyse(ds, settings, None, root_key=(tuple(sorted(data.items())), seed))


def _mean_over_seeds(settings=Settings(), seeds=C.SWEEP_SEEDS, **kw):
    """Mittel (mit Streuung und Spanne) aller Kennzahlen über die festen Sweep-Datensätze für eine Datenkonfiguration."""
    return _summarise(None, [_record(_analyse_seed(s, settings, kw)) for s in seeds])


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel, Streuung und Spanne der Kennzahlen der vier Detektoren über die festen Sweep-Datensätze in Abhängigkeit von einem Regler (alle anderen wie in `base`)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        if parameter in SETTING_PARAMETERS:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, parameter: x}), **base)
        else:
            row = _mean_over_seeds(settings, **{**base, parameter: x})
        row["x"] = x
        rows.append(row)
    return rows


def _ecod_only(seed, kw):
    """Aufnahme und ihre Wahrheit ohne Wald und Wurzel: (X, Anomalie-Maske)."""
    ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **kw})
    return ds.X, ds.anomaly


SCENARIOS = (
    ("Standardfall", {}),
    ("dichte Gruppe 10 %", {"kind": "cluster"}),
    ("dichte Gruppe 30 %", {"kind": "cluster", "contamination": 30}),
    ("Lücke, 2 Betriebsarten", {"kind": "gap", "n_modes": 2}),
    ("Lücke, 3 Betriebsarten", {"kind": "gap", "n_modes": 3}),
    ("Korrelationsbruch", {"kind": "decorrelated"}),
    ("Korrelationsbruch, p = 30", {"kind": "decorrelated", "p": 30}),
    ("45 % verstreut", {"contamination": 45}),
    ("40 Rauschmerkmale", {"n_noise": 40}),
)


def scenario_table(settings=Settings(), scenarios=SCENARIOS, **base):
    """Die vier Detektoren in den Szenarien: AUC und F1 mit bekanntem Anteil (Mittel über die festen Sweep-Datensätze)."""
    rows = []
    for label, extra in scenarios:
        row = _mean_over_seeds(settings, **{**base, **extra})
        row["scenario"] = label
        rows.append(row)
    return rows


def variants_table(**base):
    """Die drei Varianten von ECOD in den Szenarien: AUC und F1 mit bekanntem Anteil (nur ECOD, ohne Vergleichsdetektoren)."""
    rows = []
    for label, extra in SCENARIOS:
        cells = {v: [] for v in VARIANTS}
        for seed in C.SWEEP_SEEDS:
            X, y = _ecod_only(seed, {**base, **extra})
            for v in VARIANTS:
                s = ecod_score(X, v)
                cells[v].append((roc_auc(s, y), flag_metrics(isf.flag_top(s, y.mean()), y)["f1"]))
        rows.append({"scenario": label, **{v: {"auc": float(np.mean([c[0] for c in cs])), "oracle_f1": float(np.mean([c[1] for c in cs]))} for v, cs in cells.items()}})
    return rows


def calibration_table(quantile=C.DEFAULT_ECOD_QUANTILE, **base):
    """Fehlalarmrate der Fisher-Schwelle bei (fast) reinen Normalen je Merkmalszahl p: mit den korrelierten Kennzahlen des Szenarios und mit unabhängigen Merkmalen (Kontrolle); nominell 1 - Quantil. Je Variante paper und twosided."""
    rows = []
    for p in (2, 5, 12, 20, 30):
        thr = fisher_threshold(p, quantile)
        cells = {"paper": [], "twosided": [], "paper_indep": [], "twosided_indep": []}
        for seed in C.SWEEP_SEEDS:
            X, y = _ecod_only(seed, {**base, "p": p, "contamination": 1})
            for v in ("paper", "twosided"):
                cells[v].append(float((ecod_score(X, v)[~y] > thr).mean()))
            Z = np.random.default_rng([seed, p]).standard_normal((len(X), p))
            for v in ("paper", "twosided"):
                cells[v + "_indep"].append(float((ecod_score(Z, v) > thr).mean()))
        rows.append({"p": p, "nominal": 1.0 - quantile, **{k: float(np.mean(v)) for k, v in cells.items()}})
    return rows


def cost_table(settings=Settings(), **base):
    """Rechenzeit (Wert je Tour bestimmen) von ECOD, LOF und Isolation Forest über die Tourenzahl bei 12 und bei 52 Merkmalen (40 Rauschmerkmale)."""
    times = []
    for n_noise in (0, 40):
        for n in (100, 300, 600):
            rows = [_analyse_seed(seed, settings, {**base, "n": n, "n_noise": n_noise}) for seed in C.SWEEP_SEEDS[:3]]
            times.append({"n": n, "p": 12 + n_noise, **{d: float(np.mean([a.seconds[d] for a in rows])) for d in ("ecod", "lof", "iforest")}})
    return {"times": times}


def threshold_table(settings=Settings(), **base):
    """Schwelle: (1) Kennzahlen über das Fisher-Quantil (ECOD); (2) F1 aller vier Detektoren bei ½-, 1- und 2-fach angenommenem Anteil."""
    cut = sweep("ecod_quantile", settings=Settings(**{**settings.__dict__, "threshold_kind": "standard"}), **base)
    true_share = base.get("contamination", C.DEFAULT_CONTAMINATION)
    wrong = []
    for factor in (0.5, 1.0, 2.0):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "threshold_kind": "share", "share": int(round(true_share * factor))}), **base)
        row["x"], row["factor"] = int(round(true_share * factor)), factor
        wrong.append(row)
    return {"cutoff": cut, "wrong_share": wrong}


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.05             # AUC-Abstand, ab dem ein Detektor als besser gilt
ECOD_WIN_MARGIN = 0.03
GAP_AUC = 0.8
BLIND_AUC = 0.6
THRESHOLD_F1_DROP = 0.15


def verdict(a):
    """(Art, Code, Kennzahlen): Lücke (keiner findet sie), blind (ECOD unter 0.6, ein anderer über 0.8), andere besser, ECOD besser, falsche Schwelle bei guter Rangfolge, sonst gleichauf."""
    ds = a.ds
    e = a.scores["ecod"]
    others = {d: a.scores[d]["auc"] for d in ("lof", "iforest", "robust")}
    best_other = max(others.values())
    data = {"n": ds.n, "p": a.p_total, "n_noise": ds.n_noise, "n_modes": ds.n_modes, "kind": ds.kind, "contamination": 100.0 * ds.anomaly.mean(), "n_anomalies": int(ds.anomaly.sum()), "variant": a.settings.variant,
            "best_other_auc": best_other, **{f"oracle_{d}": a.oracle_f1[d] for d in DETECTORS}, **{f"{d}_{k}": v for d in DETECTORS for k, v in a.scores[d].items()}}
    if ds.kind == "gap" and max(best_other, e["auc"]) < GAP_AUC:
        return "warning", "gap", data
    if e["auc"] < BLIND_AUC and best_other >= GAP_AUC:
        return "warning", "blind", data
    if best_other - e["auc"] >= WIN_MARGIN:
        return "warning", "others_win", data
    if e["auc"] - best_other >= ECOD_WIN_MARGIN:
        return "success", "ecod_wins", data
    if e["auc"] >= 0.95 and a.oracle_f1["ecod"] - e["f1"] > THRESHOLD_F1_DROP:
        return "warning", "threshold_off", data
    return "success", "comparable", data
