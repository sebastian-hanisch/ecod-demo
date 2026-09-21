"""Defaults, Regler-Grenzen und Presets für die LOF-Demo (Anomalie-Erkennung an Lieferrouten-Kennzahlen; Szenario, Isolation Forest und Vergleichsschätzer aus den Vorgänger-Demos)."""

# --- Merkmale: die 12 Kennzahlen der PCA-Demo (Name, Einheit, Mittelwert, typische Streuung), dazu Zusatzmerkmale für den Fall n < p --------------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_BASE_FEATURES = len(FEATURES)
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_BASE_FEATURES))
# Reihenfolge, in der die ersten p Merkmale gewählt werden: reihum durch die vier Gruppen, damit schon p = 2 beide latenten Faktoren sieht (Distanz und Zeitfenster-Enge)
FEATURE_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)
EXTRA_MEAN, EXTRA_SCALE = 50.0, 10.0                   # Zusatzmerkmale 13 ... p (zufällige Mischungen der latenten Faktoren plus eigenes Rauschen)

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 20, 600
DEFAULT_P = 12
DEFAULT_N_NOISE = 0
N_NOISE_MIN, N_NOISE_MAX = 0, 40
P_MIN, P_MAX = 2, 30
N_MODES_MIN, N_MODES_MAX = 1, 3
DEFAULT_N_MODES = 1
DEFAULT_CURVATURE = 0.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_CONTAMINATION = 10                             # Prozent
CONTAMINATION_MIN, CONTAMINATION_MAX = 1, 45
KINDS = ("scattered", "cluster", "gap", "decorrelated")
KIND_LABELS = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke zwischen den Betriebsarten", "decorrelated": "Korrelationsbruch (Randverteilungen normal)"}
DEFAULT_KIND = "scattered"
DEFAULT_STRENGTH = 6.0
STRENGTH_MIN, STRENGTH_MAX = 3.0, 12.0
DEFAULT_SUPPORT = 0.5                                   # Stützanteil h/n (0.5 = größter Bruchpunkt); 1.0 = alle Punkte = klassische Schätzung
SUPPORT_MIN, SUPPORT_MAX = 0.5, 1.0
DEFAULT_QUANTILE = 0.975                                # chi^2-Quantil der Schwelle
QUANTILE_MIN, QUANTILE_MAX = 0.90, 0.999
DEFAULT_REWEIGHT = True
DEFAULT_SEED = 7

# --- ECOD und Vergleichsdetektoren -----------------------------------------------------------------------------------------
VARIANTS = ("paper", "twosided", "auto")
DEFAULT_VARIANT = "paper"                               # ECOD wie veröffentlicht: Maximum der drei Summen
DEFAULT_ECOD_QUANTILE = 0.975                           # Quantil der Gamma(p, 1)-Verteilung (Fisher) als Schwelle ohne Vorwissen über den Anteil
ECOD_QUANTILE_MIN, ECOD_QUANTILE_MAX = 0.90, 0.999
DEFAULT_TREES = 100                                     # Isolation Forest im Vergleich
DEFAULT_PSI = 256
DEFAULT_CUTOFF_IF = 0.5                                 # nominelle Score-Schwelle des Isolation Forest
DEFAULT_CUTOFF_LOF = 1.5                                # LOF-Schwelle im Vergleich
DEFAULT_LOF_K = 20                                      # LOF im Vergleich: k = 20 (höchstens n / 2)
THRESHOLD_KINDS = ("standard", "share")
THRESHOLD_LABELS = {"standard": "Standard (Fisher-Quantil bzw. LOF 1.5 bzw. Score 0.5 bzw. χ²-Quantil)", "share": "erwarteter Anteil (für alle vier)"}
DEFAULT_THRESHOLD_KIND = "standard"
DEFAULT_SHARE = 10                                      # angenommener Anteil der Anomalien [%] (= der wahre im Standardfall)
SHARE_MIN, SHARE_MAX = 1, 45

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
Q = 2                                                   # latente Faktoren (fest; die PCA-Demo variiert sie, hier geht es um Anomalien)
CROSS_LOADING = 0.15
WITHIN_LOADINGS = (0.95, 0.9, 0.85)
CURVATURE_FREQUENCY = 1.6
CURVATURE_AMPLITUDE = 2.0
LAYOUT_SEED = 20240915                                  # dieselben festen Matrizen wie in der PCA-Demo
EXTRA_LAYOUT_SEED = LAYOUT_SEED + 2
MODE_RADIUS = 2.2                                       # Betriebsarten liegen auf einem Kreis dieses Radius im Faktorraum
MODE_SD = 0.6                                           # Streuung innerhalb einer Betriebsart (bei nur einer Betriebsart 1, wie in der PCA-Demo)
CLUSTER_SD = 0.3                                        # Streuung der dichten Anomalie-Gruppe
GAP_SD = 0.3                                            # Streuung der Anomalien in der Lücke
CLUSTER_ANGLE = 0.6                                     # Richtung der dichten Gruppe im Faktorraum (Bogenmaß)

# --- Auswertung --------------------------------------------------------------------------------------------------------
MCD_STARTS = 500                                        # zufällige Startmengen (je zwei C-Schritte)
MCD_KEEP = 10                                           # die besten davon laufen bis zur Konvergenz
MCD_INITIAL_STEPS = 2
MCD_MAX_STEPS = 50
RIDGE = 1e-9                                            # relative Regularisierung der Kovarianz (n < p)
SWEEP_SEEDS = tuple(100_000 + i for i in range(5))

# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(n=DEFAULT_N_TOURS, p=DEFAULT_P, n_noise=DEFAULT_N_NOISE, n_modes=DEFAULT_N_MODES, curvature=DEFAULT_CURVATURE, noise=DEFAULT_NOISE, contamination=DEFAULT_CONTAMINATION,
                kind=DEFAULT_KIND, strength=DEFAULT_STRENGTH, variant=DEFAULT_VARIANT, threshold_kind=DEFAULT_THRESHOLD_KIND, ecod_quantile=DEFAULT_ECOD_QUANTILE, quantile=DEFAULT_QUANTILE,
                share=DEFAULT_SHARE, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Standardfall (Original)": _preset(),
    "Standardfall, zweiseitig (Fisher)": _preset(variant="twosided"),
    "Dichte Gruppe abseits (30 %)": _preset(kind="cluster", contamination=30),
    "Korrelationsbruch": _preset(kind="decorrelated"),
    "Lücke, 2 Betriebsarten": _preset(n_modes=2, kind="gap"),
    "Viele Anomalien (45 %)": _preset(contamination=45),
    "40 Rauschmerkmale": _preset(n_noise=40),
}
PRESET_HELP = {
    "Standardfall (Original)": "Im Mittel über fünf Aufnahmen: 300 Touren, 12 Merkmale, 10 % verstreute Anomalien, ECOD wie veröffentlicht. Die Rangfolge ist gut (AUC 0.99), aber die Fisher-Schwelle (Gamma-Quantil 0.975, ohne Vorwissen) "
                              "markiert 20 % der normalen Touren (Sollwert 2.5 %): F1 0.52 gegen 0.94 (LOF), 0.96 (Isolation Forest) und 0.84 (robust); mit bekanntem Anteil 0.83.",
    "Standardfall, zweiseitig (Fisher)": "Im Mittel über fünf Aufnahmen: dieselben Daten, aber die zweiseitige Variante mit p-Werten, für die die Fisher-Schwelle hergeleitet ist. AUC 1.00, Fehlalarmrate 4.5 % statt 20 % und F1 0.83 (Recall 0.99) - "
                                        "ohne einen einzigen Parameter; mit bekanntem Anteil 0.89. Bei mehr Merkmalen wächst die Fehlalarmrate trotzdem (Experiment Kalibrierung).",
    "Dichte Gruppe abseits (30 %)": "Im Mittel über fünf Aufnahmen: 30 % der Touren (90) bilden eine dichte Gruppe abseits. ECOD hat AUC 0.86 (Recall 0.86 an der Schwelle), der Isolation Forest 0.70, LOF 0.47 (k = 20 < Gruppe), robust 0.49 "
                                    "(die Gruppe verschiebt die Schätzung): die Randverteilungen sehen die Gruppe ohne k und ohne Kovarianz.",
    "Korrelationsbruch": "Im Mittel über fünf Aufnahmen: die Merkmale der Anomalien sind je Spalte aus den Normalen neu gezogen - identische Randverteilungen, zerstörte Abhängigkeit. ECOD hat AUC 0.40 (unter Raten), "
                         "LOF 0.99, robust 1.00, Isolation Forest 0.80: ECOD sieht nur Randverteilungen.",
    "Lücke, 2 Betriebsarten": "Im Mittel über fünf Aufnahmen: zwei Betriebsarten, 10 % Anomalien in der Lücke dazwischen. ECOD hat AUC 0.08 - die Lücken-Touren liegen in der Mitte der Randverteilungen und wirken normaler als die Normalen; "
                              "LOF 0.53, Isolation Forest 0.54, robust 0.40: keiner findet sie.",
    "Viele Anomalien (45 %)": "Im Mittel über fünf Aufnahmen: 45 % verstreute Anomalien (135 Touren). ECOD hat AUC 0.98 und F1 0.84 (Fehlalarmrate 0 %) - wie der Isolation Forest (1.00, 0.81), während LOF (0.64) und die Wurzel (0.87) einbrechen: "
                              "die empirische Verteilung wird von so vielen Anomalien nicht verzerrt.",
    "40 Rauschmerkmale": "Im Mittel über fünf Aufnahmen: 40 unabhängige Rauschmerkmale. ECOD hat AUC 0.95 (LOF und Isolation Forest 0.99, robust 0.88), aber an der Fisher-Schwelle Recall 0.84 und F1 0.66 - LOF findet dort nichts (Recall 0.00), "
                         "der Isolation Forest 0.39. Jedes Rauschmerkmal addiert im Mittel 1 und Streuung 1 zur Summe.",
}
# Bänder (Seed des Presets; mit dem ausgelieferten Code kalibriert, bewusst weit): Kennzahlen der Detektoren (ecod_*, lof_*, iforest_*, robust_*) und erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Original)": {"ecod_auc": (0.97, 1.0), "ecod_f1": (0.3, 0.75), "ecod_false_alarm": (0.1, 0.35), "lof_f1": (0.85, 1.0), "verdict": ("threshold_off",)},
    "Standardfall, zweiseitig (Fisher)": {"ecod_auc": (0.97, 1.0), "ecod_f1": (0.7, 1.0), "ecod_false_alarm": (0.0, 0.1), "verdict": ("comparable",)},
    "Dichte Gruppe abseits (30 %)": {"ecod_auc": (0.75, 0.95), "lof_auc": (0.3, 0.6), "iforest_auc": (0.55, 0.85), "robust_auc": (0.3, 0.6), "verdict": ("ecod_wins",)},
    "Korrelationsbruch": {"ecod_auc": (0.0, 0.6), "lof_auc": (0.9, 1.0), "robust_auc": (0.95, 1.0), "verdict": ("blind",)},
    "Lücke, 2 Betriebsarten": {"ecod_auc": (0.0, 0.35), "lof_auc": (0.4, 0.85), "robust_auc": (0.0, 0.6), "verdict": ("gap",)},
    "Viele Anomalien (45 %)": {"ecod_auc": (0.95, 1.0), "lof_auc": (0.5, 0.85), "robust_auc": (0.9, 1.0), "ecod_false_alarm": (0.0, 0.05), "verdict": ("comparable",)},
    "40 Rauschmerkmale": {"ecod_auc": (0.9, 1.0), "lof_recall": (0.0, 0.2), "ecod_recall": (0.6, 1.0), "verdict": ("comparable", "others_win")},
}
