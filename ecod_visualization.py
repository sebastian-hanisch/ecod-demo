"""Plotly-Visualisierungen der ECOD-Demo: Touren in der Ebene der größten Streuung, empirische Verteilungsfunktionen mit den Positionen zweier Touren, Beiträge je Merkmal, Wert-Histogramm mit Schwelle,
ROC-Kurven, Kennzahlen-Balken, Sweeps und die Experimente (Szenarien, Varianten, Kalibrierung der Fisher-Schwelle, Kosten). Alle Figuren laufen durch `lock_axes`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ecod_ee_algorithm as alg

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL, PINK = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f", "#c2185b"
DETECTOR_COLORS = {"ecod": PINK, "lof": ORANGE, "iforest": PURPLE, "robust": TEAL}
DETECTOR_NAMES = {"ecod": "ECOD", "lof": "LOF", "iforest": "Isolation Forest", "robust": "robust (MCD)"}
VARIANT_COLORS = {"paper": PINK, "twosided": GREEN, "auto": GRAY}
VARIANT_NAMES = {"paper": "Paper (Original)", "twosided": "zweiseitig (Fisher)", "auto": "nur nach Schiefe"}
KIND_NAMES = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke", "decorrelated": "Korrelationsbruch"}
DETECTORS = ("ecod", "lof", "iforest", "robust")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# --- Projektion -----------------------------------------------------------------------------------------------------------------


def standardise(X):
    m, s = X.mean(axis=0), X.std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    return (X - m) / s


def projection(X, robust):
    """Die Touren in der Ebene der zwei größten Streuungsrichtungen der robusten Kovarianz (standardisierte Kennzahlen): (Punkte n x 2, Achsen)."""
    s = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    axes = alg.projection_axes(robust.covariance / np.outer(s, s))
    return standardise(X) @ axes, axes


def _points(P, anomaly, flagged=None):
    normal = ~anomaly
    traces = [go.Scatter(x=P[normal, 0], y=P[normal, 1], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55), hoverinfo="skip", name="normale Touren"),
              go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond"), hoverinfo="skip", name="Sonderfahrten (Wahrheit)")]
    if flagged is not None and flagged.any():
        traces.append(go.Scatter(x=P[flagged, 0], y=P[flagged, 1], mode="markers", marker=dict(size=13, color="black", line=dict(width=2), symbol="circle-open"), hoverinfo="skip", name="als Anomalie markiert"))
    return traces


def build_scatter(P, anomaly, flagged=None, height=380):
    """Touren in der Projektionsebene (blau = normal, rote Rauten = Sonderfahrten, Kreise = als Anomalie markiert)."""
    fig = go.Figure(_points(P, anomaly, flagged))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1 (standardisiert)", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False),
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_features(X, anomaly, names, i=0, j=1):
    """Zwei Rohmerkmale gegeneinander (Einheiten wie gemessen)."""
    j = min(j, X.shape[1] - 1)
    fig = go.Figure(_points(np.stack([X[:, i], X[:, j]], axis=1), anomaly))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=names[i], zeroline=False), yaxis=dict(title=names[j], zeroline=False), showlegend=False)
    return lock_axes(fig)


# --- Schritte ---------------------------------------------------------------------------------------------------------------------


def build_ecdf(X, anomaly, names, picks, i, j):
    """Empirische Verteilungsfunktionen (links) zweier Merkmale mit den Positionen zweier Touren: die Höhe der Kurve an der Position ist die Schwanzwahrscheinlichkeit F_links (rechts: 1 - F_links + 1/n).
    `picks` = [(Index, Name, Farbe)]; die Punkte liegen auf der Kurve."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(names[i], names[j]), horizontal_spacing=0.10)
    n = len(X)
    for col, f in ((1, i), (2, j)):
        xs = np.sort(X[:, f])
        ys = np.arange(1, n + 1) / n
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=3, shape="hv"), hoverinfo="skip", showlegend=False), row=1, col=col)
        for idx, name, color in picks:
            y = float(np.searchsorted(xs, X[idx, f], side="right") / n)
            fig.add_trace(go.Scatter(x=[X[idx, f]], y=[y], mode="markers", marker=dict(size=13, color=color, line=dict(width=2, color="black")), name=name, hoverinfo="skip", showlegend=col == 1), row=1, col=col)
    fig.update_yaxes(title="F(x) = Anteil der Touren ≤ x", range=[0, 1.03], col=1)
    fig.update_yaxes(range=[0, 1.03], col=2)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_contributions(names, entries, top=12):
    """Beitrag jedes Merkmals (negative Log-Schwanzwahrscheinlichkeit) zum ECOD-Wert zweier Touren; die größten `top` Merkmale der Sonderfahrt, dazu die Summe je Tour. `entries` = [(Name, Farbe, Beiträge [p])]."""
    order = np.argsort(-entries[-1][2])[:top]
    labels = [names[k] if k < len(names) else f"Merkmal {k + 1}" for k in order]
    fig = go.Figure()
    for name, color, contrib in entries:
        fig.add_trace(go.Bar(x=labels, y=[float(contrib[k]) for k in order], name=f"{name} (Summe {float(contrib.sum()):.1f})", marker_color=color, hoverinfo="skip"))
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=10, b=90), yaxis=dict(title="Beitrag −log F"), legend=dict(orientation="h", y=-0.55), xaxis=dict(tickangle=-40))
    return lock_axes(fig)


def build_score_hist(values, anomaly, thr, title="ECOD-Wert"):
    """Histogramm des ECOD-Werts (normale Touren blass, Sonderfahrten kräftig); senkrechte Linie = Schwelle."""
    hi = float(max(values.max(), thr * 1.05))
    lo = float(min(values.min(), thr * 0.95))
    bins = dict(start=lo, end=hi, size=(hi - lo) / 50.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values[~anomaly], xbins=bins, marker_color=BLUE, opacity=0.55, name="normale Touren", hoverinfo="skip"))
    if anomaly.any():
        fig.add_trace(go.Histogram(x=values[anomaly], xbins=bins, marker_color=RED, opacity=0.9, name="Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(thr), line=dict(color="black", dash="dash"), annotation_text="Schwelle", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title=title), yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_rank_compare(ecod_values, robust_values, anomaly):
    """Rang von ECOD gegen Rang der Wurzel (robuste Mahalanobis-Abstände) je Tour: Touren, die ECOD und die Wurzel gleich einschätzen, liegen auf der Diagonalen; Sonderfahrten als Rauten."""
    n = len(ecod_values)
    re_ = np.argsort(np.argsort(ecod_values)) / (n - 1)
    rr = np.argsort(np.argsort(robust_values)) / (n - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr[~anomaly], y=re_[~anomaly], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.5), name="normale Touren", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rr[anomaly], y=re_[anomaly], mode="markers", marker=dict(size=9, color=RED, symbol="diamond"), name="Sonderfahrten", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Rang bei der Wurzel (robuste Mahalanobis-Abstände)", range=[-0.02, 1.02]),
                      yaxis=dict(title="Rang bei ECOD", range=[-0.02, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_roc(curves):
    """ROC-Kurven (Fehlalarmrate gegen Trefferquote) der vier Detektoren; `curves` = {Detektor: (fpr, tpr, auc)}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=DETECTOR_COLORS[name], width=3), name=f"{DETECTOR_NAMES[name]} (AUC {auc:.2f})", hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fehlalarmrate", range=[0, 1]), yaxis=dict(title="Trefferquote", range=[0, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_method_bars(scores):
    """Kennzahlen der vier Detektoren nebeneinander: AUC, Recall, Precision, Fehlalarmrate (bei der gewählten Schwelle)."""
    keys = (("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision"), ("false_alarm", "Fehlalarmrate"))
    fig = go.Figure()
    for det in DETECTORS:
        y = [scores[det][k] for k, _ in keys]
        fig.add_trace(go.Bar(x=[lab for _, lab in keys], y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


# --- Sweeps und Experimente ----------------------------------------------------------------------------------------------------------


# --- Sweeps und Experimente ----------------------------------------------------------------------------------------------------------


def _band(fig, xs, rows, key, color, col):
    y, sd = np.array([r[key] for r in rows]), np.array([r[key + "_std"] for r in rows])
    fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(np.nan_to_num(y + sd)) + list(np.nan_to_num(y - sd))[::-1], fill="toself", fillcolor=color, opacity=0.13, line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=col)


def build_sweep(rows, xlabel, current=None):
    """Links AUC der vier Detektoren (mit Streuung über die Sweep-Datensätze), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge)", "F1 und Fehlalarmrate bei der Schwelle"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    for det in DETECTORS:
        color = DETECTOR_COLORS[det]
        _band(fig, xs, rows, f"{det}_auc", color, 1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=color, width=3), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_f1"] for r in rows], mode="lines+markers", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_false_alarm"] for r in rows], mode="lines+markers", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------


def build_scenarios(rows):
    """AUC der vier Detektoren in den Szenarien (links) und F1 mit bekanntem Anteil (rechts); die Linie bei 0.5 ist Raten."""
    labels = [r["scenario"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC", "F1 mit bekanntem Anteil"), horizontal_spacing=0.10)
    for det in DETECTORS:
        for col, key in ((1, f"{det}_auc"), (2, f"{det}_oracle_f1")):
            y = [r[key] for r in rows]
            fig.add_trace(go.Bar(x=labels, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], showlegend=col == 1, text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=7),
                                 hoverinfo="skip"), row=1, col=col)
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=1)
    fig.update_yaxes(range=[0, 1.15])
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(height=520, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)


def build_variants(rows):
    """AUC der drei Varianten von ECOD in den Szenarien."""
    labels = [r["scenario"] for r in rows]
    fig = go.Figure()
    for v in ("paper", "twosided", "auto"):
        y = [r[v]["auc"] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=VARIANT_NAMES[v], marker_color=VARIANT_COLORS[v], text=[f"{a:.2f}" for a in y], textposition="outside", textfont=dict(size=7), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_yaxes(range=[0, 1.15], title="AUC")
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(height=430, barmode="group", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.45))
    return lock_axes(fig)


def build_calibration(rows):
    """Fehlalarmrate der Fisher-Schwelle bei reinen Normalen je Merkmalszahl: zweiseitige Variante (grün) und Paper-Variante (rosa), mit den korrelierten Kennzahlen des Szenarios (durchgezogen) und mit unabhängigen Merkmalen
    (gestrichelt); die waagerechte Linie ist der Sollwert."""
    xs = [str(r["p"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["twosided"] for r in rows], mode="lines+markers", name="zweiseitig, korrelierte Kennzahlen", line=dict(color=GREEN, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["twosided_indep"] for r in rows], mode="lines+markers", name="zweiseitig, unabhängige Merkmale", line=dict(color=GREEN, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["paper"] for r in rows], mode="lines+markers", name="Paper, korrelierte Kennzahlen", line=dict(color=PINK, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["paper_indep"] for r in rows], mode="lines+markers", name="Paper, unabhängige Merkmale", line=dict(color=PINK, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_hline(y=rows[0]["nominal"], line=dict(color="black", dash="dot"), annotation_text="Sollwert", annotation_position="top left")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Merkmale p", type="category"), yaxis=dict(title="Fehlalarmrate bei den Normalen", range=[0, 1.0]), legend=dict(orientation="h", y=-0.45))
    return lock_axes(fig)


def build_cutoff(rows):
    """ECOD: F1 (durchgezogen), Recall (gepunktet) und Fehlalarmrate (gestrichelt) über das Fisher-Quantil der Schwelle."""
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["ecod_f1"] for r in rows], mode="lines+markers", name="F1", line=dict(color=PINK, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ecod_recall"] for r in rows], mode="lines+markers", name="Recall", line=dict(color=PINK, width=2, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ecod_false_alarm"] for r in rows], mode="lines+markers", name="Fehlalarmrate", line=dict(color=PINK, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fisher-Quantil der Schwelle", type="category"), yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_wrong_share(rows):
    """F1 der vier Detektoren, wenn der angenommene Anteil das ½-, 1- und 2-fache des wahren ist."""
    xs = [f"{r['x']} % ({r['factor']:g}×)" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_f1"] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="angenommener Anteil (Vielfaches des wahren)"), yaxis=dict(title="F1", range=[0, 1.15]),
                      legend=dict(orientation="h", y=-0.4))
    return lock_axes(fig)


def build_costs(times):
    """Rechenzeit (Wert je Tour bestimmen) von ECOD, LOF und Isolation Forest über die Tourenzahl bei 12 und 52 Merkmalen."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("12 Merkmale", "52 Merkmale (40 Rauschmerkmale)"), horizontal_spacing=0.10)
    for col, p in ((1, 12), (2, 52)):
        rows = [t for t in times if t["p"] == p]
        for key, color, name in (("ecod", PINK, "ECOD"), ("lof", ORANGE, "LOF"), ("iforest", PURPLE, "Isolation Forest")):
            y = [t[key] for t in rows]
            fig.add_trace(go.Bar(x=[f"n = {t['n']}" for t in rows], y=y, name=name, marker_color=color, showlegend=col == 1, text=[f"{v:.4f} s" for v in y], textposition="outside", textfont=dict(size=8), hoverinfo="skip"), row=1, col=col)
    fig.update_yaxes(title="Sekunden", col=1)
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)
