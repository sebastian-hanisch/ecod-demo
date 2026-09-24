"""ECOD - Ausreißer aus den Randverteilungen, ohne einen einzigen Parameter - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - ECOD - und lässt stattdessen das Beispiel wachsen.
Sechstes Stück der Anomalie-Erkennung-Linie der "Konzepte"-Reihe: ein eigener Ast direkt nach der Wurzel (Elliptic Envelope), neben LOF und Isolation Forest. ECOD braucht weder ein k noch Teilmengen noch eine Kovarianz -
und sieht dafür nur die Randverteilungen. Gemessen wird, was das bringt - und was nicht. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import ecod_algorithm as ecod
import ecod_constants as C
from ecod_evaluation import (
    SWEEP_LABELS,
    SWEEP_VALUES,
    VARIANT_LABELS,
    Settings,
    analyse_for,
    calibration_table,
    cost_table,
    roc_curve,
    scenario_table,
    sweep,
    threshold_table,
    variants_table,
    verdict,
)
from ecod_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    kind_options,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from ecod_visualization import (
    VARIANT_NAMES,
    build_calibration,
    build_contributions,
    build_costs,
    build_cutoff,
    build_ecdf,
    build_features,
    build_method_bars,
    build_rank_compare,
    build_roc,
    build_scatter,
    build_scenarios,
    build_score_hist,
    build_sweep,
    build_variants,
    build_wrong_share,
    projection,
)

st.set_page_config(page_title="ECOD – Sebastian Hanisch", layout="wide")
BLUE_TXT, RED_TXT = "#1f77b4", "#d62728"


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings, values):
    return sweep(parameter, values=values, settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _scenarios(settings):
    return scenario_table(settings)


@st.cache_data(show_spinner=False)
def _variants():
    return variants_table()


@st.cache_data(show_spinner=False)
def _calibration(quantile):
    return calibration_table(quantile)


@st.cache_data(show_spinner=False)
def _threshold(base, settings):
    return threshold_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _costs(base, settings):
    return cost_table(settings, **dict(base))


st.title("📈 ECOD – Ausreißer aus den Randverteilungen")
st.markdown(
    """
Die Vorgänger dieser Linie brauchten immer etwas: die Wurzel eine Gauß-Wolke, der **LOF** ein passendes k, **Feature Bagging** Teilmengen und Glück, der Isolation Forest einen Anteil für die Schwelle.
**ECOD** (Empirical Cumulative distribution functions for Outlier Detection) braucht **nichts davon**: je Merkmal wird die **empirische Verteilungsfunktion** bestimmt, und eine Tour ist um so auffälliger, je **unwahrscheinlicher ihr Wert in den Enden** dieser Verteilungen liegt -
summiert über alle Merkmale. Kein k, keine Teilmengen, keine Kovarianz, keine Zufallszahlen. Der Preis steckt im Namen: ECOD sieht nur **Randverteilungen**, die **Abhängigkeit zwischen den Merkmalen** bleibt unsichtbar.
Diese Demo misst, was das bringt - und wo es blind macht.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - sechstes Stück der Anomalie-Erkennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Szenario, LOF, Isolation Forest und die robuste Schätzung der Wurzel sind wortgleich aus den Vorgänger-Demos übernommen (die klassische Schätzung entfällt, damit die Balken lesbar bleiben); neu ist die Anomalie-Art **Korrelationsbruch**. "
    "Die Linie hat keinen Konvergenzpunkt; ECOD ist ein eigener Ast, seine Nachbarn sind One-Class SVM, Deep SVDD und ein Autoencoder (noch nicht gebaut)."
)

with st.expander("So funktioniert ECOD", expanded=True):
    st.markdown(
        """
1. **Randverteilung je Merkmal.** Für jedes Merkmal $j$ zählt man, welcher Anteil der Touren kleiner oder gleich $x$ ist ($F_l(x)$) und welcher größer oder gleich ($F_r(x)$) - die **Schwanzwahrscheinlichkeiten**, zwischen $1/n$ und 1.
2. **Beitrag.** $-\\log F$ ist 0 in der Mitte und $\\log n$ am äußersten Rand: je seltener der Wert, desto größer der Beitrag. Jedes Merkmal trägt seinen Teil bei.
3. **Summe.** Der ECOD-Wert einer Tour ist die Summe der Beiträge über alle Merkmale. Im Original ($\\text{paper}$) ist es das **Maximum von drei Summen**: nur linke Schwänze, nur rechte Schwänze und die Seite, auf die die Schiefe des Merkmals zeigt.
   Die Demo vergleicht auch eine **zweiseitige Variante mit p-Werten** ($-\\log \\min(1, 2\\min(F_l, F_r))$ je Merkmal).
4. **Schwelle ohne Vorwissen?** Sind die Merkmale unabhängig, ist jeder Beitrag der zweiseitigen Variante $\\text{Exp}(1)$-verteilt und die Summe $\\text{Gamma}(p, 1)$ (Fisher-Methode): daraus folgt ein Quantil als Schwelle, ohne den Anteil der Anomalien zu kennen. Ob das bei den **korrelierten** Kennzahlen des Szenarios gilt, misst ein Experiment unten.

Was **nicht** vorausgesetzt wird: eine Verteilungsform, ein k, Teilmengen, Zufall. Was ECOD **voraussetzt**: dass Anomalien sich in **mindestens einer Randverteilung** verraten.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider(
        "Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=10,
        help="Anzahl der Touren. Die AUC von ECOD ist bei 20 / 30 / 50 / 100 / 200 Touren 0.99 / 0.99 / 0.98 / 0.99 / 0.99 (LOF und Isolation Forest 1.00); der F1 an der Fisher-Schwelle 0.75 / 0.59 / 0.60 / 0.61 / 0.58 "
             "(Fehlalarmrate 8 % / 14 % / 15 % / 15 % / 16 %), beim LOF 0.96 / 0.94 / 0.88 / 0.86 / 0.91, beim Isolation Forest 0.60 / 0.67 / 0.78 / 0.89 / 0.96.",
    )
    p_features = st.slider(
        "Merkmale", *bounds("p_slider"), key="p_slider",
        help="Anzahl der Kennzahlen je Tour (ab 13 zusätzliche Mischungen der versteckten Faktoren). AUC von ECOD bei 2 / 5 / 8 / 12 / 20 / 30 Merkmalen: 0.98 / 0.97 / 0.99 / 0.99 / 0.99 / 0.99. Die Fehlalarmrate der Fisher-Schwelle "
             "wächst dabei von 0.3 % auf 6 % / 13 % / 20 % / 19 % / 22 %: mit mehr (korrelierten) Merkmalen streut die Summe stärker als hergeleitet.",
    )
    n_noise = st.slider(
        "Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider", step=5,
        help="Zusätzliche unabhängige Spalten ohne Zusammenhang mit den Faktoren. Die AUC von ECOD sinkt bei 0 / 5 / 10 / 20 / 30 / 40 nur langsam: 0.99 / 0.98 / 0.98 / 0.97 / 0.96 / 0.95 (LOF und Isolation Forest 0.99-1.00, robust 1.00 → 0.88). "
             "Recall an der Fisher-Schwelle 0.99 / 0.99 / 0.96 / 0.91 / 0.88 / 0.84, beim LOF 1.00 / 0.93 / 0.71 / 0.18 / 0.03 / 0.00, beim Isolation Forest 0.99 / 0.95 / 0.91 / 0.71 / 0.58 / 0.39.",
    )
    n_modes = st.slider(
        "Betriebsarten", *bounds("n_modes_slider"), key="n_modes_slider",
        help="Aus wie vielen Gruppen (Stadt, Land, Fernverkehr) die normalen Touren stammen. Die AUC von ECOD ist bei 1 / 2 / 3 Betriebsarten 0.99 / 0.99 / 0.99 (robust 1.00 / 0.95 / 0.85): die Randverteilungen bleiben trennscharf.",
    )
    curvature = st.slider(
        "Krümmung des Normalbereichs", *bounds("curvature_slider"), key="curvature_slider", step=0.25,
        help="Biegt die normale Fläche (nicht mehr konvex). Die AUC von ECOD bleibt bei 0.98-1.00, aber die Fisher-Schwelle wird besser: F1 0.52 / 0.62 / 0.75 / 0.85 / 0.88 bei 0 / 0.25 / 0.5 / 0.75 / 1 "
             "(Fehlalarmrate 20 % → 2.9 %), weil die Krümmung die Kennzahlen teilweise entkorreliert (mittlere absolute Korrelation der Normalen 0.51 → 0.31). Der LOF verliert dagegen (F1 0.94 → 0.74).",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen der Kennzahlen. Die AUC von ECOD ist bei 0 / 0.25 / 0.5 / 1.0 0.98 / 0.99 / 0.99 / 0.99; der F1 an der Fisher-Schwelle 0.51 / 0.52 / 0.57 / 0.70 (mehr Rauschen entkoppelt die Merkmale, die Fehlalarmrate sinkt von 21 % auf 9 %).",
    )
    contamination = st.slider(
        "Anteil der Anomalien [%]", *bounds("contamination_slider"), key="contamination_slider",
        help="Wie viele Touren Sonderfahrten sind. AUC von ECOD bei 2 / 5 / 10 / 20 / 30 / 40 / 45 % verstreuten Anomalien: 0.99 / 0.99 / 0.99 / 0.99 / 0.98 / 0.98 / 0.98 (LOF 1.00 / 1.00 / 1.00 / 0.99 / 0.90 / 0.72 / 0.64, robust 1.00 bis 0.87): "
             "ECOD ist gegen viele verstreute Anomalien unempfindlich. F1 an der Schwelle 0.13 / 0.29 / 0.52 / 0.83 / 0.88 / 0.85 / 0.84 - bei wenigen Anomalien ist die Fehlalarmrate von 20-27 % das Problem.",
    )
    kind = st.selectbox(
        "Art der Anomalien", kind_options(n_modes), key="kind_select", format_func=lambda k: C.KIND_LABELS[k],
        help="Verstreut: jede Anomalie in einer anderen Richtung. Dichte Gruppe: alle beieinander - bei 10 % AUC ECOD 0.99, Isolation Forest 0.95, LOF 0.35, robust 1.00; bei 30 % ECOD 0.86, Isolation Forest 0.70, LOF 0.47, robust 0.49. "
             "In der Lücke (ab zwei Betriebsarten): ECOD 0.08 (weit unter Raten), LOF 0.53, Isolation Forest 0.54, robust 0.40. Korrelationsbruch: Merkmale der Anomalien je Spalte aus den Normalen neu gezogen - ECOD 0.40, LOF 0.99, robust 1.00, Isolation Forest 0.80.",
    )
    if kind not in ("gap", "decorrelated"):
        seed_widget("strength_slider")
        strength = st.slider(
            "Abstand der Anomalien (Faktor-σ)", *bounds("strength_slider"), key="strength_slider", step=0.5,
            help="Wie weit die Anomalien im Faktorraum vom Normalen entfernt sind.",
        )
        st.session_state["_strength_kept"] = strength
    else:
        strength = float(st.session_state.get("_strength_kept", C.DEFAULT_STRENGTH))

    st.markdown("**ECOD**")
    variant = st.selectbox(
        "Variante", C.VARIANTS, key="variant_select", format_func=lambda v: VARIANT_LABELS[v],
        help="Wie die Merkmale zusammengefasst werden. Original (Paper): Maximum der drei Summen - AUC 0.99, an der Fisher-Schwelle Fehlalarmrate 20 % und F1 0.52. Zweiseitig mit p-Werten: AUC 1.00, Fehlalarmrate 4.5 %, F1 0.83. "
             "Nur nach Schiefe (O_auto): AUC nur 0.69 - eine Seite pro Merkmal sieht bei allen Richtungen der Anomalien nur die Hälfte (dafür bei einer dichten Gruppe abseits 1.00 / 0.99 bei 10 % / 30 %).",
    )
    threshold_kind = st.selectbox(
        "Schwelle", C.THRESHOLD_KINDS, key="threshold_kind_select", format_func=lambda k: C.THRESHOLD_LABELS[k],
        help="Standard: ECOD über dem Fisher-Quantil (Gamma(p, 1)), LOF über 1.5, Isolation Forest über Score 0.5, robust über dem χ²-Quantil. Erwarteter Anteil: bei allen vieren werden die größten Werte markiert - "
             "dann entscheidet nur die Rangfolge, aber der Anteil muss bekannt sein.",
    )
    if threshold_kind == "standard":
        seed_widget("ecod_quantile_slider")
        ecod_quantile = st.slider(
            "Schwelle: Fisher-Quantil (ECOD)", *bounds("ecod_quantile_slider"), key="ecod_quantile_slider", step=0.001, format="%.3f",
            help="Quantil der Gamma(p, 1)-Verteilung als Schwelle für den ECOD-Wert. Bei 0.9 / 0.95 / 0.975 / 0.99 / 0.999: Fehlalarmrate 39 % / 28 % / 20 % / 13 % / 3 %, F1 0.37 / 0.45 / 0.52 / 0.62 / 0.81, "
                 "Recall 1.00 / 0.99 / 0.99 / 0.98 / 0.88 (Original-Variante, Standardfall) - die Schwelle passt erst bei einem viel höheren Quantil als dem Sollwert.",
        )
        seed_widget("quantile_slider")
        quantile = st.slider(
            "Schwelle: χ²-Quantil (robust)", *bounds("quantile_slider"), key="quantile_slider", step=0.001, format="%.3f",
            help="Ab welchem Anteil der χ²-Verteilung eine Tour bei der robusten Schätzung als Anomalie gilt. Bei 0.9 / 0.95 / 0.975 / 0.99 / 0.999: F1 der robusten Schätzung 0.67 / 0.77 / 0.84 / 0.89 / 0.90.",
        )
        st.session_state["_ecod_quantile_kept"] = ecod_quantile
        st.session_state["_quantile_kept"] = quantile
        share = int(st.session_state.get("_share_kept", C.DEFAULT_SHARE))
    else:
        seed_widget("share_slider")
        share = st.slider(
            "Angenommener Anteil der Anomalien [%]", *bounds("share_slider"), key="share_slider",
            help="Wie viele Touren als Anomalie markiert werden (die größten Werte, für alle vier Detektoren). Beim wahren Anteil 10 % ist der F1 bei angenommenen 2 / 5 / 10 / 20 / 40 % bei ECOD "
                 "0.33 / 0.66 / 0.83 / 0.65 / 0.40, bei LOF und Isolation Forest 0.33 / 0.67 / 0.97 / 0.67 / 0.40, bei robust 0.33 / 0.67 / 0.90 / 0.66 / 0.40.",
        )
        st.session_state["_share_kept"] = share
        ecod_quantile = float(st.session_state.get("_ecod_quantile_kept", C.DEFAULT_ECOD_QUANTILE))
        quantile = float(st.session_state.get("_quantile_kept", C.DEFAULT_QUANTILE))
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren und die Anomalien.")

sync_query_params({
    "n_tours_slider": int(n_tours), "p_slider": int(p_features), "n_noise_slider": int(n_noise), "n_modes_slider": int(n_modes), "curvature_slider": float(curvature), "noise_slider": float(noise),
    "contamination_slider": int(contamination), "kind_select": kind, "strength_slider": float(strength), "variant_select": variant, "threshold_kind_select": threshold_kind,
    "ecod_quantile_slider": float(ecod_quantile), "quantile_slider": float(quantile), "share_slider": int(share), "seed_input": int(seed),
})

data_params = (int(n_tours), int(p_features), int(n_noise), int(n_modes), float(curvature), float(round(noise, 2)), int(contamination), kind, float(strength), int(seed))
settings = Settings(variant=variant, threshold_kind=threshold_kind, ecod_quantile=float(ecod_quantile), quantile=float(quantile), share=int(share))
with st.spinner("Rechne die Randverteilungen..."):
    a = _analysis(data_params, settings)
level, code, vd = verdict(a)
ds = a.ds
es, ls, ifs, rs = (a.scores[d] for d in ("ecod", "lof", "iforest", "robust"))
n_anom = int(ds.anomaly.sum())
p_total = a.p_total
base_data = tuple(sorted({"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(round(noise, 2)),
                          "contamination": int(contamination), "kind": kind, "strength": float(strength)}.items()))
data_key = data_params + (settings,)

# --- ECOD in Aktion -------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 ECOD in Aktion")
STEP_LABELS = {1: "1 · Touren", 2: "2 · Randverteilungen", 3: "3 · Beiträge", 4: "4 · Wert", 5: "5 · Ergebnis"}
if "ecod_step" not in st.session_state or st.session_state.get("ecod_step_owner") != data_key:
    st.session_state["ecod_step"] = 1
    st.session_state["ecod_step_owner"] = data_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="ecod_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()

P, axes = projection(ds.X, a.robust)
values = a.values["ecod"]
normal_idx = np.flatnonzero(~ds.anomaly)
anom_idx = np.flatnonzero(ds.anomaly)
i_anom = int(anom_idx[np.argsort(values[anom_idx])[len(anom_idx) // 2]])                          # eine typische Sonderfahrt: die mit dem mittleren Wert
i_norm = int(normal_idx[np.argmin(np.abs(P[normal_idx]).sum(axis=1))])                             # eine normale Tour nahe der Mitte
picks = [(i_norm, "normale Tour", BLUE_TXT), (i_anom, "Sonderfahrt", RED_TXT)]
# Merkmalspaar für die Darstellung: beim Korrelationsbruch das am stärksten korrelierte Paar der Normalen, sonst die ersten zwei
if ds.kind == "decorrelated" and ds.p >= 2:
    corr = np.abs(np.corrcoef(ds.X[~ds.anomaly][:, : ds.p].T))
    np.fill_diagonal(corr, 0.0)
    pair_i, pair_j = (int(v) for v in np.unravel_index(np.argmax(corr), corr.shape))
else:
    pair_i, pair_j = 0, min(1, ds.p - 1)
left, right = ecod.tail_probabilities(ds.X)
names_all = ds.names


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns(2)
            c1.markdown("**Die Touren in der Ebene ihrer größten Streuung** (rote Rauten = Sonderfahrten)")
            c1.plotly_chart(build_scatter(P, ds.anomaly), width="stretch", key="step_scatter")
            c2.markdown(f"**Zwei Rohmerkmale: {names_all[pair_i]} gegen {names_all[pair_j]}** (Einheiten wie gemessen)" + (" - das am stärksten korrelierte Paar der Normalen" if ds.kind == "decorrelated" else ""))
            c2.plotly_chart(build_features(ds.X, ds.anomaly, names_all, pair_i, pair_j), width="stretch", key="step_features")
        elif current_step == 2:
            st.markdown(f"**Empirische Verteilungsfunktionen von {names_all[pair_i]} und {names_all[pair_j]}** mit den Positionen einer normalen Tour und einer Sonderfahrt (Höhe = Anteil der Touren mit kleinerem oder gleichem Wert)")
            st.plotly_chart(build_ecdf(ds.X, ds.anomaly, names_all, picks, pair_i, pair_j), width="stretch", key="step_ecdf")
        elif current_step == 3:
            st.markdown("**Beitrag jedes Merkmals** (negative Log-Schwanzwahrscheinlichkeit) zum Wert einer normalen Tour und einer Sonderfahrt - die zwölf größten Beiträge der Sonderfahrt")
            st.plotly_chart(build_contributions(names_all, [("normale Tour", BLUE_TXT, a.contributions[i_norm]), ("Sonderfahrt", RED_TXT, a.contributions[i_anom])]), width="stretch", key="step_contributions")
        elif current_step == 4:
            c1, c2 = st.columns(2)
            c1.markdown("**ECOD-Wert je Tour** (Schwelle gestrichelt)")
            c1.plotly_chart(build_score_hist(values, ds.anomaly, es["threshold"]), width="stretch", key="step_score_hist")
            c2.markdown("**ECOD gegen die Wurzel**: Rang jeder Tour bei beiden")
            c2.plotly_chart(build_rank_compare(values, a.values["robust"], ds.anomaly), width="stretch", key="step_rank_compare")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Kennzahlen bei der gewählten Schwelle**")
            c1.plotly_chart(build_method_bars(a.scores), width="stretch", key="step_bars")
            c2.markdown("**ROC-Kurven** (unabhängig von der Schwelle)")
            curves = {d: (*roc_curve(a.values[d], ds.anomaly), a.scores[d]["auc"]) for d in ("ecod", "lof", "iforest", "robust")}
            c2.plotly_chart(build_roc(curves), width="stretch", key="step_roc")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{ds.n} Touren mit {p_total} Kennzahlen" + (f" ({ds.n_noise} davon reines Rauschen)" if ds.n_noise else "") + f", davon {n_anom} Sonderfahrten ({n_anom / ds.n:.0%}; {C.KIND_LABELS[ds.kind]}), "
               f"{ds.n_modes} Betriebsart{'en' if ds.n_modes > 1 else ''}. Die Ebene ist die der zwei größten Streuungsrichtungen der robusten Schätzung in standardisierten Kennzahlen; ECOD sieht jedes Merkmal einzeln."
               + (" Beim Korrelationsbruch haben die Anomalien in jedem Merkmal normale Werte - nur die Kombination stimmt nicht." if ds.kind == "decorrelated" else ""))
elif step == 2:
    st.caption(f"Die normale Tour liegt bei {names_all[pair_i]} bei F_links = {left[i_norm, pair_i]:.2f} und bei {names_all[pair_j]} bei {left[i_norm, pair_j]:.2f} - mitten in den Verteilungen. Die Sonderfahrt bei {left[i_anom, pair_i]:.2f} und {left[i_anom, pair_j]:.2f} "
               f"(rechts: {right[i_anom, pair_i]:.2f} und {right[i_anom, pair_j]:.2f}); der kleinere der beiden Werte je Merkmal ist ihre Schwanzwahrscheinlichkeit."
               + (" Beim Korrelationsbruch liegt auch die Sonderfahrt in jedem einzelnen Merkmal unauffällig." if ds.kind == "decorrelated" else ""))
elif step == 3:
    c_n, c_a = a.contributions[i_norm], a.contributions[i_anom]
    st.caption(f"Summe der Beiträge: normale Tour {c_n.sum():.1f}, Sonderfahrt {c_a.sum():.1f} (größtmöglicher Beitrag je Merkmal: log n = {np.log(ds.n):.1f}, bei {p_total} Merkmalen also höchstens {p_total * np.log(ds.n):.0f}). "
               f"Variante: {VARIANT_LABELS[variant]}. Jedes Rauschmerkmal trägt im Mittel 1 bei (Streuung 1) - die Summe über {ds.n_noise} Rauschmerkmale ist selbst ein Rauschen von etwa ±{np.sqrt(max(ds.n_noise, 0)):.1f}." if ds.n_noise else
               f"Summe der Beiträge: normale Tour {c_n.sum():.1f}, Sonderfahrt {c_a.sum():.1f} (größtmöglicher Beitrag je Merkmal: log n = {np.log(ds.n):.1f}, bei {p_total} Merkmalen also höchstens {p_total * np.log(ds.n):.0f}). Variante: {VARIANT_LABELS[variant]}.")
elif step == 4:
    st.caption(f"Mittlerer Wert der normalen Touren {values[~ds.anomaly].mean():.1f}, der Sonderfahrten {values[ds.anomaly].mean():.1f}; die Fisher-Schwelle liegt bei {es['threshold']:.1f} und markiert {es['n_flagged']} von {ds.n} Touren. "
               f"Rechts: liegen die Punkte nahe der Diagonalen, urteilen ECOD und die Wurzel gleich; Sonderfahrten, die bei der Wurzel oben und bei ECOD unten liegen, sind für ECOD unsichtbar (Korrelationsbruch).")
else:
    st.caption("Die ROC-Kurve zeigt die Rangfolge (AUC), die Balken die Wirkung der Schwelle: eine perfekte Rangfolge kann trotzdem viele Fehlalarme oder verpasste Anomalien haben, wenn die Schwelle nicht passt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Detektoren gefunden haben – ECOD gegen LOF gegen Isolation Forest gegen Wurzel")
st.caption(
    "Anomalie = Tour über der Schwelle (Standard: ECOD über dem Fisher-Quantil, LOF über 1.5, Isolation Forest über Score 0.5, robust über dem χ²-Quantil). **AUC**: Wahrscheinlichkeit, dass eine zufällige Sonderfahrt einen größeren Wert hat als eine "
    "zufällige normale Tour (1 = perfekte Rangfolge, 0.5 = Raten, darunter: die Anomalien wirken normaler als die Normalen). **Recall**: Anteil der gefundenen Sonderfahrten. **Fehlalarmrate**: Anteil der normalen Touren, die markiert werden."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC (ECOD)", f"{es['auc']:.2f}", help="Rangfolge der Anomalie-Werte; darunter die anderen Detektoren.")
m1.caption(f"LOF {ls['auc']:.2f} · Isolation Forest {ifs['auc']:.2f} · robust {rs['auc']:.2f}")
m2.metric("Recall (ECOD)", _pct(es["recall"]), help=f"Anteil der {n_anom} Sonderfahrten, die bei der Schwelle markiert werden.")
m2.caption(f"LOF {_pct(ls['recall'])} · Isolation Forest {_pct(ifs['recall'])} · robust {_pct(rs['recall'])}")
m3.metric("Fehlalarmrate (ECOD)", f"{es['false_alarm']:.1%}", help="Anteil der normalen Touren, die als Anomalie markiert werden.")
m3.caption(f"LOF {ls['false_alarm']:.1%} · Isolation Forest {ifs['false_alarm']:.1%} · robust {rs['false_alarm']:.1%}")
m4.metric("F1 (ECOD)", f"{es['f1']:.2f}", help="Harmonisches Mittel aus Precision und Recall bei der Schwelle; darunter die F1 der anderen Detektoren und die von ECOD, wenn der wahre Anteil bekannt wäre.")
m4.caption(f"LOF {ls['f1']:.2f} · Isolation Forest {ifs['f1']:.2f} · robust {rs['f1']:.2f} · ECOD mit bekanntem Anteil {a.oracle_f1['ecod']:.2f}")

_t = vd
if code == "ecod_wins":
    st.success(f"✅ ECOD ist besser: AUC {_t['ecod_auc']:.2f} gegen {_t['best_other_auc']:.2f} beim besten anderen Detektor (LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}) - "
               f"ohne k, ohne Kovarianz, ohne Zufall. An der Schwelle: Recall {_pct(_t['ecod_recall'])}, F1 {_t['ecod_f1']:.2f}, Fehlalarmrate {_t['ecod_false_alarm']:.0%}.")
elif code == "comparable":
    st.success(f"✅ ECOD ist ebenbürtig: AUC {_t['ecod_auc']:.2f} (LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}); bei der Schwelle F1 {_t['ecod_f1']:.2f} (LOF {_t['lof_f1']:.2f}, "
               f"Isolation Forest {_t['iforest_f1']:.2f}, robust {_t['robust_f1']:.2f}), mit bekanntem Anteil {_t['oracle_ecod']:.2f}. ECOD braucht dafür keinen einzigen Parameter.")
elif code == "threshold_off":
    st.warning(f"⚠️ Die Schwelle passt nicht: die Rangfolge ist gut (AUC {_t['ecod_auc']:.2f}), aber die Fisher-Schwelle {_t['ecod_threshold']:.1f} markiert {_t['ecod_false_alarm']:.0%} der normalen Touren (Sollwert {1 - ecod_quantile:.1%}): "
               f"F1 {_t['ecod_f1']:.2f} gegen {_t['oracle_ecod']:.2f} mit bekanntem Anteil. Die Herleitung gilt nur für unabhängige Merkmale (und die zweiseitige Variante); hier sind die Kennzahlen korreliert. "
               f"LOF {_t['lof_f1']:.2f}, Isolation Forest {_t['iforest_f1']:.2f}.")
elif code == "blind":
    st.warning(f"⚠️ ECOD ist blind: AUC {_t['ecod_auc']:.2f} (unter Raten), während LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f} und robust {_t['robust_auc']:.2f} erreichen. "
               "Die Anomalien haben in jedem einzelnen Merkmal normale Werte - nur die Abhängigkeit zwischen den Merkmalen stimmt nicht, und die sieht ECOD nicht.")
elif code == "gap":
    st.warning(f"⚠️ Die Anomalien liegen in der Lücke zwischen den Betriebsarten: AUC {_t['ecod_auc']:.2f} bei ECOD (weit unter Raten: die Lücke liegt in der Mitte der Randverteilungen, die Anomalien wirken normaler als die Normalen), "
               f"{_t['lof_auc']:.2f} beim LOF, {_t['iforest_auc']:.2f} beim Isolation Forest, {_t['robust_auc']:.2f} robust: keiner findet sie.")
elif code == "others_win":
    st.warning(f"⚠️ Ein anderer Detektor ist besser: AUC ECOD {_t['ecod_auc']:.2f}, LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}. "
               "Bei vielen Rauschmerkmalen verrauscht die Summe der Beiträge (jedes Rauschmerkmal addiert im Mittel 1 mit Streuung 1); die Wälder und der LOF ranken dort noch etwas besser.")

d1, d2c = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    rows = [("AUC", "auc", "{:.2f}"), ("mittlere Präzision (AP)", "ap", "{:.2f}"), ("Precision", "precision", "{:.2f}"), ("Recall", "recall", "{:.2f}"), ("F1", "f1", "{:.2f}"),
            ("Fehlalarmrate", "false_alarm", "{:.3f}"), ("Schwelle", "threshold", "{:.2f}")]
    st.table({"Kennzahl": [r[0] for r in rows], "ECOD": [r[2].format(es[r[1]]) for r in rows], "LOF": [r[2].format(ls[r[1]]) for r in rows], "Isolation Forest": [r[2].format(ifs[r[1]]) for r in rows],
              "robust (MCD)": [r[2].format(rs[r[1]]) for r in rows]})
with d2c:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Rechenzeit", "Parameter", "Bewertung", "mittlerer Wert (normal / Anomalie)"],
              "ECOD": [f"{a.seconds['ecod'] * 1000:.1f} ms", "keine", "Randverteilungen, Summe der −log F", f"{values[~ds.anomaly].mean():.1f} / {values[ds.anomaly].mean():.1f}"],
              "LOF": [f"{a.seconds['lof'] * 1000:.1f} ms", "k = 20 (höchstens n / 2)", "Dichte gegenüber den Nachbarn", f"{a.values['lof'][~ds.anomaly].mean():.2f} / {a.values['lof'][ds.anomaly].mean():.2f}"],
              "Isolation Forest": [f"{a.seconds['iforest'] * 1000:.0f} ms", f"{len(a.forest_if.trees)} Bäume × ψ = {a.forest_if.psi}", "Pfadlänge in zufälligen Bäumen", f"{a.values['iforest'][~ds.anomaly].mean():.2f} / {a.values['iforest'][ds.anomaly].mean():.2f}"]})
    st.caption("ECOD ist deterministisch und braucht keine Standardisierung (nur die Ränge der Werte je Merkmal zählen: eine monotone Umrechnung eines Merkmals ändert nichts). Die klassische Schätzung der Wurzel entfällt in dieser Demo.")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Daten und Variante ab?")
sweep_options = [k for k in SWEEP_LABELS if not ((kind in ("gap", "decorrelated") and k in ("strength",)) or (kind == "gap" and k == "n_modes") or (threshold_kind == "share" and k in ("ecod_quantile", "quantile")) or (threshold_kind == "standard" and k == "share"))]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
current = {"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(noise), "contamination": int(contamination), "strength": float(strength),
           "ecod_quantile": float(ecod_quantile), "quantile": float(quantile), "share": int(share), "variant": None}[sweep_param]
if st.button("Sweep über 5 feste Datensätze berechnen (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows_sweep = _sweep(sweep_param, tuple(kv for kv in base_data if kv[0] != sweep_param), settings, SWEEP_VALUES[sweep_param])
    if sweep_param == "variant":
        rows_sweep = [{**r, "x": VARIANT_NAMES[r["x"]]} for r in rows_sweep]
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Links die Rangfolge (AUC), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle. "
               "Bei der Variante ändern sich nur die Linien von ECOD.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wo ECOD trifft und wo nicht: neun Szenarien")
if st.button("Die vier Detektoren in neun Szenarien vergleichen (dauert etwa 30 Sekunden)", key="scenarios_start"):
    st.session_state["scenarios_on"] = True
if st.session_state.get("scenarios_on"):
    with st.spinner("Rechne 9 Szenarien × 5 Aufnahmen..."):
        sc_rows = _scenarios(settings)
    st.plotly_chart(build_scenarios(sc_rows), width="stretch", key="scenarios_chart")
    st.caption("Mittel über 5 feste Datensätze, ECOD wie im Original (Variante Paper). **ECOD trifft**: dichte Gruppe abseits (AUC 0.99 bei 10 %, 0.86 bei 30 % - Isolation Forest 0.95 und 0.70, LOF 0.35 und 0.47, robust 1.00 und 0.49) und viele verstreute Anomalien "
               "(0.98 bei 45 %; LOF 0.64, robust 0.87). **ECOD ist blind**: Korrelationsbruch (AUC 0.40 und 0.47 bei p = 30; LOF 0.99 und 1.00, robust 1.00, Isolation Forest 0.80 und 0.83) und die Lücke (0.08 und 0.04 - weit unter Raten, die Lücken-Touren liegen in der Mitte der Randverteilungen und wirken "
               "normaler als die Normalen; die anderen 0.27-0.54). Bei 40 Rauschmerkmalen 0.95 (LOF und Isolation Forest 0.99, robust 0.88). Im Standardfall 0.99 - aber das F1 mit bekanntem Anteil ist nur 0.83 (LOF und Isolation Forest 0.97): die Spitze der Rangfolge ist unschärfer.")

st.markdown("---")

st.subheader("🔬 Varianten: Paper, zweiseitig, nach Schiefe")
if st.button("Die drei Varianten von ECOD in neun Szenarien vergleichen (dauert etwa 10 Sekunden)", key="variants_start"):
    st.session_state["variants_on"] = True
if st.session_state.get("variants_on"):
    with st.spinner("Rechne 9 Szenarien × 3 Varianten × 5 Aufnahmen..."):
        vt = _variants()
    st.plotly_chart(build_variants(vt), width="stretch", key="variants_chart")
    st.caption("AUC der drei Varianten (Mittel über 5 feste Datensätze). Paper (Maximum der drei Summen) und zweiseitig ranken fast gleich (Standardfall 0.99 und 1.00, 45 % verstreut 0.98 und 1.00); die Variante **nur nach Schiefe** sieht bei symmetrisch verteilten Anomalien nur die Hälfte "
               "(0.69 im Standardfall, 0.67 bei 45 % verstreut) - dafür ist sie bei der **dichten Gruppe abseits** die beste (1.00 bei 10 %, 0.99 bei 30 %: die Gruppe liegt auf einer Seite, und die Schiefe zeigt dorthin). "
               "Beim Korrelationsbruch ist keine Variante über 0.56, in der Lücke keine über 0.46.")

st.markdown("---")

st.subheader("🔬 Die Fisher-Schwelle: ohne Vorwissen, aber nur bei unabhängigen Merkmalen")
if st.button("Fehlalarmrate der Fisher-Schwelle je Merkmalszahl berechnen (dauert etwa 15 Sekunden)", key="calibration_start"):
    st.session_state["calibration_on"] = True
if st.session_state.get("calibration_on"):
    with st.spinner("Rechne 5 Merkmalszahlen × 5 Aufnahmen..."):
        cal = _calibration(float(ecod_quantile))
    st.plotly_chart(build_calibration(cal), width="stretch", key="calibration_chart")
    st.caption(f"Fehlalarmrate bei (fast) reinen Normalen (Mittel über 5 feste Datensätze, Quantil {ecod_quantile:.3f}, Sollwert {1 - ecod_quantile:.1%}; die Zahlen im Text gelten für 0.975). Bei **unabhängigen Merkmalen** hält die zweiseitige Variante den Sollwert (1.3-1.9 %, "
               "etwas darunter), die Paper-Variante liegt beim Doppelten (4.5-5.4 %, das Maximum aus drei Summen verschiebt die Verteilung). Mit den **korrelierten Kennzahlen des Szenarios** steigt die Fehlalarmrate der zweiseitigen Variante mit der Merkmalszahl: "
               "1.9 % / 6.5 % / 12 % / 18 % / 21 % bei p = 2 / 5 / 12 / 20 / 30, die der Paper-Variante auf 1.8 % / 14 % / 29 % / 28 % / 32 % - die Summe korrelierter Beiträge streut stärker als die Herleitung annimmt. "
               "Die mittlere absolute Korrelation der Normalen wächst von 0.28 (p = 2) auf 0.58 (p = 30).")

st.markdown("---")

st.subheader("🔬 Rauschmerkmale und kleine Stichproben")
if st.button("Rauschmerkmale 0-40 für Paper und zweiseitig durchfahren (dauert etwa 40 Sekunden)", key="noise_start"):
    st.session_state["noise_on"] = True
if st.session_state.get("noise_on"):
    with st.spinner("Rechne 6 Werte × 2 Varianten × 5 Aufnahmen..."):
        nb = tuple(kv for kv in base_data if kv[0] != "n_noise")
        n_paper = _sweep("n_noise", nb, Settings(variant="paper", threshold_kind=threshold_kind, ecod_quantile=float(ecod_quantile), quantile=float(quantile), share=int(share)), SWEEP_VALUES["n_noise"])
        n_two = _sweep("n_noise", nb, Settings(variant="twosided", threshold_kind=threshold_kind, ecod_quantile=float(ecod_quantile), quantile=float(quantile), share=int(share)), SWEEP_VALUES["n_noise"])
    c1, c2 = st.columns(2)
    c1.markdown("**Original (Paper)**")
    c1.plotly_chart(build_sweep(n_paper, SWEEP_LABELS["n_noise"]), width="stretch", key="noise_paper_chart")
    c2.markdown("**Zweiseitig mit p-Werten (Fisher)**")
    c2.plotly_chart(build_sweep(n_two, SWEEP_LABELS["n_noise"]), width="stretch", key="noise_twosided_chart")
    st.caption("Mittel über 5 feste Datensätze. Die AUC von ECOD sinkt langsam: Paper 0.99 → 0.95, zweiseitig 1.00 → 0.97 bei 40 Rauschmerkmalen (LOF und Isolation Forest 0.99, robust 0.88). Jedes Rauschmerkmal addiert im Mittel 1 mit Streuung 1 zur Summe - bei 40 Rauschmerkmalen ±6, "
               "gegen etwa 5.7 (log 300) je informativem Merkmal am äußersten Rand. An der Fisher-Schwelle **verliert ECOD dabei nicht**: zweiseitig F1 0.83 → 0.74 (Recall 0.99 → 0.68, Fehlalarmrate 1.5 %), während der LOF auf 0.00 fällt (Recall 0.00) und der Isolation Forest auf 0.56 (Recall 0.39). "
               "Kleine Stichproben: die AUC von ECOD ist schon bei n = 20 0.99 (Sweep über die Tourenzahl oben).")

st.markdown("---")

st.subheader("🔬 Dichte Gruppen und viele Anomalien")
if st.button("Anteil der dichten Gruppe von 2 bis 45 % durchfahren (dauert etwa 25 Sekunden)", key="cluster_start"):
    st.session_state["cluster_on"] = True
if st.session_state.get("cluster_on"):
    with st.spinner("Rechne 7 Anteile × 5 Aufnahmen..."):
        cl = _sweep("contamination", tuple(kv for kv in base_data if kv[0] not in ("contamination", "kind", "n_modes")) + (("kind", "cluster"), ("n_modes", 1)), settings, SWEEP_VALUES["contamination"])
    st.plotly_chart(build_sweep(cl, SWEEP_LABELS["contamination"]), width="stretch", key="cluster_chart")
    st.caption("Dichte Gruppe abseits, eine Betriebsart (Mittel über 5 feste Datensätze). AUC von ECOD bei 2 / 5 / 10 / 20 / 30 / 40 / 45 %: 1.00 / 1.00 / 0.99 / 0.95 / 0.86 / 0.71 / 0.62 - **ohne k und ohne Kovarianz**, wo der LOF bei 10 % unter Raten liegt (0.35) und die robuste Schätzung ab 30 % kippt (0.49). "
               "Die Gruppe verschiebt die Randverteilungen erst, wenn sie einen großen Teil der Touren stellt; bei 40 % und 45 % ist ECOD mit 0.71 und 0.62 der beste von vier, aber kaum noch brauchbar. Die Fehlalarmrate an der Fisher-Schwelle liegt bei allen Anteilen bei 19-38 %.")

st.markdown("---")

st.subheader("🔬 Die Schwelle")
if st.button("Fisher-Quantile vergleichen (dauert etwa 20 Sekunden)", key="threshold_start"):
    st.session_state["threshold_on"] = True
if st.session_state.get("threshold_on"):
    with st.spinner("Rechne 5 Quantile × 5 Datensätze und drei angenommene Anteile..."):
        tt = _threshold(base_data, settings)
    c1, c2 = st.columns(2)
    c1.markdown("**ECOD: F1, Recall und Fehlalarmrate je Fisher-Quantil**")
    c1.plotly_chart(build_cutoff(tt["cutoff"]), width="stretch", key="cutoff_chart")
    c2.markdown("**F1 bei falsch angenommenem Anteil** (½×, 1×, 2× des wahren)")
    c2.plotly_chart(build_wrong_share(tt["wrong_share"]), width="stretch", key="wrong_share_chart")
    st.caption("Links (Standardfall, Original-Variante): F1 0.37 / 0.45 / 0.52 / 0.62 / 0.81 bei den Quantilen 0.9 / 0.95 / 0.975 / 0.99 / 0.999, Fehlalarmrate 39 % / 28 % / 20 % / 13 % / 3 %, Recall 1.00 bis 0.88 - die Schwelle passt erst bei einem Quantil, das weit über dem Sollwert liegt. "
               "Rechts: ein falsch angenommener Anteil kostet alle vier Detektoren fast gleich viel (0.83 → 0.65 bei ECOD, 0.97 → 0.67 bei LOF und Isolation Forest), denn dann entscheidet nur die Rangfolge.")

st.markdown("---")

st.subheader("🔬 Kosten")
if st.button("Rechenzeit berechnen (dauert etwa 20 Sekunden)", key="cost_start"):
    st.session_state["cost_on"] = True
if st.session_state.get("cost_on"):
    with st.spinner("Messe die Rechenzeit..."):
        ct = _costs(tuple(kv for kv in base_data if kv[0] not in ("n", "n_noise")), settings)
    st.plotly_chart(build_costs(ct["times"]), width="stretch", key="cost_chart")
    st.caption("Mittel über 3 feste Datensätze (Zeiten rechnerabhängig, nur die Größenordnungen zählen). ECOD sortiert jedes Merkmal einmal (O(n · p · log n)) und braucht höchstens etwa 5 ms (600 Touren, 52 Merkmale); der Isolation Forest ist mit 65-415 ms etwa 100-mal langsamer. "
               "Der LOF (0.4-17 ms) wächst quadratisch mit der Tourenzahl: ab 300 Touren ist ECOD schneller (bei 600 Touren 3- bis 15-mal), bei 100 Touren und 52 Merkmalen aber nicht - dort ist der LOF mit 0.4 ms gegen 0.9 ms schneller.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Anomalien verraten sich in einer Randverteilung** | Nicht beim **Korrelationsbruch**: die Merkmale der Anomalien haben normale Randverteilungen, ECOD hat AUC 0.40 (unter Raten), LOF 0.99, robust 1.00, Isolation Forest 0.80. Auch mit 30 Merkmalen: 0.47. | Wurzel (Kovarianz), LOF (Nachbarschaft) |
| **Anomalien liegen am Rand** | Nicht in der **Lücke** zwischen Betriebsarten: AUC 0.08 (2) und 0.04 (3 Betriebsarten) - die Lücke liegt in der Mitte der Randverteilungen, die Anomalien wirken **normaler als die Normalen**. Die anderen Detektoren scheitern auch (0.27-0.54), aber nicht unter Raten. | LOF mit passendem k (siehe LOF-Demo) |
| **Die Fisher-Schwelle braucht keinen Anteil** | Nur bei unabhängigen Merkmalen: mit den korrelierten Kennzahlen des Szenarios markiert die Original-Variante 20 % der Normalen (Sollwert 2.5 %, F1 0.52), die zweiseitige 4.5 % (F1 0.83) - und bei 30 Merkmalen 21 % (zweiseitig) bzw. 32 %. | erwarteter Anteil (PyOD), Kalibrierung |
| **Alle Merkmale sind informativ** | Jedes Rauschmerkmal addiert im Mittel 1 mit Streuung 1 zur Summe: bei 40 Rauschmerkmalen sinkt die AUC von 0.99 auf 0.95 (zweiseitig 0.97), an der Fisher-Schwelle bleibt ECOD trotzdem brauchbar (F1 0.74 zweiseitig; LOF 0.00). | Feature Bagging, Merkmalsauswahl |
| **Der Rang an der Spitze ist scharf** | Auch im Standardfall (AUC 0.99) ist das F1 mit bekanntem Anteil nur 0.83 (LOF und Isolation Forest 0.97): einzelne normale Touren mit mehreren extremen Merkmalen mischen sich in die Spitze. | Isolation Forest, LOF |
| **Sehr große Anomaliegruppen** | Bei 40 % und 45 % dichter Gruppe fällt ECOD auf 0.71 und 0.62 - der beste von vier, aber kaum brauchbar (Isolation Forest 0.40 und 0.27, LOF 0.49 und 0.50, robust 0.39 und 0.35). Die Gruppe *ist* dann der Normalbereich. | (keiner) |
"""
)
st.caption(
    "Die Nachbarn der Anomalie-Erkennung-Linie: die Wurzel Elliptic Envelope, LOF, Feature Bagging, Isolation Forest und Extended IF (gebaut), One-Class SVM und Deep SVDD, ein Autoencoder (noch nicht gebaut). "
    "Keiner ist überlegen: ECOD gewinnt bei dichten Gruppen und vielen verstreuten Anomalien - ohne einen einzigen Parameter, ohne Zufall, in Millisekunden - und ist blind für gebrochene Abhängigkeiten und für die Lücke."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Randverteilungen.** Für $n$ Touren und Merkmal $j$: $F_j^{l}(x) = \frac1n \#\{i : x_{ij} \le x\}$ und $F_j^{r}(x) = \frac1n \#\{i : x_{ij} \ge x\}$ (Bindungen zählen auf beiden Seiten mit), Werte in $[1/n, 1]$. Beitrag: $U_j^{l} = -\log F_j^{l}$, $U_j^{r} = -\log F_j^{r}$.

**Wert.** $O^{l}(i) = \sum_j U_j^{l}(x_{ij})$, $O^{r}(i) = \sum_j U_j^{r}(x_{ij})$, $O^{\text{auto}}(i) = \sum_j \big[\gamma_j < 0 \,?\, U_j^{l} : U_j^{r}\big]$ mit der Schiefe $\gamma_j$. Original: $\text{ECOD}(i) = \max\{O^{l}, O^{r}, O^{\text{auto}}\}$.
Zweiseitig: $\text{ECOD}_2(i) = \sum_j -\log\min\{1, 2\min(F_j^{l}, F_j^{r})\}$.

**Schwelle ohne Vorwissen.** Sind die Merkmale unabhängig und stetig, ist $-\log(2\min(F^{l}, F^{r})) \sim \text{Exp}(1)$ je Merkmal (ein zweiseitiger p-Wert), also $\text{ECOD}_2 \sim \text{Gamma}(p, 1)$ und $2\,\text{ECOD}_2 \sim \chi^2_{2p}$ (Fisher-Methode); als Schwelle dient das $q$-Quantil.
Bei korrelierten Merkmalen ist die Summe überstreut - die Fehlalarmrate liegt über $1 - q$.

**Rauschmerkmale.** Jedes unabhängige Rauschmerkmal addiert im Mittel $1$ und mit Varianz $1$ (zweiseitig) zur Summe; $r$ Rauschmerkmale also $r \pm \sqrt r$ - gegen höchstens $\log n$ je informativem Merkmal.

**Ränge.** Nur die Rangfolge je Merkmal geht ein: monotone Umrechnungen einzelner Merkmale ändern den Wert nicht (LOF und Mahalanobis-Abstände schon).

**Grenzen.** (1) Keine Abhängigkeit zwischen Merkmalen. (2) Anomalien in der Mitte der Randverteilungen wirken normaler als die Normalen. (3) Die Fisher-Schwelle gilt nur für unabhängige Merkmale. (4) Rauschmerkmale verrauschen die Summe.

Implementiert in `ecod_algorithm.py` (Schwanzwahrscheinlichkeiten, Schiefe, Beiträge, Varianten), `ecod_lof.py`, `ecod_isolation_forest.py` und `ecod_ee_algorithm.py` (LOF, Isolation Forest, klassisch und MCD, wortgleich aus den Vorgängern),
`ecod_scenario.py` (Touren mit Betriebsarten, Krümmung, Anomalien, Rauschmerkmalen, Korrelationsbruch), `ecod_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
