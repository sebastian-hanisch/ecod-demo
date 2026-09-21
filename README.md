# ECOD – Ausreißer aus den Randverteilungen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-ecod-demo.streamlit.app/)**

Sechstes Stück der **Anomalie-Erkennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **ECOD** (Li, Zhao, Hu, Botta, Ionescu und Chen) – an einem wachsenden Beispiel, gegen den [LOF](../lof-demo), den [Isolation Forest](../isolation-forest-demo)
und die robuste Schätzung der Wurzel ([elliptic-envelope-demo](../elliptic-envelope-demo)). Vehikel: dieselben **Lieferrouten-Kennzahlen** wie in den Vorgängern und der [pca-demo](../pca-demo)
(Szenario, LOF, Isolation Forest und Wurzel-Schätzer wortgleich übernommen, per Test gegen eingefrorene Werte geprüft; die klassische Schätzung entfällt, damit die Balken lesbar bleiben). Neu ist die Anomalie-Art **Korrelationsbruch**.

**Einordnung in die Reihe (die Kanten des Graphen):** ECOD ist ein **eigener Ast direkt nach der Wurzel** – der Kontrast zu allem, was vorher kam: die Wurzel braucht eine Gauß-Wolke, der LOF ein passendes k, Feature Bagging Teilmengen und Glück, der Isolation Forest einen Anteil für die Schwelle.
ECOD braucht **keinen einzigen Parameter**: je Merkmal die empirische Verteilungsfunktion, je Tour die Summe der negativen Log-Schwanzwahrscheinlichkeiten. Der Preis: ECOD sieht nur **Randverteilungen**, die **Abhängigkeit zwischen den Merkmalen** bleibt unsichtbar.
Ergebnis in Kürze: **ECOD trennt verstreute Anomalien fast perfekt (AUC 0.99), sieht dichte Gruppen, an denen LOF, Isolation Forest und die Wurzel scheitern, und ist gegen 45 % Anomalien unempfindlich – aber es ist blind für gebrochene Abhängigkeiten (AUC 0.40, unter Raten) und für die Lücke zwischen Betriebsarten (AUC 0.08),
und seine parameterfreie Schwelle stimmt bei den korrelierten Kennzahlen nicht (Fehlalarmrate 20 % statt 2.5 %).** Die Linie hat **keinen Konvergenzpunkt**; ECOD ist ein Kontrast, keine Fortsetzung.
```
elliptic-envelope-demo (Wurzel: robuste Ellipse)
  ├─ ecod-demo                  (Kontrast: verteilungsfrei)                              [dieses Stück]
  ├─ lof-demo → feature-bagging-demo (lokale Dichte; Ensembles gegen viele Merkmale)     [beide gebaut]
  ├─ One-Class SVM → Deep SVDD  (gelernte Grenze)                                        [nicht gebaut]
  ├─ isolation-forest-demo → extended-isolation-forest-demo (Zufallsbäume)               [beide gebaut]
  └─ autoencoder-anomalie-demo  (Rekonstruktionsfehler)                                  [gebaut]
```

| Frage | Ergebnis (300 Touren, 12 Merkmale, 10 % verstreute Anomalien im Abstand 6 Faktor-σ, ein Normalbereich, Rauschen 0.25; ECOD wie im Original (Maximum der drei Summen), Schwelle Gamma(p, 1)-Quantil 0.975, LOF k = 20 und Schwelle 1.5, Isolation Forest 100 Bäume × ψ = 256 und Schwelle 0.5, robust χ²-Quantil 0.975; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Standardfall | ✅/⚠️ **Rangfolge gut, Schwelle daneben.** AUC 0.99 (LOF und Isolation Forest 1.00, robust 0.99), aber die Fisher-Schwelle markiert **20 %** der normalen Touren (Sollwert 2.5 %): F1 **0.52** gegen 0.94 (LOF), 0.96 (Isolation Forest) und 0.84 (robust); mit bekanntem Anteil 0.83 (LOF und Isolation Forest 0.97) – die Spitze der Rangfolge ist unschärfer |
| Zweiseitige Variante (p-Werte) | ✅ dieselben Daten mit der Variante, für die die Schwelle hergeleitet ist: AUC 1.00, Fehlalarmrate **4.5 %**, F1 **0.83** (Recall 0.99), mit bekanntem Anteil 0.89. Nur nach Schiefe (O_auto): AUC **0.69** – eine Seite pro Merkmal sieht bei allen Richtungen der Anomalien nur die Hälfte |
| Dichte Gruppe abseits | ✅ **Hier gewinnt ECOD, ohne k und ohne Kovarianz.** AUC bei 10 % / 30 % der Touren: **0.99 / 0.86** (Isolation Forest 0.95 / 0.70, LOF 0.35 / 0.47 (k = 20 unter der Gruppengröße), robust 1.00 / 0.49 (die Gruppe verschiebt die Schätzung)). Bei 2 / 5 / 10 / 20 / 30 / 40 / 45 %: 1.00 / 1.00 / 0.99 / 0.95 / 0.86 / 0.71 / 0.62 – bei 40 % und 45 % der beste von vier, aber kaum noch brauchbar. Die Variante nur nach Schiefe: 1.00 (10 %) und 0.99 (30 %) |
| Viele verstreute Anomalien | ✅ AUC bei 2 / 5 / 10 / 20 / 30 / 40 / 45 %: 0.99 / 0.99 / 0.99 / 0.99 / 0.98 / 0.98 / 0.98 (LOF 1.00 / 1.00 / 1.00 / 0.99 / 0.90 / 0.72 / 0.64, robust 1.00 bis 0.87, Isolation Forest 1.00 überall). F1 an der Schwelle 0.13 / 0.29 / 0.52 / 0.83 / 0.88 / 0.85 / 0.84 – bei wenigen Anomalien ist die Fehlalarmrate von 20–27 % das Problem |
| **Korrelationsbruch** | ❌ **ECOD ist blind.** Die Merkmale der Anomalien sind je Spalte aus den Normalen neu gezogen (identische Randverteilungen, zerstörte Abhängigkeit): AUC **0.40** (unter Raten; mit 30 Merkmalen 0.47), LOF 0.99, robust 1.00, Isolation Forest 0.80. Keine Variante kommt über 0.56 |
| **Lücke zwischen Betriebsarten** | ❌ ECOD AUC **0.08** (zwei Betriebsarten) und **0.04** (drei) – weit unter Raten: die Lücke liegt in der **Mitte** der Randverteilungen, die Anomalien wirken **normaler als die Normalen**. Die anderen scheitern auch (0.27–0.54), aber nicht unter Raten; keine Variante von ECOD über 0.46 |
| Fisher-Schwelle | ⚠️ **Nur bei unabhängigen Merkmalen kalibriert.** Fehlalarmrate bei reinen Normalen (Quantil 0.975): unabhängige Merkmale zweiseitig 1.3–1.9 % (etwas unter dem Sollwert), Original 4.5–5.4 %; mit den **korrelierten** Kennzahlen des Szenarios wächst sie mit der Merkmalszahl: zweiseitig 1.9 / 6.5 / 12 / 18 / 21 % bei p = 2 / 5 / 12 / 20 / 30, Original 1.8 / 14 / 29 / 28 / 32 % (mittlere absolute Korrelation der Normalen 0.28 → 0.58). Fisher-Quantile 0.9 / 0.95 / 0.975 / 0.99 / 0.999: F1 0.37 / 0.45 / 0.52 / 0.62 / 0.81 |
| Rauschmerkmale | ✅/❌ AUC 0.99 → 0.95 bei 40 Rauschmerkmalen (zweiseitig 1.00 → 0.97; LOF und Isolation Forest 0.99, robust 0.88). An der Schwelle **verliert ECOD weniger**: Recall 0.99 / 0.96 / 0.91 / 0.88 / 0.84 (0 / 10 / 20 / 30 / 40 Rauschmerkmale), LOF 1.00 / 0.71 / 0.18 / 0.03 / **0.00**, Isolation Forest 0.99 / 0.91 / 0.71 / 0.58 / 0.39; zweiseitig F1 0.83 → 0.74 (Fehlalarmrate 1.5 %). Jedes Rauschmerkmal addiert im Mittel 1 mit Streuung 1 zur Summe |
| Krümmung, Rauschen, Betriebsarten | ➖ AUC bleibt bei 0.98–1.00. Aber die **Schwelle wird besser** – F1 0.52 / 0.62 / 0.75 / 0.85 / 0.88 bei Krümmung 0 / 0.25 / 0.5 / 0.75 / 1 (Fehlalarmrate 20 % → 2.9 %), weil die Krümmung die Kennzahlen teilweise entkorreliert (mittlere absolute Korrelation 0.51 → 0.31); ebenso Messrauschen 0 → 1.0 (F1 0.51 → 0.70). Der LOF verliert bei Krümmung (0.94 → 0.74) |
| Kleine Stichproben | ➖ AUC 0.99 schon bei 20 Touren; F1 bei 20 / 30 / 50 / 100 / 200 Touren 0.75 / 0.59 / 0.60 / 0.61 / 0.58 (Fehlalarmrate 8 % → 16 %), LOF 0.96 / 0.94 / 0.88 / 0.86 / 0.91, Isolation Forest 0.60 / 0.67 / 0.78 / 0.89 / 0.96 |
| Falsch angenommener Anteil | ➖ kostet alle vier fast gleich: F1 bei ½× / 1× / 2× des wahren Anteils (10 %) ECOD 0.66 / 0.83 / 0.65, LOF und Isolation Forest 0.67 / 0.97 / 0.67 – dann entscheidet nur die Rangfolge |
| Rechenzeit | ➖ ECOD höchstens etwa 5 ms (600 Touren, 52 Merkmale), Isolation Forest 65–415 ms (etwa 100-mal), LOF 0.4–17 ms: ab 300 Touren ist ECOD schneller (bei 600 Touren 3- bis 15-mal), bei 100 Touren und 52 Merkmalen ist der LOF schneller – rechnerabhängig |

## Was die Demo zeigt

1. **ECOD in Aktion** (Schritt-Slider + Abspielen): **Touren** (Ebene der größten Streuung und zwei Rohmerkmale; beim Korrelationsbruch das am stärksten korrelierte Paar) → **Randverteilungen** (empirische Verteilungsfunktionen zweier Merkmale mit den Positionen einer normalen Tour und einer Sonderfahrt) →
   **Beiträge** (−log F je Merkmal, Summe = Wert) → **Wert** (Verteilung der ECOD-Werte mit Schwelle; Rang bei ECOD gegen Rang bei der Wurzel) → **Ergebnis** (Kennzahlen und ROC-Kurven der vier Detektoren).
2. **Was die Detektoren gefunden haben:** AUC, Recall, Fehlalarmrate, F1 (dazu der F1 mit bekanntem Anteil), Urteil (Codes: Lücke → ECOD blind (Korrelationsbruch) → anderer Detektor besser → ECOD besser → falsche Schwelle bei guter Rangfolge → ebenbürtig), Detailtabelle mit Rechenzeiten.
3. **📐 Sweeps** über Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil und Abstand der Anomalien, Fisher-Quantil, χ²-Quantil, angenommenen Anteil und die Variante (feste Datensätze ab 100000, Streuung).
4. **🔬 Experimente auf Abruf:** neun Szenarien × vier Detektoren, die drei Varianten von ECOD, Kalibrierung der Fisher-Schwelle je Merkmalszahl (korreliert und unabhängig), Rauschmerkmale (Original und zweiseitig), dichte Gruppe von 2 bis 45 %, Schwelle (Fisher-Quantile und falsch angenommener Anteil), Kosten.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Randverteilungen genügen, Anomalien am Rand, Schwelle ohne Anteil, alle Merkmale informativ, scharfe Spitze, sehr große Anomaliegruppen).

Regler: Touren (20–600), Merkmale (2–30), Rauschmerkmale (0–40), Betriebsarten (1–3), Krümmung, Rauschen, Anteil der Anomalien (1–45 %), Art (verstreut / dichte Gruppe / in der Lücke – ab zwei Betriebsarten / **Korrelationsbruch**), Abstand (bei "Lücke" und "Korrelationsbruch" ausgeblendet, Wert bleibt erhalten),
**Variante** (Original / zweiseitig mit p-Werten / nur nach Schiefe), **Schwelle** (Standard: Fisher-Quantil und χ²-Quantil, ausgeblendet beim erwarteten Anteil; sonst der angenommene Anteil für alle vier).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | AUC ECOD | F1 ECOD | AUC LOF | F1 LOF | AUC IF | F1 IF | AUC robust | F1 robust | Urteil |
|---|---|---|---|---|---|---|---|---|---|
| Standardfall (Original) | 1.00 | 0.55 | 1.00 | 0.97 | 1.00 | 0.98 | 1.00 | 0.85 | Schwelle passt nicht |
| Standardfall, zweiseitig (Fisher) | 1.00 | 0.91 | 1.00 | 0.97 | 1.00 | 0.98 | 1.00 | 0.85 | ebenbürtig |
| Dichte Gruppe abseits (30 %) | 0.86 | 0.69 | 0.44 | 0.00 | 0.69 | 0.21 | 0.39 | 0.02 | ECOD besser |
| Korrelationsbruch | 0.37 | 0.03 | 0.98 | 0.83 | 0.78 | 0.23 | 1.00 | 0.84 | ECOD blind |
| Lücke, 2 Betriebsarten | 0.17 | 0.00 | 0.60 | 0.00 | 0.63 | 0.05 | 0.35 | 0.00 | keiner findet sie |
| Viele Anomalien (45 %) | 0.99 | 0.82 | 0.64 | 0.16 | 1.00 | 0.83 | 0.96 | 0.85 | ebenbürtig |
| 40 Rauschmerkmale | 0.96 | 0.64 | 0.98 | 0.06 | 0.99 | 0.42 | 0.90 | 0.36 | ebenbürtig |

## Modell und Verfahren

- **Szenario** (`ecod_scenario.py`): wortgleich aus den Vorgängern (zwei versteckte Faktoren, 12 Kennzahlen der PCA-Demo, Betriebsarten, Krümmung, Anomalien verstreut / dichte Gruppe / in der Lücke mit exaktem Anteil, Zusatzmerkmale bis p = 30, Rauschmerkmale).
  **Neu: Korrelationsbruch** – die Merkmale der Anomalien werden je Spalte unabhängig aus den Normalen gezogen (gleiche Randverteilungen, zerstörte Abhängigkeit); als letzte Ziehung angehängt, damit alle bisherigen Datensätze bit-identisch bleiben (per Test).
- **ECOD** (`ecod_algorithm.py`, numpy von Grund auf): Schwanzwahrscheinlichkeiten F_links = #{x_i ≤ x} / n und F_rechts = #{x_i ≥ x} / n je Merkmal (Bindungen zählen auf beiden Seiten), Beitrag −log F, Schiefe je Merkmal. **Varianten:** *paper* = Maximum von O_links, O_rechts und O_auto (Seite nach Vorzeichen der Schiefe);
  *zweiseitig* = Σ −log min(1, 2 · min(F_links, F_rechts)) – unter Unabhängigkeit exakt Gamma(p, 1), Grundlage der Schwelle (Fisher-Methode: 2 · Summe ~ χ²(2p)); *auto* = nur O_auto; dazu *pyod* (je Merkmal das Maximum aus links, rechts und Schiefe-Term, dann summiert – **eine andere Rechnung als im Papier**), nur zum Abgleich mit PyOD.
- **Isolation Forest** (`ecod_isolation_forest.py`), **LOF** (`ecod_lof.py`, k = 20, höchstens n / 2, standardisierte Merkmale) und **Wurzel** (`ecod_ee_algorithm.py`: χ² ohne scipy, klassisch, FastMCD): wortgleich aus den Vorgängern.
- **Auswertung** (`ecod_evaluation.py`): AUC, mittlere Präzision, Precision, Recall, F1, Fehlalarmrate für vier Detektoren; **F1 mit bekanntem Anteil** als Referenz für die Schwelle; Sweeps, Experiment-Tabellen, Urteil.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutungen (vor dem Bau gemessen):** (1) "ECOD trennt verstreute Anomalien perfekt und ist das billigste Verfahren" – **halb bestätigt**: AUC 0.99 (nicht 1.00), die Spitze der Rangfolge ist unscharf (F1 mit bekanntem Anteil 0.83 gegen 0.97); billig ja, aber nicht überall das billigste (bei 100 Touren und 52 Merkmalen ist der LOF schneller).
  (2) "Dichte Gruppe: ECOD sieht sie ohne Parameter" – **bestätigt** (AUC 0.99 gegen 0.35 beim LOF), auch bei 30 %. (3) "Korrelationsbruch: ECOD bei etwa 0.5" – **schlimmer**: 0.40, unter Raten. (4) "Lücke: ECOD blind, aber nicht schlechter als die Vorgänger" – **widerlegt**: 0.08 und 0.04, weit unter Raten (die anderen 0.27–0.54).
  (5) "Die Fisher-Schwelle ist bei den korrelierten Kennzahlen nicht kalibriert" – **bestätigt**, und sie wird mit der Merkmalszahl schlechter; die zweiseitige Variante ist nur bei kleinem p ordentlich (1.9 % bei p = 2, 12 % bei p = 12). (6) "Rauschmerkmale schaden ECOD weniger als LOF" – **bestätigt an der Schwelle** (Recall 0.84 gegen 0.00 bei 40), **nicht in der Rangfolge** (0.95 gegen 0.99).
  (7) "Bei stark gebogener Fläche ändert sich die AUC nicht, die Schwelle schon" – bestätigt, aber in die **falsche Richtung**: die Schwelle wird *besser*, weil die Krümmung die Kennzahlen entkorreliert.
- **Die Variante nur nach Schiefe (O_auto)** taugt für symmetrische Anomalien nicht (AUC 0.69), gewinnt aber bei der dichten Gruppe (1.00 / 0.99 bei 10 % / 30 %). Auch das **Maximum der drei Summen** verliert bei gemischten Vorzeichen etwas gegen die zweiseitige Variante (Standardfall 0.99 gegen 1.00; F1 0.52 gegen 0.83).
- **PyOD rechnet anders als das Papier:** PyOD nimmt je Merkmal das Maximum und summiert, das Papier das Maximum der drei Summen. Die Demo hat beide; der Abgleich mit PyOD gilt für die *pyod*-Variante (Differenz < 1e-14). Bei einem ersten Abgleich mit dem rohen Schiefe-Wert statt seinem Vorzeichen stimmten die Werte **nicht** überein – die Testsuite fängt das ab.
- **Synthetische Daten:** zwei Faktoren, lineare Mischung, weißes Gauß'sches Rauschen, feste Betriebsarten-Geometrie; die Rauschmerkmale sind unabhängige Spalten. Der Korrelationsbruch ist ein Extremfall (Anomalien mit exakt normalen Randverteilungen); reale Anomalien liegen meist dazwischen. Literatur nur mit Namen: Li, Zhao, Hu, Botta, Ionescu und Chen (ECOD); Fisher (Kombination von p-Werten).

## Verifikation

- ECOD: Handinstanz der Schwanzwahrscheinlichkeiten mit Bindungen, Handinstanz der drei Summen, **eine Referenzimplementierung mit expliziten Schleifen** (paper, auto, zweiseitig, mit Bindungen), Schiefe-Richtung, **Kreuzprüfung gegen PyOD** (falls installiert, < 1e-9),
  **Rang-Invarianz** unter monotonen Umrechnungen je Merkmal (die Mahalanobis-Abstände ändern sich, ECOD nicht), Determinismus und Zeilen-Äquivarianz, die zweiseitige Summe unter Unabhängigkeit ist Gamma(p, 1) (Simulation, Kolmogorov-Smirnov) und das Quantil trifft den Sollwert, die Original-Variante überschreitet ihn.
- Übernommene Bausteine: LOF, Isolation Forest, Wurzel-Schätzer (χ² gegen scipy). Szenario: normale Zeilen wie in der PCA-Demo (eingefrorene Zeilensummen), eingefrorener Standardfall, `n_noise` ändert nur angehängte Spalten, exakter Anomalie-Anteil, Geometrie der Arten;
  **Korrelationsbruch**: Spalten der Anomalien stammen aus den Normalen, Korrelation zerstört, Randverteilungen gleich, die Normalen bleiben bit-identisch.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Szenarien-, Varianten-, Kalibrierungs-, Rauschmerkmale-, Dichte-Gruppe-, Schwellen- und Kostentabellen; jeweils Mittel über die festen Sweep-Datensätze, positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung/Verhältnis);
  alle 7 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt bei 2, 12 und 30 Merkmalen, jede Variante mit und ohne Korrelationsbruch, Abspielen ohne doppelte Schlüssel, ausgeblendete Regler behalten ihre Werte, Sweep-Optionen folgen Schwellenart und Art der Anomalien, Experimente auf Abruf),
  Achsensperre und explizite eindeutige Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Szenarien, Varianten, Kalibrierung, Rauschmerkmale, dichte Gruppe, Schwelle, Kosten), 🚧 Grenzen, Mathe |
| `ecod_algorithm.py` | Schwanzwahrscheinlichkeiten, Schiefe, Beiträge, Varianten (paper, zweiseitig, auto, pyod) |
| `ecod_lof.py`, `ecod_isolation_forest.py`, `ecod_ee_algorithm.py` | LOF, Isolation Forest, χ²-Verteilung, klassische Schätzung, FastMCD (wortgleich aus den Vorgängern) |
| `ecod_scenario.py`, `ecod_constants.py` | Touren mit Betriebsarten, Krümmung, Anomalien (auch Korrelationsbruch) und Rauschmerkmalen; Konstanten, Presets |
| `ecod_evaluation.py` | Kennzahlen, Analyse, Schwellen, Sweeps, Experimente, Urteil |
| `ecod_presets.py`, `ecod_visualization.py` | Permalink/Presets (ausgeblendete Regler), Plotly-Figuren (achsengesperrt) |
| `tests/` | ECOD (Handinstanzen, Referenzschleife, PyOD, Invarianz, Fisher-Verteilung), Szenario und Kennzahlen, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
