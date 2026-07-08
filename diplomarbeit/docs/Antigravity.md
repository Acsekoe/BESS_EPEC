# Dies ist der Änderungsverlauf, verfasst von Gemini 3 Pro (High), beginnend mit 28.01.2026.

---

## 28.01.2026 - Feature Toggles & Basismodell

### Kurzzusammenfassung des Tages
Einführung von Feature-Toggles zur Deaktivierung instabiler Modellteile (aFRR, Curtailment) und Dokumentation des stabilen Basismodells.

#### Besprochene Ziele & Aufgaben

1. Schritt zurück: Deaktivierung von aFRR und Curtailment mittels Variablen.
2. Verifikation, dass das Modell nun wieder dem Stand von "epec_v9" (Grundmodell + Degradation) entspricht.
3. Dokumentation dieses Grundmodells in LaTeX.
4. Aufzeigen der Unterschiede zum ursprünglichen Startmodell.


#### Umgesetzte Änderungen

1. `bess_epec_model.py`: Hinzufügen von `CALCULATE_AFRR = False` und `CURTAILMENT = False`.
2. `bess_epec_model.py`: Bedingte Initialisierung von Variablen (R_up, R_down, Deficit, P_curtail) und Constraints (aFRR-Bilanz, Curtailment-Limitationen) basierend auf den Toggles.
3. `docs/Modell.md`: Erstellung einer detaillierten LaTeX-Beschreibung des resultierenden Basismodells inkl. Degradation.
4. `docs/Modell.md`: Hinzufügen einer Vergleichstabelle zum Startmodell.

`Wichtiger Code zur Erklärung`:
```python
# --- FEATURE TOGGLES ---
CALCULATE_AFRR = False
CURTAILMENT = False

# ...

if CALCULATE_AFRR:
    m.R_up = pyo.Var(...)
    # ... weitere aFRR Variablen & Constraints
```


#### Offene Punkte (ToDo's)

- Solver `ipopt` Installation prüfen/fixen, um volle Ausführbarkeit sicherzustellen (aktuell via Code-Review verifiziert, Run scheitert an fehlender Executable).
- Schrittweise Re-Aktivierung und Stabilisierung der aFRR Komponente.

---

## 28.01.2026 - Anpassung Last- und Windprofile

### Kurzzusammenfassung
Anpassung der Eingangsdaten für Last und Wind, um realistischere "Stress"-Szenarien (Duck Curve, volatile Einspeisung) abzubilden.

#### Umgesetzte Änderungen

1. `bess_epec_model.py`:
    - **Lastprofil (`daily_pattern`)**: Änderung zu einer "Canyon Curve" (starker Einbruch zu Mittag durch PV-Eigenverbrauch), um Speicher-Einsatz attraktiver zu machen.
    - **Windprofil (`wind_pattern`)**: Einführung einer "Morgen-Flaute" (nahe 0 Einspeisung am Vormittag), um Knappheitssignale zu erzeugen.

#### Offene Punkte
- Validierung, wie sich diese Profile auf die Investitionsanreize (Arbitrage vs. Kapazität) auswirken.

## 28.01.2026 - Integration realer Profildaten (AGPT)

### Kurzzusammenfassung
Ersetzung der manuell erstellten Profile durch reale Daten aus `AGPT...csv` (11.09.2025).

#### Umgesetzte Änderungen
1. `bess_epec_model.py`:
    - **Extraction**: Daten für Wind, Solar und Last wurden via Script (`analyze_csv.py`) aus der CSV extrahiert und normalisiert.
    - **Load Profile**: Update auf reale Lastkurve (Peak morgens & abends).
    - **Wind Profile**: Extrem volatile Einspeisung (sehr hoch nachts/morgens, fast Null tagsüber).
    - **PV Profile**: Typische Glockenkurve basierend auf realen Messwerten.

### 28.01.2026 - Reskalierung des Modells

#### Hintergrund
Die bisherigen Last- und Erzeugungswerte waren Platzhalter. Um realistische Knappheitssignale (Notwendigkeit konventioneller Kraftwerke) zu erzeugen, wurden die Daten basierend auf der CSV-Analyse skaliert.

#### Kennzahlen
- **System-Spitzenlast**: 1200 MW (skaliert, um zu den 400 MW Leitungslimits zu passen).
- **Wind-Kapazität**: 370 MW (~31% der Spitzenlast, Knoten N1).
- **PV-Kapazität**: 430 MW (~36% der Spitzenlast, Knoten N3).
- **Lastverteilung**:
    - Zentrum (N2, N3, N4): je 30%.
    - Peripherie (N1, N5): je 5%.

Dies stellt sicher, dass in Dunkelflauten fast die gesamte thermische Kapazität (1200 MW aus G_Base + G_Peak) benötigt wird.
#### Offene Punkte
- Erneute Verifikation der Modellergebnisse unter den neuen, volatileren Bedingungen.

### 31.01.2026 - Solver Switch zu MA97

#### Anpassungen
- **Solver**: Wechsel von `mumps` (Standard) zu `ma97` für bessere Performance/Stabilität.
- **Pfad-Logik**: Die `get_solver()` Funktion wurde angepasst, um primär im Conda-Environment (`miniforge3\envs\pyomo_env`) nach `ipopt.exe` zu suchen. Dies war notwendig, da die DLLs (`libhsl.dll`) dort korrekt hinterlegt sind, während die lokale `.venv` Version fehlerhaft war (Crash 3221225477).
- **Import Fix**: Fehlender Import von `TerminationCondition` wurde behoben.

#### Status
Modell läuft nun erfolgreich mit dem MA97 Linear Solver.

### 31.01.2026 - Parallelisierung (Jacobi-Verfahren)

#### Anpassungen
- **Parallel-Execution**: Implementierung von `ProcessPoolExecutor`, um die 4 Investoren gleichzeitig auf 4 Kernen zu berechnen (`GAUSS_JACOBI = True`).
- **Refactoring**: Extraktion der `solve_single_investor_process` Funktion für Multiprocessing-Support.
- **Konvergenz**: Das Verfahren wurde von Gauß-Seidel (sequenziell) auf Jacobi (parallel) umgestellt. Die Iterationen laufen nun parallel, was die CPU-Auslastung maximiert.

#### Status
Modell läuft parallel stabil und nutzt die verfügbare Hardware (12 Kerne / 16 Threads) effizienter.
    - Der Output `Def: 0.00` in der Konsole steht für **aFRR Deficit**.
    - Es beziffert die Menge an benötigter, aber nicht lieferbarer Regelleistung (in MW).
    - Da aFRR aktuell deaktiviert ist (`CALCULATE_AFRR = False`), ist der Wert 0.

### 01.02.2026 - Erweiterte Analyse & Visualisierung

#### Anpassungen
- **CSV Export**: Das Modell exportiert nun nach Konvergenz eine `results_final.csv` mit allen Zeitreihen (Preise, Leistungen, SOCs, Residuallast) für jeden Knoten.
- **Knoten-Plots (Refactoring)**:
    - Neue 2-Fenster Darstellung pro Knoten.
    - **Oben (Bilanz)**: Preis (Linie) vs. Zuflüsse/Abflüsse (Stacked Areas).
        - *Fix*: Darstellung nutzt nun **Netto-Flüsse** für BESS, um visuelle Artefakte ("gleichzeitiges Laden/Entladen") zu vermeiden.
    - **Unten (Speicher)**: Detaillierte Fahrpläne der einzelnen Investoren + Summenlinie.
- **Daten-Logik**: Berechnung von Import/Export und NetInjection direkt im Post-Processing (`extract_results`).

#### Status
Visualisierung ist nun vollständig konsistent und detailliert. Datenexport ermöglicht externe Analyse.

### 01.02.2026 - Re-Aktivierung aFRR Markt

#### Anpassungen
- **aFRR Aktiviert**: `CALCULATE_AFRR = True`.
- **Demand Erhöhung**: Nachfrage von 1 MW auf 50 MW pro Block erhöht, um signifikanten Markteinfluss zu erzeugen.
- **Visualisierung**: Neuer Plot `plot_afrr_provision` zeigt die reservierte Leistung (Up/Down) je Knoten und Investor.
- **Daten**: CSV-Export enthält nun `R_Up_MW` und `R_Down_MW`.

#### Status
Der Markt ist aktiv. Erste Tests zeigen:
- Investoren reagieren und halten Leistung vor.
- Das System ist deutlich "steifer" (öfter Infeasibilities in den ersten Iterationen), stabilisiert sich aber meist über den Deficit-Mechanismus.

### Ökonomische Validierung (Beobachtung)
**Beobachtung**: Investor 1 (I1) tätigt die geringsten Gesamtinvestitionen, sichert sich aber fast den kompletten aFRR-Markt.
**Erklärung**:
- I1 hat den geringsten Diskontsatz (`r=8%` vs. 12-20% bei anderen).
- **Wettbewerbsvorteil**: Durch niedrigere Kapitalkosten kann I1 Kapazität günstiger vorhalten (`Hold Capacity` Kosten < Konkurrenz).
- **Markt-Segmentierung**: 
    - Der "geduldige" Investor (I1) gewinnt den stabilen, aber begrenzten Kapazitätsmarkt (aFRR).
    - Die "risikofreudigeren" Investoren (I3, I4) weichen auf den Spotmarkt/Arbitrage aus, um ihre höheren Kapitalkosten durch dort höhere Spreads zu decken.
- Dies bestätigt, dass das Modell fundamentale ökonomische Anreize korrekt abbildet.


## 22.02.2026 - Modelldokumentation überarbeitet

### Kurzzusammenfassung des Tages
Aktualisierung der Modelldokumentation, um die mathematische Formulierung (Wirkungsgrade, NPV, Zinssatz) präziser darzustellen und die drei Ausbaustufen des Pyomo-Modells zu unterscheiden.

#### Besprochene Ziele & Aufgaben
1. Überarbeitung des Grundmodells, um Degradation und aFRR auszuschließen.
2. Darstellung des Modells mit aFRR und Degradation (Co-Optimierung von Spot und Reserve).
3. Gegenüberstellung der notwendigen Schritte, um vom ersten in das zweite Modell zu gelangen.
4. Korrekte Anwendung der Optimierungs-Notation ($\max_{X_{i,n}} \text{NPV}_{i}$) in den Zielfunktionen.
5. Integration vernachlässigter Parameter wie Wirkungsgrad ($\eta$) und Zinssatz ($r_i$).
6. **Klärung bezüglich Wirkungsgrad in der Zielfunktion:** Wurde geprüft. Der Wirkungsgrad fehlt nicht, da die Leistung am Netzanschlusspunkt als finanzielle Grundlage dient (siehe Umgesetzte Änderungen).
7. Konkretisierung und detailliertere mathematische Ausformulierung der PTDF Berechnungen (DC-OPF) im Market-Clearing (Lower Level).

#### Umgesetzte Änderungen
1. Die Datei `docs/Modell.md` wurde strukturell angepasst, um strikt die 3 gewünschten Abschnitte abzubilden.
2. Die Zielfunktion des Grundmodells wurde auf die reine $\max \text{NPV}_{i}$ Notation umgestellt. Der Diskontierungsfaktor $1 / (1 + r_i)$ wurde in die Mathematik aufgenommen, um die finanzielle Bewertung korrekt abzubilden.
3. In den Speicherrestriktionen des "Lower Level Problems" wurde explizit der Wirkungsgrad $\eta$ für das Laden/Entladen innerhalb der Speicherbilanz-Gleichung ($SOC_{t}$) aufgeführt.
4. Der zweite Abschnitt beschreibt nun exakt das Zusammenwirken von Degradations-Kosten und aFRR-Erlösen im Co-optimierten Setup.
5. In Abschnitt 3 wurde eine tabellarische Gegenüberstellung eingefügt, die klar aufzeigt, welche Variablen, Restriktionen und Zielfunktions-Terme modifiziert werden mussten, um vom Grundmodell zur komplexeren Ausbaustufe zu gelangen.
6. Erklärt (im Chat mit dem User), wieso in der Zielfunktion für die Erlösrechnung kein Wirkungsgrad ($\eta$) stehen darf: Die Variablen `P_charge` und `P_discharge` beschreiben die Leistung am Netzanschlusspunkt, welche gehandelt wird. Verluste treten lediglich intern beim Speichern in der Batterie ($SOC$) auf.
7. Der DC-OPF mit PTDF Ansatz in `docs/Modell.md` wurde detaillierter ausgeführt. Die kurze Limit-Gleichung wurde durch den 3-Schritt-Prozess ersetzt: 1. Berechnung der `NetInjection` pro Knoten. 2. Berechnung der Leitungslasten `Flow` über die PTDF-Matrix und 3. Evaluierung des `Limit_l`.

#### Offene Punkte (ToDo's)
- Prüfung, ob die neu notierte mathematische Darstellung exakt den Vorstellungen entspricht.
- Gegebenenfalls Abgleich der Notation mit der geplanten Formatierung für eine LaTeX-Ausgabe der Diplomarbeit.
