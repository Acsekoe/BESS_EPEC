Protokoll: Entwicklung eines EPEC-Modells für strategische Batteriespeicher-Investitionen

Datum: 27. Januar 2026
Thema: Implementierung eines Equilibrium Problem with Equilibrium Constraints (EPEC) in Python/Pyomo zur Modellierung von Speicherinvestitionen im Stromnetz.

1. Projektziel und Anforderungen

Das Ziel der Diplomarbeit ist die Entwicklung eines Modells, das die strategischen Investitionsentscheidungen (Leistung in MW, Energie in MWh) konkurrierender Investoren in Batteriespeicher (BESS) abbildet.

Kernanforderungen:

Modellstruktur: Bilevel-Spiel (Leader-Follower).

Leader (Upper Level): Strategische Investoren maximieren ihren Profit (NPV).

Follower (Lower Level): Marktbetreiber (ISO) minimiert Systemkosten (DC-OPF).

Methodik: EPEC (Equilibrium Problem with Equilibrium Constraints), gelöst via Diagonalisierung (Gauss-Seidel).

Physik: Netzrestriktionen (Engpässe) und PTDF-Matrix.

Technologie: Python, Pyomo, Solver Ipopt (für nicht-lineare KKT-Systeme).

2. Entwicklungsphasen und Meilensteine

Phase 1: Grundgerüst und erste Implementierung

Initialisierung: Aufbau des Pyomo-Modells für einen Investor (MPEC).

Mathematik: Formulierung des Lower-Level-Problems (LLP) als Karush-Kuhn-Tucker (KKT) Bedingungen, um es in das Upper-Level-Problem zu integrieren (MPCC).

Herausforderung: Der Solver Ipopt scheiterte initial ("Infeasible") aufgrund schlechter Skalierung und harter KKT-Bedingungen.

Lösung:

Skalierung der Zielfunktion (Faktor $10^{-6}$).

Relaxierung der Komplementarität ($x \cdot y \le \epsilon$ statt $=0$).

Robustere Behandlung von Solver-Fehlern im Loop.

Phase 2: Netztopologie und Physik (IEEE 5-Bus)

Erweiterung: Wechsel von einem simplen Strang-Netz auf das vermaschte IEEE 5-Bus Test System.

Neuerung: Automatische Berechnung der PTDF-Matrix (Power Transfer Distribution Factors) basierend auf den Leitungsreaktanzen ($X$), um Loop-Flows physikalisch korrekt abzubilden.

Visualisierung: Implementierung von Plots für Netztopologie (networkx) und Lastprofile (matplotlib).

Phase 3: Zeitliche Auflösung und Profile

Erweiterung: Erhöhung des Zeithorizonts auf 24 Stunden.

Profile: Implementierung realistischer Tagesgänge für Last (Duck Curve), PV (Glockenkurve) und Wind (stochastisch).

Speicher: Zyklische Randbedingungen für den Speicherfüllstand ($SOC_{t=24} = SOC_{t=0}$), damit der Tag repräsentativ für ein Jahr ist.

Phase 4: Robustheit und Konvergenz (Die "Soft Constraints")

Problem: In frühen Iterationen stürzte das Modell oft ab, weil alle Investoren gleichzeitig denselben Knoten "überbuchten" (Verletzung des Shared Limits).

Lösung: Einführung von Soft Constraints.

Verletzung des Limits ist erlaubt, kostet aber eine enorme Strafe (Penalty).

Dies verhindert den Absturz ("Infeasible") und führt das Modell sanft in den zulässigen Bereich zurück.

Stabilisierung:

Warm-Start: Initialisierung der Variablen mit den Werten der Vorrunde.

Damping: Nur teilweise Übernahme neuer Ergebnisse ($x_{neu} = (1-\alpha)x_{alt} + \alpha x_{calc}$), um Oszillationen ("Schweinezyklus") zu verhindern.

Phase 5: Markterweiterung (aFRR & Curtailment)

Co-Optimierung: Erweiterung des Modells um den Regelenergiemarkt (aFRR). Der Speicher optimiert nun simultan zwischen Spotmarkt-Arbitrage und Vorhaltung von Regelreserve (Opportunitätskosten).

Österreichisches Design: Abbildung von 4-Stunden-Blöcken für die Leistungsvorhaltung.

Curtailment (Abregelung): Einführung der Möglichkeit, Erneuerbare (Wind/PV) bei negativen Preisen abzuregeln (Economic Curtailment), um extrem negative Preise (-500 €) zu verhindern.

Herausforderung: Explodierende Preise ($10^{16}$) im aFRR-Markt bei Knappheit.

Lösung: Einführung eines "Deficit"-Mechanismus mit Penalty-Preis (Price Cap) im aFRR-Markt.

3. Zusammenfassung der methodischen Entscheidungen (für den Betreuer)

Umgesetzte Elemente

Vollständiges EPEC: Erfolgreiche Implementierung des iterativen Lösungsansatzes für strategische Interaktionen.

Physikalisches Netz: Korrekte DC-OPF Abbildung mittels PTDFs in einem vermaschten Netz.

Technische Details: SOC-Bilanzierung, Wirkungsgrade, Degradationskosten (linearisiert aus Literatur).

Markt-Komplexität: Co-Optimierung von Energie- und Reservemarkt (Leistungspreis) unter Berücksichtigung von Opportunitätskosten.

Notwendige Kompromisse (Modellgrenzen)

Kontinuität vs. Ganzzahligkeit:

Marktregeln wie Mindestgebotsgrößen (z.B. +/- 1 MW Schritte) wurden relaxiert (kontinuierlich modelliert).

Grund: Der Solver Ipopt benötigt stetige, differenzierbare Funktionen. Binäre Variablen (MIPEC) wären numerisch instabil und kaum lösbar gewesen.

Argumentation: Annahme eines Pools/Aggregators, der die Kleinteiligkeit ausgleicht.

Deterministik (Perfect Foresight):

Das Modell kennt die Zukunft (Preise, Wind) perfekt. Ausgleichsenergie (durch Prognosefehler) wird daher nicht abgebildet.

Grund: Stochastische Optimierung würde die Rechenzeit explodieren lassen.

Degradation:

Verwendung eines "Marginal Cost"-Ansatzes (Kosten pro MWh Durchsatz) statt komplexer Rainflow-Counting-Algorithmen.

Grund: Rainflow-Counting ist nicht differenzierbar und daher nicht direkt in das Optimierungsproblem integrierbar.

Erkenntnisse zur Lösbarkeit

EPEC-Modelle neigen zu Oszillationen und Infeasibility.

Die Kombination aus Soft-Constraints, Damping und Warm-Starts war zwingend erforderlich, um stabile Nash-Gleichgewichte zu finden.

Die Ergebnisse sind sensitiv bezüglich der Startwerte und Dämpfungsfaktoren, was auf die Existenz multipler Gleichgewichte im Markt hinweist (ökonomisch plausibel).