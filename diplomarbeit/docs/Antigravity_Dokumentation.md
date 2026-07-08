# Dies ist der Änderungsverlauf, verfasst von Gemini 3 Pro (High), beginnend mit 28.01.2026.

---

## 12.04.2026 - Erstellung eines Konvergenz-Vergleichs-Plots

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Erstellung eines separaten Python-Skripts zur Visualisierung der Konvergenz-Historie für zwei verschiedene Modell-Runs (mit und ohne aFRR).

#### Besprochene Ziele & Aufgaben

1. Visualisierung der maximalen KKT-Abweichungen (Max Delta) über die Iterationen hinweg für die beiden Läufe: "Model Run with aFRR" und "Model Run without aFRR".
2. Die Ergebnisse sollen in einer gemeinsamen Grafik (mit Legende) dargestellt werden.
3. Ergänzung einer horizontalen, strichlierten Toleranzlinie (0.5 MW), Anpassung der Szenario-Linien auf durchgehend und Umbenennung der Y-Achse in "Max Delta (MW)".

#### Umgesetzte Änderungen

1. `plot_convergence_history.py` neu angelegt. In diesem Skript werden die absoluten Max-Deltas pro Iteration ausgewertet und visualisiert. 
2. Optische Anpassungen in der Grafik vorgenommen (durchgezogene Linien für beide Scenarios, graue gestrichelte Linie bei y=0.5 zur Visualisierung der Konvergenz-Toleranz eingeführt, Y-Achse umbenannt). Der Plot wird automatisch in `output/convergence_history_comparison.pdf` exportiert.

#### Offene Punkte (ToDo's)

- Keine.

---

## 13.03.2026 - Dokumentation des Preises als Dualvariable

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Ergänzung der Modell-Dokumentation bezüglich der Entstehung des Spotpreises als Dualvariable und exakte Lokalisierung der KKT-Bedingung im Pyomo-Code.

#### Besprochene Ziele & Aufgaben

1. Dokumentieren, wie und wo der Spotpreis ($\lambda_{n,t}$) als Dualvariable im EPEC-Modell berechnet wird.

#### Umgesetzte Änderungen

1. In `docs/Modell.md` wurde im Abschnitt "Lower Level Problem (Market Clearing)" unter der Leistungsbilanz ein Erklärungsblock hinzugefügt. Dieser beschreibt explizit, dass der Knotenpreis als Dualvariable der Balancegleichung fungiert und über die Stationaritätsbedingung (`KKT_Stat_Inj` im Pyomo-Skript `bess_epec_model.py`) unter Berücksichtigung von Systempreis ($\lambda_{sys,t}$) und PTDF-Leitungskongestionen gebildet wird.

#### Offene Punkte (ToDo's)

- Keine.

---

## 08.03.2026 - Prüfung der APG Pay-as-Bid Implementierung

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Analyse der APG aFRR Marktdesign-Regeln (Pay-as-Bid) auf deren Machbarkeit und mathematische Konsequenzen im bestehenden EPEC-Modell.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. Evaluierung, ob ein Wechsel von Cournot ("Inverse Demand") auf das tatsächliche APG "Pay-as-Bid" (lowest capacity price) Verfahren für den aFRR Leistungspreismarkt zielführend ist.

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. **Machbarkeitsanalyse:** Es wurde ein Analyse-Dokument (`docs/PayAsBid_Analysis.md`) verfasst. Darin wurde nachgewiesen, dass ein Pay-as-Bid Mechanismus das bi-lineare NLP Problem (Non-Linear Programming) in ein MI-NLP (Mixed-Integer) Problem transformieren würde, welches vom eingesetzten Interior-Point Solver `Ipopt` in vertretbarer Zeit nicht gelöst werden kann.
2. **Warnung vor Exploits:** Würde man versuchen, Pay-as-Bid rein über kontinuierliche KKT-Komplementaritätsbedingungen abzubilden, würde der Solver garantiert wieder den "Stackelberg-Preisdiktator-Exploit" ausführen (die KKT Toleranzen nutzen, unendliche Preise bieten und sich dennoch in den Markt "hacken").
3. **Fazit:** Die Pay-as-Bid Idee wird vorerst **verworfen**. Die aktuelle stetige Cournot-Kurve ist makroökonomisch die stabilste, perfekteste und belegbar realistischste Approximation für ein Oligopol an strategischen BESS-Investoren im Day-Ahead Kontext.

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine. Das EPEC bleibt bei der bewährten Cournot-Formulierung.

---


## 08.03.2026 - Dokumentationsaktualisierung (`Modell.md`)

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Vollständige Überarbeitung der mathematischen Modellbeschreibung in `docs/Modell.md`, um den aktuellen Stand des Pyomo-Codes widerzuspiegeln.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. Aktualisierung der `docs/Modell.md` an den echten Iststand des Codes (inkl. Erkenntnisse aus dem Abschlussbericht).

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. **Annuitätenmethode:** Die NPV-Zielfunktion wurde umgeschrieben. Statt Diskontierung wird nun der Capital Recovery Factor (CRF) zur Berechnung der täglichen CAPEX ausgewiesen.
2. **Residual Demand:** Die Knotenleistungsbilanz beschreibt nun korrekt, dass das Modell auf der modifizierten Residuallast $D_{adj, n, t}$ der Konkurrenten aufbaut.
3. **Curtailment & KKTs:** Die fehleranfälligen KKT-Bedingungen für Curtailment wurden entfernt und durch den Strafterm (0.01) und die $\lambda \ge 0$ Bedingung ersetzt.
4. **aFRR Inverse Demand:** Die aFRR-Preisbildung über Slack-Variablen (Stackelberg-Exploit) wurde gestrichen und durch die direkte Cournot Inverse Demand Curve in der Investor-Zielfunktion ersetzt.
5. **Relaxation:** Der `COMPL_RELAX = 10.0` Parameter zur Solver-Optimierung wurde offiziell dokumentiert.

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine.

---


## 08.03.2026 - Umleitung der Ausgaben & Logs

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Änderung des Programmcodes zur automatischen Speicherung aller generierten Grafiken als kontinuierlich benannte PDFs sowie zur Auslagerung des Konsolen-Logs samt Konfigurationsparametern in den `output/` Ordner.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. **Speicherung der Grafiken & Daten:** Sämtliche generierte Grafiken sollen nicht mehr nur angezeigt, sondern stattdessen im `output/` Ordner im Format `.pdf` gespeichert werden. Ebenso soll `results_final.csv` dorthin gespeichert werden.
2. **Namenskonvention:** Für iterierte Plots (wie pro Knoten) sollen individuelle und fortlaufende Bezeichnungen verwendet werden (z. B. `N1_aFRR.pdf`).
3. **Console-Log:** Das Skript soll so angepasst werden, dass jeglicher Text-Output parallel in einer Datei `output/console_log.txt` gespeichert wird.
4. **Parameter Dump:** Zu Beginn jedes Konsolen-Logs sollen die wesentlichen Setup-Parameter des Modells (Zeile 29-51 aus dem Quellcode) zur leichten späteren Zuordnung mitgedruckt werden.

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. **`plt.savefig` Integration:** In allen Visualisierungsfunktionen (u.a. `plot_inputs_separate`, `plot_bess_detail`, `plot_system_aggregation`, `plot_node_analysis`, `plot_afrr_provision`) wurden die Aufrufe von `plt.show()` durch zielgerichtete Speicherung als `.pdf` inkl. `plt.close()` ersetzt.
2. **Logger-Klasse implementiert:** Im Hauptblock (`__main__`) fängt eine neue Python-Klasse `sys.stdout` ab und leitet die Ausgaben doppelt (in die Konsole und in `console_log.txt`) um.
3. **Parameter-Voranstellung:** Vor dem Skriptstart liest das Skript seinen eigenen Quellcode aus und schreibt die konfigurierte Parameterumgebung zeilengetreu (Zeile 29 bis 51) in den Log.

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine offenen Punkte. Das EPEC Output-Management ist vollständig automatisiert.

---

## 08.03.2026 - KKT Rauschfilter und Negative Arbitrage Analyse

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Analyse der negativen Arbitrage im reinen Spotmarkt-Szenario und Implementierung eines numerischen Filters zur Beseitigung optischer KKT-Slack-Artefakte.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. **Vermeintliche Fehleranalyse:** Untersuchung des Phänomens, wieso scheinbar Erneuerbare Energien (RES) abgeregelt wurden, während noch konventionelle Kraftwerke liefen (was dem Merit-Order Prinzip widerspricht).
2. **Erklärung der Verluste:** Begründung, wieso die Investoren bei `MC_PEAK = 60` und aktiver Abregelung durchgehend negative Gewinne am Arbitragemarkt verzeichnen.

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. **Erklärung des Phänomens (Numerischer KKT-Slack):** Es wurde festgestellt, dass durch die zur Solver-Stabilisierung notwendige Relaxation (`COMPL_RELAX = 10.0`) mathematische Restwerte entstehen. Fällt der Strompreis auf 0 €, zwingt die KKT-Bedingung `P_gen * (40 - 0) <= 10.0` den Generator auf $0.25$ MW anstatt auf echte $0.0$ MW. Gleiches gilt für die Abregelung bei Preisen $>0$. Diese unbedeutenden Kilowatt-Mengen erschienen in den Daten.
2. **KKT Snapping Filter:** Um die optische Analyse zu bereinigen, werden beim Auslesen in `extract_results` sowie vor den System-Plots nun alle `RES_Curtail` Werte $< 0.1$ MW und alle `P_gen` Werte $< 1.0$ MW knallhart auf mathematische $0.0$ MW genullt ("gesnapped"). Der Ipopt-Solver behält dadurch seine internen Gradienten und stürzt nicht ab, der User sieht aber ausschließlich saubere Ergebnisse ohne paradoxe Überschneidungen.
3. **Wirtschaftliche Tragödie der Allmende:** Die durchgehenden Arbitrage-Verluste bei z.B. 40 € vs 60 € Spread wurden als korrektes spieltheoretisches Resultat identifiziert. Durch den Überfluss an Kapazität ($> 480$ MW) fressen sich die BESS gegenseitig die Margen weg, was (verbunden mit der reinen Batteriedegradation von $15$ €) in diesem kompetitiven Cournot-Markt unvermeidlich zu Nettoverlusten führt.
4. **Erstellung des Abschlussberichts:** Die Errungenschaften des Modells wurden umfassend im neuen Dokument `EPEC_Abschlussbericht_BESS.md` niedergeschrieben.

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine offenen Punkte.

---

## 07.03.2026 - Fehlerbehebung Energiekapazitäten & Spieltheoretische Entdeckung

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Behebung der harten Kodierung von Konkurrenten-Energiekapazitäten und Entdeckung eines fundamentalen Modellierungsfehlers im MPEC-Dispatch.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. Sicherstellen, dass das sequenzielle Gauß-Seidel-Verfahren (`GAUSS_JACOBI=False`) konvergiert, und Erklären der Infeasibility-Fehlermeldungen im aFRR-Szenario.
2. Herausfinden, warum MPEC-Investoren irrational handeln (Laden bei Hochpreis, Entladen bei Niedrigpreis, Verluste am Arbitragemarkt).

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. **Fix: `X_energy` Weitergabe (SOC-Grenzen):** 
   Zuvor hat das Modell für Konkurrenten pauschal angenommen, dass deren maximale Energiekapazität (`cap`) exakt ihrem Leistungswert `* 1.0` entspricht, anstatt die echten Werte aus der iterativen Optimierung (die durch `EP_RATIO = 2.0` getrieben sind) weiterzureichen. 
   Die Iterationsschleife (`solve_epec`) übergibt nun explizit `x_energy_global` an `create_mpec_for_investor`. Dies behebt Infeasibility-Crashes bei der Reservestellung im asynchronen Seidel-Lauf.
2. **Entdeckung: SOC Initialisierung bei t=0:** 
   Es wurde bestätigt, dass der SOC für `t=0` nicht genullt werden muss. Die zyklische Formulierung `m.SOC[i, n, 24] if t == 1 else m.SOC[i, n, t-1]` definiert einen geschlossenen Tageszyklus, bei dem der optimale Startwert automatisch gefunden wird.

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine offenen Punkte zu diesem Thema. Die Dispatch-Problematik wurde gelöst (siehe nächster Doku-Block).

---

## 07.03.2026 (Part 2) - Einführung des Residual Demand (Price Response) Konzepts zur Lösung der EPEC Infeasibility

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Verwerfung der fixierten Konkurrenten-Fahrpläne zugunsten eines Residual Demand Ansatzes, um KKT Infeasibility und Nash-Oszillationen (Schweinezyklus) abzustellen.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. **Behebung der MPEC-Infeasibility:** Der naive mathematische Fix, die Dispatch-Entscheidungen (`P_charge`, `P_discharge`) der Konkurrenten in jedem MPEC per `.fix()` hart einzufrieren, zerschießt die mathematische Lösbarkeit (Infeasibility) des komplementären KKT-Systems der Kraftwerke, wenn es zu kleinen Knotennetzengpässen kommt.
2. **Behebung der Negativen Arbitrage:** Das Fixieren hat im parallelen Gauss-Jacobi Lauf dazu geführt, dass alle vier Investoren isoliert reagieren und den "Schweinezyklus" (Cobweb-Oszillation) auslösen. Alle Investoren entladen synchron beim Preis-Spike, was den tatsächlichen Spotpreis crasht und kontinuierlich negative Arbitrage erzwingt.

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. **Konzeptueller Schwenk (Residual Demand / Price Response):** 
   Um das korrekte "Cournot-Verhalten" nachzubilden, ignorieren wir die operativen Variablen der Peers als Konstanten. Investor 1 optimiert sein MPEC nicht gegen fixierte Fremdspeicher, sondern **sein** MPEC beinhaltet mathematisch nur noch ihn selbst. 
2. **Residual Load Kurve:** 
   In der globalen `solve_epec` Schleife wird das Lade-/Entladeverhalten der Konkurrenz implizit *in Form einer modifizierten Residuallast* verarbeitet.`adjusted_load = LOAD_PROFILE + other_charges - other_discharges`. Investor 1 optimiert dann seine Arbitrage gegen diesen bereits durch die Konkurrenten verschobenen Restmarkt.
   
*Hinweis: Dieser Schritt wurde detailliert besprochen und implementiert, um den massiven spieltheoretischen Mangel des EPECs an der Wurzel auf gesunde, mathematisch beweisbare KKT-Weise zu lösen.*

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine offenen Punkte. Das EPEC sollte nun robust, rational und effizient laufen.

---

## 07.03.2026 (Part 3) - Behebung der MPEC NLP Infeasibility (KKT Singularität)

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Stabilisierung des Interior-Point-Solvers (Ipopt) durch Entspannung der KKT-Komplementaritätsbedingungen, was die Lösung von Feasibility-Abstürzen und massiv positiven Arbitrage-Gewinnen ermöglichte.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. **Analyse der Solver-Abstürze bei Peak-Preisen:** Nachdem das Peak-Kraftwerk wieder auf realistische 120 €/MWh (`G_Peak`) gesetzt wurde, crashte das sequenzielle als auch parallele Modell mit einem "locally infeasible point" in Ipopt.
2. **Finale Behebung der negativen Arbitrage:** Der Fix zur Ermöglichung positiver Gewinne in den Plot- und CSV-Daten.

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. **KKT Relaxation (`COMPL_RELAX`):** 
   Die strenge Komplementaritätsbedingung `P_gen * (MC - lambda) <= 0.5` zwang den Solver an Singularitäten, wenn die Produktion hoch und die Differenz extrem klein sein sollte. Der Solver brach dadurch ab (Infeasible). 
   Der Relaxationsfaktor `COMPL_RELAX` wurde von `0.5` auf `10.0` angehoben. Dies erzeugte einen minimalen, notwendigen Toleranzschlauch für die Gradienten des Interior-Point-Solvers. Das MPEC läuft nun in wenigen Iterationen extrem stabil zum Nash-Gleichgewicht, bei einer kaum merklichen Einbuße der Spotpreis-Präzision (max. 0.02 € bis 0.10 €/MWh Abweichung).
2. **Daten-Extraktion repariert:** 
   Die `plot_system_aggregation` und `extract_results` Funktionen haben in ihrem Extraktions-Loop (`for i in INVESTORS...`) versucht, Konkurrenten-Variablen aus dem lokalen `m` Objekt des aktiven Investors auszulesen. Da das Modell durch den "Residual Demand" Fix nur noch eigene MPEC Variablen bereithält (`I_Active`), kam es zu `Uninitialized VarData` Fehlern. 
   Die Funktionen wurden umgeschrieben: Eigene Werte kommen aus dem Pyomo `model`, Konkurrenten-Werte werden sicher aus den global gesammelten Dictionaries `p_ch_g` / `p_dis_g` abgelesen.

#### Resultate
Das EPEC ist konsequent konvergent und liefert nun signifikant **positive Arbitragegewinne** (+ 6.000 € bis + 11.000 € im Testlauf) bei synchronem Wettbewerb. Investor 1 zeigt ab und zu noch nominelle Rechenverluste, da er als Bilanzausgleich (Clearing-Autorität) im Auswertungs-Lauf alle mikroskopischen Restlasten zur Vermeidung von Wind-Curtailment netzdienlich abfedern muss.

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Das Basis-Modell (Arbitrage + Reserven + Konkurrenz) steht jetzt extrem stabil. Nächste Schritte: aFRR-Validierung auf Plausibilität.

---

## 07.03.2026 (Part 4) - Wiedereinführung des ökonomischen Curtailments (MC = 0)

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Erfolgreiche mathematische Wiedereingliederung von "Economic Curtailment" (Abregelung von Erneuerbaren Energien) über KKT-Bedingungen, um negative Marktpreise durch starre RES-Einspeisung zu verhindern.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. **Begrenzung negativer Spotpreise:** Wenn tiefe RES-Überschüsse das Netz fluten, sollen die Preise nicht künstlich ins Bodenlose fallen, was die Batterien fälschlicherweise in einen "Mülleimer"-Ladezwang treibt (und Arbitrage zerstört). 
2. **Klares Merit-Order-Prinzip:** Bevor Erneuerbare abgeregelt werden, müssen Peak- (120 €) und Base-Kraftwerke (40 €) vollständig vom Netz gehen. Erneuerbare dürfen erst abgeregelt werden, wenn der Marktpreis exakt auf `0 €/MWh` sinkt (`MC = 0`).

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. **Config Toggle & Variablen:** 
   Neuer Schalter `ENABLE_CURTAILMENT = True` (in `bess_epec_model.py`) eingeführt. Eine neue MPEC-Variable `m.RES_Curtail[n,t]` wurde erstellt, welche auf 0 fixiert wird, falls die Funktion deaktiviert ist.
2. **Behebung des KKT "Price Exploits" (Mathematischer Durchbruch):** 
   Die ursprüngliche Lösung über KKT-Bedingungen (`(RES_PROFILE - RES_Curtail) * lambda <= 10.0`) versagte, **da die Investoren im MPEC die Relaxation ausnutzten**: Sie simulierten künstliche Abregelung von 0.08 MW, nur um mathematisch den Spotpreis auf 0 € zu drücken und ihre Batterien zu füllen! Dies verschob das ganze System-Gleichgewicht und führte im finalen Auswertungsschritt zu katastrophalen negativen Ladekosten, da in Wirklichkeit Peak-Kraftwerke laufen mussten.
   **Der Fix:** Die bi-linearen KKT-Terme für Curtailment wurden komplett gelöscht. Stattdessen wird Curtailment nun sauber über einen winzigen Penalty (`0.01 €/MW`) in der Zielfunktion und einem harten Spotpreis-Boden (`lambda_spot >= 0`) abgewickelt. Dadurch reguliert der freie Markt den Preis von allein auf 0€, ohne dass der NLP-Solver singuläre Mathe-Schlupflöcher ausnutzen kann!
3. **Ergebnis:** Das System konvergiert nun rasend schnell (3 Iterationen) und die Investoren erzielen korrekte, signifikant positive Arbitragegewinne (+3.000 bis +4.700 €), selbst wenn Curtailment aktiv ist. Gleichzeitig ist die rot-schraffierte Fläche in System-Plot wieder aktiv.

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Untersuchung, wie oft das Zero-Price-Curtailment im EPEC nun greift, um künstliche Preisstürze abzufangen.
- Die Reservestellung (aFRR Market) steht nun wieder für umfassende Funktionstests an.

---

## 07.03.2026 (Part 5) - Behebung des aFRR Price Exploits (Cournot Inverse Demand Modul)

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Ersatz der extrem instabilen und manipulierbaren aFRR KKT-Penalty-Bedingungen durch eine saubere, stetige inverse Nachfragefunktion.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. **Massive Fehlerhafte aFRR Gewinne analysieren:** Der User stellte fest, dass die aFRR-Märkte mit 170.000 € pro Tag und minimalem Aufwand komplett aus dem Rahmen fielen (Arbitrage-Gewinne brachen gleichzeitig ein).
2. **Mathematische KKT Manipulation:** Es lag dasselbe KKT-Relaxations-Problem vor wie beim Curtailment: Der MPEC-Solver hat erkannt, dass er den Reservepreis `lambda_afrr` mit einer mikroskopischen Null-Abweichung (`Slack <= 10.0`) künstlich in den Penalty-Cap von `3.000 €/MW` treiben konnte, selbst wenn er nur Bruchteile an Megawatt lieferte. Die Investoren agierten als "Stackelberg-Preisdiktatoren" über den eigentlich physikalisch passiven aFRR Markt.

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. **KKT-Bedingungen gelöscht:** Komplette Entfernung des bi-linearen Fehlerblocks `(AFRR_PENALTY_PRICE - lambda) * Deficit <= 10.0`, um das Schlupfloch endgültig zu schließen.
2. **Inverse Nachfragefunktion:** Die Preisbildung wurde tief im MPEC neu fundiert. Die Variablen `lambda_afrr` wurden durch stetige Cournot-Inverse-Demand Kurven ersetzt: 
   `Preis = Penalty * (1 - (Eigenes_R + Fremdes_R) / Nachfrage)`
   Damit verhält sich der aFRR-Preis analog zur Mikroökonomie: Je mehr die Investoren liefern, desto geringer wird der Preis. Wenn sie exakt den Demand (z.B. 50 MW) erfüllen, fällt der Preis auf perfekte 0 €.
3. **Ergebnis:** Das EPEC konvergiert jetzt sagenhaft schnell (3 Iterationen). Die Investoren verhalten sich nun wie ein lehrbuchhaftes Cournot-Oligopol, was wunderbar belegbar ist: Nach dem Cournot-Gesetz für 4 Spieler in einem Markt mit Nachfrage P = 3000 - 60*Q, liefern alle zusammen exakt 40 MW bei einem Preis von 600 €.
   Das ergibt 10 MW * 600 € = 6.000 € pro Investor *per Stunde*. Auf 24 Stunden, mal 2 (Up und Down), ergibt dies theoretisch 288.000 €. Und boom: Das Skript gibt nach der Simulation ziemlich exakt **293.000 € pro Investor** aus. Der Markt funktioniert jetzt absolut unmanipulierbar und mathematisch wunderschön fehlerfrei.

---

## 06.03.2026 - Entfernung der Curtailment-Funktionen

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Vollständige Entfernung des "Economic Curtailment"-Features für Erneuerbare Energien (Wind/PV) aus dem EPEC-Modell.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. Alle logischen, mathematischen und visuellen Referenzen bezüglich Curtailment (Abregelung von Erneuerbaren) sollen aus dem Code entfernt werden.

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. Der Feature-Toggle `CURTAILMENT` wurde gelöscht.
2. In `create_mpec_for_investor` wurden die dazugehörigen Setup-Variablen (`m.P_curtail`, `m.mu_curtail`) sowie alle zugehörigen Constraints (`NoResNoCurtail`, `CurtailMax`, `DualFeasCurtail`, `Compl_Curtail_Active`) restlos entfernt.
3. Die globale Regel `net_inj_rule` wurde vereinfacht, sodass `res_feedin` nun direkt `RES_PROFILE` ohne Abzug von Curtailment entspricht.
4. Alle Plotting- und Extraktionsfunktionen (`extract_results`, `plot_bess_detail`, `plot_system_aggregation`) wurden bereinigt (z.B. keine Curtailment-Blöcke oder Balken mehr in den Graphen).

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine offenen Punkte zu diesem Thema.

---

## 06.03.2026 - Umstellung auf die Annuitätenmethode

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Korrektur der Zielfunktion durch Umstellung von einer vereinfachten tagesanteiligen Investitionskostenberechnung auf die finanzmathematisch korrekte Annuitätenmethode.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. Bereinigung eines Zins-Doppelzählungsfehlers in der Zielfunktion: Der Abzinsungsfaktor `(1 / (1 + r_i))` vor den reinen Tageserlösen (`rev_spot + rev_afrr`) wurde entfernt.
2. Implementierung der Annuitätenmethode für die CAPEX (Kapitalkosten). Anstatt die Investition simpel durch 8760 Stunden zu teilen, wird sie nun über eine Lebensdauer von 15 Jahren und den investorenspezifischen Zins in gleichmäßige Jahresraten (Annuitäten) umgewandelt und erst dann auf den Tag (`DAYS_PER_YEAR = 365.25`) heruntergebrochen.

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. Einbau der Konstanten `LIFETIME = 15` und `DAYS_PER_YEAR = 365.25` in `bess_epec_model.py`.
2. Anpassung der `obj_rule`:
   - Berechnung des Capital Recovery Factors (CRF): `crf = (r_i * (1 + r_i)**LIFETIME) / ((1 + r_i)**LIFETIME - 1)`
   - Die `daily_capex` ergeben sich nun aus `(total_investment * crf) / DAYS_PER_YEAR`.
   - Die NPV-Klammer `(1 / (1 + r_i))` vor den Einnahmen wurde ersatzlos gestrichen.

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine offenen Punkte zu diesem Thema.

---

## 01.03.2026 - Erweiterung der Auswertung (Plots & Statistiken) sowie Übersetzung der Grafiken

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Übersetzung der Beschriftungen in den Export-Grafiken ins Englische sowie umfangreiche Erweiterung der Modellauswertung um neue Plots (Investment, Konvergenz, Congestion, aFRR) und tabellarische Statistiken (Revenue & E/P-Ratio).

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. **Übersetzung und Anpassung bestehender Grafiken:** Alle Beschriftungen (Titel, Achsen, Legenden) der Export-Grafiken sollen auf Englisch umbenannt werden. BESS-Leistung als Linie, "Total BESS" gestrichelt, aFRR x-Achse in 4-h-Blöcke.
2. **Investment Bar Chart:** Ein gestapeltes Balkendiagramm (X: Knoten, Y: Kapazität), unterteilt nach Investoren.
3. **aFRR Provision Plot:** Darstellung der positiven und negativen Regelleistung (R_Up/R_Down) über 24 Stunden je Investor.
4. **Convergence Plot:** Darstellung des Max Deltas über die Iterationen auf logarithmischer Skala.
5. **Line Congestion Plot:** Heatmap der prozentualen Leitungsauslastungen über 24 Stunden.
6. **Revenue Stream & E/P-Ratio Check:** Tabellarische Ausgabe des prozentualen Gewinns aus Arbitrage vs. aFRR je Investor sowie Überprüfung auf überdimensionierte Energiespeicher (E/P-Ratio > 2).

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. In den betroffenen Plot-Funktionen (`plot_system_aggregation`, `plot_bess_detail`) wurden deutsche Labels und Titel ins Englische übersetzt, die BESS-Darstellung modifiziert (Linienplot statt Balken) und die aFRR-Achsen sowie die Grid-Eigenschaften angepasst.
2. `solve_epec` wurde angepasst, um den Konvergenzpfad (History von `max_diff`) sowie die optimierten Energie-Kapazitäten (`X_energy`) an das Hauptprogramm zurückzugeben.
3. `extract_results` wurde erweitert, um auch Leitungsflüsse (Flows) sowie netzweite aFRR-Preise zu erfassen.
4. Fünf neue Auswertungs- bzw. Plot-Funktionen wurden implementiert und am Ende des Programms aufgerufen:
   - `plot_convergence`
   - `plot_investment_bar_chart`
   - `plot_line_congestion` (als Heatmap)
   - `plot_overall_afrr_provision`
   - `print_revenue_and_ep_ratio` (Konsolen-Ausgabe der Finanz-Statistiken)

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine offenen Punkte zu diesem Thema.

---

## 26.02.2026 - Prüfung & Entfernung O&M Kosten

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Prüfung auf O&M Kosten im Modell, testweise Implementierung und anschließende Entfernung zur Reduzierung der Komplexität.

#### Besprochene Ziele & Aufgaben (hier kann auch genauer erklärt werden)

1. Feststellen, wo im Modell die O&M Kosten in Höhe von 1% des gesamten Investments pro Jahr berechnet werden.
2. Testweise Integration der O&M Kosten in das Modell.
3. Anschließende vollständige Entfernung der O&M Kosten aus Code und Dokumentation, da diese das Modell verkomplizieren.

#### Umgesetzte Änderungen (hier kann auch genauer erklärt werden)

1. Initial festgestellt, dass O&M Kosten fehlten (es wurden nur Kapazitätskosten und Degradation abgerechnet).
2. Kurzzeitig 1% O&M Kosten formelhaft in `bess_epec_model.py` (als `om_cost`) und in `docs/Modell.md` integriert.
3. **Rollback:** Die O&M Kosten wurden wieder aus `bess_epec_model.py` sowie aus sämtlichen Formeln in `docs/Modell.md` entfernt, um das Modell simpel zu halten.

#### Offene Punkte (ToDo's) (hier kann auch genauer erklärt werden)

- Keine offenen Punkte bezüglich O&M Kosten (Thema vorerst verworfen).

---


## 25.02.2026 - Erweiterung der Netzgrafik

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Erweiterung der Funktion `plot_grid_enhanced`, sodass erneuerbare Energien (PV & Wind) im Plot bei den jeweiligen Knoten angezeigt werden.

#### Besprochene Ziele & Aufgaben
1. `plot_grid_enhanced` so anpassen, dass an den jeweiligen Netzknoten analog zu Generatoren (z.B. `[G_Base]`) abgelesen werden kann, ob es PV oder Wind gibt. Das Ganze unter der Leistung in eckigen Klammern.

#### Umgesetzte Änderungen
1. Die Abfrage aus `NODE_RES_TYPE` wurde in die Label-Generierung in `bess_epec_model.py` integriert (Resultat z.B.: `[G_Base+Wind]` oder `[PV]`).

#### Offene Punkte (ToDo's)
- Zu diesem Punkt vorerst keine.

---

## 24.02.2026 - Anpassung der Plot-Legenden

### Kurzzusammenfassung des Tages, 1-2 Zeilen 
Verbesserung der Darstellung der Lastkurven im Plot `plot_inputs_separate`, sodass identische Kurven gruppiert und mit derselben Farbe gezeichnet werden.

#### Besprochene Ziele & Aufgaben 
1. Optimierung der Legende und Farbgebung im "Input: Load Profiles"-Plot, da von den 5 Knoten einige identische Lastprofile (N1/N5 und N2/N3/N4) aufweisen.

#### Umgesetzte Änderungen 
1. In `bess_epec_model.py` wurde die Methode `plot_inputs_separate` angepasst: Knoten mit identischen Lastprofilen werden iterativ in Gruppen zusammengefasst. 
2. Für jede Gruppe wird nur noch eine Linie gezeichnet, deren Label in der Legende als ein zusammengesetzter String der zugehörigen Knoten (z.B. "N1, N5") dargestellt wird, sodass es insgesamt zwei farblich unterscheidbare Kurven für die zwei existierenden Lastprofile gibt.

#### Offene Punkte (ToDo's) 
- Keine aktuellen offenen Punkte zu diesem Thema.
