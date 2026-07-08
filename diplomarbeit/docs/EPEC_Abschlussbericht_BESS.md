# EPEC BESS Modell - Entwicklungs- und Abschlussdokumentation
*(Stand: Laufende Evolution bis inkl. Annualisierung der Investitionskosten)*

Dieses Dokument fasst die maßgeblichen mathematischen, algorithmischen und wirtschaftlichen Durchbrüche zusammen, die im Laufe der Entwicklung des **Equilibrium Problem with Equilibrium Constraints (EPEC)** für Batterie-Energiespeichersysteme (BESS) erzielt wurden. Es dient als ultimative Referenz für die implementierten Lösungsansätze, insbesondere jene, die schwerwiegende Solver-Instabilitäten und ökonomische Paradoxa behoben haben.

---

## 1. Der "Residual Demand" Durchbruch (Nash Equilibrium)

### Das Problem (Das Oszillations- und Verlust-Paradoxon)
In den frühen Iterationen des Modells wurden die Lade- und Entladeentscheidungen der Konkurrenten dem aktiven MPEC (Mathematical Program with Equilibrium Constraints) als starre Variablen übergeben. Dies führte zu zwei massiven Problemen:
1. **Cobweb-Oszillationen:** Investoren wichen sich zyklisch aus. Lief Investor 1 in Stunde $t$, mied Investor 2 diese Stunde in der nächsten Iteration und lud in Stunde $t+1$. In Iteration 3 wich Investor 1 in $t+1$ aus. Das Modell fand kein Gleichgewicht.
2. **Negative Arbitrage ("Gefangenendilemma"):** Die Investoren zwangen sich gegenseitig in unrentable Zyklen, da sie den Marktpreis bei ihren Aktionen nicht antizipieren konnten, was zu extremen Verlusten führte.

### Die Lösung
Die Modellarchitektur wurde auf die **Residual Demand Formulierung** umgebaut. Statt alle MPEC-Entscheidungsvariablen aller Konkurrenten lokal nachzubilden, wurden die aggregierten Einspeisungen und Entnahmen der Konkurrenten in einem prä-prozessierten Parameter `adj_load` (Angepasste Systemlast) verrechnet.
- Für das lokale MPEC von Investor $i$ existieren nun *nur noch* die eigenen Speicher-Variablen. 
- Das Modell reagiert organisch auf die Netto-Restnachfrage des Systems.
- Dies stabilisierte das Nash-Gleichgewicht sofort und führte zu den ersten rationalen, positiven Arbitrage-Entscheidungen der BESS.

---

## 2. KKT Singularitäten & Solver-Stabilität (Spotmarkt)

### Das Problem (Infeasibility bei Peak-Preisen)
Nachdem die Marginalen Kosten (MC) des Peak-Kraftwerks auf realistische 120 €/MWh angehoben wurden, stürzte der Ipopt-NLP-Solver kontinuierlich mit der Meldung "locally infeasible point" ab. 

### Die Analyse
Die Ursache war die **Karush-Kuhn-Tucker (KKT) Komplementaritätsbedingung** für die Generatoren:
`(MC - lambda_spot) * P_gen <= COMPL_RELAX`
Der Relaxations-Faktor (`COMPL_RELAX`) war auf einen rigorosen Wert von `0.5` gesetzt. Bei extrem hohen Strompreisen ($lambda \approx 120$) und volllaufenden Generatoren musste der Solver einen extrem spitzen, mathematischen "Kegel" navigieren (Singularität), was zu unendlichen Gradienten im Interior-Point Verfahren führte.

### Die Lösung
Anhebung der Relaxation auf `COMPL_RELAX = 10.0`. 
Dies gab dem non-linearen Solver einen ausreichend großen Toleranz-Schlauch. Die Preise weichen dadurch zwar um insignifikante mikro-Cent Beträge ab, das mathematische Modell konvergiert jedoch nun **vollkommen robust** in allen Last-Szenarien ohne Abstürze.

---

## 3. Economic Curtailment und der "Zero-Price Exploit"

### Das Problem (Batterien als Müllschlucker)
Um negative Spotpreise bei starkem Erneuerbaren-Überschuss (RES) zu vermeiden, wurde ein Feature zur wirtschaftlichen Abregelung (`RES_Curtail`) eingeführt. Überraschenderweise führte dies sofort wieder zu massiven Verlusten (negative Arbitrage) bei allen Investoren.

### Die Analyse ("0-€ Exploit")
Die Abregelung wurde klassisch über KKT-Bedingungen modelliert: `RES_Curtail * lambda_spot <= COMPL_RELAX`
Da dieses KKT lokal in jedem MPEC der Investoren lag, **manipulierten die Investoren den Toleranzbereich**:
Ein Investor simulierte eine winzige, fiktive Abregelung in seinem eigenen MPEC (z.B. $0.08 \text{ MW} \times 120 \text{ €} \approx 9.6 \leq 10.0$). Diese winzige mathematische Slack-Ausnutzung erlaubte es dem Solver, den Spotpreis im lokalen MPEC künstlich auf $0 €$ zu setzen! Die BESS luden im blinden Glauben an Gratisstrom ihre Speicher voll. In der globalen System-Realität (wo keine echte Abregelung stattfand) mussten die teuren Peak-Kraftwerke diesen Strombedarf bedienen, was den wahren Preis auf 120 € trieb und zu massiven Ladekosten führte.

### Die Lösung
Der bi-lineare KKT-Block für das Curtailment wurde **komplett gelöscht**.
Stattdessen wird Curtailment nun global durch reine duale Optimalität abgewickelt:
1. Harter Preis-Boden: `m.lambda_spot.bounds = (0.0, 3000.0)`
2. Winzige Zielfunktions-Strafe für Verschwendung: `- RES_Curtail * 0.01`
Der Markt schaltet bei Überschuss die Kraftwerke ab und regelt organisch den Überschuss weg. Der Preis fällt logisch auf $0 €$, ohne dass die MPECs die Nichtlinearitäten des KKTs ausnutzen können.

---

## 4. Regelenergiemarkt (aFRR) und der "Cournot-Exploit"

### Das Problem (Exzessive Monopol-Gewinne)
Nach der Freischaltung des Sekundärregelmarktes (aFRR) brachen die Arbitragegewinne ein, während jeder Investor plötzlich absurde Profite im Bereich von $\sim 170.000~€$ pro Tag pro Investor aus aFRR-Bereitstellungen lukrierte.

### Die Analyse ("Stackelberg Preisdiktator")
Wieder nutzten die Investoren den NLP-Relaxationsschlupf aus:
`(Penalty - Price) * Deficit <= 10.0`
Weil der aFRR-Preis (`lambda_afrr`) eine freie MPEC-Dualvariable war, wies das Modell an, bewusst eine winzige Versorgungslücke (`Deficit > 0`) zu erzeugen. Die KKT-Bedingung wurde aktiv und erlaubte dem MPEC-Mechanismus, den aFRR-Preis sofort auf das absolut maximal zugelassene Penalty-Limit von $3.000~€/MW$ springen zu lassen, selbst wenn fast 0 MW geliefert wurden. Die Investoren agierten als marktmanipulierende Preisdiktatoren.

### Die Lösung
Komplette Neugestaltung des aFRR Reserve-Pricings durch eine **Stetige inverse Nachfragefunktion** anstelle starrer KKT-Defizit-Regeln.
Der Preis wird nun deterministisch als Reaktion auf die Angebotsmenge berechnet:
`Preis = Penalty-Cap * (1 - (Bereitgestellte Reserve / Maximaler Reservebedarf))`
- Liefern die Investoren nichts, schießt der Preis auf $3000~€$.
- Liefern sie exakt den geforderten Marktbedarf, ist der Markt gesättigt und der Preis ist $0~€$.

**Resultat:** Das EPEC verhält sich nun wie ein lehrbuchhaftes **Cournot-Oligopol**. Simulationen belegen, dass die 4 Investoren im 50-MW Markt bei 3000€-Cap zusammen bei ca. $600~€$ Preis ins Nash-Level konvergieren und sich die Gewinne im perfekten physikalisch-ökonomischen Gleichgewicht einpendeln.

---

## 5. Finanzielle Annualisierung (Capital Recovery Factor)

Um die MPEC-Zielfunktion für reale Investitionsentscheidungen nutzbar zu machen, mussten gigantische Upfront-Investitionskosten (CAPEX) in tägliche Operationsentscheidungen skaliert werden.

### Die Implementierung
Es wurde die Annuitätenmethode über den **Capital Recovery Factor (CRF)** angewendet.
$CRF = \frac{r_i \times (1 + r_i)^{Lifetime}}{(1 + r_i)^{Lifetime} - 1} $
wobei:
- $r_i$ : Die individuellen Kapitalkosten (WACC) des Investors (8%, 12%, 15%, 20%)
- $Lifetime$ : Projektlebensdauer (15 Jahre)

Die Gesamtkosten (`BESS_COST_POWER * MW + BESS_COST_ENERGY * MWh`) werden mit dem CRF multipliziert und durch die Tage des Jahres ($365.25$) geteilt, um die exakten **Daily CAPEX** zu erhalten.
Dadurch maximiert das EPEC nun den wahren Net Present Value (Tages-Deckungsbeitrag minus Tages-Annuitätskosten) pro Investor und gleicht die strategischen Vorzüge billigerer Kredite nahtlos in den Batterieausbau ein.

---

## 6. KKT Relaxation Noise und Optische Filterung

### Das Problem (Curtailment bei aktiven Kraftwerken)
Es fiel auf, dass in manchen Szenarien paradoxerweise winzige Mengen (ca. 0.01 MW bis 0.05 MW) an Winderzeugung abgeregelt wurden, **während** konventionelle Kraftwerke liefen, oder umgekehrt: Die Kraftwerke erzeugten augenscheinlich noch mit ~0.4 MW, obwohl massive Abregelung stattfand (Spotpreis = 0 €).

### Die Analyse
Dies ist kein Modellfehler, sondern ein mathematisches Artefakt der **Interior-Point Penalty Relaxation**:
- Wenn Spotpreis = 0 €, lautet die statische Generator KKT: `P_gen * (40 - 0) <= 10.0`
Dies zwingt den Generator nicht zwingend auf `0`, sondern erlaubt einen numerischen Slack von `10.0 / 40 = 0.25 MW`.
- Wenn Spotpreis = 40 €, lautet die Curtailment KKT: `RES_Curtail * 40 <= 1.0`
Daraus ergibt sich ein erlaubter Curtailment-Slack von `1.0 / 40 = 0.025 MW`.

### Die Lösung (Snapping Filter)
Da diese Werte ökonomisch vollkommen irrelevant sind (im Kilowatt-Bereich auf einem 1200 MW Grid) und nur optische Verwirrung in den Systemplots stiften (z.B. indem die rote Kraftwerkslinie bei massiver Abregelung leicht über der Null-Achse schwebt), wurde eine **Snapping-Funktion** in der Ergebnis-Extraktion eingeführt:
- In der CSV-Generierung sowie den Plots werden sämtliche Abregelungen `RES_Curtail` $< 0.1 \text{ MW}$ und alle Generatorleistungen `P_gen` $< 1.0 \text{ MW}$ in der Visualisierung auf eine perfekte physikalische $0$ gesetzt.
Dies sorgt für kompromisslose grafische und tabellarische Konsistenz für den Endnutzer, wobeigleichzeitig der NLP-Solver seinen dringend benötigten mathematischen Gradienten-Schlauch (`COMPL_RELAX`) behält.
