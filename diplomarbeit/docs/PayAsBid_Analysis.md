# APG Pay-as-Bid aFRR Modellprüfung im BESS-EPEC

## Zusammenfassung
Die APG nutzt für den aFRR Leistungspreis (Capacity Auction) auf dem Primärmarkt tatsächlich ein **Pay-as-Bid** Verfahren ("lowest capacity price", "pay-as-bid approach").

Eine exakte Implementierung dieses Verfahrens im vorliegenden kontinuierlichen EPEC-Modell ist jedoch aus wahrscheinlichkeitstheoretischen, spieltheoretischen und algorithmischen Gründen **nicht zielführend** und würde die in vorherigen Schritten mühsam gelösten Probleme (KKT-Singularitäten, Stackelberg-Exploit, Infeasibility) unweigerlich wieder einführen.

---

## 1. Mathematische Implikationen von Pay-as-Bid im EPEC

### Das Problem der Merit-Order KKTs
Im aktuellen Cournot-Modell ist der Preis eine deterministische, stetige Funktion der Gesamtmenge: $\lambda = P_{cap}(1 - \frac{Q}{D})$. 
Ein Pay-as-Bid Modell erfordert jedoch eine exakte Merit-Order (Reihung nach Gebotspreis) im Lower Level Problem (ISO). 

Das bedeutet:
1. **Neue ULP Variablen:** Die Investoren müssen nicht nur die Einsatzmenge ($R_{up}$), sondern auch einen **Gebotspreis ($Bid$)** in ihrem Upper Level Problem als freie Variable festlegen.
2. **Bi-Lineare Zielfunktion:** Die Zielfunktion des Investors wäre nicht mehr $Preis \cdot Menge$, sondern $Bid \cdot Akzeptierte\_Menge$.
3. **KKTs für den ISO:** Der ISO minimiert $\sum (Bid_i \cdot Akzeptierte\_Menge_i)$ abhängig von der Grenze $Akzeptierte\_Menge_i \le Angebotene\_Menge_i$. 

Da $Bid_i$ aus Sicht des Lower Levels (ISO) ein Parameter sein muss, aus Sicht des Upper Levels (Investor) aber eine Variable ist, entsteht ein hochgradig nicht-konvexes, bi-lineares Gleichungssystem.

### Wiederkehr des Stackelberg-Exploits
Im Pay-as-Bid Modell wird das Gleichgewicht ausschließlich über KKT-Komplementaritätsbedingungen für das Marktclearing diktiert (z.B. `(Bid_i - lambda_system) * Slack <= 10.0`). 
Wie wir bereits beim Spotmarkt-Curtailment und der alten aFRR-Penalty-Formel gesehen haben, **missbraucht der Ipopt-NLP-Solver diese KKT-Relaxation**. 

Der Solver eines Investors würde Folgendes tun:
- Er setzt seinen $Bid$ auf $2999.99$ €.
- Er nutzt die KKT-Toleranz aus, um den ISO mathematisch auszutricksen, sodass er trotz des absurden Preises komplett akzeptiert wird (indem er fiktive Defizite oder Slack-Grenzen ausnutzt).
- Ergebnis: Wir landen exakt wieder bei dem Fehler, dass jeder Investor den maximalen Preis diktiert, ohne in Wettbewerb zu treten.

### Diskrete vs. Kontinuierliche Mathematik
Eine saubere Pay-as-Bid Auktion in EPECs erfordert **MIP (Mixed-Integer Programming)** Variablen (z.B. binäre Variablen $\delta \in \{0, 1\}$, ob ein Gebot akzeptiert wird). Derzeit benutzen wir `Ipopt`, einen Solver für Interior-Point **NLP (Non-Linear Programming)**, der ausschließlich stetige Gradienten verarbeiten kann. Die Einführung von Integer-Logik würde das Modell auf "MI-NLP" heben, was EPECs in der Regel komplett unlösbar ("computationally intractable") macht. Die Rechenzeit würde von 4 Sekunden auf Stunden oder Unlösbarkeit ansteigen.

---

## 2. Warum das Cournot-Oligopol die bessere Wahl ist

Für Investitionsmodelle (Langfrist- oder Designstudien) betrachtet man Märkte makroökonomisch. Das aktuell implementierte **Cournot-Oligopol** (Inverse Demand Curve) ist das akademische Standardwerkzeug für genau diesen Zweck:
- Es bildet ab, dass Marktmacht den Preis beeinflusst (je mehr geboten wird, desto stärker sinkt der Preis).
- Es ist **strikt konvex** und stetig differenzierbar, was den Solver rasend schnell und zu 100% robust konvergieren lässt.
- Im Endeffekt spiegelt die Cournot-Kurve das durchschnittliche Gleichgewichtsergebnis eines Pay-as-Bid Marktes unter unvollkommenem Wettbewerb mit mehreren Akteuren perfekt wider, ohne den diskreten Bidding-Prozess mikroskopisch modellieren zu müssen.

## 3. Fazit und Handlungsempfehlung

**Empfehlung: Verwerfen der Pay-as-Bid Idee.**

Die Implementierung würde den mathematischen Kern des MPECs zerstören und das Modell in Infeasibility oder alte Exploit-Muster zwingen. Das kürzlich stabilisierte stetige Cournot-Verfahren für den aFRR Markt ist der State-of-the-Art Ansatz für Continuous EPECs und erfüllt den Zweck der Preis-Mengen-Dynamik im Hinblick auf oligopolistische Marktstrukturen bereits exzellent. 

Sollten die Gewinne im Cournot-Modell künstlich zu hoch sein, ist der korrekte Hebel eine Anpassung des `AFRR_PENALTY_PRICE` (die Y-Achsenabschnitt-Obergrenze), und nicht der Umbau der Marktclearing-KKTs.
