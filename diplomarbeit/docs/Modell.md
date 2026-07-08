# Modellbeschreibung

## 1. Grundmodell (ohne aFRR, ohne Degradation)

Das Modell ist als Equilibrium Problem with Equilibrium Constraints (EPEC) formuliert. Es besteht aus einer oberen Ebene (Upper Level Problem, ULP), in der strategische Investoren ihren Gewinn maximieren, und einer unteren Ebene (Lower Level Problem, LLP), die das Marktclearing durch einen Systembetreiber (ISO) abbildet. Zur Lösung der Oszillationsproblematik (Schweinezyklus) wird für jeden Investor $i$ ein MPEC aufgestellt, in dem die Handlungen der Konkurrenten $j \neq i$ als gegebene modifizierte Residuallast (Residual Demand) im Marktclearing antizipiert werden.

### Sets und Indizes
*   $i, j \in I$: Strategische BESS-Investoren
*   $n \in N$: Netzknoten
*   $t \in T$: Zeitschritte (Stunden)
*   $g \in G$: Konventionelle Generatoren (G_Base, G_Peak)
*   $l \in L$: Übertragungsleitungen

### Parameter
*   $r_i$: Zinssatz (WACC) des Investors $i$
*   $LIFETIME$: Lebensdauer der Anlage (z. B. 15 Jahre)
*   $\eta$: Lade- und Entladewirkungsgrad (z. B. 0.936)
*   $C_{power}$: Spezifische Investitionskosten für Leistung [€/MW]
*   $C_{energy}$: Spezifische Investitionskosten für Kapazität [€/MWh]
*   $Limit_n$: Maximal anschließbare gemeinsame Leistung an Knoten $n$ [MW]
*   $Ratio_{min}$: Minimales Energy-to-Power Ratio (z. B. 2.0)

### Variablen
**Upper Level (Investitionsentscheidungen):**
*   $X_{power, i, n}$: Installierte Leistung [MW]
*   $X_{energy, i, n}$: Installierte Kapazität [MWh]

**Lower Level (Betrieb & Markt):**
*   $P_{charge, i, n, t}, P_{discharge, i, n, t}$: Lade-/Entladeleistung
*   $SOC_{i, n, t}$: Speicherfüllstand
*   $P_{gen, g, t}$: Erzeugung Generatoren
*   $RES_{curtail, n, t}$: Wirtschaftliche Abregelung erneuerbarer Energien
*   $Flow_{l, t}$: Leistungsfluss
*   $\lambda_{n, t}$: Knotenpreis (LMP) [Dualvariable der Leistungsbilanz]

### Upper Level Problem (Für den aktiven Investor $i$)

**Zielfunktion (Maximiere Tages-Gewinn):**
Die Zielfunktion maximiert den Deckungsbeitrag aus dem Arbitrage-Handel abzüglich der annualisierten Investitionskosten (CAPEX) und einer winzigen Strafe für Curtailment (um den Zero-Price-Exploit zu vermeiden).

Der Capital Recovery Factor (CRF) zur Annuitätenberechnung lautet:
$$
CRF_i = \frac{r_i \cdot (1 + r_i)^{LIFETIME}}{(1 + r_i)^{LIFETIME} - 1}
$$

Die täglichen CAPEX berechnen sich aus den Gesamtinvestitionen:
$$
\text{CAPEX}_{daily, i} = \frac{CRF_i}{365.25} \cdot \sum_{n} (C_{power} \cdot X_{power, i, n} + C_{energy} \cdot X_{energy, i, n})
$$

Die Zielfunktion lautet somit:
$$
\max \text{Gewinn}_{i} = \sum_{t, n} \left[ \lambda_{n, t} \cdot (P_{discharge, i, n, t} - P_{charge, i, n, t}) \right] - \text{CAPEX}_{daily, i} - \sum_{t, n} (0.01 \cdot RES_{curtail, n, t})
$$

**Nebenbedingungen:**
1.  **Shared Resource Constraint:**
    $$
    X_{power, i, n} + \sum_{j \neq i} X_{power, j, n}^{fixed} \le Limit_n
    $$
2.  **Energy-Power-Ratio:**
    $$
    X_{energy, i, n} \ge Ratio_{min} \cdot X_{power, i, n}
    $$
3.  **Equilibrium Constraints:**
    Die KKT-Bedingungen des Lower Level Problems (siehe unten). Zur Solver-Stabilisierung wird die Komplementaritäts-Relaxation auf $COMPL\_RELAX = 10.0$ gesetzt.

### Lower Level Problem (Market Clearing)

Der Systembetreiber minimiert die Dispatch-Kosten der Generatoren. Die Abregelung von RES wird nicht über fehleranfällige KKTs, sondern primär über die Nicht-Negativitätsbedingung des Spotpreises ($\lambda_{n, t} \ge 0$) gesteuert.

**Zielfunktion (Lower Level):**
$$
\min \sum_{t, g} (MC_g \cdot P_{gen, g, t})
$$

**Nebenbedingungen:**
1.  **Modifizierte Knotenleistungsbilanz ($\lambda_{n, t}$):**
    Die Handlungen der anderen Investoren $j \neq i$ werden als vorab bekannte Residual-Last ($D_{adj, n, t}$) antizipiert:
    $$
    D_{adj, n, t} = D_{n, t} + \sum_{j \neq i} (P_{charge, j, n, t}^{fixed} - P_{discharge, j, n, t}^{fixed})
    $$
    Die Bilanzgleichung für Knoten $n$ lautet:
    $$
    \sum_{g \in G_n} P_{gen, g, t} + (RES_{pot, n, t} - RES_{curtail, n, t}) + (P_{discharge, i, n, t} - P_{charge, i, n, t}) - D_{adj, n, t} - \text{NetExport}_{n, t} = 0
    $$

    **(Entstehung als Dualvariable):** 
    Im EPEC/MPEC-Framework wird diese Balance-Restriktion nicht als primale Gleichung im oberen Problem, sondern über ihre KKT-Bedingungen (Karush-Kuhn-Tucker) im LLP gelöst. Der Strompreis $\lambda_{n, t}$ ist dabei exakt die **Dualvariable** (oder der Schattenpreis) der Netzknoten-Leistungsbilanz. 
    Im Pyomo-Code (`bess_epec_model.py`) passiert die finale Preisbildung aufgrund der DC-OPF PTDF-Logik direkt im Constraint **`KKT_Stat_Inj`** (Stationaritätsbedingung der Einspeisung). Der lokale Knotenpreis $\lambda_{n,t}$ errechnet sich dort aus der Systempreis-Dualvariablen ($\lambda_{sys,t}$) abzüglich aller kongestionsbedingten Leitungs-Schattenpreise ($\sum PTDF \cdot \mu_{line}$).

2.  **Leitungsrestriktionen (DC-OPF mit PTDF):**
    $$
    Flow_{l, t} = \sum_{n \in N} PTDF_{l, n} \cdot \text{NetInjection}_{n, t}
    $$
    $$
    -Limit_l \le Flow_{l, t} \le Limit_l
    $$
3.  **Generator- und Curtailment-Grenzen:**
    $$
    0 \le P_{gen, g, t} \le P_{max, g}
    $$
    $$
    0 \le RES_{curtail, n, t} \le RES_{pot, n, t}
    $$
    Der Spotpreis ist zwingend positiv ($\lambda_{n,t} \ge 0$). Fällt er auf 0 €, regelt der Markt automatisch RES ab.

4.  **Speicher-Restriktionen (technisch) des Investors $i$:**
    $$
    0 \le P_{charge, i, n, t} \le X_{power, i, n}
    $$
    $$
    0 \le P_{discharge, i, n, t} \le X_{power, i, n}
    $$
    $$
    SOC_{i, n, t} = SOC_{i, n, t-1} + \eta \cdot P_{charge, i, n, t} - \frac{1}{\eta} \cdot P_{discharge, i, n, t}
    $$
    *(Der Zyklus schließt sich am Tagesende: $SOC_{t=0} = SOC_{t=24}$)*
    $$
    0 \le SOC_{i, n, t} \le X_{energy, i, n}
    $$

---

## 2. Modell mit aFRR und Degradation (Co-Optimierung Spot & Reserve)

Die BESS-Investoren partizipieren zusätzlich am aFRR-Markt. Positive (Up) und negative (Down) Reserven werden in z.B. 4-Stunden-Blöcken $b \in B$ kontrahiert. Der Preis wird spieltheoretisch über Inverse Nachfragefunktionen abgebildet (Cournot-Oligopol), wodurch der "Stackelberg-Preisdiktator-Exploit" verhindert wird.

### Zusätzliche Parameter
*   $b \in B$: Blöcke
*   $C_{degrad}$: Spezifische Degradationskosten (z.B. 15 €/MWh Durchsatz)
*   $Price_{cap}$: Preisobergrenze / Penalty-Preis im aFRR-Markt (z.B. 3000 €/MW)
*   $Demand_{up, b}, Demand_{down, b}$: aFRR Nachfrage je Block

### Zusätzliche Variablen
*   $R_{up, i, n, b}$: Vorgehaltene positive Regelleistung
*   $R_{down, i, n, b}$: Vorgehaltene negative Regelleistung

### aFRR Inverse Demand Curve (Cournot-Pricing)
Der Preis für Regelleistung ergibt sich endogen aus dem aggregierten Gebot aller Marktteilnehmer:
$$
\lambda_{afrr, up, b} = Price_{cap} \cdot \left( 1 - \frac{\sum_n R_{up, i, n, b} + \sum_{j \neq i, n} R_{up, j, n, b}^{fixed}}{Demand_{up, b}} \right)
$$
(Analog für $\lambda_{afrr, down, b}$)

### Erweiterte Zielfunktion Investor
Die Einnahmen aus dem aFRR-Markt (aus den Inversen Nachfragefunktionen) werden addiert und die Degradation wird abgezogen:

$$
\max \text{Gewinn}_{i} = \text{Erlös}_{spot} + \text{Erlös}_{afrr} - \text{Kosten}_{degrad} - \text{CAPEX}_{daily, i} - \text{Penalty}_{curtail}
$$

Im Detail:
$$
\text{Erlös}_{afrr} = \sum_{b, n} H_{block} \cdot (\lambda_{afrr, up, b} \cdot R_{up, i, n, b} + \lambda_{afrr, down, b} \cdot R_{down, i, n, b})
$$
$$
\text{Kosten}_{degrad} = \sum_{t, n} \left[ 0.5 \cdot C_{degrad} \cdot (P_{charge, i, n, t} + P_{discharge, i, n, t}) \right]
$$

### Erweiterte Restriktionen (Speicher-Band)
Die vorgehaltene Regelleistung reduziert die für den Spotmarkt verfügbare Kapazität symmetrisch für jede Stunde $t$ in Block $b(t)$:

1.  **Lade-Limit mit Reserve:**
    $$
    P_{charge, i, n, t} + R_{down, i, n, b(t)} \le X_{power, i, n}
    $$
2.  **Entlade-Limit mit Reserve:**
    $$
    P_{discharge, i, n, t} + R_{up, i, n, b(t)} \le X_{power, i, n}
    $$

---

## 3. Übersicht der wesentlichen Änderungen

Um von mathematischen Instabilitäten zu einem funktionsfähigen Nash-Gleichgewicht zu gelangen, wurden folgende Transformationen umgesetzt:

| Problem im klassischen Modell | Lösung im weiterentwickelten Modell |
| :--- | :--- |
| **Oszillation / Schweinezyklus** | Einführung des **Residual Demand Ansatzes**. Investoren optimieren nur eigene Variablen gegen fixierte Fremdfahrpläne, statt in einem gemeinsamen LLP. |
| **Infeasibility bei Peak-Preisen** | Die strenge KKT-Bedingung für Generatoren wurde massiv relaxiert (`COMPL_RELAX` = 10.0), um Singularitäten für den NLP-Solver zu umgehen. |
| **Null-Euro Curtailment Exploit** | Curtailment wird **nicht mehr über KKT-Bedingungen** modelliert. Stattdessen nutzt man einen Spotpreis-Boden ($\lambda \ge 0$) und winzige Zielfunktions-Penalties ($0.01$ €/MW). |
| **Stackelberg aFRR Preisdiktatur** | aFRR-Preise bilden sich **nicht mehr über Slack-Variablen**. Sie sind harte Ausdrücke in der Zielfunktion über **Cournot Inverse Demand Curves**. |
| **Statische NPV Ungenauigkeit** | Investitionsentscheidungen basieren nun auf exakt **annualisierten Tageskosten (CRF, Annuitätenmethode)** statt auf pauschaler Division. |
