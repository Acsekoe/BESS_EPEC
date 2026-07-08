Du bist ein Coding-Assistent der sich auf Energiesystem-Modellierung fokussiert. Du hilfst mir bestmöglich beim Erreichen meiner Aufgabestellungen im Rahmen meiner Diplomarbeit.
Aufgabenstellung:
Implementiere ein Equilibrium Problem with Equilibrium Constraints (EPEC) in Python, um die strategischen Investitionsentscheidungen (Leistung in MW, Energie in MWh) von konkurrierenden Batteriespeicher-Investoren in einem Stromnetz zu modellieren.
Das Modell soll als Bilevel-Spiel (Leader-Follower) strukturiert sein:
Obere Ebene (Leader): Mehrere strategische BESS-Investoren (i), die individuell ihren Nettogegenwartswert (NPV) maximieren. Jeder Investor löst sein eigenes MPEC.
Untere Ebene (Follower): Ein zentraler Markt-/Netzbetreiber (TSO/ISO), der einen kostenminimalen, netzbeschränkten Dispatch (DC-OPF) durchführt, gegeben die Investitionsentscheidungen der Leader.
Kernanforderungen:
Strategische Investoren: Das Modell muss die Investitionsentscheidungen (MW und MWh) an jedem Netzknoten für jeden Investor endogen bestimmen.
Geteilte Ressource: Die Summe aller BESS-Leistungsinvestitionen an einem Knoten darf eine vordefinierte maximale Anschlussleistung nicht überschreiten (Shared Constraint).
Risikoaversion: Die Zielfunktion jedes Investors soll eine investorspezifische Diskontrate ($r_i$) verwenden, um unterschiedliche Risikoaversionen (Forderung nach schnellerer Amortisation) abzubilden.
Markt-Clearing: Das untere Problem muss ein DC-OPF sein, das knotenscharfe Strompreise (LMPs) als Dualvariablen der Leistungsbilanz erzeugt.
Lösungsstrategie: Das EPEC soll über einen Diagonalisierungs-Algorithmus (Gauss-Seidel) gelöst werden.
Technologie-Stack:
Sprache: Python
Modellierungs-Framework: Pyomo, da es die Formulierung von komplexen Optimierungsproblemen und die Handhabung von Dualvariablen (KKT-Bedingungen) ermöglicht.
Solver: Ein Solver, der nichtlineare Probleme (NLPs) lösen kann, die aus der MPEC-Formulierung resultieren (z.B. IPOPT).
1. Mathematische Modellformulierung
Sets und Indizes
$I$: Menge der strategischen BESS-Investoren (z.B. i = 1, 2, 3)
$N$: Menge der Netzknoten (z.B. n = 1, ..., 5)
$L$: Menge der Übertragungsleitungen
$G$: Menge der konventionellen Generatoren
$T$: Menge der repräsentativen Zeitstunden (z.B. t = 1, ..., 24 für einen Tag)
Parameter (Beispieldaten bereitstellen)
$D_{n,t}$: Stromnachfrage an Knoten n zur Stunde t [MW]
$P_{res,n,t}^{verfügbar}$: Verfügbare erneuerbare Einspeisung an n zu t [MW]
$C_g^{var}$: Variable Kosten des Generators g [€/MWh]
$P_g^{max}$: Maximale Leistung des Generators g [MW]
$Limit_l$: Thermisches Limit der Leitung l [MW]
$PTDF_{l,n}$: Power Transfer Distribution Factor (Einfluss von Knoten n auf Leitung l)
$C^{power}, C^{energy}, C^{fix\_om}$: Annualisierte BESS-Investitions- und Betriebskosten [€/MW, €/MWh, €/MW/Jahr]
$\eta_c, \eta_d, SOC_{min}, SOC_{max}$: Technische BESS-Parameter (Wirkungsgrade, SOC-Grenzen)
$Budget_i$: Investitionsbudget für Investor i [€]
$r_i$: Investorspezifische Diskontrate für Investor i [p.u.]
$Max\_Anschlussleistung_n$: Geteilte Ressource an Knoten n [MW]
Variablen des Oberen Levels (ULP) - Für jeden Investor $i$
$X_{power,i,n}$: Zu installierende BESS-Leistung an Knoten n [MW]
$X_{energy,i,n}$: Zu installierende BESS-Energiekapazität an Knoten n [MWh]
Variablen des Unteren Levels (LLP) - Für jede Stunde $t$
Primalvariablen: $P_{g,n,t}$, $P_{res,n,t}$, $P_{charge,i,n,t}$, $P_{discharge,i,n,t}$, $SOC_{i,n,t}$, $Flow_{l,t}$
Dualvariablen: $\lambda_{n,t}$ (Preis), $\mu_{l,t}^{min/max}$ (Engpasskosten), $\beta_{...}$ (BESS-Grenzen), $\alpha_{...}$ (Gen-Grenzen)
2. Das Untere Problem (LLP): Kostenminimaler Dispatch
Zielfunktion (Minimiere Systemkosten):
$$\min \sum_{t \in T} \sum_{n \in N} \sum_{g \in G} (C_{g}^{var} \cdot P_{g,n,t})$$
Nebenbedingungen (für alle $t \in T$):
Knotenleistungsbilanz (Dualvariable $\lambda_{n,t}$):
$$\sum_{g} P_{g,n,t} + P_{res,n,t} + \sum_{i} P_{discharge,i,n,t} - D_{n,t} - \sum_{i} P_{charge,i,n,t} - \sum_{l} Flow_{l,n,t} = 0 \quad \forall n$$
Leistungsfluss & Leitungsgrenzen (Dualvariablen $\mu_{l,t}^{min/max}$):
$$Flow_{l,t} = \sum_{n} PTDF_{l,n} \cdot \left( \sum_{g} P_{g,n,t} + P_{res,n,t} + \sum_{i} P_{discharge,i,n,t} - D_{n,t} - \sum_{i} P_{charge,i,n,t} \right) \quad \forall l$$
$$-Limit_{l} \le Flow_{l,t} \le Limit_{l} \quad \forall l$$
Generator- und EE-Grenzen:
$$0 \le P_{g,n,t} \le P_g^{max}$$
$$0 \le P_{res,n,t} \le P_{res,n,t}^{verfügbar}$$
BESS-Betriebsgrenzen (wichtig: gekoppelt an ULP-Variablen $X$):
$$0 \le P_{charge,i,n,t} \le X_{power,i,n} \quad \forall i,n$$
$$0 \le P_{discharge,i,n,t} \le X_{power,i,n} \quad \forall i,n$$
$$SOC_{i,n,t} = SOC_{i,n,t-1} - \frac{P_{discharge,i,n,t}}{\eta_{d}} + (P_{charge,i,n,t} \cdot \eta_{c}) \quad \forall i,n$$
$$SOC_{min} \cdot X_{energy,i,n} \le SOC_{i,n,t} \le SOC_{max} \cdot X_{energy,i,n} \quad \forall i,n$$
$$SOC_{i,n,t=T} = SOC_{i,n,t=0} \quad \forall i,n$$
3. Das Obere Problem (ULP): MPEC für Investor $i$
Zielfunktion (Maximiere $NPV_i$ unter Verwendung von LLP-Variablen):
(Hier wird $Y=1$ angenommen, d.h. die Erlöse sind annualisiert; $r_i$ ist die Diskontrate)
$$\max_{X_{power,i}, X_{energy,i}} \frac{1}{(1 + r_i)} \cdot \sum_{t \in T} \sum_{n \in N} \left( \lambda_{n,t} \cdot P_{discharge,i,n,t} \cdot \eta_{d} - \frac{\lambda_{n,t} \cdot P_{charge,i,n,t}}{\eta_{c}} \right)$$
$$- \sum_{n \in N} (C^{power} \cdot X_{power,i,n} + C^{energy} \cdot X_{energy,i,n} + C^{fix\_om} \cdot X_{power,i,n})$$
Nebenbedingungen (für Investor $i$):
Investitionsspezifische Nebenbedingungen:
$$\sum_{n \in N} (C^{power} \cdot X_{power,i,n} + C^{energy} \cdot X_{energy,i,n}) \le Budget_i$$
$$X_{energy,i,n} \ge E/P_{min} \cdot X_{power,i,n} \quad \forall n$$
$$X_{power,i,n} \ge 0, \quad X_{energy,i,n} \ge 0 \quad \forall n$$
Geteilte Nebenbedingung (Shared Constraint):
$$X_{power,i,n} + \sum_{j \neq i} (X_{power,j,n}^{\text{fixed}}) \le Max\_Anschlussleistung_n \quad \forall n$$
(Wobei $X_{power,j,n}^{\text{fixed}}$ die (fixierten) Investitionen der anderen Investoren aus der vorherigen Iteration sind.)
Equilibrium Constraints (KKTs des LLP):
Füge hier die gesamten KKT-Bedingungen (Primale Zulässigkeit, Duale Zulässigkeit, Stationarität, Komplementarität) des oben definierten Unteren Problems (LLP) als Nebenbedingungen ein.
Dies ist der Kern des MPEC. In Pyomo kann dies z.B. über pyomo.environ.KKT oder durch manuelles Ausformulieren der Bedingungen geschehen.
4. Lösungsalgorithmus: Diagonalisierung (Gauss-Seidel)


Implementiere eine Funktion, die das EPEC-Gleichgewicht iterativ findet:
Python

def solve_epec_diagonalization():
    # 1. Initialisierung
    X_power_global = initialize_investments(investors, nodes) # z.B. alles auf Null
    X_energy_global = initialize_investments(investors, nodes)
    
    converged = False
    max_iterations = 50
    tolerance = 1e-3
    
    for iteration in range(max_iterations):
        if converged:
            break
            
        converged = True
        X_power_old_iteration = X_power_global.copy()
        
        for i in investors:
            # 2. Löse MPEC für Investor i
            # Übergebe X_power_global und X_energy_global als Parameter (fixed)
            # für alle j != i
            
            mpec_model = create_mpec_for_investor(i, X_power_global, X_energy_global)
            
            # Das MPEC ist ein NLP (oder MIP, wenn Komplementarität linearisiert wird)
            solver = SolverFactory('ipopt') 
            results = solver.solve(mpec_model)
            
            # 3. Update der globalen Investitionsentscheidung
            new_power_i = get_results(mpec_model, 'X_power')
            new_energy_i = get_results(mpec_model, 'X_energy')
            
            # 4. Konvergenzprüfung
            if check_convergence(new_power_i, X_power_old_iteration[i], tolerance) == False:
                converged = False
                
            # Update für die nächste Iteration (für Investor i+1)
            X_power_global[i] = new_power_i
            X_energy_global[i] = new_energy_i
            
    return X_power_global, X_energy_global
5. Beispieldaten (3 Investoren, 5 Knoten)
Bitte verwende die folgende (oder eine ähnliche) Topologie und Parameter für das Testmodell:
Knoten (5):
N1 (Nord): Hohe Wind-Einspeisung (P_res), niedrige Last.
N2: Transitknoten.
N3: Mittlere Last, mittlere PV (P_res).
N4: Transitknoten.
N5 (Süd): Hohe Last (D), teurer Gaskraftwerk (G1).
Leitungen (4):
L12 (N1-N2), L23 (N2-N3), L34 (N3-N4), L45 (N4-N5).
Setze Limit_l für L34 auf einen niedrigen Wert, um einen Engpass zu erzeugen.
Investoren (3):
I1: Neutral (r_1 = 0.08), hohes Budget.
I2: Risikoavers (r_2 = 0.12), mittleres Budget.
I3: Sehr risikoavers (r_3 = 0.15), niedriges Budget.
Geteilte Ressource:
Max_Anschlussleistung an N1 und N5 (hohe Nachfrage) = 100 MW.
Max_Anschlussleistung an N4 (nahe Engpass) = 50 MW.
Max_Anschlussleistung an N2, N3 = 200 MW.
Zeit: T=24 (repräsentativer Tag mit PV-Profil auf N3, Wind-Profil auf N1).