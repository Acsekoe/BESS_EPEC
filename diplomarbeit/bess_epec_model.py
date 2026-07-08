# Readme:
# v5: plot inputs, visualize grid, main
# v6: ieee 5 bus test system, ptdf calculation
# v7: added SOC & visualization of best investor bess charge curve
# v8: added new visualizations
# v9: added degradation cost in obj fct
# v10: added co-optimization for aFRR, plot all nodes
# v11: plot bess detail second axis
# v12: added EP_Ratio
# v13: Added Economic Curtailment for RES (Wind/PV) with negative strike prices.
# v14:
# - Fixes oscillations by reducing DAMPING_FACTOR to 0.25
# - Fixes 1e16 prices by introducing aFRR Deficit (Slack) mechanism with Penalty Price
# - Cleaned up plots

# (c)opyright Daniel Horvath, 11941270, TU Wien

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.contrib.latex_printer import latex_printer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==============================================================================
# 1. KONFIGURATION & PARAMETER
# ==============================================================================

CALCULATE_AFRR = True          # False = Nur Arbitrage, True = Arbitrage + aFRR
ENABLE_CURTAILMENT = True       # True = Erlaubt Abregelung von RES bei Spotpreis <= 0

# --- Algorithmus Tuning ---
MAX_ITER = 40           # Etwas mehr Iterationen, da Dämpfung stärker ist
TOLERANCE = 0.5        # Toleranz
DAMPING_FACTOR = 0.25   # WICHTIG: Nur 25% Mix aus Neu, 75% Alt -> Verhindert Oszillation!
COMPL_RELAX = 10      # Relaxation factor for KKT complementarity to avoid Ipopt infeasibility
GAUSS_JACOBI = True     # True = Parallel (Jacobi), False = Sequenziell (Gauß-Seidel). Parallel nutzt CPU besser, ändert aber Konvergenzpfad.
CONVERGENCE_CRITERIA = 'Power' # 'Power', 'Energy', 'PowerAndEnergy' - User switch

# -- Wirtschaftsdaten --
BESS_COST_POWER = 6600 
BESS_COST_ENERGY = 18800 
LIFETIME = 15  # Jahre
DAYS_PER_YEAR = 365.25 
OBJ_SCALE = 1e-6 
DEGRADATION_COST = 15.0
EP_RATIO = 2.0 # minimale EP Ratio
MC_BASE = 40
MC_PEAK = 80

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# --- Szenario Dimensionen ---
NODES = ['N1', 'N2', 'N3', 'N4', 'N5']
GENS = ['G_Base', 'G_Peak']
TIME = range(1, 25) 
INVESTORS = ['I1', 'I2', 'I3', 'I4']

# --- Österreichisches aFRR Design ---
BLOCKS = range(1, 7) 
HOURS_PER_BLOCK = 4
TIME_TO_BLOCK = {t: ((t-1) // 4) + 1 for t in TIME}

# aFRR Parameter
AFRR_DEMAND_UP = {b: 50.0 for b in BLOCKS}
AFRR_DEMAND_DOWN = {b: 50.0 for b in BLOCKS}
AFRR_PENALTY_PRICE = 3000.0 # WICHTIG: Maximalpreis für aFRR (statt unendlich)


NODE_RES_TYPE = {'N1': 'Wind', 'N2': 'None', 'N3': 'PV', 'N4': 'None', 'N5': 'None'}


# Netz
RAW_LINES = [
    {'id': 'L12', 'from': 'N1', 'to': 'N2', 'x': 0.0281, 'limit': 400},
    {'id': 'L14', 'from': 'N1', 'to': 'N4', 'x': 0.0304, 'limit': 400},
    {'id': 'L15', 'from': 'N1', 'to': 'N5', 'x': 0.0064, 'limit': 400},
    {'id': 'L23', 'from': 'N2', 'to': 'N3', 'x': 0.0108, 'limit': 400},
    {'id': 'L34', 'from': 'N3', 'to': 'N4', 'x': 0.0297, 'limit': 240}, 
    {'id': 'L45', 'from': 'N4', 'to': 'N5', 'x': 0.0297, 'limit': 240}, 
]
LINES = [l['id'] for l in RAW_LINES]
LINE_LIMITS = {l['id']: l['limit'] for l in RAW_LINES}
SHARED_LIMITS = {'N1': 100, 'N2': 100, 'N3': 100, 'N4': 100, 'N5': 100}

# PTDF
def calculate_ptdf(nodes, raw_lines, slack_node='N3'):
    n_nodes = len(nodes)
    node_map = {n: i for i, n in enumerate(nodes)}
    slack_idx = node_map[slack_node]
    B = np.zeros((n_nodes, n_nodes))
    for line in raw_lines:
        i, j = node_map[line['from']], node_map[line['to']]
        b = 1.0 / line['x']
        B[i, j] -= b
        B[j, i] -= b
        B[i, i] += b
        B[j, j] += b
    non_slack = [i for i in range(n_nodes) if i != slack_idx]
    B_reduced = B[np.ix_(non_slack, non_slack)]
    try:
        X_reduced = np.linalg.inv(B_reduced)
    except:
        raise ValueError("Singuläre B-Matrix")
    X_bus = np.zeros((n_nodes, n_nodes))
    r_map = {r: f for r, f in enumerate(non_slack)}
    for ri in range(len(non_slack)):
        for rj in range(len(non_slack)):
            X_bus[r_map[ri], r_map[rj]] = X_reduced[ri, rj]
    ptdf_dict = {}
    for line in raw_lines:
        i, j = node_map[line['from']], node_map[line['to']]
        factor = 1.0 / line['x']
        for n_name in nodes:
            n = node_map[n_name]
            ptdf_dict[(line['id'], n_name)] = factor * (X_bus[i, n] - X_bus[j, n])
    return ptdf_dict

PTDF = calculate_ptdf(NODES, RAW_LINES)
# MC: 40 base load, 80 peak load
GEN_DATA = {'G_Base': {'node': 'N1', 'mc': MC_BASE, 'pmax': 600}, 'G_Peak': {'node': 'N5', 'mc': MC_PEAK, 'pmax': 600}}

LOAD_PROFILE = {}
RES_PROFILE = {}
daily_pattern = [0.699, 0.674, 0.659, 0.63, 0.643, 0.706, 0.865, 0.995, 1.0, 0.906, 0.803, 0.765, 0.749, 0.741, 0.731, 0.758, 0.811, 0.834, 0.842, 0.927, 0.898, 0.704, 0.619, 0.573]
pv_pattern = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.035, 0.206, 0.438, 0.687, 0.879, 0.996, 1.0, 0.914, 0.749, 0.518, 0.24, 0.032, 0.0, 0.0, 0.0, 0.0, 0.0]
wind_pattern = [0.759, 0.88, 1.0, 0.959, 0.993, 0.866, 0.752, 0.579, 0.434, 0.206, 0.114, 0.068, 0.049, 0.028, 0.027, 0.03, 0.03, 0.042, 0.027, 0.031, 0.042, 0.029, 0.035, 0.072]

# --- Scaling ---
# Total System Peak Load = 1200 MW to fit grid constraints (400 MW lines)
# Ratios from CSV: Wind ~31%, PV ~36% of Peak Load
TOTAL_PEAK_LOAD = 1200
WIND_CAPACITY = 370
PV_CAPACITY = 430

# Load Distribution: Central nodes (N2, N3, N4) take 30% each, peripheral (N1, N5) 5%
NODE_LOAD_SHARE = {'N1': 0.05, 'N2': 0.30, 'N3': 0.30, 'N4': 0.30, 'N5': 0.05}

for t_idx, t in enumerate(TIME):
    for n in NODES:
        # Load
        base_load = TOTAL_PEAK_LOAD * NODE_LOAD_SHARE[n]
        LOAD_PROFILE[(n,t)] = base_load * daily_pattern[t_idx]
        
        # RES
        res = 0
        if n == 'N1': res = WIND_CAPACITY * wind_pattern[t_idx]
        if n == 'N3': res = PV_CAPACITY * pv_pattern[t_idx]
        RES_PROFILE[(n,t)] = res

INV_DATA = {'I1': {'r': 0.08}, 'I2': {'r': 0.12}, 'I3': {'r': 0.15}, 'I4': {'r': 0.20}}


# ==============================================================================
# 2. MPEC MODELL
# ==============================================================================

def create_mpec_for_investor(current_investor, other_investments_power, other_investments_energy,
                             adj_load, other_p_charge, other_p_discharge, other_r_up, other_r_down,
                             current_own_investments=None):
    m = pyo.ConcreteModel()
    
    m.N = pyo.Set(initialize=NODES)
    m.T = pyo.Set(initialize=TIME)
    m.B = pyo.Set(initialize=BLOCKS)
    m.L = pyo.Set(initialize=LINES)
    m.G = pyo.Set(initialize=GENS)
    m.I_Others = pyo.Set(initialize=[i for i in INVESTORS if i != current_investor])
    
    r_i = INV_DATA[current_investor]['r']
    ETA = 0.936 
    
    def init_power_rule(model, n):
        if current_own_investments: return max(0.1, current_own_investments.get(n, 0.1))
        return 1.0
        
    # --- Variablen ---
    m.X_power = pyo.Var(m.N, within=pyo.NonNegativeReals, initialize=init_power_rule)
    m.X_energy = pyo.Var(m.N, within=pyo.NonNegativeReals, initialize=lambda m,n: init_power_rule(m,n)*1.0)
    m.slack_shared = pyo.Var(m.N, within=pyo.NonNegativeReals, initialize=0)

    # Spot & Speicher
    m.P_gen = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)
    m.P_charge = pyo.Var(INVESTORS, m.N, m.T, within=pyo.NonNegativeReals)
    m.P_discharge = pyo.Var(INVESTORS, m.N, m.T, within=pyo.NonNegativeReals)
    m.SOC = pyo.Var(INVESTORS, m.N, m.T, within=pyo.NonNegativeReals, initialize=0) 
    
    # aFRR + Deficit (Neu in v14) - CONDITIONALLY ADDED
    if CALCULATE_AFRR:
        m.R_up = pyo.Var(INVESTORS, m.N, m.B, within=pyo.NonNegativeReals, initialize=0)
        m.R_down = pyo.Var(INVESTORS, m.N, m.B, within=pyo.NonNegativeReals, initialize=0)
    else:
        # Dummy Vars / Params to avoid errors in calls, mapped to 0
        pass 
        
    # --- Residual Demand fix: Konkurrenten agieren nur als fester Last-Shift im adj_load ---
    # Daher entfernen wir die Fixierung und nutzen adj_load in der Bilanz.
    # Wir erstellen P_charge/SOC Constraints etc. nur noch für den aktuellen Investor!
    m.I_Active = pyo.Set(initialize=[current_investor])
    
    m.NetInjection = pyo.Var(m.N, m.T, within=pyo.Reals)
    m.Flow = pyo.Var(m.L, m.T, within=pyo.Reals)
    
    m.RES_Curtail = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0)
    if not ENABLE_CURTAILMENT:
        m.RES_Curtail.fix(0)
    
    # Duals
    # Spotpreis Boden ist jetzt physikalisch bei 0€ (da wir Curtailment erlauben und MC=0 für RES gilt)
    m.lambda_spot = pyo.Var(m.N, m.T, within=pyo.Reals, bounds=(0.0, 3000.0), initialize=30)
    m.lambda_sys = pyo.Var(m.T, within=pyo.Reals, bounds=(-500, 3000), initialize=30)
    
    # aFRR Preise (Cournot Inverse Demand Curve)
    if CALCULATE_AFRR:
        def afrr_price_up_rule(m, b):
            own_up = sum(m.R_up[current_investor, n, b] for n in m.N)
            other_up = sum(other_r_up.get((i_oth, n, b), 0) for i_oth in m.I_Others for n in m.N)
            return AFRR_PENALTY_PRICE * (1 - (own_up + other_up) / AFRR_DEMAND_UP[b])
        m.afrr_price_up = pyo.Expression(m.B, rule=afrr_price_up_rule)
        
        def afrr_price_down_rule(m, b):
            own_dn = sum(m.R_down[current_investor, n, b] for n in m.N)
            other_dn = sum(other_r_down.get((i_oth, n, b), 0) for i_oth in m.I_Others for n in m.N)
            return AFRR_PENALTY_PRICE * (1 - (own_dn + other_dn) / AFRR_DEMAND_DOWN[b])
        m.afrr_price_down = pyo.Expression(m.B, rule=afrr_price_down_rule)
        
    m.mu_line = pyo.Var(m.L, m.T, within=pyo.NonNegativeReals, initialize=0)
    m.mu_gen = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals, initialize=0)
    
    # --- Zielfunktion ---
    def obj_rule(m):
        rev_spot = sum(m.lambda_spot[n, t] * (m.P_discharge[current_investor, n, t] - m.P_charge[current_investor, n, t]) 
                      for n in m.N for t in m.T)
        
        rev_afrr = 0
        if CALCULATE_AFRR:
            rev_afrr = sum(HOURS_PER_BLOCK * (
                            m.afrr_price_up[b] * sum(m.R_up[current_investor, n, b] for n in m.N) + 
                            m.afrr_price_down[b] * sum(m.R_down[current_investor, n, b] for n in m.N)
                        ) for b in m.B)
        
        total_investment = sum(BESS_COST_POWER * m.X_power[n] + BESS_COST_ENERGY * m.X_energy[n] for n in m.N)
        
        # Annuitätenmethode: Capital Recovery Factor (CRF)
        crf = (r_i * (1 + r_i)**LIFETIME) / ((1 + r_i)**LIFETIME - 1)
        annual_capex = total_investment * crf
        daily_capex = annual_capex / DAYS_PER_YEAR
        
        penalty = sum(m.slack_shared[n] for n in m.N) * 1e6 
        wear_cost = sum((m.P_charge[current_investor, n, t] + m.P_discharge[current_investor, n, t]) * 0.5 * DEGRADATION_COST
                        for n in m.N for t in m.T)
        curtail_penalty = sum(m.RES_Curtail[n,t] for n in m.N for t in m.T) * 0.01

        return (rev_spot + rev_afrr - wear_cost - daily_capex - penalty - curtail_penalty) * OBJ_SCALE 
    m.Obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # --- Constraints ---
    
    def shared_cap_rule(m, n):
        others = sum(other_investments_power[(i, n)] for i in m.I_Others)
        return m.X_power[n] + others <= SHARED_LIMITS[n] + m.slack_shared[n]
    m.SharedCap = pyo.Constraint(m.N, rule=shared_cap_rule)
    m.EPRatio = pyo.Constraint(m.N, rule=lambda m, n: m.X_energy[n] >= EP_RATIO * m.X_power[n])

    def net_inj_rule(m, n, t):
        g = sum(m.P_gen[g, t] for g in m.G if GEN_DATA[g]['node'] == n)
        # Nur NOCH der eigene Speicher interagiert hier dynamisch. 
        # Das Verhalten der anderen steckt nun fix in adj_load!
        b_own = m.P_discharge[current_investor, n, t] - m.P_charge[current_investor, n, t]
        res_feedin = RES_PROFILE[(n,t)] - m.RES_Curtail[n,t]
        # NetInjection: Generation + Eigene Batterie + RES - Modifizierte Last
        return m.NetInjection[n,t] == g + res_feedin + b_own - adj_load[(n,t)]
    m.DefNetInj = pyo.Constraint(m.N, m.T, rule=net_inj_rule)
    
    m.SysBalance = pyo.Constraint(m.T, rule=lambda m, t: sum(m.NetInjection[n,t] for n in m.N) == 0)
    m.DefFlow = pyo.Constraint(m.L, m.T, rule=lambda m, l, t: m.Flow[l,t] == sum(PTDF[(l,n)] * m.NetInjection[n,t] for n in m.N))
    
    # BESS Operation (ONLY FOR ACTIVE INVESTOR)
    def bess_ch_rule(m, i, n, t):
        block = TIME_TO_BLOCK[t]
        limit = m.X_power[n]
        reserve = m.R_down[i,n,block] if CALCULATE_AFRR else 0
        return m.P_charge[i,n,t] + reserve <= limit
    m.C_BessCh = pyo.Constraint(m.I_Active, m.N, m.T, rule=bess_ch_rule)
    
    def bess_dis_rule(m, i, n, t):
        block = TIME_TO_BLOCK[t]
        limit = m.X_power[n]
        reserve = m.R_up[i,n,block] if CALCULATE_AFRR else 0
        return m.P_discharge[i,n,t] + reserve <= limit
    m.C_BessDis = pyo.Constraint(m.I_Active, m.N, m.T, rule=bess_dis_rule)

    def soc_balance_rule(m, i, n, t):
        prev_soc = m.SOC[i, n, 24] if t == 1 else m.SOC[i, n, t-1]
        return m.SOC[i,n,t] == prev_soc + (m.P_charge[i,n,t] * ETA) - (m.P_discharge[i,n,t] / ETA)
    m.C_SOC_Balance = pyo.Constraint(m.I_Active, m.N, m.T, rule=soc_balance_rule)
    
    def soc_limit_rule(m, i, n, t):
        cap = m.X_energy[n]
        return m.SOC[i,n,t] <= cap
    m.C_SOC_Limit = pyo.Constraint(m.I_Active, m.N, m.T, rule=soc_limit_rule)
    
    # --- KKTs (Markt Clearing) ---
    def add_compl(model, name, dual_func, slack_expr, index_set):
        slack = pyo.Var(index_set, within=pyo.NonNegativeReals, initialize=1.0)
        setattr(model, f"slack_{name}", slack)
        def slack_def(m, i): return slack_expr(m, i) == getattr(m, f"slack_{name}")[i]
        setattr(model, f"def_slack_{name}", pyo.Constraint(index_set, rule=slack_def))
        def compl_con(m, i): 
            return dual_func(i) * getattr(m, f"slack_{name}")[i] <= COMPL_RELAX
        setattr(model, f"compl_{name}", pyo.Constraint(index_set, rule=compl_con))

    # 1. Netz-Restriktionen
    for l_id in LINES:
        add_compl(m, f"LineMax_{l_id}", 
                  lambda t, l=l_id: m.mu_line[l, t], 
                  lambda m, t, l=l_id: LINE_LIMITS[l] - m.Flow[l, t], m.T)

    # 2. Generator Restriktionen
    for g in m.G:
        add_compl(m, f"GenMax_{g}", 
                  lambda t, g_val=g: m.mu_gen[g_val, t],
                  lambda m, t, g_val=g: GEN_DATA[g_val]['pmax'] - m.P_gen[g_val, t], m.T)

    # 3. aFRR Markt (Cournot Logic)
    # Preisbildung erfolgt direkt über die pyo.Expression m.afrr_price_up/down in der Zielfunktion.
    # Keine fehleranfälligen KKTs mehr nötig!

    def stat_inj(m, n, t):
        congestion = sum(PTDF[(l,n)] * m.mu_line[l, t] for l in m.L)
        return m.lambda_spot[n,t] == m.lambda_sys[t] - congestion
    m.KKT_Stat_Inj = pyo.Constraint(m.N, m.T, rule=stat_inj)
    
    def compl_gen_active(m, g, t):
        n = GEN_DATA[g]['node']
        term = GEN_DATA[g]['mc'] - m.lambda_spot[n,t] + m.mu_gen[g,t]
        return m.P_gen[g,t] * term <= COMPL_RELAX
    m.Compl_Gen_Active = pyo.Constraint(m.G, m.T, rule=compl_gen_active)

    m.DualFeasGen = pyo.Constraint(m.G, m.T, rule=lambda m,g,t: GEN_DATA[g]['mc'] - m.lambda_spot[GEN_DATA[g]['node'],t] + m.mu_gen[g,t] >= -0.1)

    if ENABLE_CURTAILMENT:
        m.C_MaxCurtail = pyo.Constraint(m.N, m.T, rule=lambda m,n,t: m.RES_Curtail[n,t] <= RES_PROFILE[(n,t)])
        
        # KKT Bedingung für ökonomisches Curtailment: (Preis * Curtailment = 0)
        # Verhindert, dass Investoren böswillig Wind abregeln, um Kraftwerke zu starten.
        # Wir limitieren den Exploit auf 1.0 (sehr strikt), sodass bei Preisen > 0 kein Curtailment möglich ist.
        def compl_res_curtail_rule(m, n, t):
            return m.RES_Curtail[n,t] * m.lambda_spot[n,t] <= 1.0
        m.Compl_Res_Curtail = pyo.Constraint(m.N, m.T, rule=compl_res_curtail_rule)

    return m

# ==============================================================================
# 3. MAIN (MIT FIX FÜR PLOTS)
# ==============================================================================

def get_solver():
    """Helper to get ipopt solver with correct path"""
    import os
    executable = 'ipopt' # default fallback to PATH
    
    # Potential paths to check
    possible_paths = [
        r"C:\Users\Daniel\miniforge3\envs\pyomo_env\Library\bin\ipopt.exe", # Pfad zu CyIpopt von miniforge
        #os.path.join(os.path.dirname(__file__), '.venv', 'Scripts', 'ipopt.exe'), # Alter Pfad zum .venv Ipopt
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            executable = path
            break
        
    # Force 'nl' interface to ensure we use the executable (and standard options dict)
    # instead of potentially falling back to a direct cyipopt binding which lacks .options
    solver = pyo.SolverFactory('ipopt', executable=executable, solver_io='nl')
    
    # --- Parallelization for HSL MA97 ---
    # Set OMP_NUM_THREADS to utilize multiple cores (Ryzen 3900X has 12/24 cores, using 16 is safe/good)
    os.environ['OMP_NUM_THREADS'] = '16'  
    
    solver.options['max_iter'] = 3000
    solver.options['tol'] = 1e-2
    solver.options['linear_solver'] = 'ma97' # User requested HSL MA97
    # solver.options['print_level'] = 0 
    return solver

def solve_single_investor_process(inv, x_power_global_copy, x_energy_global_copy,
                                  p_charge_copy, p_dis_copy, r_up_copy, r_down_copy,
                                  iteration, damping_factor):
    """
    Worker function for parallel execution (Jacobi).
    Solves for one investor based on the copy of global state.
    """
    my_current_vals = {n: x_power_global_copy[(inv, n)] for n in NODES}
    
    # Calculate Residual Load for this specific investor
    adj_load = LOAD_PROFILE.copy()
    for other_inv in INVESTORS:
        if other_inv != inv:
            for n in NODES:
                for t in TIME:
                    # Wenn Kollege lädt -> mehr Systemlast (aus Sicht des Restmarktes)
                    adj_load[(n,t)] += p_charge_copy.get((other_inv, n, t), 0)
                    # Wenn Kollege entlädt -> weniger Systemlast
                    adj_load[(n,t)] -= p_dis_copy.get((other_inv, n, t), 0)

    model = create_mpec_for_investor(inv, x_power_global_copy, x_energy_global_copy,
                                     adj_load, p_charge_copy, p_dis_copy, r_up_copy, r_down_copy,
                                     my_current_vals)
    
    solver = get_solver()
    solver.options['print_level'] = 0 
    
    status = 'error'
    try:
        results = solver.solve(model, tee=False)
        status = results.solver.termination_condition
    except Exception as e:
        # print(f"Error solving {inv}: {e}")
        status = 'error'
        
    result_data = {
        'inv': inv,
        'status': status,
        'new_vals': {},
        'current_change': 0,
        'slack_used': 0,
        'def_up': 0
    }

    if status in [TerminationCondition.optimal, TerminationCondition.maxIterations]:
        # Slack Check
        try: result_data['slack_used'] = sum(pyo.value(model.slack_shared[n]) for n in NODES)
        except: pass
        
        # Deficit Check
        if CALCULATE_AFRR:
            try: result_data['def_up'] = sum(pyo.value(model.Deficit_Up[b]) for b in BLOCKS)
            except: pass
        
        max_diff = 0
        for n in NODES:
            # Power Difference
            old_p = x_power_global_copy[(inv, n)]
            try: new_p = pyo.value(model.X_power[n])
            except: new_p = old_p
            new_p = max(0, new_p)
            damped_p = (1 - damping_factor) * old_p + damping_factor * new_p
            result_data['new_vals'][n] = damped_p
            
            diff_p = abs(damped_p - old_p)
            
            # Energy Difference
            old_e = x_energy_global_copy[(inv, n)]
            try: new_e = pyo.value(model.X_energy[n])
            except: new_e = old_e
            new_e = max(0, new_e)
            damped_e = (1 - damping_factor) * old_e + damping_factor * new_e
            if 'new_vals_e' not in result_data: result_data['new_vals_e'] = {}
            result_data['new_vals_e'][n] = damped_e
            
            diff_e = abs(damped_e - old_e)
            
            # Apply switch logic
            if CONVERGENCE_CRITERIA == 'Power' and diff_p > max_diff:
                max_diff = diff_p
            elif CONVERGENCE_CRITERIA == 'Energy' and diff_e > max_diff:
                max_diff = diff_e
            elif CONVERGENCE_CRITERIA == 'PowerAndEnergy':
                if diff_p > max_diff: max_diff = diff_p
                if diff_e > max_diff: max_diff = diff_e
                
        # --- Extract and damp operational variables ---
        result_data['p_charge'] = {}
        result_data['p_dis'] = {}
        result_data['r_up'] = {}
        result_data['r_down'] = {}
        
        for n in NODES:
            for t in TIME:
                old_ch = p_charge_copy.get((inv, n, t), 0)
                try: new_ch = pyo.value(model.P_charge[inv, n, t])
                except: new_ch = old_ch
                damped_ch = (1 - damping_factor) * old_ch + damping_factor * new_ch
                result_data['p_charge'][(inv, n, t)] = damped_ch
                
                old_dis = p_dis_copy.get((inv, n, t), 0)
                try: new_dis = pyo.value(model.P_discharge[inv, n, t])
                except: new_dis = old_dis
                damped_dis = (1 - damping_factor) * old_dis + damping_factor * new_dis
                result_data['p_dis'][(inv, n, t)] = damped_dis
            
            if CALCULATE_AFRR:
                for b in BLOCKS:
                    old_up = r_up_copy.get((inv, n, b), 0)
                    try: new_up = pyo.value(model.R_up[inv, n, b])
                    except: new_up = old_up
                    damped_up = (1 - damping_factor) * old_up + damping_factor * new_up
                    result_data['r_up'][(inv, n, b)] = damped_up
                    
                    old_down = r_down_copy.get((inv, n, b), 0)
                    try: new_down = pyo.value(model.R_down[inv, n, b])
                    except: new_down = old_down
                    damped_down = (1 - damping_factor) * old_down + damping_factor * new_down
                    result_data['r_down'][(inv, n, b)] = damped_down

        result_data['current_change'] = max_diff
        
    return result_data

def solve_epec():
    x_power_global = {(i, n): 1.0 for i in INVESTORS for n in NODES}
    x_energy_global = {(i, n): 1.0 for i in INVESTORS for n in NODES}
    
    # Global state for operations
    p_charge_global = {(i, n, t): 0.0 for i in INVESTORS for n in NODES for t in TIME}
    p_dis_global = {(i, n, t): 0.0 for i in INVESTORS for n in NODES for t in TIME}
    r_up_global = {(i, n, b): 0.0 for i in INVESTORS for n in NODES for b in BLOCKS}
    r_down_global = {(i, n, b): 0.0 for i in INVESTORS for n in NODES for b in BLOCKS}
    convergence_history = []
    print(f"Start EPEC v14 (Stabilized). Iter: {MAX_ITER}, Damping: {DAMPING_FACTOR}")
    
    for it in range(MAX_ITER):
        print(f"\n--- Iteration {it+1} ---")
        max_diff = 0
        all_ok = True
        
        # --- PARALLEL (GAUSS-JACOBI) ---
        if GAUSS_JACOBI:
            # Create a copy of the global state for the workers
            # (In Jacobi, everyone ignores peer updates within the same iteration)
            x_power_copy = x_power_global.copy()
            x_energy_copy = x_energy_global.copy()
            p_charge_copy = p_charge_global.copy()
            p_dis_copy = p_dis_global.copy()
            r_up_copy = r_up_global.copy()
            r_down_copy = r_down_global.copy()
            
            results_list = []
            # Use max_workers=len(INVESTORS) or limit to e.g. 4
            with ProcessPoolExecutor(max_workers=len(INVESTORS)) as executor:
                # Submit tasks
                futures = {
                    executor.submit(solve_single_investor_process, inv, x_power_copy, x_energy_copy,
                                    p_charge_copy, p_dis_copy, r_up_copy, r_down_copy,
                                    it, DAMPING_FACTOR): inv 
                    for inv in INVESTORS
                }
                
                # Collect results
                for future in as_completed(futures):
                    inv = futures[future]
                    try:
                        res = future.result()
                        results_list.append(res)
                    except Exception as exc:
                        print(f"  Inv {inv} generated an exception: {exc}")
                        all_ok = False

            # Update Global State (Synchronous Update)
            results_list.sort(key=lambda x: x['inv']) # Sort for consistent print order
            for res in results_list:
                inv = res['inv']
                status = res['status']
                
                if status in [TerminationCondition.optimal, TerminationCondition.maxIterations]:
                    # Print Status
                    print(f"  Inv {inv}: {status}. Max Delta: {res['current_change']:.3f} (Def: {res['def_up']:.2f})")
                    
                    if res['def_up'] > 0.1: print(f"    WARNUNG: aFRR Deficit Up: {res['def_up']:.2f} MW")
                    
                    # Apply updates
                    for n, val in res['new_vals'].items():
                        x_power_global[(inv, n)] = val
                    if 'new_vals_e' in res:
                        for n, val in res['new_vals_e'].items():
                            x_energy_global[(inv, n)] = val
                            
                    # Update operational state
                    p_charge_global.update(res.get('p_charge', {}))
                    p_dis_global.update(res.get('p_dis', {}))
                    r_up_global.update(res.get('r_up', {}))
                    r_down_global.update(res.get('r_down', {}))
                        
                    if res['current_change'] > max_diff: max_diff = res['current_change']
                else:
                    print(f"  Inv {inv}: FEHLGESCHLAGEN ({status})")
                    all_ok = False

        # --- SEQUENTIAL (GAUSS-SEIDEL) ---
        else:
            for inv in INVESTORS:
                my_current_vals = {n: x_power_global[(inv, n)] for n in NODES}
                
                # Residual Load Update (Live)
                adj_load = LOAD_PROFILE.copy()
                for other_inv in INVESTORS:
                    if other_inv != inv:
                        for n in NODES:
                            for t in TIME:
                                adj_load[(n,t)] += p_charge_global.get((other_inv, n, t), 0)
                                adj_load[(n,t)] -= p_dis_global.get((other_inv, n, t), 0)
                
                model = create_mpec_for_investor(inv, x_power_global, x_energy_global,
                                                 adj_load, p_charge_global, p_dis_global, 
                                                 r_up_global, r_down_global, 
                                                 my_current_vals)
                
                solver = get_solver()
                solver.options['print_level'] = 0 
                
                try:
                    results = solver.solve(model, tee=False)
                    status = results.solver.termination_condition
                except: status = 'error'

                if status in [TerminationCondition.optimal, TerminationCondition.maxIterations]:
                    def_up = 0
                    if CALCULATE_AFRR:
                        def_up = sum(pyo.value(model.Deficit_Up[b]) for b in BLOCKS)
                        if def_up > 0.1: print(f"  WARNUNG: aFRR Deficit Up: {def_up:.2f} MW")
                    
                    current_change = 0
                    for n in NODES:
                        # Power Update
                        old_p = x_power_global[(inv, n)]
                        try: new_p = pyo.value(model.X_power[n])
                        except: new_p = old_p
                        new_p = max(0, new_p)
                        damped_p = (1 - DAMPING_FACTOR) * old_p + DAMPING_FACTOR * new_p
                        x_power_global[(inv, n)] = damped_p
                        diff_p = abs(damped_p - old_p)
                        
                        # Energy Update
                        old_e = x_energy_global[(inv, n)]
                        try: new_e = pyo.value(model.X_energy[n])
                        except: new_e = 0
                        new_e = max(0, new_e)
                        damped_e = (1 - DAMPING_FACTOR) * old_e + DAMPING_FACTOR * new_e
                        x_energy_global[(inv, n)] = damped_e
                        diff_e = abs(damped_e - old_e)
                        
                        # Switch Logic
                        if CONVERGENCE_CRITERIA == 'Power' and diff_p > current_change:
                            current_change = diff_p
                        elif CONVERGENCE_CRITERIA == 'Energy' and diff_e > current_change:
                            current_change = diff_e
                        elif CONVERGENCE_CRITERIA == 'PowerAndEnergy':
                            if diff_p > current_change: current_change = diff_p
                            if diff_e > current_change: current_change = diff_e
                    
                    # Update global operational states directlly
                    for n in NODES:
                        for t in TIME:
                            old_ch = p_charge_global.get((inv, n, t), 0)
                            try: new_ch = pyo.value(model.P_charge[inv, n, t])
                            except: new_ch = old_ch
                            p_charge_global[(inv, n, t)] = (1 - DAMPING_FACTOR) * old_ch + DAMPING_FACTOR * new_ch
                            
                            old_dis = p_dis_global.get((inv, n, t), 0)
                            try: new_dis = pyo.value(model.P_discharge[inv, n, t])
                            except: new_dis = old_dis
                            p_dis_global[(inv, n, t)] = (1 - DAMPING_FACTOR) * old_dis + DAMPING_FACTOR * new_dis
                        
                        if CALCULATE_AFRR:
                            for b in BLOCKS:
                                old_up = r_up_global.get((inv, n, b), 0)
                                try: new_up = pyo.value(model.R_up[inv, n, b])
                                except: new_up = old_up
                                r_up_global[(inv, n, b)] = (1 - DAMPING_FACTOR) * old_up + DAMPING_FACTOR * new_up
                                
                                old_down = r_down_global.get((inv, n, b), 0)
                                try: new_down = pyo.value(model.R_down[inv, n, b])
                                except: new_down = old_down
                                r_down_global[(inv, n, b)] = (1 - DAMPING_FACTOR) * old_down + DAMPING_FACTOR * new_down
                    
                    if current_change > max_diff: max_diff = current_change
                    print(f"  Inv {inv}: {status}. Max Delta: {current_change:.3f} (Def: {def_up:.2f})")
                else:
                    print(f"  Inv {inv}: FEHLGESCHLAGEN ({status})")
                    all_ok = False

        convergence_history.append(max_diff)

        if max_diff < TOLERANCE and all_ok:
            print(f"\nKONVERGENZ! Delta < {TOLERANCE}")
            break
    return x_power_global, x_energy_global, p_charge_global, p_dis_global, r_up_global, r_down_global, convergence_history

# --- Visualisierung ---
def plot_inputs_separate():
    df_load = pd.DataFrame(index=TIME)
    df_res = pd.DataFrame(index=TIME)
    for n in NODES:
        df_load[n] = [LOAD_PROFILE[(n, t)] for t in TIME]
        df_res[n] = [RES_PROFILE[(n, t)] for t in TIME]
    
    plt.figure(figsize=(10, 4))
    
    # Gruppierung gleicher Lastprofile für eine saubere Legende
    plot_groups = []
    colors = ['tab:blue','tab:orange']
    c_idx = 0
    
    for n in NODES:
        prof = df_load[n].tolist()
        found = False
        for g in plot_groups:
            if g['profile'] == prof:
                g['nodes'].append(n)
                found = True
                break
        if not found:
            plot_groups.append({'profile': prof, 'nodes': [n], 'color': colors[c_idx % len(colors)]})
            c_idx += 1
            
    for g in plot_groups:
        label = ", ".join(g['nodes'])
        plt.plot(df_load.index, g['profile'], marker='.', alpha=0.7, color=g['color'], label=label)

    plt.title("Input: Load Profiles")
    plt.ylabel("Load [MW]")
    plt.xlabel("Time [h]")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/Inputs_Load_Profiles.pdf')
    plt.close()
    
    cols_res = [c for c in df_res.columns if df_res[c].sum() > 0]
    if cols_res:
        plt.figure(figsize=(10, 4))
        plt.plot(df_res[cols_res], marker='.', alpha=0.7)
        plt.title("Input: Renewable Generation Profiles")
        plt.ylabel("Generation [MW]")
        plt.xlabel("Time [h]")
        plt.legend(cols_res, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/Inputs_RES_Profiles.pdf')
        plt.close()

def plot_grid_enhanced():
    G = nx.Graph()
    G.add_nodes_from(NODES)
    edge_labels, edge_colors = {}, []
    
    for line in RAW_LINES:
        u, v = line['from'], line['to']
        G.add_edge(u, v)
        edge_labels[(u,v)] = f"{line['limit']}MW"
        edge_colors.append('red' if line['limit'] < 300 else 'black')
        
    node_labels, node_colors = {}, []
    for n in NODES:
        limit_val = SHARED_LIMITS[n]
        label_text = f"{n}\n{limit_val} MW"
        components = [g for g in GENS if GEN_DATA[g]['node'] == n]
        res_type = NODE_RES_TYPE.get(n, 'None')
        if res_type != 'None':
            components.append(res_type)
            
        if components:
            c_names = "\n".join(components)
            label_text += f"\n{c_names}"
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgreen')
        node_labels[n] = label_text

    pos = {'N1': (0.5, 1.0), 'N2': (0.0, 0.7), 'N3': (0.2, 0.3), 'N4': (0.8, 0.3), 'N5': (0.5, 0.0)}
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors, edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_color='black', font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Grid topology: Node and line limits")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('output/Grid_Topology.pdf')
    plt.close()


def plot_bess_detail(investor, node, final_inv, final_inv_energy, p_ch_g, p_dis_g, r_up_g, r_down_g):
    print(f"\n--- Detail-Analyse für {investor} an {node} ---")
    
    adj_load = LOAD_PROFILE.copy()
    for other_inv in INVESTORS:
        if other_inv != investor:
            for n in NODES:
                for t in TIME:
                    adj_load[(n,t)] += p_ch_g.get((other_inv, n, t), 0)
                    adj_load[(n,t)] -= p_dis_g.get((other_inv, n, t), 0)
                    
    model = create_mpec_for_investor(investor, final_inv, final_inv_energy, adj_load, p_ch_g, p_dis_g, r_up_g, r_down_g, None)
    my_cap = final_inv[(investor, node)]
    if my_cap < 0.1: return
    
    # FIX: EP_RATIO beachten
    model.X_power[node].fix(my_cap)
    model.X_energy[node].fix(final_inv[(investor, node)] * EP_RATIO)
    
    solver = get_solver()
    solver.solve(model, tee=False)
    
    hours = list(TIME)
    soc, power, spot_price = [], [], []
    afrr_up_bid, afrr_down_bid, afrr_price_up, afrr_price_down = [], [], [], []
    curtail = []
    
    for t in TIME:
        b = TIME_TO_BLOCK[t]
        # SOC berechnung
        e_cap = pyo.value(model.X_energy[node])
        soc_val = (pyo.value(model.SOC[investor, node, t]) / e_cap * 100) if e_cap > 0 else 0
        soc.append(soc_val)
        
        power.append(pyo.value(model.P_discharge[investor, node, t]) - pyo.value(model.P_charge[investor, node, t]))
        spot_price.append(pyo.value(model.lambda_spot[node, t]))
        
        if CALCULATE_AFRR:
            afrr_up_bid.append(pyo.value(model.R_up[investor, node, b]))
            afrr_down_bid.append(-1 * pyo.value(model.R_down[investor, node, b])) 
            afrr_price_up.append(pyo.value(model.lambda_afrr_up[b]))
            afrr_price_down.append(pyo.value(model.lambda_afrr_down[b]))
        else:
             afrr_up_bid.append(0)
             afrr_down_bid.append(0)
             afrr_price_up.append(0)
             afrr_price_down.append(0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Preise
    line1 = ax1.plot(hours, spot_price, label='Spot Price', color='black')
    ax1.set_ylabel('Spot (€/MWh)')
    ax1.grid(True, alpha=0.3)
    
    ax1t = ax1.twinx()
    line2 = ax1t.plot(hours, afrr_price_up, label='aFRR Up (€/MW)', color='orange', linestyle='--')
    line3 = ax1t.plot(hours, afrr_price_down, label='aFRR Down (€/MW)', color='purple', linestyle='--')
    ax1t.set_ylabel('aFRR (€/MW)', color='purple')
    
    lns = line1+line2+line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    ax1.set_title(f"Prices at {node} (with aFRR Cap {AFRR_PENALTY_PRICE}€)")

    # Plot 2: Leistung
    ax2.step(hours, power, where='mid', label='Spot Net', color='blue')
    ub = [p + r for p, r in zip(power, afrr_up_bid)]
    lb = [p + r for p, r in zip(power, afrr_down_bid)]
    ax2.fill_between(hours, power, ub, step='mid', color='orange', alpha=0.3, label='Res Up')
    ax2.fill_between(hours, power, lb, step='mid', color='purple', alpha=0.3, label='Res Down')
    
    ax2.legend(loc='upper right')
    ax2.set_ylabel('MW')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: SOC
    ax3.plot(hours, soc, color='green', marker='.')
    ax3.set_ylabel('SOC %')
    ax3.set_ylim(-5, 105)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'output/{node}_{investor}_Detail.pdf')
    plt.close()

def plot_system_aggregation(final_inv, final_inv_energy, p_ch_g, p_dis_g, r_up_g, r_down_g):
    print("Berechne System-Aggregation...")
    
    # Für die Systemaggregation brauchen wir ein Modell, das alle Speicher als Bestandteil des Marktes sieht.
    # Behelfsweise bauen wir das Modell für 'I1' und fixieren alle Variablen von I1 exakt,
    # während wir die Residuallast wie immer durch die anderen berechnen lassen.
    adj_load = LOAD_PROFILE.copy()
    for other_inv in INVESTORS:
        if other_inv != 'I1':
            for n in NODES:
                for t in TIME:
                    adj_load[(n,t)] += p_ch_g.get((other_inv, n, t), 0)
                    adj_load[(n,t)] -= p_dis_g.get((other_inv, n, t), 0)
                    
    model = create_mpec_for_investor('I1', final_inv, final_inv_energy, adj_load, p_ch_g, p_dis_g, r_up_g, r_down_g, None)
    for n in NODES:
        model.X_power[n].fix(final_inv[('I1', n)])
        model.X_energy[n].fix(final_inv[('I1', n)] * EP_RATIO)
    
    solver = get_solver()
    solver.solve(model, tee=False)
    
    system_data = {'Load': [], 'RES_Potential': [], 'RES_Actual': [], 'Curtailed_RES': [], 'Conv': [], 'BESS_Net': []}
    for t in TIME:
        total_load = sum(LOAD_PROFILE[(n, t)] for n in NODES)
        total_res_pot = sum(RES_PROFILE[(n, t)] for n in NODES)
        
        # Filter Curtailment Noise
        if ENABLE_CURTAILMENT:
            clean_curtails = [pyo.value(model.RES_Curtail[n,t]) if pyo.value(model.RES_Curtail[n,t]) >= 0.1 else 0 for n in NODES]
            total_res_act = sum(RES_PROFILE[(n, t)] - c for n, c in zip(NODES, clean_curtails))
        else:
            total_res_act = total_res_pot
            
        total_curtail = total_res_pot - total_res_act
        
        # Filter Generator KKT Slack Noise (< 1.0 MW)
        total_conv = sum(pyo.value(model.P_gen[g, t]) if pyo.value(model.P_gen[g, t]) >= 1.0 else 0 for g in GENS)
        
        total_bess = 0
        for i in INVESTORS:
            for n in NODES:
                if i == 'I1':
                    total_bess += (pyo.value(model.P_discharge[i, n, t]) - pyo.value(model.P_charge[i, n, t]))
                else:
                    total_bess += (p_dis_g.get((i, n, t), 0) - p_ch_g.get((i, n, t), 0))
        
        system_data['Load'].append(total_load)
        system_data['RES_Potential'].append(total_res_pot)
        system_data['RES_Actual'].append(total_res_act)
        system_data['Curtailed_RES'].append(total_curtail)
        system_data['Conv'].append(total_conv)
        system_data['BESS_Net'].append(total_bess)
        
    df_sys = pd.DataFrame(system_data, index=TIME)
    plt.figure(figsize=(10, 6))
    plt.plot(df_sys.index, df_sys['Load'], color='black', linewidth=2, label='Total Load', linestyle='--')
    plt.plot(df_sys.index, df_sys['RES_Potential'], color='green', linestyle=':', label='RES Potential')
    plt.plot(df_sys.index, df_sys['RES_Actual'], color='green', label='Total RES Actual')
    
    if ENABLE_CURTAILMENT:
        plt.fill_between(df_sys.index, df_sys['RES_Actual'], df_sys['RES_Potential'], color='red', alpha=0.3, hatch='//', label='Curtailed RES')
        
    plt.plot(df_sys.index, df_sys['Conv'], color='red', label='Total Conventional')
    plt.plot(df_sys.index, df_sys['BESS_Net'], color='blue', linewidth=2, alpha=0.7, label='BESS Net')
    
    plt.title("System Balance (Sum over all nodes)")
    plt.ylabel("Power [MW]")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/System_Balance.pdf')
    plt.close()

def print_scenario_overview():
    print("\n" + "="*60)
    print(f"SZENARIO ÜBERSICHT (Simulationsdauer: {len(TIME)} h)")
    print("="*60)
    print("\n--- Netz-Restriktionen (Leitungen) ---")
    df_lines = pd.DataFrame(list(LINE_LIMITS.items()), columns=['Line', 'Limit (MW)'])
    print(df_lines.to_string(index=False))
    print("\n--- Shared Constraints (Anschluss-Limits) ---")
    df_shared = pd.DataFrame(list(SHARED_LIMITS.items()), columns=['Node', 'MaxSharedCap (MW)'])
    print(df_shared.to_string(index=False))
    print("="*60 + "\n")

def extract_results(final_inv, final_inv_energy, p_ch_g, p_dis_g, r_up_g, r_down_g):
    print("Extrahiere Ergebnisse für CSV und Plots...")
    
    adj_load = LOAD_PROFILE.copy()
    for other_inv in INVESTORS:
        if other_inv != 'I1':
            for n in NODES:
                for t in TIME:
                    adj_load[(n,t)] += p_ch_g.get((other_inv, n, t), 0)
                    adj_load[(n,t)] -= p_dis_g.get((other_inv, n, t), 0)
                    
    model = create_mpec_for_investor('I1', final_inv, final_inv_energy, adj_load, p_ch_g, p_dis_g, r_up_g, r_down_g, None)
    for n in NODES:
        model.X_power[n].fix(final_inv[('I1', n)])
        model.X_energy[n].fix(final_inv_energy[('I1', n)])
    
    solver = get_solver()
    solver.solve(model, tee=False)
    
    data = []
    for t in TIME:
        for n in NODES:
            # Shared Data
            price = pyo.value(model.lambda_spot[n, t])
            load = LOAD_PROFILE[(n, t)]
            res = RES_PROFILE[(n,t)]
            res_act = res
            
            # Filter Generator KKT Slack Noise (< 1.0 MW)
            conv = sum(pyo.value(model.P_gen[g, t]) if pyo.value(model.P_gen[g, t]) >= 1.0 else 0 for g in GENS if GEN_DATA[g]['node'] == n)
            
            # Net Injection & Import/Export
            net_inj = pyo.value(model.NetInjection[n,t])
            # NetInjection = Gen + BESS + RES - Load
            # If NetInjection > 0: Export to Grid (Outflow from node perspective) -> Actually, for the balance view:
            # Sources (Inflow to Node): RES, Conv, Import, BESS_Discharge
            # Sinks (Outflow from Node): Load, Export, BESS_Charge
            
            # Physics: Inflow = Outflow
            # Import + Gen + RES + BESS_Dis = Load + Export + BESS_Ch
            
            imp_mw = max(0, -net_inj) # If NetInj is negative, we import
            exp_mw = max(0, net_inj)  # If NetInj is positive, we export
            # Filter KKT relaxation noise
            raw_curtail = pyo.value(model.RES_Curtail[n,t]) if ENABLE_CURTAILMENT else 0
            clean_curtail = 0 if raw_curtail < 0.1 else raw_curtail

            row = {
                'Time': t,
                'Node': n,
                'Load_MW': load,
                'RES_MW': res,
                'RES_Act_MW': res_act,
                'RES_Curtail_MW': clean_curtail,
                'Conv_MW': conv,
                'Import_MW': imp_mw,
                'Export_MW': exp_mw,
                'Price_EUR': price
            }
            
            b = TIME_TO_BLOCK[t]
            if CALCULATE_AFRR:
                row['aFRR_Price_Up_EUR'] = pyo.value(model.afrr_price_up[b])
                row['aFRR_Price_Down_EUR'] = pyo.value(model.afrr_price_down[b])
            else:
                row['aFRR_Price_Up_EUR'] = 0
                row['aFRR_Price_Down_EUR'] = 0
                
            for l in LINES:
                row[f'Flow_{l}_MW'] = pyo.value(model.Flow[l, t])
                row[f'Limit_{l}_MW'] = LINE_LIMITS[l]
            
            total_bess_p = 0
            # Investor Data
            for i in INVESTORS:
                if i == 'I1':
                    p_ch = pyo.value(model.P_charge[i, n, t])
                    p_dis = pyo.value(model.P_discharge[i, n, t])
                else:
                    p_ch = p_ch_g.get((i, n, t), 0)
                    p_dis = p_dis_g.get((i, n, t), 0)
                    
                net_p = p_dis - p_ch
                total_bess_p += net_p
                
                # SOC calculation
                inv_cap = final_inv[(i,n)]
                if i == 'I1':
                    soc_energy = pyo.value(model.SOC[i, n, t])
                else:
                    # We don't have SOC global explicitly tracked, so we reconstruct it based on charge/discharge
                    # For plotting purposes, we can just use the final extracted SOC if needed, but it's complex.
                    # Or we just leave SOC for competitors as 0 in this specific 'results_final.csv' extraction.
                    soc_energy = 0 # To reconstruct this properly we'd need to simulate or track it.
                    
                soc_perc = (soc_energy / (inv_cap * EP_RATIO) * 100) if inv_cap > 0.01 else 0.0
                
                # aFRR (Block based, mapped to Time)
                b = TIME_TO_BLOCK[t]
                r_up = 0
                r_down = 0
                if CALCULATE_AFRR:
                    if i == 'I1':
                        r_up = pyo.value(model.R_up[i, n, b])
                        r_down = pyo.value(model.R_down[i, n, b])
                    else:
                        r_up = r_up_g.get((i, n, b), 0)
                        r_down = r_down_g.get((i, n, b), 0)

                row[f'{i}_Power_MW'] = net_p
                row[f'{i}_SOC_Perc'] = soc_perc
                row[f'{i}_Ch_MW'] = p_ch
                row[f'{i}_Dis_MW'] = p_dis
                row[f'{i}_R_Up_MW'] = r_up
                row[f'{i}_R_Down_MW'] = r_down
            
            row['Total_BESS_Power_MW'] = total_bess_p
            data.append(row)
            
    return pd.DataFrame(data)

def export_results_to_csv(df, filename='output/results_final.csv'):
    df.to_csv(filename, index=False)
    print(f"Ergebnisse gespeichert in {filename}")

def plot_node_analysis(df):
    hours = df['Time'].unique()
    for n in NODES:
        df_n = df[df['Node'] == n]
        
        # Setup Plot with 2 Subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # --- SUBPLOT 1: System Balance & Price ---
        # Left Axis: Price
        p_price, = ax1.plot(hours, df_n['Price_EUR'], 'k-', label='Price', linewidth=2, zorder=10)
        ax1.set_ylabel('Price (€/MWh)', color='black')
        ax1.grid(True, alpha=0.3)
        
        # Right Axis: Flows (Stacked)
        ax1r = ax1.twinx()
        
        # Prepare Data for Stacking
        # Sources (Positive)
        y_conv = df_n['Conv_MW'].values
        y_res = df_n['RES_Act_MW'].values
        y_imp = df_n['Import_MW'].values
        
        # Calculate NET BESS Flows for the Balance Plot to avoid simultaneous Charge/Discharge visual artifacts
        # We take the Total Net Power (which is consistent with Subplot 2) and split it into Pos (Discharge) and Neg (Charge) parts
        total_bess_net = df_n['Total_BESS_Power_MW'].values
        y_bess_dis_net = np.maximum(0, total_bess_net)
        y_bess_ch_net = np.maximum(0, -total_bess_net) # Positive magnitude for stacking
        
        # BESS Discharge (Net)
        y_bess_dis = y_bess_dis_net
        
        # Sinks (Negative)
        y_load = -1 * df_n['Load_MW'].values
        y_exp = -1 * df_n['Export_MW'].values
        # BESS Charge (Net) - Make negative for plotting
        y_bess_ch = -1 * y_bess_ch_net
        
        # Stackplot logic manually with fill_between for control
        # Positive Stack
        base = np.zeros(len(hours))
        p_conv = ax1r.fill_between(hours, base, base+y_conv, color='brown', alpha=0.3, label='Conventional')
        base += y_conv
        p_res = ax1r.fill_between(hours, base, base+y_res, color='green', alpha=0.3, label='RES')
        base += y_res
        p_imp = ax1r.fill_between(hours, base, base+y_imp, color='blue', alpha=0.2, label='Import (Grid)')
        base += y_imp
        p_bdis = ax1r.fill_between(hours, base, base+y_bess_dis, color='orange', alpha=0.5, label='BESS Discharge')
        
        # Negative Stack
        base_neg = np.zeros(len(hours))
        p_load = ax1r.fill_between(hours, base_neg, base_neg+y_load, color='black', alpha=0.1, label='Load')
        base_neg += y_load
        p_exp = ax1r.fill_between(hours, base_neg, base_neg+y_exp, color='cyan', alpha=0.2, label='Export (Grid)')
        base_neg += y_exp
        p_bch = ax1r.fill_between(hours, base_neg, base_neg+y_bess_ch, color='purple', alpha=0.5, label='BESS Charge')
        
        ax1r.set_ylabel('Power Flow (MW)')
        
        # Legend S1
        lns = [p_price, p_conv, p_res, p_imp, p_bdis, p_load, p_exp, p_bch]
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
        ax1.set_title(f"Output: Node {n} Balance & Prices")

        # --- SUBPLOT 2: BESS Detail ---
        # Individual Lines
        colors = ['red', 'green', 'blue', 'orange']
        for idx, i in enumerate(INVESTORS):
            ax2.plot(hours, df_n[f'{i}_Power_MW'], label=f'{i}', color=colors[idx % len(colors)], linestyle='-', alpha=0.6, linewidth=2)
            
        # Total Sum (Normal Line Width, Dashed)
        ax2.plot(hours, df_n['Total_BESS_Power_MW'], label='Total', color='black', linestyle='--', linewidth=2)
        
        ax2.set_ylabel('BESS Net Power (MW)')
        ax2.set_xlabel('Hour')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1), fontsize='small')
        ax2.axhline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'output/{n}_Balance.pdf')
        plt.close()

def plot_afrr_provision(df):
    if not CALCULATE_AFRR: return
    hours = df['Time'].unique()
    
    for n in NODES:
        df_n = df[df['Node'] == n]
        
        # Check if any aFRR provided
        total_up = sum(df_n[f'{i}_R_Up_MW'].sum() for i in INVESTORS)
        total_down = sum(df_n[f'{i}_R_Down_MW'].sum() for i in INVESTORS)
        
        if total_up < 0.1 and total_down < 0.1:
            print(f"Node {n}: Kein aFRR.")
            continue
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot Up
        base = np.zeros(len(hours))
        colors = ['red', 'green', 'blue', 'orange']
        for idx, i in enumerate(INVESTORS):
            y = df_n[f'{i}_R_Up_MW'].values
            ax1.fill_between(hours, base, base+y, label=f'{i}', color=colors[idx % len(colors)], alpha=0.6)
            base += y
            
        ax1.set_title(f"Output: Node {n} aFRR Provision UP")
        ax1.set_ylabel("MW")
        ax1.set_xticks(range(0, 25, 4))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Plot Down
        base = np.zeros(len(hours))
        for idx, i in enumerate(INVESTORS):
            y = df_n[f'{i}_R_Down_MW'].values
            ax2.fill_between(hours, base, base+y, label=f'{i}', color=colors[idx % len(colors)], alpha=0.6)
            base += y
            
        ax2.invert_yaxis() # Down reserves go "down" visually? Or just positive magnitude? 
        # Requirement: "Leistung(+/-) vorhält". Traditionally reserves are positive quantities.
        # But visualizing them in opposite direction makes sense.
        
        ax2.set_title(f"Output: Node {n} aFRR Provision DOWN")
        ax2.set_ylabel("MW")
        ax2.set_xlabel("Hour")
        ax2.set_xticks(range(0, 25, 4))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'output/{n}_aFRR.pdf')
        plt.close()

def plot_convergence(conv_hist):
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(conv_hist) + 1), conv_hist, marker='o', color='blue')
    plt.axhline(TOLERANCE, color='red', linestyle='--', label=f'Tolerance ({TOLERANCE} MW)')
    plt.yscale('log')
    plt.title("Output: Convergence History (Max Delta per Iteration)")
    plt.ylabel("Max Change (MW)")
    plt.xlabel("Iteration")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/Convergence.pdf')
    plt.close()

def plot_investment_bar_chart(final_inv):
    df = pd.DataFrame(index=NODES, columns=INVESTORS)
    for (i, n), val in final_inv.items(): df.at[n, i] = val
    
    ax = df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab10')
    plt.title("Output: Installed BESS Power Capacity by Node and Investor")
    plt.ylabel("Installed Capacity (MW)")
    plt.xlabel("Node")
    plt.legend(title="Investor", loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/Investment_Capacities.pdf')
    plt.close()

def plot_line_congestion(df):
    hours = df['Time'].unique()
    line_data = []
    line_names = []
    
    for l in LINES:
        df_l = df[df['Node'] == 'N1'] # Line limits/flows are the same per time step regardless of node filter
        flow = df_l[f'Flow_{l}_MW'].values
        limit = df_l[f'Limit_{l}_MW'].values[0]
        
        utilization = np.abs(flow) / limit * 100
        line_data.append(utilization)
        line_names.append(l)
        
    line_data = np.array(line_data)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(line_data, aspect='auto', cmap='Reds', interpolation='none', vmin=0, vmax=100)
    plt.colorbar(label='Line Loading (%)')
    plt.yticks(range(len(line_names)), line_names)
    plt.xticks(range(len(hours)), hours)
    plt.title("Output: Line Congestion Heatmap")
    plt.ylabel("Line")
    plt.xlabel("Hour")
    plt.tight_layout()
    plt.savefig('output/Line_Congestion.pdf')
    plt.close()

def plot_overall_afrr_provision(df):
    if not CALCULATE_AFRR: return
    hours = df['Time'].unique()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = ['red', 'green', 'blue', 'orange']
    
    # Sum over all nodes for each investor
    # groupby 'Time' and sum
    df_grouped = df.groupby('Time').sum()
    
    base_up = np.zeros(len(hours))
    base_down = np.zeros(len(hours))
    
    for idx, i in enumerate(INVESTORS):
        y_up = df_grouped[f'{i}_R_Up_MW'].values
        y_down = df_grouped[f'{i}_R_Down_MW'].values
        
        ax1.fill_between(hours, base_up, base_up+y_up, label=f'{i}', color=colors[idx % len(colors)], alpha=0.6)
        base_up += y_up
        
        ax2.fill_between(hours, base_down, base_down+y_down, label=f'{i}', color=colors[idx % len(colors)], alpha=0.6)
        base_down += y_down
        
    ax1.set_title("Output: Total aFRR Provision UP (All Nodes)")
    ax1.set_ylabel("MW")
    ax1.set_xticks(range(0, 25, 4))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    ax2.invert_yaxis()
    ax2.set_title("Output: Total aFRR Provision DOWN (All Nodes)")
    ax2.set_ylabel("MW")
    ax2.set_xlabel("Hour")
    ax2.set_xticks(range(0, 25, 4))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/Total_aFRR.pdf')
    plt.close()

def print_revenue_and_ep_ratio(final_inv, final_inv_energy, results_df):
    print("\n" + "="*60)
    print("FINANCE & DESIGN STATISTICS")
    print("="*60)
    
    stats = []
    df_n1 = results_df[results_df['Node'] == 'N1'] 
    hours = df_n1['Time'].values
    
    for i in INVESTORS:
        arb_rev = 0
        afrr_rev = 0
        total_p = sum(final_inv[(i, n)] for n in NODES)
        total_e = sum(final_inv_energy[(i, n)] for n in NODES)
            
        for t in hours:
            for n in NODES:
                df_tn = results_df[(results_df['Time'] == t) & (results_df['Node'] == n)].iloc[0]
                price = df_tn['Price_EUR']
                p_ch = df_tn[f'{i}_Ch_MW']
                p_dis = df_tn[f'{i}_Dis_MW']
                arb_rev += price * (p_dis - p_ch)
                
                if CALCULATE_AFRR:
                    r_up = df_tn[f'{i}_R_Up_MW']
                    r_down = df_tn[f'{i}_R_Down_MW']
                    p_up = df_tn['aFRR_Price_Up_EUR']
                    p_down = df_tn['aFRR_Price_Down_EUR']
                    afrr_rev += (r_up * p_up + r_down * p_down) 

        total_rev = arb_rev + afrr_rev
        pct_arb = (arb_rev / total_rev * 100) if total_rev > 0 else 0
        pct_afrr = (afrr_rev / total_rev * 100) if total_rev > 0 else 0
        
        ep_ratio_actual = (total_e / total_p) if total_p > 0.01 else 0
        
        stats.append({
            'Investor': i,
            'Rate (%)': INV_DATA[i]['r'] * 100,
            'Total MW': total_p,
            'Total MWh': total_e,
            'Actual E/P': ep_ratio_actual,
            'Arbitrage €': arb_rev,
            'aFRR €': afrr_rev,
            'Profit % Arb': pct_arb,
            'Profit % aFRR': pct_afrr
        })
        
    df_stats = pd.DataFrame(stats)
    print(df_stats.round(2).to_string(index=False))

if __name__ == "__main__":
    import os
    import sys
    os.makedirs('output', exist_ok=True)
    
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger("output/console_log.txt")
    
    with open(__file__, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("--- Parameter Konfiguration ---")
        for i in range(28, 51):
            print(lines[i].strip())
        print("-------------------------------")

    try:
        # # --- Latex Printer ---
        # print("Initialisiere Latex Printer...")
        # x_power_dummy = {(i, n): 1.0 for i in INVESTORS for n in NODES}
        # model_latex = create_mpec_for_investor('I1', x_power_dummy, None)
        # print("Drucke Modell zu 'model_latex.txt'...")
        # try:
        #    latex_printer(model_latex, ostream='model_latex.txt')
        #    print("Latex Printer erfolgreich.")
        # except Exception as e:
        #    print(f"Latex Printer fehlgeschlagen: {e}")
        # # ---------------------
 
        print("1. Visualisiere Inputs...")
        plot_inputs_separate()
        plot_grid_enhanced()
        
        print_scenario_overview()
        final_inv, final_inv_energy, p_ch_g, p_dis_g, r_up_g, r_down_g, conv_hist = solve_epec()
        
        print("\n" + "="*60)
        print("EPEC ENDERGEBNISSE")
        print("="*60)    
        print("\n--- Investitionen (MW) ---")
        df = pd.DataFrame(index=NODES, columns=INVESTORS)
        for (i, n), val in final_inv.items(): df.at[n, i] = val
        df['Total_Node'] = df.sum(axis=1)
        df.loc['Total_Inv'] = df.sum(axis=0)
        print(df.round(2))
        
        print("\n--- Auslastung Shared Limits ---")
        for n in NODES:
            total = sum(final_inv[(i,n)] for i in INVESTORS)
            limit = SHARED_LIMITS[n]
            util = (total / limit) * 100 if limit > 0 else 0
            print(f"  {n}: {total:6.2f} / {limit} MW ({util:5.1f}%)")
    
        # --- NEU: CSV Export & Erweiterte Analyse ---
        results_df = extract_results(final_inv, final_inv_energy, p_ch_g, p_dis_g, r_up_g, r_down_g)
        export_results_to_csv(results_df)
        
        print("\nErstelle System-Gesamtbilanz...")
        plot_system_aggregation(final_inv, final_inv_energy, p_ch_g, p_dis_g, r_up_g, r_down_g)
        
        print("\nErstelle Knoten-Analysen...")
        plot_node_analysis(results_df)

        print("\nErstelle aFRR-Analysen (Knoten-Ebene)...")
        plot_afrr_provision(results_df)
        
        print("\nZeige Konvergenz-Historie...")
        plot_convergence(conv_hist)
        
        print("\nZeige Investment Bar Chart...")
        plot_investment_bar_chart(final_inv)
        
        print("\nZeige Line Loading Congestion Heatmap...")
        plot_line_congestion(results_df)
        
        print("\nZeige aFRR Gesamtbereitstellung...")
        plot_overall_afrr_provision(results_df)
        
        print_revenue_and_ep_ratio(final_inv, final_inv_energy, results_df)
                
    except Exception as e:
        print(f"KRITISCHER FEHLER: {e}")
        import traceback
        traceback.print_exc()