"""Physical, settlement and independently differentiated KKT regressions."""
import sys
from pathlib import Path
from dataclasses import replace

import pytest
import pyomo.environ as pyo
from pyomo.core.expr.calculus.derivatives import differentiate, Modes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import afrr
import iso_market
import mpec_afrr
import capacity_game as game
from investors import InvestorConfig, three_investors
from market_data import load_market_data, validate_afrr
from solvers import SolverSettings, maximum_bound_violation


@pytest.fixture
def toy():
    return replace(load_market_data(), nodes=('N',), generators=('G',), lines=(),
        generators_at_node={'N': ('G',)}, generation_cost={'G': 50.0}, generation_offer=None,
        generation_capacity={('G', t): 300.0 for t in range(1, 25)},
        demand_el={('N', t): 100.0 for t in range(1, 25)}, line_limit={}, ptdf={},
        node_connection_limit={'N': 200.0}, afrr_enabled=True,
        afrr_eligible_generators=('G',), afrr_generator_offer_eur_per_mw_hour={'G': 25.0})


def clear(data, power, energy):
    return iso_market.clear_market(data, SolverSettings(),
        {('I1', 'N'): power}, {('I1', 'N'): energy}, {'I1': 15.0})


def test_zero_energy_cannot_supply_reserves(toy):
    m = clear(toy, 20.0, 0.0)
    assert max(pyo.value(v) for name in ('R_up', 'R_dn')
               for v in getattr(m, name).values()) < 1e-6


def test_block_prices_revenue_and_expected_wear(toy):
    m = clear(toy, 20.0, 40.0)
    for b in m.B:
        assert afrr.price(m, toy, 'up', b) == pytest.approx(100.0, abs=1e-5)
        assert afrr.price(m, toy, 'dn', b, hourly=True) == pytest.approx(25.0, abs=1e-5)
    s = iso_market.settle(m, toy, InvestorConfig('I1'), {('I1', 'N'): 20}, {('I1', 'N'): 40})
    assert s.afrr_capacity_revenue == pytest.approx(24*25*40, abs=1e-3)
    assert s.afrr_expected_degradation == pytest.approx(24*0.5*15*0.1*40, abs=1e-3)
    assert s.profit == pytest.approx(s.storage_settlement+s.afrr_capacity_revenue
        +s.owned_generation_rent+s.owned_generation_afrr_surplus-s.degradation-s.capex)


def test_shortage_sets_analytic_price_cap(toy):
    toy = replace(toy, afrr_eligible_generators=())
    m = clear(toy, 0.0, 0.0)
    assert pyo.value(m.afrr_short_up[1]) == pytest.approx(50, abs=1e-5)
    assert afrr.price(m, toy, 'up', 1) == pytest.approx(12000, abs=1e-4)
    assert afrr.price(m, toy, 'dn', 1, hourly=True) == pytest.approx(3000, abs=1e-4)


def test_soc_guard_includes_both_blocks_at_boundary(toy):
    m = clear(toy, 20.0, 40.0)
    assert (1, 0) in m.AFRR_BK and (6, 24) in m.AFRR_BK
    assert (1, 4) in m.AFRR_BK and (2, 4) in m.AFRR_BK
    m.SOC['I1', 'N', 0].set_value(0)
    m.R_up['I1', 'N', 1].set_value(20)
    assert pyo.value(m.afrr_soc_up['I1', 'N', 1, 0].body) < -10


def test_generators_cannot_sell_unbacked_reserve(toy):
    toy = replace(toy, generation_capacity={('G', t): 120.0 for t in toy.times})
    m = clear(toy, 0, 0)
    for t in m.T:
        assert pyo.value(m.P_gen['G', t]+m.r_up['G', toy.afrr_block(t)]) <= 120+1e-6
        assert pyo.value(m.r_dn['G', toy.afrr_block(t)]) <= pyo.value(m.P_gen['G', t])+1e-6


def test_all_new_reduced_costs_are_lagrangian_derivatives(toy):
    m = mpec_afrr.build_capacity_mpec(toy, investor=InvestorConfig('I1'), initial_power_mw=20)
    rows = [('nodal_balance','lam'), ('system_balance','lam_sys'),
        ('generation_capacity_bound','nu_gen'), ('line_upper_bound','mu_up'),
        ('line_lower_bound','mu_dn'), ('charge_power_bound','rho_ch'),
        ('discharge_power_bound','sig_dis'), ('soc_transition','gam'),
        ('soc_capacity_bound','del_soc'), ('soc_periodicity','rho_per'),
        ('afrr_balance_up','Gamma_up'), ('afrr_balance_dn','Gamma_dn'),
        ('afrr_gen_down_headroom','kappa_gen_dn'), ('afrr_soc_up','xi_soc_up'),
        ('afrr_soc_down','zeta_soc_dn')]
    lagrangian = m.primal_objective.expr
    for constraint_name, dual_name in rows:
        for key, c in getattr(m, constraint_name).items():
            rhs = c.upper if c.upper is not None else c.lower
            lagrangian -= getattr(m, dual_name)[key] * (c.body-rhs)
    # Nonzero arbitrary values expose wrong signs even on inactive constraints.
    for index, v in enumerate(m.component_data_objects(pyo.Var)):
        v.set_value((-1 if v.ub is not None and v.ub == 0 else 1)*(1+index/1000), skip_validation=True)
    cases = []
    for g, t in m.GT:
        cases.append((m.P_gen[g,t], mpec_afrr._gen_reduced_cost(m,toy,{'G':['N']},g,t)))
    for i,n in m.IN:
        for t in m.T:
            cases += [(m.P_charge[i,n,t], mpec_afrr._charge_reduced_cost(m,toy.eta,{'I1':15},i,n,t)),
                      (m.P_discharge[i,n,t], mpec_afrr._discharge_reduced_cost(m,toy.eta,{'I1':15},i,n,t))]
        for k in m.T_SOC:
            cases.append((m.SOC[i,n,k],mpec_afrr._soc_reduced_cost(m,i,n,k,24)))
        for b in m.B:
            for d in ('up','dn'):
                cases.append((getattr(m,'R_'+d)[i,n,b],
                    mpec_afrr._reserve_reduced_cost(m,toy,{'I1':15},d,i,n,b)))
    for g,b in m.AFRR_GB:
        for d in ('up','dn'):
            cases.append((getattr(m,'r_'+d)[g,b],mpec_afrr._generator_reserve_reduced_cost(m,toy,d,g,b)))
    for b in m.B:
        for d in ('up','dn'):
            cases.append((getattr(m,'afrr_short_'+d)[b],
                4*toy.afrr_penalty_eur_per_mw_hour-getattr(m,'Gamma_'+d)[b]))
    for variable, expected in cases:
        actual = differentiate(lagrangian, wrt=variable, mode=Modes.reverse_symbolic)
        assert pyo.value(actual) == pytest.approx(pyo.value(expected), abs=1e-8), variable.name


@pytest.mark.parametrize('formulation', ['relaxed-kkt', 'strong-duality'])
def test_exact_qp_seed_satisfies_full_kkt_and_settlement(toy, formulation):
    investor = InvestorConfig('I1')
    config = game.GameConfig(investors=(investor,), lower_level=formulation,
        market_design='afrr', proximal_penalty=0, solver=SolverSettings())
    p, e = {('I1','N'):20.0}, {('I1','N'):40.0}
    m = game.build_best_response_model(toy, config, investor, p, e)
    # Strong-duality residual is the aggregate EUR objective gap, so use the
    # same absolute tolerance here as in the explicit gap check below.
    assert maximum_bound_violation(m) < 1e-4
    diag = mpec_afrr.complementarity_diagnostics(m)
    assert diag['absolute_primal_dual_gap_eur_per_day'] < 1e-4
    assert diag['maximum_product'] < 1e-5
    lower = clear(toy,20,40)
    assert pyo.value(m.unregularized_profit) == pytest.approx(
        iso_market.settle(lower,toy,investor,p,e).profit,abs=1e-3)
    assert pyo.value(m.primal_objective) == pytest.approx(pyo.value(lower.objective),abs=1e-4)


def test_renewables_explicitly_ineligible_and_validation():
    data = load_market_data()
    assert not any(g.startswith('RES_') for g in data.afrr_eligible_generators)
    with pytest.raises(ValueError):
        validate_afrr(replace(data,afrr_response_hours=0))
    with pytest.raises(ValueError):
        validate_afrr(replace(data,afrr_demand_up_mw=float('nan')))


def test_zero_demand_recovers_energy_only_objective_and_profit(toy):
    zero = replace(toy, afrr_demand_up_mw=0, afrr_demand_down_mw=0)
    old = replace(zero, afrr_enabled=False)
    m0, m1 = clear(old,20,40), clear(zero,20,40)
    assert pyo.value(m0.objective) == pytest.approx(pyo.value(m1.objective),abs=1e-4)
    p,e={('I1','N'):20}, {('I1','N'):40}
    investor=InvestorConfig('I1')
    assert iso_market.settle(m0,old,investor,p,e).profit == pytest.approx(
        iso_market.settle(m1,zero,investor,p,e).profit,abs=1e-4)


def test_full_network_ipopt_seed_has_small_unscaled_kkt_products():
    data = replace(load_market_data(), afrr_enabled=True)
    config = game.GameConfig(investors=three_investors(data),market_design='afrr')
    state = game.initial_state(data,config)
    model = game.build_best_response_model(data,config,config.investors[0],state.power,state.energy)
    diag = mpec_afrr.complementarity_diagnostics(model)
    assert maximum_bound_violation(model) < 1e-6
    assert diag['maximum_product'] < 1e-6
    assert diag['absolute_primal_dual_gap_eur_per_day'] < 1e-4
