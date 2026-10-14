"""Analytical checks of dual signs, optimal faces, and the capacity MPEC."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
import pyomo.environ as pyo

from primal_llp import build_primal
from dual_llp import build_dual, dual_profit
from prepare_input import load_case, scale_flexible_demand
from solve import make_nlp_solver, make_solver, require_optimal, residual
from mpec import build_mpec, chosen_profile, complementarity_pairs, complete_sos1_start, solve_mpec
from mpec_relaxed import build_mpec_relaxed, initialize_from_market, solve_relaxed, start_profile
from solution_space import extrema, face_models
from capacity_clearing import clear_capacity, inner_equilibrium, own_limits, verify_equilibrium

IPOPT = make_nlp_solver().available(exception_flag=False)


def case(two_hours=False, duplicate=False):
    times = [1, 2] if two_hours else [1]
    gens = ["cheap", "peak"] + (["twin"] if duplicate else [])
    data = dict(nodes=["N"], generators=gens, times=times, soc_times=[0] + times,
        lines=[], line_limit={}, ptdf=[], eta=1.0, voll=1000.,
        generators_at_node={"N": gens}, generation_cost={g: 40. if g == "peak" else 10. for g in gens},
        generation_capacity=[dict(generator=g, hour=t, capacity_mw=(20. if g == "peak" else 10. if t == 1 else 0.)) for g in gens for t in times],
        demand_el=[dict(node="N", hour=t, demand_mw=5. if two_hours and t == 1 else 10.) for t in times])
    config = dict(wacc=0., lifetime_years=1, cost_power_eur_per_mw=365.25,
        cost_energy_eur_per_mwh=365.25, degradation_eur_per_mwh=0.,
        ratio_min=1., ratio_max=1., power_mw={}, energy_mwh={}, owned_generation_shares={})
    return data, {"I": config}


def bid_case():
    """Charge at 10 in hours 1 and 3. Hour 2: 8 MW at 12 and the peaker at 40
    against 10 MW demand, so discharge above 2 MW drops the price from 40 to 12.
    Hour 4: only the peaker at 40."""
    data, profile = case()
    times = [1, 2, 3, 4]
    capacity = {("cheap", 1): 30., ("cheap", 3): 30., ("low", 2): 8., ("peak", 2): 20., ("peak", 4): 20.}
    data.update(generators=["cheap", "low", "peak"], times=times, soc_times=[0] + times,
        generators_at_node={"N": ["cheap", "low", "peak"]}, generation_cost={"cheap": 10., "low": 12., "peak": 40.},
        generation_capacity=[dict(generator=g, hour=t, capacity_mw=capacity.get((g, t), 0.))
                             for g in ("cheap", "low", "peak") for t in times],
        demand_el=[dict(node="N", hour=t, demand_mw=10.) for t in times])
    return data, profile


def solve_relaxed_from(m, data, profile, start_power):
    initialize_from_market(m, data, start_profile(m, profile, start_power=start_power))
    return solve_relaxed(m, [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])[-1]["status"]


class CoreTests(unittest.TestCase):
    def test_price_interval_at_supply_kink(self):
        data, profile = case()
        primal, dual, _ = face_models(data, profile, tolerance=0)
        self.assertAlmostEqual(pyo.value(primal.market_cost), 100.)
        limits = extrema(dual, dual.price["N", 1])
        self.assertAlmostEqual(limits["minimum"], 10.)
        self.assertAlmostEqual(limits["maximum"], 40.)

    def test_dispatch_degeneracy_with_unique_price(self):
        data, profile = case(duplicate=True)
        primal, dual, _ = face_models(data, profile, tolerance=0)
        limits = extrema(primal, primal.generation["cheap", 1])
        self.assertAlmostEqual(limits["minimum"], 0.)
        self.assertAlmostEqual(limits["maximum"], 10.)
        prices = extrema(dual, dual.price["N", 1])
        self.assertAlmostEqual(prices["minimum"], 10.)
        self.assertAlmostEqual(prices["maximum"], 10.)

    def test_shedding_and_price_sign(self):
        data, profile = case()
        data["demand_el"][0]["demand_mw"] = 35.
        m, dual, _ = face_models(data, profile, tolerance=0)
        self.assertAlmostEqual(pyo.value(m.market_cost), pyo.value(dual.dual_value), places=7)
        self.assertAlmostEqual(pyo.value(m.load_shed["N", 1]), 5.)
        self.assertAlmostEqual(m.dual[m.nodal_balance["N", 1]], 1000.)
        self.assertLess(residual(m), 1e-8)

    def test_optimistic_mpec_and_independent_reclear(self):
        data, profile = case(two_hours=True)
        m = build_mpec(data, profile, "I", node_limit=10)
        require_optimal(m)
        self.assertFalse(any(v.is_binary() for v in m.component_data_objects(pyo.Var)))
        self.assertAlmostEqual(pyo.value(m.X_power["N"]), 5., places=5)
        self.assertAlmostEqual(pyo.value(m.X_energy["N"]), 5., places=5)
        self.assertAlmostEqual(pyo.value(m.profit), 140., places=5)
        self.assertAlmostEqual(pyo.value(m.profit), pyo.value(m.profit_linear), places=5)
        self.assertLess(max(abs(pyo.value(s * mu)) for s, mu in complementarity_pairs(m)), 1e-6)
        selected = chosen_profile(m)
        p, d, _ = face_models(data, selected, tolerance=0)
        self.assertAlmostEqual(pyo.value(m.market_cost), pyo.value(p.market_cost), places=5)
        payoff = extrema(d, dual_profit(d, "I"))
        self.assertAlmostEqual(payoff["minimum"], -10., places=5)
        self.assertAlmostEqual(payoff["maximum"], 140., places=5)

    def test_rival_degradation_and_generator_ownership_identity(self):
        data, profile = case(two_hours=True)
        profile["I"]["owned_generation_shares"] = {"cheap": .4}
        rival = copy.deepcopy(profile["I"])
        rival.update(power_mw={"N": 1.}, energy_mwh={"N": 1.}, degradation_eur_per_mwh=2., owned_generation_shares={})
        profile["R"] = rival
        for eps in (0., 1e-4):
            m = build_mpec(data, profile, "I", node_limit=10, balancing_eps=eps)
            require_optimal(m)
            # The optimistic profit is 112 for every X in [4, 5] at eps = 0.
            self.assertAlmostEqual(pyo.value(m.profit), 112. if eps == 0 else 112.0361, places=4)
            self.assertAlmostEqual(pyo.value(m.profit), pyo.value(m.profit_linear), places=5)
            self.assertLess(max(abs(pyo.value(s * mu)) for s, mu in complementarity_pairs(m)), 1e-5)
            self.assertLess(abs(pyo.value(m.market_cost - m.dual_value)), 1e-5)
            self.assertLess(residual(m), 1e-5)

    def test_balancing_makes_price_unique_at_the_step(self):
        # Exact LP: at X = 5 the hour-1 price is anywhere in [10, 40]. With
        # balancing b = eps * price, the price stays 10 until X = 5 + 10 eps,
        # then ramps to 40 by X = 5 + 40 eps. The MPEC optimum is the ramp start.
        data, profile = case(two_hours=True)
        eps = 1e-4
        m = build_mpec(data, profile, "I", node_limit=10, balancing_eps=eps)
        require_optimal(m)
        self.assertAlmostEqual(pyo.value(m.X_power["N"]), 5 + 10 * eps, places=6)
        self.assertAlmostEqual(pyo.value(m.profit), 28 * (5 + 10 * eps), places=4)
        self.assertAlmostEqual(pyo.value(m.profit), pyo.value(m.profit_linear), places=4)
        # Prices are pinned only to about FeasibilityTol / eps, so reclear tightly.
        selected = chosen_profile(m)
        solver = make_solver()
        solver.options.update({"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9,
                               "BarConvTol": 1e-12, "BarQCPConvTol": 1e-12})
        d = build_dual(data, selected, eps)
        require_optimal(d, solver=solver)
        self.assertAlmostEqual(pyo.value(d.price["N", 1]), 10., places=3)
        self.assertAlmostEqual(pyo.value(dual_profit(d, "I")), pyo.value(m.profit), places=2)

    def test_quadratic_curtailment_cost(self):
        # Hour 1: 30 MW of cost-0 supply against 5 MW demand. Owner slopes 1 (I, 60%)
        # and 2 (R, 40%) give a plant slope of 1.4, so curtailing u MW sets the
        # price -1.4 * u. Without storage u = 25: price -35, market cost 0.5*1.4*25^2.
        data, profile = case(two_hours=True)
        data["generation_cost"]["cheap"] = 0.
        data["generation_capacity"][0]["capacity_mw"] = 30.
        profile["I"].update(owned_generation_shares={"cheap": .6},
                            curtailment_slope_eur_per_mwh_per_mw={"cheap": 1.})
        rival = copy.deepcopy(profile["I"])
        rival.update(owned_generation_shares={"cheap": .4}, curtailment_slope_eur_per_mwh_per_mw={"cheap": 2.})
        profile["R"] = rival
        primal, dual, _ = face_models(data, profile)
        self.assertAlmostEqual(pyo.value(primal.market_cost), 0.5 * 1.4 * 25 ** 2 + 40 * 10, places=4)
        self.assertAlmostEqual(primal.dual[primal.nodal_balance["N", 1]], -35., places=4)
        self.assertAlmostEqual(pyo.value(dual.price["N", 1]), -35., places=4)
        # Storage X <= 10 shifts X MW to hour 2 (price 40), so u = 25 - X. I's profit
        # X*(40 + 1.4u) - 2X + 0.6*(-1.4u*(30 - u) - 0.5u^2) rises in X: at X = 10,
        # 610 - 20 - 256.5 = 333.5. Settlement and strong-duality profit must agree.
        m = build_mpec(data, profile, "I", node_limit=10)
        require_optimal(m)
        self.assertAlmostEqual(pyo.value(m.X_power["N"]), 10., places=4)
        self.assertAlmostEqual(pyo.value(m.price["N", 1]), -21., places=4)
        self.assertAlmostEqual(pyo.value(m.profit), 333.5, places=4)
        self.assertAlmostEqual(pyo.value(m.profit_linear), 333.5, places=4)
        self.assertLess(abs(pyo.value(m.market_cost - m.dual_value)), 1e-5)
        self.assertLess(residual(m), 1e-5)

    def test_rising_generation_cost(self):
        # Hour 2: the peaker's marginal cost is 40 + 4 g against 10 MW demand, so
        # price 80 and market cost 40*10 + 0.5*4*10^2 (+ 10*5 in hour 1).
        data, profile = case(two_hours=True)
        data["generation_cost_slope"] = {"peak": 4.}
        data["generation_capacity"][0]["capacity_mw"] = 30.
        primal, dual, _ = face_models(data, profile)
        self.assertAlmostEqual(pyo.value(primal.market_cost), 50. + 400. + 200., places=4)
        self.assertAlmostEqual(primal.dual[primal.nodal_balance["N", 2]], 80., places=4)
        self.assertAlmostEqual(pyo.value(dual.price["N", 2]), 80., places=4)
        # Storage X shifts X MW from the peaker (price 80 - 4X) to cheap energy at 10:
        # profit (80 - 4X) X - 10 X - 2 X (capex 1 per MW and 1 per MWh) is largest at
        # X = 8.5 (price 46, profit 289). Settlement and strong-duality profit agree.
        m = build_mpec(data, profile, "I", node_limit=20)
        require_optimal(m)
        self.assertAlmostEqual(pyo.value(m.X_power["N"]), 8.5, places=4)
        self.assertAlmostEqual(pyo.value(m.price["N", 2]), 46., places=4)
        self.assertAlmostEqual(pyo.value(m.profit), 289., places=4)
        self.assertAlmostEqual(pyo.value(m.profit_linear), 289., places=4)
        self.assertLess(abs(pyo.value(m.market_cost - m.dual_value)), 1e-5)
        profile["I"]["owned_generation_shares"] = {"peak": 1.}
        with self.assertRaises(ValueError):
            build_mpec(data, profile, "I")

    def test_flexible_demand(self):
        # Hour 1: cheap 10 MW at 10 against 5 MW demand; flexible demand with
        # willingness to pay 30 - 3 q takes the other 5 MW, and the peaker at 40
        # is too dear, so the price is the willingness to pay at 5 MW: 15.
        # Hour 2: peaker at 40 above 30, no flexible demand. Market cost
        # 10*10 + 40*10 less the benefit 30*5 - 1.5*5^2 = 387.5.
        data, profile = case(two_hours=True)
        data["flexible_demand"] = {"N": {"max_mw": 10., "max_price_eur_per_mwh": 30.}}
        primal, dual, _ = face_models(data, profile)
        self.assertAlmostEqual(pyo.value(primal.market_cost), 387.5, places=4)
        self.assertAlmostEqual(pyo.value(primal.flexible_demand["N", 1]), 5., places=4)
        self.assertAlmostEqual(pyo.value(primal.flexible_demand["N", 2]), 0., places=4)
        self.assertAlmostEqual(primal.dual[primal.nodal_balance["N", 1]], 15., places=4)
        self.assertAlmostEqual(pyo.value(dual.price["N", 1]), 15., places=4)
        self.assertAlmostEqual(pyo.value(dual.price["N", 2]), 40., places=4)
        # Storage X charges in hour 1 by displacing flexible demand (price 15 + 3X)
        # and discharges at 40: profit (25 - 3X) X - 2X is largest at X = 23/6
        # (price 26.5, profit 529/12). Without flexible demand the hour-1 price
        # would jump from 10 to 40 at X = 5.
        m = build_mpec(data, profile, "I", node_limit=20)
        require_optimal(m)
        self.assertAlmostEqual(pyo.value(m.X_power["N"]), 23 / 6, places=4)
        self.assertAlmostEqual(pyo.value(m.price["N", 1]), 26.5, places=4)
        self.assertAlmostEqual(pyo.value(m.profit), 529 / 12, places=4)
        self.assertAlmostEqual(pyo.value(m.profit_linear), 529 / 12, places=4)
        self.assertLess(abs(pyo.value(m.market_cost - m.dual_value)), 1e-5)
        relaxed = build_mpec_relaxed(data, profile, "I", node_limit=20)
        self.assertEqual(solve_relaxed_from(relaxed, data, profile, start_power=1.), "optimal")
        self.assertAlmostEqual(pyo.value(relaxed.X_power["N"]), 23 / 6, places=3)
        self.assertAlmostEqual(pyo.value(relaxed.profit), 529 / 12, places=3)
        # The scale multiplies the size at the same willingness to pay; 0 removes it.
        self.assertEqual(scale_flexible_demand(data, 2)["flexible_demand"]["N"]["max_mw"], 20.)
        self.assertNotIn("flexible_demand", scale_flexible_demand(data, 0))
        data["flexible_demand"]["N"]["max_mw"] = 0.
        with self.assertRaises(ValueError):
            build_primal(data, profile)

    def test_quadratic_dispatch_cost(self):
        # Prices stay 10 (hour 1) and 40 (hour 2). Storage with slope 10 shifts c MW
        # at market cost -30c + 10c^2, so c = min(X, 1.5) is unique. Fixed X = 2 (an
        # interior optimum; at X = 1.5 the power bound is weakly active and barrier
        # accuracy drops): cost 65 + 340 + 22.5; profit = rent 0.5*10*(2*1.5^2) - capex 4.
        data, profile = case(two_hours=True)
        profile["I"]["dispatch_cost_slope_eur_per_mwh_per_mw"] = 10.
        fixed = copy.deepcopy(profile)
        fixed["I"].update(power_mw={"N": 2.}, energy_mwh={"N": 2.})
        primal, dual, _ = face_models(data, fixed)
        self.assertAlmostEqual(pyo.value(primal.market_cost), 427.5, places=4)
        self.assertAlmostEqual(pyo.value(primal.charge["I", "N", 1]), 1.5, places=4)
        self.assertAlmostEqual(pyo.value(dual_profit(dual, "I")), 18.5, places=4)
        # MPEC: profit 28X - 10X^2 for X <= 1.5, maximal at X = 1.4 (19.6). A rival
        # with a dispatch cost tests the rival rent term in profit_linear.
        rival = copy.deepcopy(profile["I"])
        rival.update(power_mw={"N": 1.}, energy_mwh={"N": 1.})
        profile["R"] = rival
        m = build_mpec(data, profile, "I", node_limit=10)
        require_optimal(m)
        self.assertAlmostEqual(pyo.value(m.X_power["N"]), 1.4, places=4)
        self.assertAlmostEqual(pyo.value(m.profit), 19.6, places=4)
        self.assertAlmostEqual(pyo.value(m.profit_linear), 19.6, places=4)
        self.assertLess(abs(pyo.value(m.market_cost - m.dual_value)), 1e-5)
        self.assertLess(residual(m), 1e-5)

    def test_strategic_bids_withhold_at_the_step(self):
        # Capacity only (E = X, capex 2 per MW): the ISO discharges X in hours 2 and
        # 4, so for X > 2 profit is 12X + 40X - 20X - 2X = 30X, best at X = 10: 300.
        # With bids the investor offers only 2 MW in hour 2 at 40:
        # 2 * 40 + 10 * 40 - 12 * 10 - 20 = 340.
        data, profile = bid_case()
        for strategic, expected in ((False, 300.), (True, 340.)):
            m = build_mpec(data, profile, "I", node_limit=20, strategic_bids=strategic)
            require_optimal(m)
            self.assertAlmostEqual(pyo.value(m.X_power["N"]), 10., places=5)
            self.assertAlmostEqual(pyo.value(m.profit), expected, places=5)
            self.assertAlmostEqual(pyo.value(m.profit_linear), expected, places=5)
            self.assertLess(abs(pyo.value(m.market_cost - m.dual_value)), 1e-5)
            self.assertLess(residual(m), 1e-5)
        self.assertAlmostEqual(pyo.value(m.discharge["I", "N", 2]), 2., places=5)
        # The exported bids are the market's input: the reclear reproduces the dispatch cost.
        selected = chosen_profile(m)
        primal, _, _ = face_models(data, selected, tolerance=0)
        self.assertAlmostEqual(pyo.value(primal.market_cost), pyo.value(m.market_cost), places=5)
        with self.assertRaises(ValueError):  # a bid above the offer would let the ISO loop the storage
            selected["I"]["charge_bid_eur_per_mwh"]["N"]["1"] = selected["I"]["discharge_offer_eur_per_mwh"]["N"]["1"] + 1
            build_primal(data, selected)

    def test_bid_selection_closest_to_truthful(self):
        # Same optimum as above: 2 MW at 40 in hour 2, recharged at 10 in hour 1
        # or 3 (LP: either). Interior dispatch pins offer(2) + SOC value = 40
        # and SOC value + charge bid = 10 in the 2 MW charging hour; the ISO's
        # SOC value is free, so the closest bids to truthful (0, d = 0) split
        # the markup: +15 and -15. All other bids return to truthful.
        data, profile = bid_case()
        m = build_mpec(data, profile, "I", node_limit=20, strategic_bids=True)
        result, (incumbent, _), selection = solve_mpec(m)
        self.assertEqual(selection["status"], "optimal")
        self.assertAlmostEqual(incumbent, 340., places=5)
        self.assertAlmostEqual(pyo.value(m.profit), 340., places=4)
        self.assertAlmostEqual(pyo.value(m.X_discharge_offer["N", 2]), 15., places=4)
        partial = 1 if pyo.value(m.charge["I", "N", 1]) < 5 else 3
        self.assertAlmostEqual(pyo.value(m.X_charge_bid["N", partial]), -15., places=4)
        for t in m.timesteps:
            if t != partial:
                self.assertAlmostEqual(pyo.value(m.X_charge_bid["N", t]), 0., places=4)
            if t != 2:
                self.assertAlmostEqual(pyo.value(m.X_discharge_offer["N", t]), 0., places=4)
        self.assertAlmostEqual(selection["squared_markup_after"], 450., places=1)
        primal, _, _ = face_models(data, chosen_profile(m), tolerance=0)
        self.assertAlmostEqual(pyo.value(primal.discharge["I", "N", 2]), 2., places=5)
        self.assertAlmostEqual(pyo.value(primal.market_cost), pyo.value(m.market_cost), places=3)  # 1e-6 MW hold

    def test_capacity_price_and_priced_headroom(self):
        # Toy: each MW (E = X) earns 40 - 10 - 2 capex = 28 up to 5 MW. A hosting
        # capacity price p per MW leaves 5 MW at p = 10 (profit 90), nothing at 30.
        data, profile = case(two_hours=True)
        for price, power, profit in ((10., 5., 90.), (30., 0., 0.)):
            m = build_mpec(data, profile, "I", node_limit=10, capacity_price={"N": price})
            require_optimal(m)
            self.assertAlmostEqual(pyo.value(m.X_power["N"]), power, places=5)
            self.assertAlmostEqual(pyo.value(m.profit_linear), profit, places=5)
        # First come, first served: a rival's 4 MW cut the headroom to 6. With
        # prices the node is shared by the price, so the investor alone may use 10.
        profile["R"] = copy.deepcopy(profile["I"])
        profile["R"]["power_mw"], profile["R"]["energy_mwh"] = {"N": 4.}, {"N": 4.}
        self.assertEqual(pyo.value(build_mpec(data, profile, "I", node_limit=10).headroom["N"]), 6.)
        self.assertEqual(pyo.value(build_mpec(data, profile, "I", node_limit=10, capacity_price={}).headroom["N"]), 10.)

    def test_hosting_rights(self):
        # Toy: each MW earns 28 up to 5 MW. A 3 MW right caps the investor at
        # 3 MW (profit 84). With a rival's 4 MW the shared limit of 10 leaves 6,
        # so a right of 8 MW still allows only 6.
        data, profile = case(two_hours=True)
        m = build_mpec(data, profile, "I", node_limit=10, own_limit={"N": 3.})
        require_optimal(m)
        self.assertAlmostEqual(pyo.value(m.X_power["N"]), 3., places=5)
        self.assertAlmostEqual(pyo.value(m.profit_linear), 84., places=5)
        rival = copy.deepcopy(profile)
        rival["R"] = copy.deepcopy(profile["I"])
        rival["R"]["power_mw"], rival["R"]["energy_mwh"] = {"N": 4.}, {"N": 4.}
        self.assertEqual(pyo.value(build_mpec(data, rival, "I", node_limit=10, own_limit={"N": 8.}).headroom["N"]), 6.)
        # At an allocated node an investor without a right may build nothing.
        rights = {"I": {"N": 2.}, "R": {"N": 2.}}
        self.assertEqual(own_limits({"I": {"N": 3.}}, "R"), {"N": 0.})
        self.assertIsNone(own_limits({}, "R"))
        if not IPOPT:
            return
        # Gauss-Seidel within the rights: both fill their 2 MW (4 MW < 5 MW).
        profile["R"] = copy.deepcopy(profile["I"])
        result, converged, _ = inner_equilibrium(data, profile, {"N": 10.}, None, order=["I", "R"],
                                                 balancing_eps=1e-4, strategic_bids=False, rounds=3, seconds=60,
                                                 allocation=rights, log=lambda text: None)
        self.assertTrue(converged)
        for i in ("I", "R"):
            self.assertAlmostEqual(result[i]["power_mw"]["N"], 2., places=3)

    @unittest.skipUnless(IPOPT, "Ipopt is not on PATH")
    def test_uniform_price_reports_a_demand_jump(self):
        # Toy: each MW earns 28 up to 5 MW, so two investors want 5 MW below a
        # price of 28 and nothing above it. A 3 MW limit has no clearing price:
        # prices 0, 10, 20 (over-full), 40, 30 (fit), 25, 27.5 (over-full), then
        # the bracket [27.5, 30] is narrow and 30 leaves the node empty.
        data, profile = case(two_hours=True)
        profile["R"] = copy.deepcopy(profile["I"])
        result, evaluations = clear_capacity(data, profile, {"N": 3.}, order=["I", "R"], balancing_eps=1e-4,
                                             strategic_bids=False, price_start=10., price_tolerance=3., rounds=3,
                                             seconds=60, log=lambda text: None)
        self.assertTrue(result["cleared"])
        self.assertEqual([e["prices"]["N"] for e in evaluations], [0., 10., 20., 40., 30., 25., 27.5, 30.])
        self.assertEqual(result["price_bracket"]["N"], [27.5, 30.])
        self.assertAlmostEqual(result["unused_mw"]["N"], 3., places=4)

    def test_relaxed_mpec_reaches_the_sos1_optimum(self):
        data, profile = case(two_hours=True)
        m = build_mpec_relaxed(data, profile, "I", node_limit=10)
        self.assertEqual(solve_relaxed_from(m, data, profile, start_power=2.), "optimal")
        self.assertAlmostEqual(pyo.value(m.X_power["N"]), 5., places=5)
        self.assertAlmostEqual(pyo.value(m.profit), 140., places=4)
        # Local and start-dependent: from 10 MW the bid MPEC stays at the
        # truthful 300 (withholding pays only below 2 MW); from 2 MW it finds 340.
        data, profile = bid_case()
        for start_power, expected in ((10., 300.), (2., 340.)):
            m = build_mpec_relaxed(data, profile, "I", node_limit=20, strategic_bids=True)
            self.assertEqual(solve_relaxed_from(m, data, profile, start_power), "optimal")
            self.assertAlmostEqual(pyo.value(m.profit), expected, places=3)
            self.assertLessEqual(max(pyo.value(s * mu) for s, mu in complementarity_pairs(m)), 1e-6 + 1e-9)

    def test_mip_start_and_verify_cutoff(self):
        # MIP start at the known optimum (5 MW / 5 MWh): a feasible KKT point with
        # one exact zero in every SOS1 pair, from which Gurobi proves 140.
        data, profile = case(two_hours=True)
        start = copy.deepcopy(profile)
        start["I"]["power_mw"], start["I"]["energy_mwh"] = {"N": 5.}, {"N": 5.}
        m = build_mpec(data, profile, "I", node_limit=10)
        initialize_from_market(m, data, start)
        complete_sos1_start(m)
        self.assertLess(residual(m), 1e-6)
        for sos in m.component_data_objects(pyo.SOSConstraint):
            self.assertEqual(min(pyo.value(v) for v in sos.get_variables()), 0.)
        result, _, _ = solve_mpec(m, warmstart=True)
        self.assertEqual(str(result.solver.termination_condition), "optimal")
        self.assertAlmostEqual(pyo.value(m.profit), 140., places=5)
        # verify: at the optimum nothing beats the cutoff, so Gurobi stops with
        # status cutoff; at zero capacity building pays, so the check refutes it.
        m = build_mpec(data, profile, "I", node_limit=10, balancing_eps=1e-4)
        require_optimal(m)
        overall, records = verify_equilibrium(data, chosen_profile(m), balancing_eps=1e-4, node_limit=10,
                                              seconds=60, log=lambda text: None)
        self.assertEqual((overall, records[0]["status"]), ("proven", "minFunctionValue"))
        overall, _ = verify_equilibrium(data, profile, balancing_eps=1e-4, node_limit=10, seconds=60,
                                        log=lambda text: None)
        self.assertEqual(overall, "refuted")
        with self.assertRaises(ValueError):
            verify_equilibrium(data, chosen_profile(m), balancing_eps=1e-4, nodes=[], seconds=60)

    def test_shared_inverter_and_cyclic_soc(self):
        data, profile = case(two_hours=True)
        data["eta"] = .9
        profile["I"].update(power_mw={"N": 3.}, energy_mwh={"N": 3.}, degradation_eur_per_mwh=2.)
        m = build_primal(data, profile)
        require_optimal(m)
        for t in data["times"]:
            self.assertLessEqual(pyo.value(m.charge["I", "N", t] + m.discharge["I", "N", t]), 3. + 1e-8)
        self.assertAlmostEqual(pyo.value(m.soc["I", "N", 0]), pyo.value(m.soc["I", "N", 2]))
        self.assertLess(residual(m), 1e-8)

    def test_congestion_prices(self):
        data, profile = case()
        data.update(nodes=["A", "B"], generators_at_node={"A": ["cheap"], "B": ["peak"]},
            lines=["L"], line_limit={"L": 5.},
            ptdf=[dict(line="L", node="A", ptdf=1.), dict(line="L", node="B", ptdf=0.)],
            demand_el=[dict(node="A", hour=1, demand_mw=0.), dict(node="B", hour=1, demand_mw=10.)])
        m, d, _ = face_models(data, profile, tolerance=0)
        self.assertAlmostEqual(pyo.value(m.market_cost), 250.)
        for n, expected in [("A", 10.), ("B", 40.)]:
            self.assertAlmostEqual(m.dual[m.nodal_balance[n, 1]], expected)
            limits = extrema(d, d.price[n, 1])
            self.assertAlmostEqual(limits["minimum"], expected)
            self.assertAlmostEqual(limits["maximum"], expected)

    def test_unbounded_price_face_is_reported(self):
        data, profile = case()
        for record in data["generation_capacity"]:
            record["capacity_mw"] = 0.
        _, d, _ = face_models(data, profile, tolerance=0)
        limits = extrema(d, d.price["N", 1])
        self.assertAlmostEqual(limits["minimum"], 1000.)
        self.assertIsNone(limits["maximum"])
        self.assertIn("unbounded", limits["maximum_status"].lower())

    def test_five_bus_market_regression(self):
        # 5-bus test system without storage: a QP from G1's rising marginal cost
        # and the pocket flexible demand (N2 10, N5 25 MW); scale 2 is the first
        # size tried (N2 20, N5 50 MW), scale 0 the system before the flexible
        # demand (recorded 2026-10-14, Gurobi 13.0.3).
        data, profile = load_case()
        for scale, expected in ((1, 250331.2321901794), (2, 240560.53533554217), (0, 260873.23465375137)):
            primal, dual, _ = face_models(scale_flexible_demand(data, scale), profile, tolerance=0)
            self.assertAlmostEqual(pyo.value(primal.market_cost), expected, delta=1e-5 + 1e-9 * expected)
            self.assertAlmostEqual(pyo.value(dual.dual_value), expected, delta=1e-5 + 1e-9 * expected)


if __name__ == "__main__":
    unittest.main()
