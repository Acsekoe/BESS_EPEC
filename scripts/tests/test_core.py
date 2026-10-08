"""Analytical checks of dual signs, optimal faces, and the capacity MPEC."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
import pyomo.environ as pyo

from primal_llp import build_primal
from dual_llp import dual_profit
from prepare_input import INPUT, load_case
from solve import require_optimal, residual
from mpec import build_mpec, chosen_profile
from solution_space import extrema, face_models


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
        m = build_mpec(data, profile, "I", node_limit=10, dual_m=10000, objective="linear")
        require_optimal(m)
        self.assertAlmostEqual(pyo.value(m.X_power["N"]), 5., places=5)
        self.assertAlmostEqual(pyo.value(m.X_energy["N"]), 5., places=5)
        self.assertAlmostEqual(pyo.value(m.profit), 140., places=5)
        self.assertAlmostEqual(pyo.value(m.profit), pyo.value(m.profit_linear), places=5)
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
        m = build_mpec(data, profile, "I", node_limit=10, dual_m=10000, objective="linear")
        require_optimal(m)
        self.assertAlmostEqual(pyo.value(m.profit), pyo.value(m.profit_linear), places=5)
        self.assertLess(abs(pyo.value(m.market_cost - m.dual_value)), 1e-5)
        self.assertLess(residual(m), 1e-5)

    def test_bilinear_objective_matches_linear_form(self):
        data, profile = case(two_hours=True)
        profile["I"]["owned_generation_shares"] = {"cheap": .4}
        rival = copy.deepcopy(profile["I"])
        rival.update(power_mw={"N": 1.}, energy_mwh={"N": 1.}, degradation_eur_per_mwh=2., owned_generation_shares={})
        profile["R"] = rival
        solved = {}
        for objective in ("linear", "bilinear"):
            m = build_mpec(data, profile, "I", node_limit=10, dual_m=10000, objective=objective)
            require_optimal(m)
            self.assertAlmostEqual(pyo.value(m.profit), pyo.value(m.profit_linear), places=5)
            self.assertLess(residual(m), 1e-5)
            solved[objective] = m
        linear, bilinear = solved["linear"], solved["bilinear"]
        self.assertAlmostEqual(pyo.value(bilinear.profit), pyo.value(linear.profit), places=4)
        self.assertAlmostEqual(pyo.value(bilinear.X_power["N"]), pyo.value(linear.X_power["N"]), places=4)
        self.assertAlmostEqual(pyo.value(bilinear.X_energy["N"]), pyo.value(linear.X_energy["N"]), places=4)

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

    def test_ieee9_market_regression(self):
        # Benchmark values recorded before replacing matrix assembly by rules.
        for filename, expected in [("capacities.json", 394587.85188079276),
                                   ("capacities_storage.json", 391977.221111562)]:
            with self.subTest(profile=filename):
                data, profile = load_case(profile_path=INPUT / filename)
                primal, dual, _ = face_models(data, profile, tolerance=0)
                self.assertAlmostEqual(pyo.value(primal.market_cost), expected, places=6)
                self.assertAlmostEqual(pyo.value(dual.dual_value), expected, places=6)


if __name__ == "__main__":
    unittest.main()
