from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyomo.environ as pyo


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import (  # noqa: E402
    _ipopt_executable,
    default_data,
    full_availability,
    solve_best_response,
    solve_market,
)


class MarketClearingTest(unittest.TestCase):
    def test_default_population_matches_real_model_portfolio_split(self) -> None:
        data = default_data()
        investors = {investor.investor_id: investor for investor in data.investors}

        self.assertEqual(set(investors), {"I1", "I2", "I3", "I4"})
        self.assertEqual(investors["I1"].owned_generation_shares, {})
        self.assertEqual(investors["I2"].owned_generation_shares, {})
        self.assertEqual(
            investors["I3"].owned_generation_shares,
            {"RES_WIND_N1": 0.8, "RES_PV_N3": 0.2},
        )
        self.assertEqual(
            investors["I4"].owned_generation_shares,
            {"RES_WIND_N1": 0.2, "RES_PV_N3": 0.8},
        )
        self.assertAlmostEqual(sum(i.power_mw for i in data.investors), 40.0)
        self.assertAlmostEqual(sum(i.energy_mwh for i in data.investors), 100.0)

    def test_full_availability_market_is_feasible_and_balanced(self) -> None:
        data = default_data()
        charge, discharge = full_availability(data)
        market = solve_market(data, charge, discharge)

        for time in data.times:
            self.assertAlmostEqual(
                sum(float(pyo.value(market.NetInjection[n, time])) for n in data.nodes),
                0.0,
                places=7,
            )
        for investor in data.investors:
            self.assertAlmostEqual(
                float(pyo.value(market.SOC[investor.investor_id, 0])),
                float(pyo.value(market.SOC[investor.investor_id, max(data.times)])),
                places=7,
            )
            for time in data.times:
                throughput = float(pyo.value(market.PCharge[investor.investor_id, time])) + float(
                    pyo.value(market.PDischarge[investor.investor_id, time])
                )
                self.assertLessEqual(throughput, investor.power_mw + 1.0e-7)

    @unittest.skipIf(_ipopt_executable() is None, "IPOPT is not installed")
    def test_joint_investment_best_response_is_feasible_and_reclears(self) -> None:
        data = default_data()
        charge, discharge = full_availability(data)
        power = {investor.investor_id: investor.power_mw for investor in data.investors}
        energy = {investor.investor_id: investor.energy_mwh for investor in data.investors}
        response = solve_best_response(
            data,
            "I1",
            charge,
            discharge,
            power_capacity_mw=power,
            energy_capacity_mwh=energy,
            endogenous_investment=True,
        )

        self.assertTrue(response.optimal)
        self.assertGreaterEqual(response.power_mw, 0.0)
        self.assertLessEqual(response.power_mw, 30.0 + 1.0e-7)
        self.assertGreaterEqual(response.energy_mwh, 2.0 * response.power_mw - 1.0e-6)
        self.assertLessEqual(response.energy_mwh, 8.0 * response.power_mw + 1.0e-6)
        self.assertTrue(
            all(value <= response.power_mw + 1.0e-6 for value in response.discharge_offer_mw.values())
        )
        self.assertLessEqual(response.maximum_lmp_reclear_gap_eur_per_mwh, 0.02)
        self.assertLessEqual(response.embedded_reclear_profit_gap_eur_per_day, 0.25)

    @unittest.skipIf(_ipopt_executable() is None, "IPOPT is not installed")
    def test_best_response_passes_independent_reclear_audit(self) -> None:
        data = default_data()
        charge, discharge = full_availability(data)
        response = solve_best_response(data, "I1", charge, discharge)

        self.assertTrue(response.optimal)
        self.assertLessEqual(response.maximum_complementarity_violation, 1.0e-8)
        self.assertLessEqual(response.absolute_primal_dual_gap_eur_per_day, 0.01)
        self.assertLessEqual(response.maximum_lmp_reclear_gap_eur_per_mwh, 0.02)
        self.assertLessEqual(response.embedded_reclear_profit_gap_eur_per_day, 0.25)


if __name__ == "__main__":
    unittest.main()
