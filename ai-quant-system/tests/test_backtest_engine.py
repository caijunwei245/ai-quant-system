import unittest

import pandas as pd

from backtest.engine import BacktestEngine


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        self.engine = BacktestEngine({
            "initial_capital": 100000,
            "commission": 0.0003,
            "min_commission": 5,
            "stamp_tax": 0.001,
            "slippage": 0.001,
        })

    def test_calendar_uses_supplied_exchange_sessions(self):
        calendar = pd.DataFrame(index=pd.to_datetime([
            "2024-01-05", "2024-01-08", "2024-01-09"
        ]))
        self.assertEqual(
            self.engine.normalize_calendar(calendar, "2024-01-06", "2024-01-09"),
            ["2024-01-08", "2024-01-09"],
        )

    def test_calendar_refuses_weekday_fallback(self):
        with self.assertRaises(ValueError):
            self.engine.normalize_calendar(pd.DataFrame(), "2024-01-01", "2024-01-31")

    def test_buy_then_mark_to_close_and_sell_with_costs(self):
        buy = self.engine.step(
            "2024-01-02", {"600000": 0.5},
            {"600000": {"open": 10, "high": 10.2, "low": 9.9,
                        "close": 11, "volume": 100000}},
        )
        self.assertEqual(self.engine.positions["600000"].shares, 5000)
        self.assertAlmostEqual(buy["market_value"], 55000)
        self.assertLess(buy["cash"], 51000)

        sell = self.engine.step(
            "2024-01-03", {},
            {"600000": {"open": 12, "high": 12.2, "low": 11.8,
                        "close": 12, "volume": 100000}},
        )
        self.assertEqual(self.engine.positions, {})
        self.assertEqual(sell["market_value"], 0)
        self.assertEqual(self.engine.trades[-1]["side"], "sell")
        self.assertGreater(self.engine.trades[-1]["stamp_tax"], 0)
        self.assertAlmostEqual(sell["nav"], self.engine.cash)

    def test_rebalance_existing_position_not_only_new_symbols(self):
        self.engine.step(
            "2024-01-02", {"600000": 0.5},
            {"600000": {"open": 10, "high": 10.2, "low": 9.9,
                        "close": 10, "volume": 100000}},
        )
        shares_before = self.engine.positions["600000"].shares
        self.engine.step(
            "2024-01-03", {"600000": 0.2},
            {"600000": {"open": 10, "high": 10.2, "low": 9.9,
                        "close": 10, "volume": 100000}},
        )
        self.assertLess(self.engine.positions["600000"].shares, shares_before)

    def test_suspended_stock_is_not_filled_and_mark_is_stale(self):
        result = self.engine.step(
            "2024-01-02", {"600000": 0.5},
            {"600000": {"open": 10, "high": 10, "low": 10,
                        "close": 10, "volume": 0}},
        )
        self.assertNotIn("600000", self.engine.positions)
        self.assertEqual(len(self.engine.trades), 0)
        self.assertEqual(result["stale_marks"], [])

    def test_daily_ledger_reconciliation_with_fixed_three_day_sample(self):
        """Cash + marked holdings must reconcile each day; costs hit cash exactly once."""
        day1 = self.engine.step(
            "2024-01-02", {"600000": 0.5},
            {"600000": {"open": 10, "high": 10.2, "low": 9.9,
                        "close": 10, "volume": 100000}},
        )
        day2 = self.engine.step(
            "2024-01-03", {"600000": 0.25},
            {"600000": {"open": 11, "high": 12, "low": 10.9,
                        "close": 12, "volume": 100000, "pre_close": 10}},
        )
        day3 = self.engine.step(
            "2024-01-04", {},
            {"600000": {"open": 12, "high": 12.1, "low": 11.9,
                        "close": 12, "volume": 100000, "pre_close": 12}},
        )

        for snapshot in (day1, day2, day3):
            holdings_value = sum(
                item["shares"] * item["last_price"]
                for item in snapshot["positions"].values()
            )
            self.assertAlmostEqual(snapshot["market_value"], holdings_value, places=8)
            self.assertAlmostEqual(
                snapshot["nav"], snapshot["cash"] + holdings_value, places=8
            )
            self.assertGreaterEqual(snapshot["cash"], -1e-8)

        self.assertEqual(day1["positions"]["600000"]["shares"], 5000)
        self.assertEqual(day2["positions"]["600000"]["shares"], 2300)
        self.assertEqual(day3["positions"], {})
        self.assertAlmostEqual(day3["nav"], day3["cash"], places=8)

        total_commission = sum(trade["commission"] for trade in self.engine.trades)
        total_stamp_tax = sum(trade["stamp_tax"] for trade in self.engine.trades)
        total_slippage = sum(trade["slippage_cost"] for trade in self.engine.trades)
        self.assertGreater(total_commission, 0)
        self.assertGreater(total_stamp_tax, 0)
        self.assertGreater(total_slippage, 0)
        self.assertTrue(all(trade["fill_price"] != trade["reference_price"]
                            for trade in self.engine.trades))
        self.assertEqual([trade["side"] for trade in self.engine.trades],
                         ["buy", "sell", "sell"])

    def test_no_leverage(self):
        with self.assertRaises(ValueError):
            self.engine.step("2024-01-02", {"600000": 1.1}, {})


if __name__ == "__main__":
    unittest.main()
