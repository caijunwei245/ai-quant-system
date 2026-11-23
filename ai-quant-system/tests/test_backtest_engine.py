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
        self.assertAlmostEqual(day1["cash"], 49934.985, places=6)
        self.assertAlmostEqual(day1["market_value"], 50000.0, places=6)
        self.assertAlmostEqual(day1["nav"], 99934.985, places=6)

        self.assertEqual(day2["positions"]["600000"]["shares"], 2300)
        self.assertAlmostEqual(day2["cash"], 79566.71361, places=6)
        self.assertAlmostEqual(day2["market_value"], 27600.0, places=6)
        self.assertAlmostEqual(day2["nav"], 107166.71361, places=6)

        self.assertEqual(day3["positions"], {})
        self.assertAlmostEqual(day3["cash"], 107103.26949, places=6)
        self.assertAlmostEqual(day3["nav"], day3["cash"], places=8)

        total_commission = sum(trade["commission"] for trade in self.engine.trades)
        self.assertAlmostEqual(total_commission, 32.18781, places=6)
        total_stamp_tax = sum(trade["stamp_tax"] for trade in self.engine.trades)
        total_slippage = sum(trade["slippage_cost"] for trade in self.engine.trades)
        self.assertGreater(total_commission, 0)
        self.assertAlmostEqual(total_stamp_tax, 57.2427, places=6)
        self.assertAlmostEqual(total_slippage, 107.3, places=6)
        self.assertTrue(all(trade["fill_price"] != trade["reference_price"]
                            for trade in self.engine.trades))
        self.assertEqual([trade["side"] for trade in self.engine.trades],
                         ["buy", "sell", "sell"])


    def test_calendar_normalizes_timezone_and_deduplicates_sessions(self):
        calendar = pd.DataFrame(index=pd.DatetimeIndex([
            "2024-01-08 09:30:00+08:00",
            "2024-01-08 15:00:00+08:00",
            "2024-01-09 15:00:00+08:00",
        ]))
        self.assertEqual(
            self.engine.normalize_calendar(calendar, "2024-01-08", "2024-01-09"),
            ["2024-01-08", "2024-01-09"],
        )

    def test_calendar_rejects_reversed_bounds(self):
        calendar = pd.DataFrame(index=pd.to_datetime(["2024-01-08"]))
        with self.assertRaises(ValueError):
            self.engine.normalize_calendar(calendar, "2024-01-09", "2024-01-08")

    def test_calendar_date_column_with_timezone_is_normalized(self):
        calendar = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-08 09:30:00+08:00",
                "2024-01-09 09:30:00+08:00",
            ])
        })
        self.assertEqual(
            self.engine.normalize_calendar(calendar, "2024-01-08", "2024-01-09"),
            ["2024-01-08", "2024-01-09"],
        )

    def test_missing_close_for_held_position_fails_closed(self):
        self.engine.step(
            "2024-01-02", {"600000": 0.5},
            {"600000": {"open": 10, "high": 10.2, "low": 9.9,
                        "close": 10, "volume": 100000}},
        )
        with self.assertRaisesRegex(ValueError, "cannot calculate a reliable NAV"):
            self.engine.step(
                "2024-01-03", {"600000": 0.5},
                {"600000": {"open": 10.5, "high": 10.8, "low": 10.4,
                            "volume": 100000}},
            )

    def test_invalid_fee_and_limit_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            BacktestEngine({"min_commission": -1})
        with self.assertRaises(ValueError):
            BacktestEngine({"default_limit_pct": 1.0})


    def test_volume_unit_lots_caps_fill_in_share_units(self):
        engine = BacktestEngine({
            "initial_capital": 100000,
            "volume_unit": "lots",
            "max_volume_participation": 0.10,
        })
        result = engine.step(
            "2024-01-02", {"600000": 0.5},
            {"600000": {"open": 10, "high": 10.2, "low": 9.9,
                        "close": 10, "volume": 10}},
        )
        self.assertEqual(result["positions"]["600000"]["shares"], 100)

    def test_locked_limit_up_buy_and_limit_down_sell_are_blocked(self):
        engine = BacktestEngine({"initial_capital": 100000})
        buy = engine.step(
            "2024-01-02", {"600000": 0.5},
            {"600000": {"open": 10, "high": 10, "low": 10,
                        "close": 10, "volume": 100000, "locked_up": True}},
        )
        self.assertEqual(buy["n_positions"], 0)
        engine.step(
            "2024-01-03", {"600000": 0.5},
            {"600000": {"open": 10, "high": 10.2, "low": 9.9,
                        "close": 10, "volume": 100000}},
        )
        shares_before = engine.positions["600000"].shares
        held = engine.step(
            "2024-01-04", {},
            {"600000": {"open": 9, "high": 9, "low": 9,
                        "close": 9, "volume": 100000, "locked_down": True}},
        )
        self.assertEqual(held["positions"]["600000"]["shares"], shares_before)

    def test_no_leverage(self):
        with self.assertRaises(ValueError):
            self.engine.step("2024-01-02", {"600000": 1.1}, {})


if __name__ == "__main__":
    unittest.main()
