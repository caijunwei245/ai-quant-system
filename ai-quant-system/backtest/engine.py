"""Deterministic daily-bar portfolio accounting for historical backtests.

Signals are formed after a session close and must be supplied for the next
session. Orders are filled at the next session's open with configurable
adverse slippage and fees; holdings are marked at that session's close.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class Position:
    shares: int
    avg_cost: float
    last_price: float


class BacktestEngine:
    """Cash + share ledger with next-open execution and close-to-close NAV."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.initial_capital = float(self.config.get("initial_capital", 1_000_000))
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be a positive finite number")
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: list[dict] = []
        self.history: list[dict] = []
        self.commission_rate = float(self.config.get("commission", 0.0003))
        self.min_commission = float(self.config.get("min_commission", 5.0))
        self.stamp_tax_rate = float(self.config.get("stamp_tax", 0.001))
        self.slippage_rate = float(self.config.get("slippage", 0.001))
        self.lot_size = int(self.config.get("lot_size", 100))
        self.max_volume_participation = float(self.config.get("max_volume_participation", 0.10))
        if not np.isfinite(self.max_volume_participation) or not 0 < self.max_volume_participation <= 1:
            raise ValueError("max_volume_participation must be in (0, 1]")
        self.default_limit_pct = float(self.config.get("default_limit_pct", 0.10))
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        for name, value in (("commission", self.commission_rate),
                            ("stamp_tax", self.stamp_tax_rate),
                            ("slippage", self.slippage_rate)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")

    @staticmethod
    def _number(row: Mapping, key: str, default: float = np.nan) -> float:
        try:
            value = float(row.get(key, default))
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def normalize_calendar(calendar: pd.DataFrame, start: str, end: str) -> list[str]:
        """Use an exchange/benchmark date index; never synthesize weekdays."""
        if calendar is None or calendar.empty:
            raise ValueError("benchmark trading calendar is empty; refusing to synthesize weekdays")
        dates = pd.to_datetime(calendar.index, errors="coerce")
        if dates.isna().all() and "date" in calendar.columns:
            dates = pd.to_datetime(calendar["date"], errors="coerce")
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        selected = sorted({d.strftime("%Y-%m-%d") for d in dates if not pd.isna(d) and start_ts <= d.normalize() <= end_ts})
        if not selected:
            raise ValueError(f"benchmark has no trading sessions in [{start}, {end}]")
        return selected

    @staticmethod
    def _bar_for(bars: Mapping[str, object], code: str) -> Optional[dict]:
        value = bars.get(code)
        if value is None:
            return None
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
        return None

    def nav(self) -> float:
        return float(self.cash + sum(p.shares * p.last_price for p in self.positions.values()))

    def _limit_pct(self, code: str, bar: Mapping) -> float:
        explicit = self._number(bar, "limit_pct")
        if np.isfinite(explicit) and 0 < explicit < 1:
            return explicit
        if code.startswith(("300", "301", "688", "689")):
            return 0.20
        if code.startswith(("4", "8")):
            return 0.30
        return self.default_limit_pct

    def _locked(self, code: str, bar: Mapping, side: str) -> bool:
        explicit = bar.get("locked_up" if side == "buy" else "locked_down")
        if explicit is not None and not pd.isna(explicit):
            return bool(explicit)
        limit_price = self._number(bar, "limit_up" if side == "buy" else "limit_down")
        open_price = self._number(bar, "open")
        high = self._number(bar, "high")
        low = self._number(bar, "low")
        if np.isfinite(limit_price) and np.isfinite(open_price):
            return abs(open_price - limit_price) <= max(0.011, limit_price * 1e-4)
        prev_close = self._number(bar, "pre_close")
        if not np.isfinite(prev_close) or prev_close <= 0 or not all(np.isfinite(x) for x in (open_price, high, low)):
            return False
        locked_flat = abs(high - low) <= max(0.011, abs(open_price) * 1e-4)
        if not locked_flat:
            return False
        pct = self._limit_pct(code, bar)
        threshold = prev_close * (1 + pct if side == "buy" else 1 - pct)
        return abs(open_price - threshold) <= max(0.02, threshold * 0.001)

    def _tradable(self, code: str, bar: Mapping, side: str) -> tuple[bool, str]:
        open_price = self._number(bar, "open")
        volume = self._number(bar, "volume", 0.0)
        if not np.isfinite(open_price) or open_price <= 0:
            return False, "missing_or_invalid_open"
        if not np.isfinite(volume) or volume <= 0:
            return False, "missing_or_zero_volume"
        if self._locked(code, bar, side):
            return False, "locked_limit_up" if side == "buy" else "locked_limit_down"
        return True, ""

    def _fee(self, notional: float, side: str) -> tuple[float, float]:
        commission = max(self.min_commission, notional * self.commission_rate) if notional > 0 else 0.0
        stamp_tax = notional * self.stamp_tax_rate if side == "sell" else 0.0
        return commission, stamp_tax

    def _record_trade(self, date: str, code: str, side: str, shares: int,
                      reference_price: float, fill_price: float, reason: str = "") -> None:
        notional = shares * fill_price
        commission, stamp_tax = self._fee(notional, side)
        if side == "buy":
            total_cost = notional + commission + stamp_tax
            if total_cost > self.cash + 1e-8:
                raise RuntimeError("internal error: attempted buy exceeds available cash")
            old = self.positions.get(code)
            old_shares = old.shares if old else 0
            old_cost = old.avg_cost * old_shares if old else 0.0
            self.cash -= total_cost
            new_shares = old_shares + shares
            self.positions[code] = Position(new_shares, (old_cost + notional) / new_shares, reference_price)
        else:
            position = self.positions[code]
            if shares > position.shares:
                raise RuntimeError("internal error: attempted sell exceeds held shares")
            self.cash += notional - commission - stamp_tax
            remaining = position.shares - shares
            if remaining:
                position.shares = remaining
                position.last_price = reference_price
            else:
                del self.positions[code]
        self.trades.append({
            "date": date, "code": code, "side": side, "shares": int(shares),
            "reference_price": float(reference_price), "fill_price": float(fill_price),
            "notional": float(notional), "commission": float(commission),
            "stamp_tax": float(stamp_tax), "slippage_cost": float(abs(fill_price-reference_price) * shares),
            "reason": reason,
        })

    def step(self, date: str, target_weights: Mapping[str, float],
             bars: Mapping[str, object]) -> dict:
        """Rebalance at today's open using yesterday's signal; mark at today's close."""
        targets: Dict[str, float] = {}
        for code, raw_weight in (target_weights or {}).items():
            weight = float(raw_weight)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"invalid target weight for {code}: {raw_weight}")
            if weight > 0:
                targets[str(code)] = weight
        if sum(targets.values()) > 1.000001:
            raise ValueError("target weights exceed 100%; cash-only portfolio does not allow leverage")

        # Mark existing holdings to today's open before sizing target shares.
        open_marks = {}
        for code, position in self.positions.items():
            bar = self._bar_for(bars, code)
            if bar:
                op = self._number(bar, "open")
                if np.isfinite(op) and op > 0:
                    open_marks[code] = op
        nav_open = self.cash + sum(
            p.shares * open_marks.get(code, p.last_price)
            for code, p in self.positions.items()
        )
        desired: Dict[str, int] = {}
        for code, weight in targets.items():
            bar = self._bar_for(bars, code)
            op = self._number(bar, "open") if bar else np.nan
            if np.isfinite(op) and op > 0:
                desired[code] = int(floor((nav_open * weight / op) / self.lot_size) * self.lot_size)

        # Sell reductions first to release cash. A full exit may sell odd lots.
        all_codes = set(self.positions) | set(desired)
        for code in sorted(all_codes):
            position = self.positions.get(code)
            if not position:
                continue
            want = desired.get(code, 0)
            if want >= position.shares:
                continue
            bar = self._bar_for(bars, code)
            if not bar:
                continue
            ok, reason = self._tradable(code, bar, "sell")
            if not ok:
                continue
            sell_shares = position.shares - want
            volume = self._number(bar, "volume", 0.0)
            max_fill = int(floor(volume * self.max_volume_participation / self.lot_size) * self.lot_size)
            sell_shares = min(sell_shares, max_fill)
            if want > 0:
                sell_shares = int(floor(sell_shares / self.lot_size) * self.lot_size)
            if sell_shares <= 0:
                continue
            ref = self._number(bar, "open")
            fill = ref * (1 - self.slippage_rate)
            self._record_trade(date, code, "sell", sell_shares, ref, fill)

        # Buy increases second, capped by settled cash and board lots.
        for code in sorted(all_codes):
            bar = self._bar_for(bars, code)
            if not bar:
                continue
            ref = self._number(bar, "open")
            if not np.isfinite(ref) or ref <= 0:
                continue
            current = self.positions.get(code).shares if code in self.positions else 0
            want = desired.get(code, current)
            add_shares = max(0, want - current)
            if add_shares <= 0:
                continue
            ok, reason = self._tradable(code, bar, "buy")
            if not ok:
                continue
            fill = ref * (1 + self.slippage_rate)
            per_share_cost = fill * (1 + self.commission_rate + self.stamp_tax_rate)
            affordable = max(0, int(floor((self.cash - self.min_commission) / per_share_cost / self.lot_size) * self.lot_size))
            volume = self._number(bar, "volume", 0.0)
            max_fill = int(floor(volume * self.max_volume_participation / self.lot_size) * self.lot_size)
            buy_shares = min(add_shares, affordable, max_fill)
            while buy_shares > 0:
                notional = buy_shares * fill
                commission, _ = self._fee(notional, "buy")
                if notional + commission <= self.cash + 1e-8:
                    break
                buy_shares -= self.lot_size
            if buy_shares > 0:
                self._record_trade(date, code, "buy", buy_shares, ref, fill)

        # Close marks are mandatory for current positions; missing close keeps the
        # last mark but is surfaced so callers can reject incomplete histories.
        stale = []
        for code, position in list(self.positions.items()):
            bar = self._bar_for(bars, code)
            close = self._number(bar, "close") if bar else np.nan
            if np.isfinite(close) and close > 0:
                position.last_price = close
            else:
                stale.append(code)
        value = self.nav()
        snapshot = {
            "date": date, "nav": value, "cash": float(self.cash),
            "market_value": float(value - self.cash),
            "gross_exposure": float((value - self.cash) / value) if value > 0 else 0.0,
            "n_positions": len(self.positions), "stale_marks": stale,
            "positions": {code: {"shares": p.shares, "avg_cost": p.avg_cost,
                                 "last_price": p.last_price,
                                 "market_value": p.shares * p.last_price}
                          for code, p in self.positions.items()},
        }
        self.history.append(snapshot)
        return snapshot
