from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .core_types import Fill, PortfolioState, Qty, Price, Cash, Signals
from .multi_data_sources import MultiMarketData
from .execution import ExecutionParams


class MultiNextBarExecutionModel:
    """
    Multi-asset исполнение на общем таймлайне.

    Контракты/упрощения для A3:
    - все символы имеют одинаковый DatetimeIndex (MultiMarketData уже выравнивает)
    - сигналы: dict[symbol] -> Signals с target_fraction ∈ [0,1]
    - long-only
    - cash общий
    - порядок исполнения в пределах бара: фиксированный (по symbols list)
    """

    def __init__(self, params: Optional[ExecutionParams] = None):
        self.params = params or ExecutionParams()
        if self.params.fill_price not in ("open", "close"):
            raise ValueError("fill_price должен быть 'open' или 'close'")

    def run(
        self,
        mmd: MultiMarketData,
        sigs: Dict[str, Signals],
        initial_state: PortfolioState,
    ) -> Tuple[List[Fill], List[PortfolioState]]:

        symbols = mmd.symbols
        idx = mmd.index

        # 1) Проверка индексов сигналов
        for sym in symbols:
            if sym not in sigs:
                raise ValueError(f"Missing signals for symbol={sym}")
            if not sigs[sym].target_position.index.equals(idx):
                raise ValueError(f"Signals index must match bars index for {sym}")

        fill_col = "Open" if self.params.fill_price == "open" else "Close"

        fills: List[Fill] = []
        states: List[PortfolioState] = []

        cash: Cash = initial_state.cash
        positions: dict[str, Qty] = dict(initial_state.positions)
        last_prices: dict[str, Price] = dict(initial_state.last_prices)

        def compute_equity(cash_: Cash, pos_: dict[str, Qty], lp_: dict[str, Price]) -> float:
            eq = float(cash_)
            for sym, qty in pos_.items():
                px = lp_.get(sym)
                if px is not None:
                    eq += float(qty) * float(px)
            return float(eq)

        fee_rate = self.params.fee_bps / 10_000.0
        slip_rate = self.params.slippage_bps / 10_000.0

        # 2) Общий проход по времени
        for ts in idx:
            # 2.1) Сначала обновляем last_prices по Close для всех символов
            for sym in symbols:
                bars = mmd.data[sym].bars
                close_px = float(bars.at[ts, "Close"])
                last_prices[sym] = close_px

            # 2.2) Затем исполняем по каждому символу
            for sym in symbols:
                bars = mmd.data[sym].bars
                row = bars.loc[ts]

                close_price = float(row["Close"])
                cur_qty: Qty = positions.get(sym, 0.0)

                # target_fraction ∈ [0,1]
                desired_raw = sigs[sym].target_position.loc[ts]
                target_fraction = 0.0 if pd.isna(desired_raw) else float(desired_raw)
                if not (0.0 <= target_fraction <= 1.0):
                    raise ValueError(f"target_fraction ∈ [0,1], got {target_fraction} for {sym} at {ts}")

                # sizing по Close
                equity_for_sizing = compute_equity(cash, positions, last_prices)
                if close_price <= 0:
                    raise ValueError(f"Close price must be positive, got {close_price} for {sym} at {ts}")
                desired_qty: Qty = (target_fraction * equity_for_sizing) / close_price

                delta: Qty = desired_qty - cur_qty
                if abs(delta) <= self.params.eps:
                    continue

                raw_price = float(row[fill_col])

                # slippage
                exec_price = raw_price * (1.0 + slip_rate) if delta > 0 else raw_price * (1.0 - slip_rate)

                # A1.1 constraint: do not overspend on buys
                if delta > 0:
                    denom = exec_price * (1.0 + fee_rate)
                    max_buy_qty = cash / denom if denom > 0 else 0.0
                    if max_buy_qty < 0:
                        max_buy_qty = 0.0
                    if delta > max_buy_qty:
                        delta = max_buy_qty

                if abs(delta) <= self.params.eps:
                    continue

                notional = abs(delta) * exec_price
                fee = notional * fee_rate

                cash -= delta * exec_price
                cash -= fee
                if abs(cash) < self.params.eps:
                    cash = 0.0

                new_qty: Qty = cur_qty + delta
                if abs(new_qty) < self.params.eps:
                    positions.pop(sym, None)
                else:
                    positions[sym] = new_qty

                fills.append(
                    Fill(
                        symbol=sym,
                        ts=pd.Timestamp(ts),
                        price=float(exec_price),
                        qty=float(delta),
                        fee=float(fee),
                    )
                )

            # 2.3) Snapshot состояния (после обработки всех символов на этом ts)
            equity = compute_equity(cash, positions, last_prices)
            states.append(
                PortfolioState(
                    cash=float(cash),
                    positions=dict(positions),
                    last_prices=dict(last_prices),
                    equity=float(equity),
                )
            )

        return fills, states
