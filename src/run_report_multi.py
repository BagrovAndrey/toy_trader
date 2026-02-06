from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from toy_trader.multi_data_sources import MultiYahooDataSource
from toy_trader.multi_engine import MultiBacktestEngine
from toy_trader.multi_execution import MultiNextBarExecutionModel
from toy_trader.execution import ExecutionParams
from toy_trader.strategies import SMACrossStrategy, SMACrossParams  # params/class names may differ in your repo
from toy_trader.reporting import metrics_from_equity, drawdown_curve

def main() -> None:
    # Minimal CLI (no configs yet)
    symbols = ["AMZN", "PLTR", "XOM"]
    start = "2020-01-01"
    end = None
    initial_cash = 10_000.0

    # Strategy params: adjust to your actual param class if needed
    # If your strategy takes a params dataclass, edit these two lines accordingly.
    strategy = SMACrossStrategy(SMACrossParams(fast=10, slow=40, shift_for_execution=1))  # <-- if your SMACrossStrategy needs params, change this

    ds = MultiYahooDataSource(symbols=symbols, start=start, end=end)
    ex = MultiNextBarExecutionModel(ExecutionParams(fill_price="open", fee_bps=1.0, slippage_bps=1.0))
    engine = MultiBacktestEngine(ds, strategy, ex, initial_cash=initial_cash)

    res = engine.run()

    idx = res.market_data.index

    # 1) Equity curve from states
    eq = pd.Series([s.equity for s in res.states], index=idx, name="equity").astype(float)

    # 2) Metrics
    m = metrics_from_equity(eq)

    print("METRICS")
    print(f"  total_return: {m.total_return:.4f}")
    print(f"  cagr:         {m.cagr:.4f}")
    print(f"  vol_annual:   {m.vol_annual:.4f}")
    print(f"  sharpe:       {m.sharpe:.4f}")
    print(f"  max_drawdown: {m.max_drawdown:.4f}")

    # 3) Per-symbol qty series
    qty_df = pd.DataFrame(
        {sym: [st.positions.get(sym, 0.0) for st in res.states] for sym in res.market_data.symbols},
        index=idx,
    )

    # 4) Plots
    plt.figure(figsize=(12, 4))
    plt.plot(eq, label="Equity ($)")
    plt.title(f"Multi-asset equity ({', '.join(symbols)})")
    plt.legend()
    plt.show()

    if drawdown_curve is not None:
        dd = drawdown_curve(eq)
        plt.figure(figsize=(12, 3))
        plt.plot(dd, label="Drawdown")
        plt.axhline(0.0)
        plt.title("Drawdown")
        plt.legend()
        plt.show()

    plt.figure(figsize=(12, 4))
    for sym in qty_df.columns:
        plt.plot(qty_df[sym], label=f"{sym} qty")
    plt.title("Position quantities (per symbol)")
    plt.legend()
    plt.show()

    print(f"bars={len(idx)}, fills={len(res.fills)}, states={len(res.states)}")
    print("last positions:", res.states[-1].positions)


if __name__ == "__main__":
    main()
