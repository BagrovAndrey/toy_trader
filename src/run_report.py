from __future__ import annotations

import pandas as pd

from toy_trader.data_sources import YahooDataSource
from toy_trader.strategies import SMACrossStrategy
from toy_trader.execution import NextBarExecutionModel
from toy_trader.engine import BacktestEngine
from toy_trader.multi_data_sources import MultiYahooDataSource

# Если ты уже сделал reporting.py — используем его.
# Если нет, можно убрать эти импорты и оставить только PnL-графики.
try:
    from toy_trader.reporting import equity_curve, basic_metrics, drawdown_curve
except Exception:
    equity_curve = None
    basic_metrics = None
    drawdown_curve = None

def main() -> None:

    # --- 0) Временно: тестирование много-ассетного лоадера

    ds = MultiYahooDataSource(symbols=["AMZN", "PLTR", "XOM"], start="2020-01-01")
    mmd = ds.get_bars()

    print(mmd.symbols)
    print(len(mmd.index), mmd.index[0], mmd.index[-1])
    for s in mmd.symbols:
        print(s, mmd.data[s].bars.shape)

    # --- 1) Настройки эксперимента (можешь менять руками)
    symbol = "PLTR"
    start = "2020-01-01"
    initial_cash = 10_000.0

    # --- 2) Сборка компонентов
    ds = YahooDataSource(symbol=symbol, start=start)
    strategy = SMACrossStrategy()
    execution = NextBarExecutionModel()

    engine = BacktestEngine(
        data_source=ds,
        strategy=strategy,
        execution_model=execution,
        initial_cash=initial_cash,
    )

    # --- 3) Прогон backtest
    res = engine.run()
    
    # A1.1 sanity: cash should not go meaningfully negative (spot, no margin)
    min_cash = min(s.cash for s in res.states)
    assert min_cash >= -1e-6, f"cash went negative: min_cash={min_cash}"
    
    symbol = res.market_data.symbol
    last_pos = res.states[-1].positions.get(symbol, 0.0)
    print("DEBUG last position qty:", last_pos)

    assert len(res.states) == len(res.market_data.bars)
    assert all(isinstance(f.qty, float) for f in res.fills)
    assert all(
        isinstance(q, float)
        for s in res.states
        for q in s.positions.values()
    )

    print(f"symbol={symbol}, start={start}")
    print(f"bars={len(res.market_data.bars)}, fills={len(res.fills)}, states={len(res.states)}")

    # --- 4) Equity curve
    if equity_curve is not None:
        eq = equity_curve(res)
    else:
        idx = res.market_data.bars.index
        eq = pd.Series([s.equity for s in res.states], index=idx, name="equity")
        
    if drawdown_curve is not None:
        dd = drawdown_curve(eq)
    else:
        dd = None

    # --- 5) Метрики (если есть reporting.basic_metrics)
    if basic_metrics is not None:
        m = basic_metrics(res)
        print("\nMETRICS")
        print(f"total_return: {m.total_return:.4f}")
        print(f"cagr:         {m.cagr:.4f}")
        print(f"vol_annual:   {m.vol_annual:.4f}")
        print(f"sharpe:       {m.sharpe:.4f}")
        print(f"max_drawdown: {m.max_drawdown:.4f}")
        print(f"fills:        {len(res.fills)}")
    else:
        print("\nMETRICS: reporting.py not found/failed to import (skipping)")

    # --- 6) Корректная визуализация для вашего текущего sizing (1 unit)
    price = res.market_data.bars["Close"]

    pnl_strategy = eq - eq.iloc[0]          # $ PnL стратегии относительно initial_cash
    qty_bh = initial_cash / price.iloc[0]    # Buy&Hold
    equity_bh = qty_bh * price
    pnl_buyhold = equity_bh - initial_cash   

    # Позиция во времени (0/1) — полезно увидеть “когда мы в рынке”
    position = res.signals.target_position

    # --- 7) Графики
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 5))
    plt.plot(price, label="Price (Close)")
    plt.plot(position * price.max() * 0.05, label="Position (scaled)")  # маленькая полоска-индикатор
    plt.legend()
    plt.title(f"{symbol} price and position")
    plt.show()

    plt.figure(figsize=(11, 5))
    plt.plot(pnl_strategy, label="Strategy PnL ($)")
    plt.plot(pnl_buyhold, label="Buy&Hold all-in PnL ($)")
    plt.axhline(0.0)
    plt.legend()
    plt.title(f"{symbol}: PnL comparison (correct for 1-unit sizing)")
    plt.show()
    
    plt.figure(figsize=(11, 4))
    plt.plot(eq, label="Equity ($)")
    plt.legend()
    plt.title(f"{symbol}: Equity curve")
    plt.show()

    if dd is not None:
        plt.figure(figsize=(11, 3))
        plt.plot(dd, label="Drawdown")
        plt.axhline(0.0)
        plt.legend()
        plt.title(f"{symbol}: Drawdown")
        plt.show()


if __name__ == "__main__":
    main()
