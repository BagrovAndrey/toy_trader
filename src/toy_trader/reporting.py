from __future__ import annotations

"""
toy_trader.reporting

Назначение (этап A2/A3):
- дать стандартизованный "research output" из результата backtest:
  equity curve, drawdown и базовые метрики.
- модуль НЕ содержит торговой логики, он только анализирует output.

Ключевой дизайн:
- для single-asset backtest удобно брать equity из BacktestResult (equity_curve(res)).
- для multi-asset backtest (MultiBacktestResult) скрипт сам собирает equity Series из states,
  а затем вызывает metrics_from_equity(eq, num_trades=...).

Мы используем calendar-based annualization (365.25) по твоему решению,
т.к. целевая среда — crypto 24/7.
"""

from dataclasses import dataclass
import math
import pandas as pd

from .engine import BacktestResult


@dataclass(frozen=True)
class BasicMetrics:
    """
    Минимальный набор метрик для сравнения стратегий.

    total_return : итоговая доходность за период (например, 0.35 = +35%)
    cagr         : годовая сложная доходность
    vol_annual   : годовая волатильность (annualized std of returns)
    sharpe       : Sharpe ratio при risk-free = 0
    max_drawdown : максимальная просадка (отрицательное число)
    num_trades   : количество сделок (len(fills) в нашем контракте)
    """
    total_return: float
    cagr: float
    vol_annual: float
    sharpe: float
    max_drawdown: float
    num_trades: int


def equity_curve(res: BacktestResult) -> pd.Series:
    """
    Equity curve ($) как Series с индексом времени.

    В single-asset BacktestResult источник времени — market_data.bars.index,
    а значения equity берём из states (PortfolioState.equity).
    """
    idx = res.market_data.bars.index
    eq = pd.Series([s.equity for s in res.states], index=idx, name="equity").astype(float)
    return eq


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """
    Drawdown curve (в долях, <= 0), по определению:
      peak(t) = max_{u<=t} equity(u)
      dd(t)   = equity(t) / peak(t) - 1

    Пример: equity упала с 100 до 80 => dd = -0.2 (-20%)
    """
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(equity: pd.Series) -> float:
    """
    Максимальная просадка (минимум drawdown curve).
    Возвращаем отрицательное число (например, -0.23 = -23%).
    """
    dd = drawdown_curve(equity)
    return float(dd.min())


def _periods_per_year(idx: pd.DatetimeIndex) -> float:
    """
    Calendar-based annualization factor.

    Мы считаем, что время непрерывное (24/7), что соответствует крипте.
    - Для дневных баров получится ~365.25
    - Для intraday оцениваем по медианному шагу времени.

    Важно: это задаёт "масштаб" Sharpe/vol. Для сравнения стратегий внутри
    одной и той же частоты данных этого достаточно и консистентно.
    """
    if len(idx) < 3:
        return 365.25

    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        return 365.25

    med = deltas.median()
    seconds = float(med.total_seconds())
    if seconds <= 0:
        return 365.25

    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / seconds


def metrics_from_equity(eq: pd.Series, num_trades: int = 0) -> BasicMetrics:
    """
    Универсальные метрики по equity curve.

    Работает и для single-asset, и для multi-asset, потому что вход — только equity Series.
    num_trades передаётся снаружи (для single мы возьмём len(res.fills), для multi аналогично).
    """
    eq = eq.astype(float)

    # bar-to-bar returns
    rets = eq.pct_change().dropna()

    # total return
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    # CAGR
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 24 * 3600)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    # annualization
    ppy = _periods_per_year(eq.index)

    # annualized vol and Sharpe (rf = 0)
    if len(rets) > 1:
        vol_annual = float(rets.std(ddof=0) * math.sqrt(ppy))
    else:
        vol_annual = float("nan")

    if len(rets) > 1 and rets.std(ddof=0) > 0:
        sharpe = float((rets.mean() / rets.std(ddof=0)) * math.sqrt(ppy))
    else:
        sharpe = float("nan")

    # max drawdown
    mdd = max_drawdown(eq)

    return BasicMetrics(
        total_return=total_return,
        cagr=cagr,
        vol_annual=vol_annual,
        sharpe=sharpe,
        max_drawdown=mdd,
        num_trades=int(num_trades),
    )


def basic_metrics(res: BacktestResult) -> BasicMetrics:
    """
    Базовые метрики из результата single-asset backtest.

    Здесь считаем:
    - eq = equity_curve(res)
    - num_trades = len(res.fills) (в нашем контракте Fill = факт сделки)
    """
    eq = equity_curve(res)
    num_trades = len(res.fills)
    return metrics_from_equity(eq, num_trades=num_trades)
