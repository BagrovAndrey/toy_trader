from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .engine import BacktestResult


@dataclass(frozen=True)
class BasicMetrics:
    total_return: float
    cagr: float
    vol_annual: float
    sharpe: float
    max_drawdown: float
    num_trades: int


def equity_curve(res: BacktestResult) -> pd.Series:
    """
    Возвращает equity curve как Series с индексом времени.
    """
    idx = res.market_data.bars.index
    eq = pd.Series([s.equity for s in res.states], index=idx, name="equity")
    return eq


def max_drawdown(equity: pd.Series) -> float:
    """
    Максимальная просадка (в долях, например -0.23 = -23%).
    """
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def basic_metrics(res: BacktestResult, bars_per_year: int = 252) -> BasicMetrics:
    """
    Базовые метрики из equity curve.
    bars_per_year:
      - дневные бары ~252
      - часовые ~24*365=8760 (приблизительно)
      - 15m ~ 4*24*365=35040
    """
    eq = equity_curve(res)
    if len(eq) < 2:
        raise ValueError("Недостаточно точек для метрик")

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    # лог-доходности для стабильности
    rets = (eq / eq.shift(1)).apply(lambda x: pd.NA if pd.isna(x) else x)
    rets = pd.to_numeric(rets, errors="coerce")
    rets = rets.dropna() - 1.0

    # CAGR по приближению через количество баров
    years = len(eq) / float(bars_per_year)
    if years <= 0:
        cagr = float("nan")
    else:
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)

    vol_annual = float(rets.std(ddof=1) * (bars_per_year ** 0.5)) if len(rets) > 1 else float("nan")
    mean_annual = float(rets.mean() * bars_per_year) if len(rets) > 0 else float("nan")
    sharpe = float(mean_annual / vol_annual) if vol_annual and vol_annual == vol_annual else float("nan")

    mdd = max_drawdown(eq)
    num_trades = len(res.fills)

    return BasicMetrics(
        total_return=total_return,
        cagr=cagr,
        vol_annual=vol_annual,
        sharpe=sharpe,
        max_drawdown=mdd,
        num_trades=num_trades,
    )
