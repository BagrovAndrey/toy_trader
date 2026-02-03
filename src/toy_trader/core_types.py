from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import pandas as pd

# ---- numeric aliases (MVP uses float) ----
Qty: TypeAlias = float
Price: TypeAlias = float
Cash: TypeAlias = float
Fee: TypeAlias = float
Equity: TypeAlias = float

@dataclass(frozen=True)
class MarketData:
    """
    Рыночные данные в формате bars (свечи).

    Контракт:
    - bars: DataFrame с индексом DateTime/Date (монотонно возрастающий)
      колонки минимум: Open, High, Low, Close, Volume
    - symbol: идентификатор инструмента (например, 'SPY', 'BTC-USD')
    """
    bars: pd.DataFrame
    symbol: str


@dataclass(frozen=True)
class Signals:
    """
    Сигналы стратегии в форме целевой позиции (target position).

    Контракт:
    - target_position: Series с индексом времени (совместим с MarketData.bars.index)
      значения: 0/1 (long-only в MVP)

    Смысл:
    - это намерение/целевое состояние, не приказы на покупку/продажу.
    """
    target_position: pd.Series


@dataclass(frozen=True)
class Fill:
    """
    Факт исполнения сделки (fill).

    Контракт:
    - symbol: инструмент
    - ts: время исполнения
    - price: цена исполнения
    - qty: количество (+N buy, -N sell)
    - fee: комиссия/издержки в валюте счёта
    """
    symbol: str
    ts: pd.Timestamp
    price: Price
    qty: Qty
    fee: Fee


@dataclass
class PortfolioState:
    """
    Состояние портфеля (обобщаемо на multi-asset).

    Контракт:
    - cash: деньги
    - positions: symbol -> qty (qty может быть <0 для short; пока не используем)
    - last_prices: symbol -> price (для mark-to-market)
    - equity: cash + sum(qty_i * price_i)
    """
    cash: float
    positions: Dict[str, Qty] = field(default_factory=dict)
    last_prices: Dict[str, Price] = field(default_factory=dict)
    equity: Equity = 0.0
