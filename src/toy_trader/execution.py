from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd

from .core_types import MarketData, Signals, Fill, PortfolioState


@dataclass(frozen=True)
class ExecutionParams:
    """
    Параметры модели исполнения (toy-уровень).

    fee_bps: комиссия в bps (1 bps = 0.01%)
    slippage_bps: проскальзывание в bps
    fill_price: какую цену бара считаем ценой исполнения
      - "open"  : исполняем на Open
      - "close" : исполняем на Close (часто некорректно для честного backtest, но полезно для экспериментов)
    """
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    fill_price: str = "open"


class ExecutionModel(ABC):
    """
    Интерфейс модели исполнения.

    Контракт:
    - вход: MarketData, Signals, начальный PortfolioState
    - выход: (fills, states) — список фактов сделок и список состояний портфеля во времени

    Важно:
    - execution НЕ генерирует сигнал
    - execution НЕ оптимизирует стратегию
    - execution только реализует target_position
    """

    @abstractmethod
    def run(
        self,
        md: MarketData,
        sig: Signals,
        initial_state: PortfolioState,
    ) -> Tuple[List[Fill], List[PortfolioState]]:
        raise NotImplementedError


class NextBarExecutionModel(ExecutionModel):
    """
    Заглушка toy-исполнения.

    Идея (которую реализуем позже):
    - на каждом баре t смотрим target_position(t)
    - если хотим быть в позиции, то держим 100% (или заданную долю) капитала в активе
    - если не хотим — держим кэш
    - сделки происходят только при смене target_position
    - цена исполнения берётся из бара (open/close) + комис/проскальзывание
    """

    def __init__(self, params: Optional[ExecutionParams] = None):
        self.params = params or ExecutionParams()
        if self.params.fill_price not in ("open", "close"):
            raise ValueError("fill_price должен быть 'open' или 'close'")

    def run(
        self,
        md: MarketData,
        sig: Signals,
        initial_state: PortfolioState,
    ) -> Tuple[List[Fill], List[PortfolioState]]:
        raise NotImplementedError
