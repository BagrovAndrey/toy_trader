from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .core_types import MarketData, Signals


class Strategy(ABC):
    """
    Интерфейс стратегии.

    Контракт:
    - вход: MarketData (bars одного инструмента)
    - выход: Signals (target_position по времени)

    Стратегия НЕ:
    - считает комиссии
    - исполняет сделки
    - знает про лаги исполнения (мы решим, где это живёт, позже)
    """

    @abstractmethod
    def generate_signals(self, md: MarketData) -> Signals:
        raise NotImplementedError


@dataclass(frozen=True)
class SMACrossParams:
    fast: int = 10
    slow: int = 40
    shift_for_execution: int = 1  # лаг "на следующий бар" (можно позже перенести в execution)


class SMACrossStrategy(Strategy):
    """
    Заглушка под SMA crossover.

    Идея:
    - считаем две SMA по Close
    - target_position = 1, если SMA_fast > SMA_slow, иначе 0

    Пока это каркас: реализацию добавим отдельно.
    """

    def __init__(self, params: Optional[SMACrossParams] = None):
        self.params = params or SMACrossParams()

        if self.params.fast <= 0 or self.params.slow <= 0:
            raise ValueError("fast и slow должны быть положительными")
        if self.params.fast >= self.params.slow:
            raise ValueError("ожидается fast < slow")

    def generate_signals(self, md: MarketData) -> Signals:
        raise NotImplementedError
