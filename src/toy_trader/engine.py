from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .core_types import MarketData, Signals, Fill, PortfolioState
from .data_sources import DataSource
from .strategies import Strategy
from .execution import ExecutionModel


@dataclass(frozen=True)
class BacktestResult:
    """
    Минимальный результат прогона.

    Позже сюда добавим:
    - equity curve (Series)
    - метрики
    - ссылки на отчёты/артефакты
    """
    market_data: MarketData
    signals: Signals
    fills: List[Fill]
    states: List[PortfolioState]


class BacktestEngine:
    """
    Оркестратор backtest-прогона.

    Задача engine — не "умничать", а:
    - вызвать модули в правильном порядке
    - передать данные между ними
    - собрать результат
    """

    def __init__(
        self,
        data_source: DataSource,
        strategy: Strategy,
        execution_model: ExecutionModel,
        initial_cash: float = 10_000.0,
    ):
        self.data_source = data_source
        self.strategy = strategy
        self.execution_model = execution_model
        self.initial_cash = float(initial_cash)

    def run(self) -> BacktestResult:
        md = self.data_source.get_bars()
        sig = self.strategy.generate_signals(md)

        initial_state = PortfolioState(cash=self.initial_cash)
        fills, states = self.execution_model.run(md, sig, initial_state)

        return BacktestResult(
            market_data=md,
            signals=sig,
            fills=fills,
            states=states,
        )
