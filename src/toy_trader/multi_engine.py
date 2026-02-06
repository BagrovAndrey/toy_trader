from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .core_types import Signals, Fill, PortfolioState
from .multi_data_sources import MultiMarketData, MultiYahooDataSource
from .strategies import Strategy
from .multi_execution import MultiNextBarExecutionModel


@dataclass(frozen=True)
class MultiBacktestResult:
    market_data: MultiMarketData
    signals: Dict[str, Signals]
    fills: List[Fill]
    states: List[PortfolioState]


class MultiBacktestEngine:
    def __init__(
        self,
        data_source: MultiYahooDataSource,
        strategy: Strategy,
        execution_model: MultiNextBarExecutionModel,
        initial_cash: float = 10_000.0,
    ):
        self.data_source = data_source
        self.strategy = strategy
        self.execution_model = execution_model
        self.initial_cash = float(initial_cash)

    def run(self) -> MultiBacktestResult:
        mmd = self.data_source.get_bars()

        # сигналы по каждому символу
        sigs: Dict[str, Signals] = {}
        for sym, md in mmd.data.items():
            sigs[sym] = self.strategy.generate_signals(md)

        fills, states = self.execution_model.run(
            mmd=mmd,
            sigs=sigs,
            initial_state=PortfolioState(cash=self.initial_cash),
        )

        return MultiBacktestResult(
            market_data=mmd,
            signals=sigs,
            fills=fills,
            states=states,
        )
