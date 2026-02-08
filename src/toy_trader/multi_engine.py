from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .core_types import Signals, Fill, PortfolioState
from .multi_data_sources import MultiMarketData, MultiYahooDataSource
from .strategies import Strategy
from .multi_execution import MultiNextBarExecutionModel
from .allocation import AllocationModel, ProportionalAllocator


@dataclass(frozen=True)
class MultiBacktestResult:
    """
    Результат multi-asset backtest.

    ВАЖНО:
    - signals здесь уже ПОСЛЕ allocation
    - т.е. target_position = портфельный вес
    """
    market_data: MultiMarketData
    signals: Dict[str, Signals]
    fills: List[Fill]
    states: List[PortfolioState]


class MultiBacktestEngine:
    """
    Оркестратор multi-asset backtest.

    Ответственность:
    - загрузить данные
    - получить raw-сигналы стратегий
    - применить allocation (portfolio construction)
    - передать веса в execution
    """

    def __init__(
        self,
        data_source: MultiYahooDataSource,
        strategy: Strategy,
        execution_model: MultiNextBarExecutionModel,
        allocator: Optional[AllocationModel] = None,
        allocation_budget: float = 1.0,
        initial_cash: float = 10_000.0,
    ):
        self.data_source = data_source
        self.strategy = strategy
        self.execution_model = execution_model

        # если allocator не задан — используем proportional (дефолт)
        self.allocator = allocator or ProportionalAllocator()

        self.allocation_budget = float(allocation_budget)
        self.initial_cash = float(initial_cash)

    def run(self) -> MultiBacktestResult:
        # 1) Загружаем и выравниваем данные
        mmd = self.data_source.get_bars()
        idx = mmd.index

        # 2) Генерируем RAW сигналы по каждому символу
        # Эти сигналы МОГУТ быть портфельно неконсистентны
        raw_sigs: Dict[str, Signals] = {}
        for sym, md in mmd.data.items():
            raw_sigs[sym] = self.strategy.generate_signals(md)

        # 3) Allocation: на каждом timestamp
        #    raw signals -> portfolio weights
        weights_by_sym: Dict[str, List[float]] = {
            sym: [] for sym in mmd.symbols
        }

        for ts in idx:
            # собираем raw intent на этот бар
            raw: Dict[str, float] = {}
            for sym in mmd.symbols:
                v = raw_sigs[sym].target_position.loc[ts]
                raw[sym] = float(v) if v == v else 0.0  # NaN -> 0

            # ПОРТФЕЛЬНЫЙ ШАГ
            w = self.allocator.allocate(
                raw,
                budget=self.allocation_budget,
            )

            # сохраняем веса по символам
            for sym in mmd.symbols:
                weights_by_sym[sym].append(float(w.get(sym, 0.0)))

        # 4) Формируем Signals, которые уже
        #    семантически = target PORTFOLIO WEIGHT
        sigs: Dict[str, Signals] = {
            sym: Signals(
                target_position=pd.Series(
                    weights_by_sym[sym],
                    index=idx,
                )
            )
            for sym in mmd.symbols
        }

        # 5) Execution (без знания про allocation)
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
