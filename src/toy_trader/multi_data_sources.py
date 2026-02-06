from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .core_types import MarketData
from .data_sources import YahooDataSource


@dataclass(frozen=True)
class MultiMarketData:
    """
    Набор MarketData для нескольких символов с общим временем (одинаковый DatetimeIndex).
    """
    data: Dict[str, MarketData]

    @property
    def symbols(self) -> List[str]:
        return list(self.data.keys())

    @property
    def index(self) -> pd.DatetimeIndex:
        # предполагаем, что уже выровнено
        first = next(iter(self.data.values()))
        return first.bars.index


class MultiYahooDataSource:
    """
    Простая обёртка над YahooDataSource для нескольких символов.

    Возвращает MultiMarketData, где бары всех символов выровнены
    по общему DatetimeIndex (берём пересечение дат).
    """

    def __init__(
        self,
        symbols: List[str],
        start: str,
        end: Optional[str] = None,
    ):
        if not symbols:
            raise ValueError("symbols must be non-empty")
        self.symbols = symbols
        self.start = start
        self.end = end

    def get_bars(self) -> MultiMarketData:
        # 1) Скачиваем данные по каждому символу
        per_symbol: Dict[str, MarketData] = {}
        for sym in self.symbols:
            ds = YahooDataSource(symbol=sym, start=self.start, end=self.end)
            md = ds.get_bars()
            per_symbol[sym] = md

        # 2) Выравниваем индекс: пересечение дат
        common_index: Optional[pd.DatetimeIndex] = None
        for md in per_symbol.values():
            common_index = md.bars.index if common_index is None else common_index.intersection(md.bars.index)

        if common_index is None or len(common_index) == 0:
            raise ValueError("No common timestamps across symbols")

        # 3) Обрезаем бары по common_index
        aligned: Dict[str, MarketData] = {}
        for sym, md in per_symbol.items():
            bars = md.bars.loc[common_index].copy()
            aligned[sym] = MarketData(bars=bars, symbol=sym)

        return MultiMarketData(data=aligned)
