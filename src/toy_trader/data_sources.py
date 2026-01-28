from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
from .core_types import MarketData


class DataSource(ABC):
    """
    Интерфейс источника рыночных данных.

    Главное: любая реализация обязана вернуть MarketData в согласованном формате.
    Это позволяет менять источник (Yahoo/CSV/IBKR/Exchange) без переписывания стратегии/движка.
    """

    @abstractmethod
    def get_bars(self) -> MarketData:
        raise NotImplementedError


class CSVDataSource(DataSource):
    """
    Заглушка для источника данных из CSV.
    Реализацию (чтение, парсинг дат, валидацию) добавим позже.
    """

    def __init__(self, symbol: str, path: str):
        self.symbol = symbol
        self.path = path

    def get_bars(self) -> MarketData:
        raise NotImplementedError


class YahooDataSource(DataSource):
    """
    Заглушка для источника данных из Yahoo (yfinance).
    Реализацию добавим позже.
    """

    def __init__(self, symbol: str, start: str, end: str | None = None, auto_adjust: bool = True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.auto_adjust = auto_adjust

    def get_bars(self) -> MarketData:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "Не найден пакет yfinance. Установите: pip install yfinance"
            ) from e

        # yfinance принимает end=None, но лучше явно передать если есть
        df = yf.download(
            tickers=self.symbol,
            start=self.start,
            end=self.end,
            auto_adjust=self.auto_adjust,
            progress=False,
            group_by="column",
        )

        if df is None or df.empty:
            raise ValueError(f"yfinance вернул пустые данные для {self.symbol} ({self.start}..{self.end})")

        # Нормализация индекса времени
        df = df.sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Ожидался DatetimeIndex от yfinance")

        # yfinance иногда возвращает MultiIndex колонок (особенно при нескольких тикерах)
        # В MVP мы ожидаем один тикер. Если MultiIndex — схлопнем уровень.
        if isinstance(df.columns, pd.MultiIndex):
            # Обычно верхний уровень — поля (Open/High/...), нижний — тикер
            # Выберем данные по нашему символу, если он присутствует
            if self.symbol in df.columns.get_level_values(-1):
                df = df.xs(self.symbol, axis=1, level=-1)
            else:
                # fallback: попробуем взять первый тикер
                df = df.droplevel(-1, axis=1)

        # Приводим имена колонок к канону
        # (Adj Close игнорируем: при auto_adjust=True close уже скорректирован;
        #  при auto_adjust=False можно было бы выбрать Adj Close, но для MVP это лишнее)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"В данных yfinance нет колонок {missing}. Доступно: {list(df.columns)}")

        bars = df[required].copy()
        bars.columns.name = None

        # Типы
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            bars[c] = pd.to_numeric(bars[c], errors="raise")

        # Санити-проверки (лучше упасть, чем молча продолжать)
        if not bars.index.is_monotonic_increasing:
            raise ValueError("Индекс времени должен быть монотонно возрастающим")
        if bars[["Open", "High", "Low", "Close"]].isna().any().any():
            raise ValueError("Обнаружены NaN в OHLC")

        return MarketData(bars=bars, symbol=self.symbol)
        
