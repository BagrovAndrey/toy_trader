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
        bars = md.bars
        close = bars["Close"]

        fast = close.rolling(
            window=self.params.fast,
            min_periods=self.params.fast,
        ).mean()

        slow = close.rolling(
            window=self.params.slow,
            min_periods=self.params.slow,
        ).mean()

        # базовый сигнал: сейчас 1 если fast > slow, иначе 0; 
        # но в принципе может быть дробным (например, входим в BTC на 0.02 от капитала)
        target = (fast > slow).astype(float)

        # лаг исполнения (на следующий бар)
        if self.params.shift_for_execution != 0:
            target = target.shift(self.params.shift_for_execution)

        # на начальных барах и после shift → flat
        target = target.fillna(0.0).astype(float)

        return Signals(target_position=target)
        
        
# ============================
# Zoo v1 — набор простых стратегий
# ============================
#
# Цель "зоопарка" на этапе A:
# - иметь несколько максимально простых и понятных стратегий,
#   чтобы сравнивать их в одинаковой инфраструктуре (одинаковый execution/reporting).
# - каждая стратегия возвращает Signals.target_position — это "целевой вес" (fraction) ∈ [0,1].
#
# ВАЖНО про честность бэктеста:
# - стратегии считают сигнал на баре t по данным, доступным на баре t
# - но сделка исполняется "на следующем баре" (shift_for_execution=1),
#   чтобы не было look-ahead (торговли на Close, который мы узнаём только после закрытия бара).
#
# Текущий execution делает sizing по Close и исполняет по Open/Close (в зависимости от params.fill_price).
# Поэтому shift_for_execution = 1 — безопасный дефолт. :contentReference[oaicite:1]{index=1}

import numpy as np


def _apply_shift_and_fill(target: pd.Series, shift_for_execution: int) -> pd.Series:
    """
    Общая утилита для всех стратегий.

    1) Делает лаг (shift) на количество баров, чтобы сигнал "вступал в силу" позже.
       Это простая модель "решение приняли на закрытии бара t, исполнили на t+1".
    2) Заполняет NaN (которые неизбежно возникают на старте rolling-окон и после shift)
       нулями => "не в рынке".
    3) Гарантирует float dtype (важно для sizing как fraction).

    Мы держим эту логику единой, чтобы:
    - стратегии отличались только торговой идеей,
    - а механика лагов была одинаковой.
    """
    if shift_for_execution != 0:
        target = target.shift(shift_for_execution)
    return target.fillna(0.0).astype(float)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """
    RSI (Relative Strength Index) в классической формулировке Уайлдера (Wilder).

    Почему делаем сами, а не через ta-lib:
    - меньше зависимостей
    - прозрачнее для обучения

    Алгоритм:
    - считаем изменения цены delta = close.diff()
    - положительные изменения -> gains, отрицательные (по модулю) -> losses
    - считаем сглаженные средние gains/losses через ewm(alpha=1/period)
      (это близко к Wilder's smoothing)
    - RSI = 100 - 100/(1 + RS), где RS = avg_gain/avg_loss
    """
    if period <= 0:
        raise ValueError("RSI period must be positive")

    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing: ewm(alpha=1/period, adjust=False)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi.astype(float)


# ------------------------------------------------------------
# 1) Buy&Hold (sanity / baseline)
# ------------------------------------------------------------

@dataclass(frozen=True)
class BuyHoldParams:
    """
    Параметры Buy&Hold.

    allocation_fraction:
    - это "намерение стратегии" (raw target).
    - в single-asset режиме обычно ставим 1.0 (all-in).
    - в multi-asset режиме это будет нормализовано allocator'ом (если он включён).
    """
    allocation_fraction: float = 1.0
    shift_for_execution: int = 1


class BuyHoldStrategy(Strategy):
    """
    Buy&Hold: всегда хотим держать позицию.

    Это полезно как:
    - sanity-check исполнения (должно быть похоже на baseline buy&hold в отчёте),
    - "нулевая" стратегия без тайминга.
    """

    def __init__(self, params: Optional[BuyHoldParams] = None):
        self.params = params or BuyHoldParams()
        if not (0.0 <= self.params.allocation_fraction <= 1.0):
            raise ValueError("allocation_fraction must be in [0,1] for long-only MVP")

    def generate_signals(self, md: MarketData) -> Signals:
        idx = md.bars.index

        # На каждом баре говорим: хотим держать allocation_fraction
        target = pd.Series(self.params.allocation_fraction, index=idx, dtype=float)

        target = _apply_shift_and_fill(target, self.params.shift_for_execution)
        return Signals(target_position=target)


# ------------------------------------------------------------
# 2) Time-Series Momentum (простая тренд-идея)
# ------------------------------------------------------------

@dataclass(frozen=True)
class TSMomParams:
    """
    lookback:
    - горизонт "смотреть назад" для измерения момента (L баров)
    - сигнал: если доходность за lookback положительна => long
    """
    lookback: int = 20
    shift_for_execution: int = 1


class TimeSeriesMomentumStrategy(Strategy):
    """
    Time-series momentum (TSMom).

    Идея (самая простая):
    - считаем return_L = Close[t] / Close[t-L] - 1
    - если return_L > 0 => target=1 (long)
    - иначе => target=0 (cash)

    Почему это полезно в зоопарке:
    - это "канонический" представитель трендовых стратегий
    - на боковике обычно проигрывает mean-reversion, что видно в сравнении.
    """

    def __init__(self, params: Optional[TSMomParams] = None):
        self.params = params or TSMomParams()
        if self.params.lookback <= 0:
            raise ValueError("lookback must be positive")

    def generate_signals(self, md: MarketData) -> Signals:
        close = md.bars["Close"].astype(float)

        # Доходность за lookback баров (грубая, без лог-доходностей)
        ret = close / close.shift(self.params.lookback) - 1.0

        target = (ret > 0.0).astype(float)

        target = _apply_shift_and_fill(target, self.params.shift_for_execution)
        return Signals(target_position=target)


# ------------------------------------------------------------
# 3) Mean Reversion по Z-score (диапазон / возврат к среднему)
# ------------------------------------------------------------

@dataclass(frozen=True)
class ZScoreReversionParams:
    """
    window:
    - окно для оценки "нормального" уровня цены и волатильности
    entry_z:
    - порог входа (обычно 1.0..2.0)
    exit_z:
    - порог выхода (обычно ближе к 0.0)

    Логика:
    - z = (Close - MA) / STD
    - если z <= -entry_z => хотим long (перепроданность)
    - если z >= -exit_z  => выходим (цена вернулась к среднему)

    Почему exit так устроен:
    - мы входим, когда сильно ниже среднего
    - выходим, когда вернулись ближе к среднему (z поднялся)
    """
    window: int = 50
    entry_z: float = 1.5
    exit_z: float = 0.0
    shift_for_execution: int = 1


class ZScoreMeanReversionStrategy(Strategy):
    """
    Mean-reversion стратегия через z-score.

    ВАЖНО: эта стратегия "состояний" (stateful):
    - сигнал 1 может держаться несколько баров, пока не сработает условие выхода.
    - поэтому мы строим position как series через простой цикл по времени.

    Почему не делаем векторно:
    - для обучения проще и прозрачнее показать "машину состояний".
    """

    def __init__(self, params: Optional[ZScoreReversionParams] = None):
        self.params = params or ZScoreReversionParams()
        if self.params.window <= 1:
            raise ValueError("window must be > 1")
        if self.params.entry_z <= 0:
            raise ValueError("entry_z must be positive")
        # exit_z может быть 0 или даже отрицательным (более ранний выход), но обычно 0

    def generate_signals(self, md: MarketData) -> Signals:
        close = md.bars["Close"].astype(float)

        ma = close.rolling(window=self.params.window, min_periods=self.params.window).mean()
        sd = close.rolling(window=self.params.window, min_periods=self.params.window).std(ddof=0)

        # Защита от деления на 0 (на спокойных инструментах std может быть очень мал)
        sd = sd.replace(0.0, np.nan)

        z = (close - ma) / sd

        # Машина состояний: position ∈ {0,1}
        pos = []
        in_pos = 0.0

        for ts in close.index:
            zt = z.loc[ts]

            if np.isnan(zt):
                # пока нет данных для rolling — не торгуем
                in_pos = 0.0
            else:
                if in_pos <= 0.0:
                    # Условие входа: сильная перепроданность
                    if zt <= -self.params.entry_z:
                        in_pos = 1.0
                else:
                    # Условие выхода: возврат к среднему
                    if zt >= -self.params.exit_z:
                        in_pos = 0.0

            pos.append(in_pos)

        target = pd.Series(pos, index=close.index, dtype=float)

        target = _apply_shift_and_fill(target, self.params.shift_for_execution)
        return Signals(target_position=target)


# ------------------------------------------------------------
# 4) RSI(2) Mean Reversion (очень популярный простой вариант)
# ------------------------------------------------------------

@dataclass(frozen=True)
class RSIReversionParams:
    """
    rsi_period:
    - период RSI (классический "попсовый" вариант для reversion — 2)
    entry:
    - вход, если RSI < entry (например, 5..20)
    exit:
    - выход, если RSI > exit (например, 50..70)

    Стратегия:
    - это тоже "машина состояний", как z-score.
    """
    rsi_period: int = 2
    entry: float = 10.0
    exit: float = 50.0
    shift_for_execution: int = 1


class RSIMeanReversionStrategy(Strategy):
    """
    RSI mean reversion (long-only).

    Вход:
    - RSI(period) < entry  => long
    Выход:
    - RSI(period) > exit   => flat

    Почему это часто работает на боковике:
    - ловит "короткие выбросы" вниз, которые затем откатываются.
    """

    def __init__(self, params: Optional[RSIReversionParams] = None):
        self.params = params or RSIReversionParams()
        if self.params.rsi_period <= 0:
            raise ValueError("rsi_period must be positive")
        if not (0.0 <= self.params.entry <= 100.0 and 0.0 <= self.params.exit <= 100.0):
            raise ValueError("RSI thresholds must be within [0,100]")
        if self.params.entry >= self.params.exit:
            raise ValueError("entry should be < exit for sensible mean-reversion")

    def generate_signals(self, md: MarketData) -> Signals:
        close = md.bars["Close"].astype(float)
        rsi = _rsi(close, self.params.rsi_period)

        pos = []
        in_pos = 0.0

        for ts in close.index:
            rt = rsi.loc[ts]
            if np.isnan(rt):
                in_pos = 0.0
            else:
                if in_pos <= 0.0:
                    if rt < self.params.entry:
                        in_pos = 1.0
                else:
                    if rt > self.params.exit:
                        in_pos = 0.0
            pos.append(in_pos)

        target = pd.Series(pos, index=close.index, dtype=float)

        target = _apply_shift_and_fill(target, self.params.shift_for_execution)
        return Signals(target_position=target)


# ------------------------------------------------------------
# 5) Donchian Breakout (канал, трендовая классика)
# ------------------------------------------------------------

@dataclass(frozen=True)
class DonchianBreakoutParams:
    """
    entry_window:
    - окно для верхней границы (breakout вверх)
    exit_window:
    - окно для нижней границы (выход по "пробою вниз")

    Классика:
    - вход, если Close > High_N (предыдущего окна)
    - выход, если Close < Low_M (предыдущего окна)

    Почему используем shift(1) для границ:
    - чтобы сегодняшняя свеча не попадала в расчёт границы,
      иначе будет лёгкий look-ahead (особенно если high/low внутри бара).
    """
    entry_window: int = 50
    exit_window: int = 20
    shift_for_execution: int = 1


class DonchianBreakoutStrategy(Strategy):
    """
    Donchian breakout (long-only).

    Это "контраст" к mean-reversion:
    - должен хорошо себя вести на трендах
    - часто страдает на боковике из-за ложных пробоев
    """

    def __init__(self, params: Optional[DonchianBreakoutParams] = None):
        self.params = params or DonchianBreakoutParams()
        if self.params.entry_window <= 1 or self.params.exit_window <= 1:
            raise ValueError("entry_window and exit_window must be > 1")

    def generate_signals(self, md: MarketData) -> Signals:
        bars = md.bars
        close = bars["Close"].astype(float)

        # Границы канала по high/low
        high_n = bars["High"].rolling(self.params.entry_window, min_periods=self.params.entry_window).max().shift(1)
        low_m = bars["Low"].rolling(self.params.exit_window, min_periods=self.params.exit_window).min().shift(1)

        pos = []
        in_pos = 0.0

        for ts in close.index:
            hn = high_n.loc[ts]
            lm = low_m.loc[ts]
            ct = close.loc[ts]

            # Пока границы не определены — не торгуем
            if np.isnan(hn) or np.isnan(lm):
                in_pos = 0.0
            else:
                if in_pos <= 0.0:
                    if ct > hn:
                        in_pos = 1.0
                else:
                    if ct < lm:
                        in_pos = 0.0

            pos.append(in_pos)

        target = pd.Series(pos, index=close.index, dtype=float)

        target = _apply_shift_and_fill(target, self.params.shift_for_execution)
        return Signals(target_position=target)
