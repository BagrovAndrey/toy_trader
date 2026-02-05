from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import pandas as pd

from .core_types import MarketData, Signals, Fill, PortfolioState, Qty, Price, Cash


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
    eps: float = 1e-12 # Round-off of nearly-zero positions


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

    #def run(
    #    self,
    #    md: MarketData,
    #    sig: Signals,
    #    initial_state: PortfolioState,
    #) -> Tuple[List[Fill], List[PortfolioState]]:
    #    raise NotImplementedError
    
    def run(
        self,
        md: MarketData,
        sig: Signals,
        initial_state: PortfolioState,
    ) -> Tuple[List[Fill], List[PortfolioState]]:
        """
        Исполняем стратегию в стиле toy-backtest.

        Идея:
        - у нас есть свечи (bars) и сигналы target_position (0 или 1)
        - target_position = 1 значит: "хочу держать 1 единицу актива"
        - target_position = 0 значит: "хочу держать 0 единиц"
        - на каждом баре мы приводим текущую позицию к желаемой
        - сделки считаем по цене Open или Close текущего бара (зависит от params.fill_price)
        - комиссия и проскальзывание учитываем в момент сделки
        - после каждого бара сохраняем снимок PortfolioState
        """

        bars = md.bars              # pandas DataFrame со свечами
        symbol = md.symbol          # например "SPY"

        # 1) Синхронизация времени
        # Очень важно: сигналы и бары должны иметь одинаковый индекс (одинаковые даты).
        # Иначе вы думаете, что торгуете на одних датах, а реально — на других.
        if not sig.target_position.index.equals(bars.index):
            raise ValueError("Signals.target_position.index должен совпадать с MarketData.bars.index")

        # 2) Какую цену бара считаем ценой исполнения сделки
        # params.fill_price у вас "open" или "close" (нижний регистр),
        # а в DataFrame колонки "Open"/"Close" (с заглавной).
        fill_col = "Open" if self.params.fill_price == "open" else "Close"

        target = sig.target_position  # Series с 0/1 по времени

        fills: List[Fill] = []        # сюда будем складывать факты сделок
        states: List[PortfolioState] = []  # сюда снимки портфеля на каждом баре

        # 3) Берём начальное состояние и делаем рабочие копии
        # Мы НЕ хотим мутировать initial_state (это плохая привычка для backtest),
        # поэтому забираем значения в локальные переменные.
        cash: Cash = initial_state.cash
        positions: dict[str, Qty] = dict(initial_state.positions)      # копия словаря позиций
        last_prices = dict(initial_state.last_prices)  # копия словаря последних цен

        # 4) Функция для расчёта equity (стоимости портфеля)
        # equity = cash + сумма(qty * last_price) по всем инструментам.
        def compute_equity(cash_: Cash, pos_: Dict[str, Qty], lp_: Dict[str, Price]) -> float:
            eq = cash_
            for sym, qty in pos_.items():
                price = lp_.get(sym)  # последняя известная цена инструмента
                if price is None:
                    # нет цены — не можем оценить вклад; для MVP можно просто пропустить
                    continue
                eq += qty * price
            return float(eq)

        # 5) Основной цикл: идём бар за баром по времени
        for ts, row in bars.iterrows():
            # ts — timestamp текущего бара (дата/время)
            # row — строка DataFrame со значениями Open/High/Low/Close/Volume

            # 5.1) Обновляем "последнюю цену" для mark-to-market
            # Для оценки портфеля удобно брать Close текущего бара.
            close_price = float(row["Close"])
            last_prices[symbol] = close_price

            # 5.2) Текущая позиция в этом инструменте (сколько единиц держим сейчас)
            cur_qty: Qty = positions.get(symbol, 0.0)  # если позиции нет — считаем 0

            # 5.3) Желаемая доля капитала (fraction) из сигналов: [0,1]
            desired_raw = target.loc[ts]

            if pd.isna(desired_raw):
                target_fraction: float = 0.0
            else:
                target_fraction = float(desired_raw)

            if not (0.0 <= target_fraction <= 1.0):
                raise ValueError(f"target_fraction ∈ [0,1], получено {target_fraction} на {ts}")

            # sizing по CLOSE
            equity_for_sizing = compute_equity(cash, positions, last_prices)
            desired_qty: Qty = (target_fraction * equity_for_sizing) / close_price
            
            # 5.4) Сколько нужно купить/продать, чтобы перейти к желаемой позиции
            # delta > 0  => надо купить delta единиц
            # delta < 0  => надо продать |delta| единиц
            delta: Qty = desired_qty - cur_qty

            # 5.5) Если позиция уже как надо — сделки нет
            if abs(delta) > self.params.eps:
                # 5.5.1) Берём базовую цену исполнения с текущего бара
                raw_price = float(row[fill_col])

                # 5.5.2) Проскальзывание (slippage)
                # В bps: 1 bps = 0.01% = 0.0001
                slip = self.params.slippage_bps / 10_000.0

                # Логика:
                # - покупка: платим чуть дороже (цена ↑)
                # - продажа: продаём чуть дешевле (цена ↓)
                if delta > 0:
                    exec_price = raw_price * (1.0 + slip)
                else:
                    exec_price = raw_price * (1.0 - slip)

                # 5.5.3) Комиссия (fee) тоже в bps от оборота (notional)
                # notional = |qty| * price
                notional = abs(delta) * exec_price
                fee = notional * (self.params.fee_bps / 10_000.0)

                # 5.5.4) Обновляем cash
                # Если delta>0 (покупка): cash уменьшается на delta*price
                # Если delta<0 (продажа): cash увеличивается, потому что "-delta*price" вычитает отрицательное
                cash -= delta * exec_price

                # Комиссия всегда уменьшает cash
                cash -= fee

                # 5.5.5) Обновляем позицию
                new_qty: Qty = cur_qty + delta
                if abs(new_qty) < self.params.eps:
                    # если стало 0 — можно удалить запись из словаря
                    positions.pop(symbol, None)
                else:
                    positions[symbol] = new_qty

                # 5.5.6) Записываем факт сделки (Fill)
                fills.append(
                    Fill(
                        symbol=symbol,
                        ts=pd.Timestamp(ts),
                        price=float(exec_price),
                        qty=delta,      # +1 buy, -1 sell (в MVP)
                        fee=float(fee),
                    )
                )

            # 5.6) В конце бара делаем "снимок" состояния портфеля
            equity = compute_equity(cash, positions, last_prices)

            states.append(
                PortfolioState(
                    cash=float(cash),
                    positions=dict(positions),       # копии, чтобы прошлое не менялось
                    last_prices=dict(last_prices),
                    equity=float(equity),
                )
            )

        return fills, states
