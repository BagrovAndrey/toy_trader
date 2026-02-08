from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


# ============================
# ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ============================

def _clean_long_only(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Приведение "сырых" сигналов к безопасному long-only виду.

    Контракт allocation слоя:
    - принимает любые float (в т.ч. NaN, отрицательные, мусор)
    - возвращает только корректные значения >= 0

    Зачем:
    - стратегии могут быть кривыми / экспериментальными
    - allocation НЕ должен падать из-за одной стратегии
    """
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            x = float(v)
        except Exception:
            x = 0.0

        # NaN check (NaN != NaN)
        if x != x:
            x = 0.0

        # long-only: всё отрицательное обнуляем
        if x < 0.0:
            x = 0.0

        out[k] = x
    return out


def _cap_and_redistribute(
    weights: Dict[str, float],
    cap: float,
    budget: float,
    eps: float,
) -> Dict[str, float]:
    """
    Применяет per-asset cap (макс. вес) с перераспределением остатка.

    Это классический water-filling:
    - сначала режем всё по cap
    - если после этого сумма < budget, перераспределяем остаток
    - делаем это итеративно, пока:
        * не исчерпаем budget
        * или не упремся во все cap'ы

    ВАЖНО:
    - deterministic
    - не зависит от порядка активов
    - гарантирует sum(w) <= budget
    """
    if cap <= 0.0:
        # если cap = 0 → всё запрещено
        return {k: 0.0 for k in weights}

    # старт: просто применяем cap
    w = {k: min(float(v), cap) for k, v in weights.items()}

    for _ in range(10_000):  # защита от бесконечных циклов
        s = sum(w.values())
        if s >= budget - eps:
            break

        leftover = budget - s

        # активы, которые ещё можно увеличивать
        eligible = [k for k, v in w.items() if v < cap - eps]
        if not eligible:
            break

        base = sum(w[k] for k in eligible)

        # если все eligible были почти 0 — делим поровну
        if base <= eps:
            add_each = leftover / len(eligible)
            changed = False
            for k in eligible:
                new_v = min(cap, w[k] + add_each)
                if abs(new_v - w[k]) > eps:
                    changed = True
                w[k] = new_v
            if not changed:
                break
        else:
            # пропорциональное перераспределение
            changed = False
            for k in eligible:
                add = leftover * (w[k] / base)
                new_v = min(cap, w[k] + add)
                if abs(new_v - w[k]) > eps:
                    changed = True
                w[k] = new_v
            if not changed:
                break

    # финальная защита от численных ошибок
    s = sum(w.values())
    if s > budget + eps:
        scale = budget / s
        w = {k: v * scale for k, v in w.items()}

    return w


# ============================
# АБСТРАКТНЫЙ ИНТЕРФЕЙС
# ============================

class AllocationModel(ABC):
    """
    Allocation / Portfolio Construction слой.

    Его задача:
    - взять "raw intent" от стратегий
    - превратить в портфельные веса
    - гарантировать инварианты:
        * w_i >= 0
        * sum(w) <= budget
    """

    @abstractmethod
    def allocate(
        self,
        raw: Dict[str, float],
        budget: float = 1.0,
    ) -> Dict[str, float]:
        raise NotImplementedError


# ============================
# РЕАЛИЗАЦИИ
# ============================

@dataclass(frozen=True)
class ProportionalAllocator(AllocationModel):
    """
    ДЕФОЛТНЫЙ аллокатор.

    Идея:
    - стратегии выдают "силу сигнала" (raw >= 0)
    - мы нормализуем их пропорционально
    - итог: sum(w) = budget

    Пример:
        raw = {A: 1, B: 2}
        → w = {A: 0.333, B: 0.666}
    """

    cap: Optional[float] = None   # max weight per asset
    eps: float = 1e-12

    def allocate(self, raw: Dict[str, float], budget: float = 1.0) -> Dict[str, float]:
        budget = float(budget)
        if budget <= 0.0:
            return {k: 0.0 for k in raw}

        clean = _clean_long_only(raw)
        s = sum(clean.values())

        # если никто не хочет в рынок — остаёмся в кэше
        if s <= self.eps:
            return {k: 0.0 for k in clean}

        # пропорциональная нормализация
        w = {k: (v / s) * budget for k, v in clean.items()}

        # опционально применяем cap
        if self.cap is not None:
            w = _cap_and_redistribute(
                w,
                cap=float(self.cap),
                budget=budget,
                eps=self.eps,
            )
        else:
            # чисто числовая защита
            sw = sum(w.values())
            if sw > budget + self.eps:
                scale = budget / sw
                w = {k: v * scale for k, v in w.items()}

        return w


@dataclass(frozen=True)
class EqualWeightAllocator(AllocationModel):
    """
    Equal-weight по активным стратегиям.

    Используется, когда:
    - сигналы бинарные (0/1)
    - хотим "не думать" о силе сигнала
    """

    cap: Optional[float] = None
    eps: float = 1e-12

    def allocate(self, raw: Dict[str, float], budget: float = 1.0) -> Dict[str, float]:
        budget = float(budget)
        if budget <= 0.0:
            return {k: 0.0 for k in raw}

        clean = _clean_long_only(raw)
        active = [k for k, v in clean.items() if v > self.eps]

        if not active:
            return {k: 0.0 for k in clean}

        w0 = budget / len(active)
        w = {k: (w0 if k in active else 0.0) for k in clean}

        if self.cap is not None:
            w = _cap_and_redistribute(
                w,
                cap=float(self.cap),
                budget=budget,
                eps=self.eps,
            )

        return w
