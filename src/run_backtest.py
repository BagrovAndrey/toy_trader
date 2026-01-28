from __future__ import annotations

# argparse — чтобы запускать файл из консоли с параметрами (например, --config ...)
import argparse

# dataclass — удобный способ описать "структурированный конфиг" как объект с полями
from dataclasses import dataclass

# Path — удобнее, чем строки, когда работаешь с путями к файлам
from pathlib import Path

# Any/Dict — типы, чтобы не гадать, что лежит в сыром YAML
from typing import Any, Dict

# PyYAML — читаем YAML-конфиг
import yaml

# Импортируем наши компоненты (все они — каркасы/заглушки, реализацию допишем позже)
from toy_trader.data_sources import YahooDataSource, CSVDataSource
from toy_trader.strategies import SMACrossStrategy, SMACrossParams
from toy_trader.execution import NextBarExecutionModel, ExecutionParams
from toy_trader.engine import BacktestEngine


@dataclass(frozen=True)
class RunConfig:
    """
    Это "нормализованный" конфиг запуска.

    Зачем он нужен:
    - YAML после чтения превращается в dict/dict/dict (не очень удобно и легко ошибиться ключом)
    - RunConfig делает явными основные секции и типы
    - в будущем его можно расширять без переписывания кучи кода
    """
    data: Dict[str, Any]
    strategy: Dict[str, Any]
    execution: Dict[str, Any]
    initial_cash: float


def load_config(path: str | Path) -> RunConfig:
    """
    Читает YAML и превращает его в RunConfig.

    Важный момент:
    - мы не пытаемся сейчас валидировать всё строго (это можно добавить позже)
    - но мы уже делаем "нормализацию": гарантируем, что нужные секции существуют
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return RunConfig(
        data=raw.get("data", {}),
        strategy=raw.get("strategy", {}),
        execution=raw.get("execution", {}),
        initial_cash=float(raw.get("initial_cash", 10_000.0)),
    )


def build_data_source(cfg: RunConfig):
    """
    Собирает конкретный DataSource на основе конфигурации.

    Почему так:
    - мы заранее хотим поддержать несколько источников данных (yahoo/csv/...)
    - поэтому выбираем реализацию по cfg.data["kind"]
    - для остального кода это будет просто "data_source.get_bars()"
    """
    kind = cfg.data.get("kind", "yahoo")

    if kind == "yahoo":
        # В этом случае источник данных — внешний API (через библиотеку yfinance)
        return YahooDataSource(
            symbol=cfg.data["symbol"],
            start=cfg.data["start"],
            end=cfg.data.get("end"),
            auto_adjust=bool(cfg.data.get("auto_adjust", True)),
        )

    if kind == "csv":
        # В этом случае источник данных — файл на диске
        return CSVDataSource(
            symbol=cfg.data["symbol"],
            path=cfg.data["path"],
        )

    # Если ввели неизвестный kind — падаем явно, чтобы не получить "тихую" ошибку
    raise ValueError(f"Unknown data.kind: {kind}")


def build_strategy(cfg: RunConfig):
    """
    Собирает стратегию.

    Почему стратегия выбирается здесь, а не внутри engine:
    - engine ничего не должен знать о типах стратегий
    - engine получает готовый объект с методом generate_signals(md)
    """
    kind = cfg.strategy.get("kind", "sma_cross")

    if kind == "sma_cross":
        # Параметры стратегии выделены в отдельный объект (SMACrossParams),
        # чтобы:
        # - было легче логировать/сохранять параметры
        # - было легче делать новые стратегии с похожим паттерном
        params = SMACrossParams(
            fast=int(cfg.strategy.get("fast", 10)),
            slow=int(cfg.strategy.get("slow", 40)),
            shift_for_execution=int(cfg.strategy.get("shift_for_execution", 1)),
        )
        return SMACrossStrategy(params=params)

    raise ValueError(f"Unknown strategy.kind: {kind}")


def build_execution(cfg: RunConfig):
    """
    Собирает модель исполнения (execution model).

    В backtest это будет симулятор.
    В будущем (live) здесь появится другой kind, который общается с брокером.

    Зачем отдельный build_*:
    - изоляция "проводки" (wiring): где создаются объекты и как они настраиваются
    - чтобы не разносить по всему проекту чтение YAML-ключей
    """
    kind = cfg.execution.get("kind", "next_bar")

    if kind == "next_bar":
        # ExecutionParams — конфиг издержек и правила выбора цены исполнения.
        # В toy-версии это грубая модель:
        # - комиссия bps
        # - проскальзывание bps
        # - цена исполнения (open/close)
        params = ExecutionParams(
            fee_bps=float(cfg.execution.get("fee_bps", 1.0)),
            slippage_bps=float(cfg.execution.get("slippage_bps", 1.0)),
            fill_price=str(cfg.execution.get("fill_price", "open")),
        )
        return NextBarExecutionModel(params=params)

    raise ValueError(f"Unknown execution.kind: {kind}")


def main():
    """
    Точка входа.

    Роль main:
    - получить путь к конфигу из CLI
    - собрать компоненты (data_source, strategy, execution)
    - собрать engine
    - запустить engine.run()
    - вывести базовую информацию о результате

    main НЕ:
    - не делает вычислений стратегии
    - не симулирует сделки
    - не строит графики
    Это всё делает библиотека в src/toy_trader/*
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # 1) выбираем источник данных
    data_source = build_data_source(cfg)

    # 2) выбираем стратегию
    strategy = build_strategy(cfg)

    # 3) выбираем модель исполнения
    execution_model = build_execution(cfg)

    # 4) собираем engine: это оркестратор, который вызовет все компоненты в правильном порядке
    engine = BacktestEngine(
        data_source=data_source,
        strategy=strategy,
        execution_model=execution_model,
        initial_cash=cfg.initial_cash,
    )

    # 5) запускаем прогон и получаем объект результата
    result = engine.run()

    # Пока печатаем минимум, чтобы убедиться, что все компоненты связались.
    # Метрики/репорты добавим отдельными модулями (metrics.py / report.py),
    # чтобы не смешивать "запуск" с "анализом результата".
    print("RUN OK")
    print("symbol:", result.market_data.symbol)
    print("bars:", len(result.market_data.bars))
    print("fills:", len(result.fills))
    print("states:", len(result.states))


# стандартный python-идиом:
# файл можно либо импортировать как модуль, либо запустить как скрипт.
# Этот блок гарантирует, что main() выполнится только при запуске напрямую.
if __name__ == "__main__":
    main()
