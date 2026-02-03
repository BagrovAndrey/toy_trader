# toy_trader — Global Overview (Roadmap + Architecture)

> Goal: evolve a small, working backtester into a **production-grade algo trader** for **Bybit Spot (crypto)**, using **small, finished commits** (no “2-year project”).
>
> Current state (STATUS_3): end-to-end backtest pipeline works; signals are **target positions**; execution is **next-bar** with fees/slippage; portfolio snapshots; project uses **src-layout**. Fractional quantities + numeric aliases are implemented.

---

## 1) Core Principles

- **Small steps:** each step ends as a clean commit with a running pipeline.
- **Separation of concerns:**
  - Strategy generates **targets** (positions / fractions), not buy/sell commands.
  - Execution simulates or executes those targets, applying costs/constraints.
  - Engine orchestrates modules; it does not implement trading logic.
- **Extensibility:** design choices should naturally generalize to:
  - multi-asset portfolios,
  - more realistic execution,
  - live trading adapters.

---

## 2) Repository Layout

toy_trader/
src/
toy_trader/
core_types.py
data_sources.py
strategies.py
execution.py
engine.py
...
run_backtest.py
run_report.py
configs/
docs/
reports/


**src-layout note:** run scripts from `toy_trader/src` (or ensure `src/` is on `PYTHONPATH`).

---

## 3) System Architecture (Pipeline)

**DataSource → MarketData → Strategy → Signals → ExecutionModel → PortfolioState → Reporting/Viz**

### Key semantics
- **Signals** represent a **target portfolio state** (e.g., target fraction or target qty), not trade commands.
- **ExecutionModel** converts targets into fills and portfolio updates.
- **Engine** wires everything together and returns a complete result artifact.

---

## 4) Data Model (Types)

### MarketData
- `bars: pandas.DataFrame`
  - `DatetimeIndex` ascending
  - columns: `Open, High, Low, Close, Volume`
- `symbol: str`

### Signals
- `target_position: pandas.Series`
  - index compatible with `bars.index`
  - values are floats (MVP: typically `0.0/1.0`, later: fractions)

### Fill
- `symbol, ts, price, qty, fee`

### PortfolioState
- `cash`
- `positions: Dict[symbol, qty]`
- `last_prices: Dict[symbol, price]`
- `equity`

### Numeric aliases
- `Qty, Price, Cash, Fee, Equity` (currently `float`, designed for future migration if needed)

---

## 5) What Works Today (Baseline)

### Data
- `YahooDataSource.get_bars()` via `yfinance`
  - returns normalized `MarketData` with standard OHLCV columns
  - `DatetimeIndex` monotonic ascending
- `CSVDataSource` is currently stub / minimal.

### Strategy
- `SMACrossStrategy`
  - SMA fast/slow on Close
  - `target_position = 1.0 if fast > slow else 0.0`
  - optional execution lag: `shift_for_execution` (currently implemented in strategy)

### Execution
- `NextBarExecutionModel`
  - validates signal index matches bar index
  - trades only on changes in desired target
  - fill price: Open or Close of the bar
  - slippage (bps): buy worse / sell worse
  - fee (bps) on notional
  - maintains snapshots per bar: `states` length equals bars length
  - supports fractional quantities (`Qty`) and epsilon-cleaning of near-zero positions

### Engine
- `BacktestEngine.run()` wires data → strategy → execution and returns `BacktestResult`.

---

## 6) Global Roadmap (Phases)

We will complete **Phase A (A1–A3)** before building a “Strategy Zoo”.

### Phase A — Research-grade backtester (offline)
Objective: produce **comparable** equity curves and basic metrics; enable multi-asset.

#### A1 — Position sizing (fractional targets)
**Problem now:** MVP “1 unit” sizing makes performance comparisons misleading.

**Target design:**
- Interpret `Signals.target_position` as **target fraction** in `[0, 1]` (long-only for now).
- Execution converts fraction → desired qty using equity/price:
  - `desired_notional = target_fraction * equity`
  - `desired_qty = desired_notional / price`

**Done when:**
- equity curves are comparable across instruments/initial cash,
- baseline buy&hold is defined consistently (same fraction logic),
- execution still passes smoke checks; no index drift.

#### A2 — Reporting + metrics (MVP)
Add a standard report runner producing:
- price chart + target position overlay,
- equity curve,
- drawdown,
- baseline comparison,
- basic metrics (minimum set):
  - total return / PnL,
  - max drawdown,
  - Sharpe (simple),
  - turnover / number of trades (optional).

**Done when:**
- `run_report.py` produces a stable, repeatable report for any config.

#### A3 — Multi-asset (minimal)
Extend the pipeline to support multiple symbols cleanly.

Options:
- `DataSource.get_bars()` returns a `Dict[str, MarketData]` or a new `MultiMarketData`.
- Strategy returns multi-symbol signals (dict of Series) or a unified structure.
- Execution iterates symbols and updates a shared `PortfolioState`.

**Done when:**
- portfolio holds multiple assets simultaneously,
- report supports multi-asset equity and per-asset exposures,
- no hidden single-symbol assumptions remain in execution/engine.

---

### Phase B — Trading-grade simulation (still offline)
Objective: avoid “backtest → live shock” by introducing constraints.

Examples:
- minimum order size / lot step / tick size,
- “insufficient cash” behavior,
- optional partial fills / delayed fills (toy realism),
- explicit risk limits (max exposure, kill switch).

---

### Phase C — Paper trading / live skeleton
Objective: same pipeline; execution swaps to a broker adapter.

- Define `BrokerAdapter` interface:
  - place orders, fetch balances/positions, read fills
  - normalized domain objects
- Build a live loop:
  - bar scheduler, idempotency, persistence, restart safety

---

### Phase D — Bybit Spot live
Objective: real money with safety and observability.

- keys/secret management,
- strict limits (max notional per day, max loss),
- monitoring, logging, reconciliation.

---

## 7) Strategy Zoo (after Phase A)
We postpone “many strategies” until Phase A is complete to ensure:
- comparable equity curves (A1),
- consistent reporting/metrics (A2),
- multi-asset support (A3).

Then we add strategies as small commits with shared evaluation harness:
- Mean reversion (z-score / bands),
- Breakout (Donchian),
- Momentum (return-based),
- Volatility targeting (position fraction scales with vol).

---

## 8) Operational Conventions

- **Every commit must:**
  - keep `run_backtest.py --config ...` working,
  - keep `run_report.py` (if present) working,
  - preserve index alignment checks and deterministic behavior.

- **Config-driven runs:**
  - backtests executed via YAML configs (`configs/*.yaml`)
  - wiring logic lives in `run_backtest.py` / `run_report.py` (not in engine).

---

## 9) Current Next Step
Proceed with **Phase A**, starting with:

1) **A1: Position sizing** (target_fraction → desired_qty via equity/price)  
2) **A2: Reporting + metrics (MVP)**  
3) **A3: Minimal multi-asset**

After A3, begin the **Strategy Zoo**.

---
