from toy_trader.multi_data_sources import MultiYahooDataSource
from toy_trader.strategies import SMACrossStrategy, SMACrossParams
from toy_trader.execution import ExecutionParams
from toy_trader.multi_execution import MultiNextBarExecutionModel
from toy_trader.multi_engine import MultiBacktestEngine

ds = MultiYahooDataSource(symbols=["AMZN","PLTR","XOM"], start="2020-01-01")
st = SMACrossStrategy(SMACrossParams(fast=10, slow=40, shift_for_execution=1))
ex = MultiNextBarExecutionModel(ExecutionParams(fill_price="open"))

engine = MultiBacktestEngine(ds, st, ex, initial_cash=10_000)
res = engine.run()

print(len(res.states), len(res.market_data.index))
print("fills:", len(res.fills))
print("last equity:", res.states[-1].equity)
print("last positions:", res.states[-1].positions)

from collections import Counter
c = Counter(f.symbol for f in res.fills)
print(c)
