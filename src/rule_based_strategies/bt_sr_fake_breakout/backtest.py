import backtrader as bt
from ..strategies.breakout_strategy import BreakoutStrategy
from ..backtest.data_feed import PandasDataFeed, load_from_csv
from ..backtest.analyzer import TradeStats

def run_backtest(cash: float = 10000.0, commission: float = 0.001):
    cerebro = bt.Cerebro()

    # Load data
    data = load_from_csv()

    cerebro.adddata(PandasDataFeed(dataname=data))

    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)  # 每次投入10%资金
       # Add analyzers
    cerebro.addanalyzer(TradeStats, _name="trade_stats")
    # Add strategy
    cerebro.addstrategy(BreakoutStrategy, period=20)

    # Set broker
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(TradeStats, _name="trade_stats")

    # Run
    results = cerebro.run()
    strat = results[0]

    stats = strat.analyzers.trade_stats.get_analysis()
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    print("Trade Stats:", stats)

    # !!! 添加这一行来生成图形报告 !!!
    cerebro.plot()
    
    return stats

if __name__ == "__main__":
    run_backtest()
