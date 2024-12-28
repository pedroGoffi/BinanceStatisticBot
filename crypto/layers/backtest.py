from dataclasses import dataclass
from typing import List, Dict, Tuple

from crypto.layers.kernel import ITradeKernel
from crypto.layers.logger import Logger
from .preprocess import CryptoCurrency, CryptoCurrencyHistory

class BacktestEngine:
    """
    Simulates strategy performance on historical data.
    """
    def __init__(self, crypto_histories: List[CryptoCurrencyHistory]):
        self.crypto_histories = crypto_histories

    def simulate_strategy(self, symbol: str, initial_cash: float, buy_threshold: float, sell_threshold: float) -> Tuple[float, List[str]]:
        """
        Simulates a buy/sell strategy using historical price data.
        :param symbol: The cryptocurrency symbol to simulate.
        :param buy_threshold: Price drop percentage to trigger a buy.
        :param sell_threshold: Price rise percentage to trigger a sell.
        :return: Tuple of total profit and a list of trades executed.
        """
        crypto_history = next((ch for ch in self.crypto_histories if ch.symbol == symbol), None)
        if not crypto_history:
            raise ValueError(f"No historical data found for symbol {symbol}")

        trades = []
        cash = initial_cash
        position = 0  # Number of units held

        for day in crypto_history.history:
            price = day.current_price

            if position == 0 and price <= buy_threshold:  # Buy condition
                position = cash / price
                cash = 0
                trades.append(f"Bought {position:.2f} {symbol} at ${price:.2f}")

            elif position > 0 and price >= sell_threshold:  # Sell condition
                cash = position * price
                position = 0
                trades.append(f"Sold {symbol} at ${price:.2f}")

        # If position remains unsold, sell at the last known price
        if position > 0:
            cash = position * crypto_history.history[-1].current_price
            trades.append(f"Final sell {symbol} at ${crypto_history.history[-1].current_price:.2f}")

        profit = cash - 1000  # Total profit or loss
        return profit, trades

class MetricsCollector:
    """
    Records key performance indicators (e.g., Sharpe ratio, win rate).
    """
    def __init__(self):
        self.metrics = {}

    def record_metrics(self, profit: float, trades: List[str]):
        """
        Records metrics for a given backtest.
        :param profit: Total profit from the strategy.
        :param trades: List of trades executed during the backtest.
        """
        self.metrics["profit"] = profit
        self.metrics["trades"] = trades
        self.metrics["win_rate"] = sum(1 for t in trades if "Sold" in t) / len(trades) if trades else 0

    def get_metrics(self) -> Dict:
        """
        Retrieves recorded metrics.
        """
        return self.metrics

class SimulationFramework:
    """
    Runs scenarios using real-time data in sandbox mode.
    """
    kernel: ITradeKernel
    logger: Logger
    def __init__(self, kernel: ITradeKernel, logger: Logger):
        self.metrics_collector  = MetricsCollector()
        self.kernel             = kernel
        self.logger             = logger

    def run_simulation(self, buy_threshold: float, sell_threshold: float):
        """
        Simulates a strategy on current market conditions.
        :param cryptos: List of CryptoCurrency objects to simulate.
        :param buy_threshold: Buy condition percentage.
        :param sell_threshold: Sell condition percentage.
        """
        results = []
        
        buy_signals, sell_signals = self.kernel.analyze_market()
        self.logger.debug((buy_signals, sell_signals, self.kernel.crypto))
        #price = crypto.current_price
        #if price <= buy_threshold:
        #    results.append(f"Simulated Buy: {crypto.symbol} at ${price:.2f}")
        #elif price >= sell_threshold:
        #    results.append(f"Simulated Sell: {crypto.symbol} at ${price:.2f}")

        return results


