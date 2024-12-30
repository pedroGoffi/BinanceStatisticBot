from typing import List, Tuple, Dict, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .logger import Logger
from .preprocess import ChartData, CryptoCurrency
from .indicators import Indicators

@dataclass
class StrategyResult:
    buy_signals: List[str]
    sell_signals: List[str]

class ITradeKernel(ABC):
    indicators: Indicators
    crypto: CryptoCurrency
    logger: Logger

    @abstractmethod
    def __init__(self, logger: Logger, crypto: CryptoCurrency) -> None: ... 

    @abstractmethod
    def analyze_market(self, chart_subset: List[ChartData]) -> StrategyResult: ...


class StrategyOne(ITradeKernel):
    indicators: Indicators
    crypto:     CryptoCurrency
    logger:     Logger
    strategy:   Callable[[ITradeKernel, List[ChartData]], StrategyResult]

    def __init__(self, logger: Logger, crypto: CryptoCurrency) -> None:
        self.crypto     = crypto        
        self.logger     = logger
        self.indicators = Indicators(self.crypto, self.logger)
        self.strategy   = StrategyOne._default_strategy
    
    def set_strategy(self, strategy: Callable[[ITradeKernel, List[ChartData]], StrategyResult]):
        self.strategy = strategy

    def analyze_market(self, chart_subset: List[ChartData]) -> StrategyResult:
        return self.strategy(self, chart_subset)

    @staticmethod    
    def _default_strategy(cls, chart_subset: List[ChartData]) -> StrategyResult:
        buy_signals: List[str] = []
        sell_signals: List[str] = []

        # Calculate indicators (example: MACD and RSI)
        macd_data: Dict[str, float] = cls.indicators.get("macd", interval=14, chart_subset=chart_subset)
        rsi_value: float = cls.indicators.get("rsi", interval=14, chart_subset=chart_subset)

        # Example trading logic based on MACD and RSI values
        if macd_data["Histogram"] > 0 and rsi_value < 30:
            buy_signals.append(cls.crypto.symbol)
        elif macd_data["Histogram"] < 0 and rsi_value > 70:            
            sell_signals.append(cls.crypto.symbol)

        # Calculate Simple Moving Average (SMA) for last 20 data points
        sma_values = [data.close for data in chart_subset[-20:]]  # Last 20 closing prices
        sma_value = sum(sma_values) / len(sma_values) if sma_values else 0  # Calculate SMA

        # Additional condition for buying (SMA strategy: price crosses above SMA)
        if chart_subset[-1].close > sma_value:
            buy_signals.append(cls.crypto.symbol)

        # Additional condition for selling (price crosses below SMA)
        if chart_subset[-1].close < sma_value:
            sell_signals.append(cls.crypto.symbol)

        # Example of adding more indicators or trading logic
        # Example: Add a condition for high trading volume (could be used for confirmation of a signal)
        if chart_subset[-1].volume > 1000000:  # Example: If volume is above 1 million, consider it a valid buy/sell condition
            buy_signals.append(cls.crypto.symbol)

        # Example: Add a time-based condition to sell after a certain number of periods
        if len(chart_subset) > 50:  # If we have more than 50 data points
            if rsi_value > 80:
                sell_signals.append(cls.crypto.symbol)

        return StrategyResult(buy_signals=buy_signals, sell_signals=sell_signals)
