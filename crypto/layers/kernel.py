from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from .logger import Logger
from .preprocess import CryptoCurrency
from .indicators import Indicators


class ITradeKernel(ABC):
    indicators: Indicators
    crypto: CryptoCurrency
    logger: Logger

    @abstractmethod
    def analyze_market(self) -> Tuple[List[str], List[str]]:
        pass

class StrategyOne(ITradeKernel):
    indicators: Indicators
    crypto: CryptoCurrency
    logger: Logger
    def __init__(self, logger: Logger, crypto: CryptoCurrency) -> None:
        self.crypto = crypto
        self.indicators = Indicators(self.crypto, logger)
        self.logger = logger
    
    def analyze_market(self) -> Tuple[List[str], List[str]]:        
        # Initialize indicators with the stored crypto data
        

        buy_signals: List[str] = []
        sell_signals: List[str] = []

        # Calculate indicators (example: MACD and RSI)
        macd_data: Dict[str, float] = self.indicators.get("macd", interval=14)
        rsi_value: float = self.indicators.get("rsi", interval=14)

        # Example trading logic based on MACD and RSI values

        if macd_data["Histogram"] > 0 and rsi_value < 30:
            buy_signals.append(self.crypto.symbol)

        elif macd_data["Histogram"] < 0 and rsi_value > 70:            
            sell_signals.append(self.crypto.symbol)
    


        return buy_signals, sell_signals

    