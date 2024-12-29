from typing import Callable, Any, Dict, List
from .logger import Logger
from dataclasses import dataclass
from .preprocess import CryptoCurrency


@dataclass
class Indicators:
    """
    A class to manage and compute indicators for a given cryptocurrency.
    """
    crypto: CryptoCurrency
    logger: Logger
    indicators: Dict[str, Callable[[int], Any]]

    def __init__(self, crypto: CryptoCurrency, logger: Logger):
        self.crypto = crypto
        self.logger = logger
        self.indicators = {}        
        self._initialize_common_indicators()

    def _initialize_common_indicators(self) -> None:
        """
        Adds default indicators like MACD and RSI to the class.
        """
        self.indicators["macd"] = self._calculate_macd
        self.indicators["rsi"] = self._calculate_rsi
        self.indicators["sma"] = self._calculate_sma
        self.indicators["ema"] = self._calculate_ema

    def _get_price_history(self) -> List[float]:
        """
        Extracts the price history from self.crypto.history.history.
        """
        if not self.crypto.history or not self.crypto.history.chart:
            raise ValueError("No historical data available.")
        
    
        return [snapshot.close for snapshot in self.crypto.history.chart]

    def _calculate_macd(self, interval: int) -> dict:
        """
        Calculates the MACD indicator using the price history.
        """
        prices = self._get_price_history()
        short_period = interval
        long_period = interval * 2
        signal_period = interval // 2

        if len(prices) < long_period:
            self.logger.error(f"Not enough data to calculate MACD in crypto: {self.crypto.symbol}")
            return {
                "MACD_Line":    0,
                "Signal_Line":  0,
                "Histogram":    0
            }            

        short_ema = sum(prices[-short_period:]) / short_period
        long_ema = sum(prices[-long_period:]) / long_period
        macd_line = short_ema - long_ema
        signal_line = macd_line * 0.9  # Placeholder for signal calculation
        histogram = macd_line - signal_line

        return {
            "MACD_Line":        macd_line,
            "Signal_Line":      signal_line,
            "Histogram":        histogram
        }

    def _calculate_rsi(self, interval: int) -> float:
        """
        Calculates the RSI indicator using the price history.
        """
        prices = self._get_price_history()
        if len(prices) < interval:
            self.logger.error(f"Not enough data to calculate RSI in crypto: {self.crypto.symbol}")
            return 0.0

        gains       = [max(prices[i] - prices[i - 1], 0) for i in range(1, len(prices))]
        losses      = [max(prices[i - 1] - prices[i], 0) for i in range(1, len(prices))]
        avg_gain    = sum(gains[-interval:]) / interval
        avg_loss    = sum(losses[-interval:]) / interval

        if avg_loss == 0:
            return 100.0  # No losses, RSI is maximum
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_sma(self, interval: int) -> float:
        """
        Calculates the SMA indicator using the price history.
        """
        prices = self._get_price_history()
        if len(prices) < interval:
            self.logger.error(f"Not enough data to calculate SMA in crypto: {self.crypto.symbol}")
            return 0.0

        return sum(prices[-interval:]) / interval

    def _calculate_ema(self, interval: int) -> float:
        """
        Calculates the EMA indicator using the price history.
        """
        prices = self._get_price_history()
        if len(prices) < interval:
            self.logger.error(f"Not enough data to calculate EMA in crypto: {self.crypto.symbol}")
            return 0.0

        ema = prices[0]  # Initialize with the first price
        multiplier = 2 / (interval + 1)
        for price in prices:
            ema = (price - ema) * multiplier + ema

        return ema

    def add_indicator(self, name: str, processor: Callable[[int], Any]) -> None:
        """
        Adds a custom indicator to the class.
        """
        self.indicators[name] = processor

    def get(self, name: str, interval: int = 14) -> Any:
        """
        Retrieves and computes an indicator based on its name and interval.
        """
        if name not in self.indicators:
            raise ValueError(f"Indicator '{name}' is not available.")

        return self.indicators[name](interval)
