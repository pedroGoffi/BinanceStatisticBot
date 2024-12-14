"""
Sistema Inteligente de Análise de Criptomoedas

Este sistema utiliza a API da Binance para coletar dados históricos de criptomoedas, calcular indicadores técnicos como MACD, RSI, Média Móvel e Z-Score, e prever tendências de preços.

As principais funcionalidades incluem:
- Coleta de dados históricos.
- Cálculo de indicadores técnicos.
- Avaliação de suporte e resistência.
- Previsão de tendência usando Z-Score.
- Decisão de compra ou venda baseada em uma combinação de indicadores.

Autor: Pedro Henrique Goffi de Paulo
Versão: 1.0
"""


import os
from tqdm import tqdm
import time
from binance.client import Client
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Literal, List, Tuple
import pandas as pd
import numpy as np
from scipy.stats import zscore  # Import zscore from scipy.stats
from multiprocessing import Pool, cpu_count  # Import Pool for multiprocessing

load_dotenv()
GAIN_THRESHOLD = 5.0  # For example, set a gain threshold at 5%
CRYPTO_MAX_PRICE = 10
API_KEY: str = os.getenv("API_KEY")
API_SECRET: str = os.getenv("API_SECRET")

client: Client = Client(API_KEY, API_SECRET)

@dataclass
class CryptoCurrency:
    symbol: str
    name: str
    current_price: float
    volume: float
    quote_volume: float
    macd: float
    macd_signal: float
    ma_50: float
    ma_200: float
    rsi: float
    support: float
    resistance: float
    z_score: float
    min_gain: float
    max_gain: float
    buy_price_threshold: float  # New field for buying threshold
    sell_price_threshold: float  # New field for selling threshold

    def update(self):
        print(f"[UPDATE]: updating crypto: {self.symbol}")
        crypto_history: CryptoCurrencyHistory = get_crypto_history(self.symbol)
        if crypto_history:
            z_score = calculate_z_score(pd.DataFrame(crypto_history.history))
            min_gain, max_gain = calculate_gain_range(crypto_history.history)
            volume = calculate_volume(crypto_history.history)
            if crypto_history.history[-1]['close'] > CRYPTO_MAX_PRICE:
                return None
            current_price = crypto_history.history[-1]['close']
            # Define dynamic buy and sell thresholds based on the current price
            buy_price_threshold = current_price * 0.7  # 30% below current price
            sell_price_threshold = current_price * 1.2  # 20% above current price
            
            self.current_price = current_price
            self.volume = volume
            self.macd = crypto_history.macd['macd'].iloc[-1]
            self.macd_signal = crypto_history.macd['signal'].iloc[-1]
            self.ma_50 = crypto_history.ma_50['ma'].iloc[-1]
            self.ma_200 = crypto_history.ma_200['ma'].iloc[-1]
            self.rsi = crypto_history.rsi['rsi'].iloc[-1]
            self.support = crypto_history.support
            self.resistance = crypto_history.resistance
            self.z_score = z_score
            self.min_gain = min_gain
            self.max_gain = max_gain
            self.buy_price_threshold = buy_price_threshold
            self.sell_price_threshold = sell_price_threshold

    def should_buy(self) -> str:
        if self.z_score > 0 and self.rsi < 30:
            return "📈 Act Now: Buy - Expecting Increase"
        elif self.z_score > 0:
            return f"⏳ Await Till Value (${self.ma_50:.2f}): RSI low but Z-score positive"
        else:
            return "🛑 Hold or Avoid - Expecting Decrease"

    def await_till_value(self) -> str:
        if self.rsi < 30:
            return f"⏳ Await Till RSI({self.rsi:.2f}) reaches a minimum threshold (30)"
        elif self.z_score > 0:
            return f"⏳ Await Till Z-score ({self.z_score:.2f}) returns to a more favorable value"
        elif self.current_price <= self.buy_price_threshold:
            return f"⏳ Await till price reaches: ${self.buy_price_threshold:.2f}"
        elif self.current_price >= self.sell_price_threshold:
            return f"⏳ Await till price drops to: ${self.sell_price_threshold:.2f}"
        else:
            return "🚫 No Action Needed"

    def should_sell(self) -> str:
        if self.z_score < 0 and self.rsi > 70:
            return "📉 Act Now: Sell - Expecting Decrease"
        elif self.z_score < 0:
            return f"⏳ Await Till Value (${self.ma_50:.2f}): RSI high but Z-score negative"
        elif self.current_price >= self.sell_price_threshold:
            return f"⏳ Await till price drops to: ${self.sell_price_threshold:.2f}"
        else:
            return "🕰️ Hold or Wait - No Immediate Action"

@dataclass
class CryptoCurrencyHistory:
    symbol: str
    name: str
    history: List[dict]
    macd: pd.DataFrame
    ma_50: pd.DataFrame
    ma_200: pd.DataFrame
    rsi: pd.DataFrame
    support: float
    resistance: float


def get_crypto_history(symbol: str, interval: Literal['15m', '1h', '4h', '1d'] = '15m') -> CryptoCurrencyHistory | None:
    klines: List[list] = client.get_historical_klines(symbol, interval, "1 day ago UTC")
    if not klines:
        return None
    history: List[dict] = []
    for kline in klines:
        history.append({
            'timestamp':    kline[0],
            'open':         float(kline[1]),
            'high':         float(kline[2]),
            'low':          float(kline[3]),
            'close':        float(kline[4]),
            'volume':       float(kline[5])
        })
    return CryptoCurrencyHistory(
        symbol=symbol,
        name=symbol[:-4],
        history=history,
        macd=calculate_macd(history),
        ma_50=calculate_moving_average(history),
        ma_200=calculate_moving_average(history, 200),
        rsi=calculate_rsi(history),
        support=calculate_support_resistance(history)[0],
        resistance=calculate_support_resistance(history)[1],    
    )


def calculate_macd(data: List[dict], short_period: int = 3, long_period: int = 10, signal_period: int = 15) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(data)
    df['close'] = df['close'].astype(float)
    df['ema_short'] = df['close'].ewm(span=short_period, min_periods=1, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long_period, min_periods=1, adjust=False).mean()
    df['macd'] = df['ema_short'] - df['ema_long']
    df['signal'] = df['macd'].ewm(span=signal_period, min_periods=1, adjust=False).mean()
    return df[['timestamp', 'macd', 'signal']]


def calculate_moving_average(data: List[dict], period: int = 10) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(data)
    df['close'] = df['close'].astype(float)
    df['ma'] = df['close'].rolling(window=period).mean()
    return df[['timestamp', 'ma']]


def calculate_rsi(data: List[dict], period: int = 14) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(data)
    df['close'] = df['close'].astype(float)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df[['timestamp', 'rsi']]


def calculate_z_score(data: pd.DataFrame) -> float:
    if data.empty or 'close' not in data.columns:
        return None

    z_scores = zscore(data['close'])
    if not z_scores.size:  # If the z_scores array is empty
        return None

    # Safely get the last element using iloc
    z_score_value = z_scores.iloc[-1]
    if np.isnan(z_score_value):
        return None
    return z_score_value


def calculate_support_resistance(data: List[dict]) -> Tuple[float, float]:
    df: pd.DataFrame = pd.DataFrame(data)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    support = df['low'].min()
    resistance = df['high'].max()
    return support, resistance


def calculate_volume(data: List[dict]) -> float:
    df: pd.DataFrame = pd.DataFrame(data)
    df['volume'] = df['volume'].astype(float)
    total_volume = df['volume'].sum()
    return total_volume

def calculate_gain_range(history: List[dict]) -> Tuple[float, float]:
    df: pd.DataFrame = pd.DataFrame(history)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    max_price = df['high'].max()
    min_price = df['low'].min()
    min_gain = ((max_price - min_price) / min_price) * 100
    max_gain = ((max_price - min_price) / min_price) * 100
    return min_gain, max_gain

def worker(ticker: dict) -> CryptoCurrency:
    symbol: str = ticker['symbol']
    if symbol.endswith('USDT'):
        crypto_history: CryptoCurrencyHistory = get_crypto_history(symbol)
        if crypto_history:
            z_score = calculate_z_score(pd.DataFrame(crypto_history.history))
            min_gain, max_gain = calculate_gain_range(crypto_history.history)
            volume = calculate_volume(crypto_history.history)
            if crypto_history.history[-1]['close'] > CRYPTO_MAX_PRICE:
                return None
            current_price = crypto_history.history[-1]['close']
            # Define dynamic buy and sell thresholds based on the current price
            buy_price_threshold = current_price * 0.7  # 30% below current price
            sell_price_threshold = current_price * 1.2  # 20% above current price
            
            crypto: CryptoCurrency = CryptoCurrency(
                symbol=symbol,
                name=crypto_history.name,
                current_price=current_price,
                volume=volume,
                quote_volume=0,
                macd=crypto_history.macd['macd'].iloc[-1],
                macd_signal=crypto_history.macd['signal'].iloc[-1],
                ma_50=crypto_history.ma_50['ma'].iloc[-1],
                ma_200=crypto_history.ma_200['ma'].iloc[-1],
                rsi=crypto_history.rsi['rsi'].iloc[-1],
                support=crypto_history.support,
                resistance=crypto_history.resistance,
                z_score=z_score,
                min_gain=min_gain,
                max_gain=max_gain,
                buy_price_threshold=buy_price_threshold,
                sell_price_threshold=sell_price_threshold
            )
            return crypto
    return None


def check_coin_state(debug: bool = False) -> List[CryptoCurrency]:
    
    tickers: List[dict] = client.get_ticker()
    with Pool(cpu_count()) as pool:
        results: List[CryptoCurrency] = pool.map(worker, tickers)

    results = [crypto for crypto in results if crypto is not None and crypto.max_gain > GAIN_THRESHOLD]
    results.sort(key=lambda crypto: crypto.max_gain, reverse=True)
    
    
    return results


def print_coin_details(crypto: CryptoCurrency) -> None:
    print("------------------------------------------------------------------")
    print(f"Symbol: {crypto.symbol}")
    print(f"Name: {crypto.name}")
    print(f"Current Price: ${crypto.current_price:.2f}")
    print(f"Resistance: ${crypto.resistance:.2f}")
    print(f"Support: ${crypto.support:.2f}")
    print(f"MACD: {crypto.macd:.2f}")
    print(f"MACD Signal: {crypto.macd_signal:.2f}")
    print(f"50-day MA: {crypto.ma_50:.2f}")
    print(f"200-day MA: {crypto.ma_200:.2f}")
    print(f"RSI: {crypto.rsi:.2f}")
    print(f"Volume: {crypto.volume:.2f}")
    print(f"Z-score: {crypto.z_score:.2f}")
    print(f"Min Gain: {crypto.min_gain:.2f}, Max Gain: {crypto.max_gain:.2f}")
    print("Action: ")
    print("    * BUY:   ",      crypto.should_buy())
    print("    * SELL:  ",    crypto.should_sell())
    print("    * AWAIT: ",   crypto.await_till_value())
    print("------------------------------------------------------------------")

def filter_crypto_list_by_names(cryptos: List[CryptoCurrency], symbols: List[str]) -> List[CryptoCurrency]:
        """
        Filtra uma lista de criptomoedas com base em uma outra lista.

        :param other_list: Lista de instâncias de CryptoCurrency.
        :return: Lista de criptomoedas que estão em ambas as listas.
        """
        filtered_cryptos = [crypto for crypto in cryptos if crypto.symbol in symbols]
        return filtered_cryptos

class Watcher:
    def __init__(self, crypto_list: list):
        """
        Inicializa o Watcher com uma lista de criptomoedas.

        :param crypto_list: Lista de instâncias de CryptoCurrency.
        """
        self.cryptos = crypto_list

    def main_loop(self):
        """
        Loop principal que monitora as criptomoedas e imprime quando deve comprar, vender ou aguardar.
        """
        while True:
            for crypto in self.cryptos:
                print("*" * 50)                
                crypto.update()                                
                print_coin_details(crypto)

            # Intervalo de tempo para o próximo loop pode ser ajustado conforme necessário
            # Por exemplo, para esperar 10 segundos antes da próxima verificação
            time.sleep(10)


if __name__ == "__main__":
    cryptos = check_coin_state(debug=True)

    filtered_cryptos: List[CryptoCurrency] = filter_crypto_list_by_names(cryptos, ["FIROUSDT"])
    

    watcher: Watcher = Watcher(filtered_cryptos)
    watcher.main_loop()
