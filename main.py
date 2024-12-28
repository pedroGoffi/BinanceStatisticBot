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


from collections import defaultdict
import textToSpeech
import os
from tqdm import tqdm
import time
from typing import TextIO
from argparse import ArgumentParser
from binance.client import Client
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Literal, List, Tuple
import pandas as pd
import numpy as np
from scipy.stats import zscore  # Import zscore from scipy.stats
from multiprocessing import Pool, cpu_count  # Import Pool for multiprocessing
from sys import stdout
load_dotenv()
LOGO = (
    "  _                           \n"
    " |_) o ._   __   _ _|_  _. _|_  _ \n"
    " |_) | | |      _>  |_ (_|  |_ _> \n"
    "==============================\n"
    "By: Pedro Henrique Goffi de Paulo."
)
tts: textToSpeech.TTS
BUY_SCORE_THRESHOLD:    int     = 2
SELL_SCOPE_THRESHOLD:   int     = -2
GAIN_THRESHOLD:         float   = float(os.getenv("GAIN_THRESHOLD"))        or 0.0
CRYPTO_MAX_PRICE:       float   = float(os.getenv("CRYPTO_MAX_PRICE"))      or 0.0
WATCH_VAR_MIN_TARGET:   float   = float(os.getenv("WATCH_VAR_MIN_TARGET"))  or 5.0
INTERVAL_TO_WATCH:      str     = os.getenv("INTERVAL_TO_WATCH")            or '15m'
WATCH_VAR_MAX_TARGET:   float   = float(os.getenv("WATCH_VAR_MAX_TARGET"))  or 20.0
API_KEY:                str     = os.getenv("API_KEY")
API_SECRET:             str     = os.getenv("API_SECRET")
client: Client = Client(API_KEY, API_SECRET)

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
    price_variance: pd.Series = None  # Nova adição
    var_24h: pd.Series = None         # Nova adição
    momentum: pd.Series = None        # Nova adição

@dataclass
class CryptoCurrency:
    """
    A class to represent a cryptocurrency and its trading indicators.
    Attributes:
    -----------
    symbol : str
        The symbol of the cryptocurrency.
    name : str
        The name of the cryptocurrency.
    current_price : float
        The current price of the cryptocurrency.
    volume : float
        The trading volume of the cryptocurrency.
    quote_volume : float
        The quote volume of the cryptocurrency.
    macd : float
        The MACD (Moving Average Convergence Divergence) value.
    macd_signal : float
        The MACD signal value.
    ma_50 : float
        The 50-day moving average.
    ma_200 : float
        The 200-day moving average.
    rsi : float
        The Relative Strength Index value.
    support : float
        The support level.
    resistance : float
        The resistance level.
    z_score : float
        The Z-score value.
    min_gain : float
        The minimum gain.
    max_gain : float
        The maximum gain.
    price_variance : float
        The price variance.
    var_24h : float
        The 24-hour variance.
    buy_price_threshold : float
        The price threshold for buying.
    sell_price_threshold : float
        The price threshold for selling.
    crypto_history : CryptoCurrencyHistory
        The historical data of the cryptocurrency.
    Methods:
    --------
    update():
        Updates the cryptocurrency data and calculates various indicators.
    should_buy() -> str:
        Determines if the cryptocurrency should be bought based on various indicators.
    create_context_str(context: list) -> str:
        Creates a formatted string from the context list.
    await_till_value() -> str:
        Provides recommendations on waiting until certain conditions are met.
    should_sell() -> str:
        Determines if the cryptocurrency should be sold based on various indicators.
    """
        
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
    price_variance: float
    var_24h: float
    buy_price_threshold: float  # New field for buying threshold
    sell_price_threshold: float  # New field for selling threshold
    crypto_history: CryptoCurrencyHistory = None
    score: int = 0


    def update(self) -> None:
        """
        Updates the cryptocurrency data for the given symbol.
        This method fetches the latest cryptocurrency history, calculates various
        metrics such as z-score, gain range, volume, MACD, moving averages, RSI,
        support, resistance, and price variance. It also defines dynamic buy and
        sell thresholds based on the current price.
        Attributes:
            current_price (float): The current price of the cryptocurrency.
            volume (float): The trading volume of the cryptocurrency.
            macd (float): The MACD value of the cryptocurrency.
            macd_signal (float): The MACD signal value of the cryptocurrency.
            ma_50 (float): The 50-day moving average of the cryptocurrency.
            ma_200 (float): The 200-day moving average of the cryptocurrency.
            rsi (float): The RSI value of the cryptocurrency.
            support (float): The support level of the cryptocurrency.
            resistance (float): The resistance level of the cryptocurrency.
            z_score (float): The z-score of the cryptocurrency.
            min_gain (float): The minimum gain range of the cryptocurrency.
            max_gain (float): The maximum gain range of the cryptocurrency.
            buy_price_threshold (float): The dynamic buy price threshold.
            sell_price_threshold (float): The dynamic sell price threshold.
            var_24h (float): The 24-hour variance of the cryptocurrency.
            price_variance (float): The price variance of the cryptocurrency.
        Returns:
            None
        """
        
        self.crypto_history = get_crypto_history(self.symbol)
        if self.crypto_history:
            z_score = calculate_z_score(pd.DataFrame(self.crypto_history.history))
            min_gain, max_gain = calculate_gain_range(self.crypto_history.history)
            volume = calculate_volume(self.crypto_history.history)
            if self.crypto_history.history[-1]['close'] > CRYPTO_MAX_PRICE:
                return None
            
            
            current_price                   = self.crypto_history.history[-1]['close']            
            # Define dynamic buy and sell thresholds based on the current price
            buy_price_threshold             = current_price * 0.7  # 30% below current price
            sell_price_threshold            = current_price * 1.2  # 20% above current price            
            self.current_price              = current_price
            self.volume                     = volume
            self.macd                       = self.crypto_history.macd['macd'].iloc[-1]
            self.macd_signal                = self.crypto_history.macd['signal'].iloc[-1]
            self.ma_50                      = self.crypto_history.ma_50['ma'].iloc[-1]
            self.ma_200                     = self.crypto_history.ma_200['ma'].iloc[-1]
            self.rsi                        = self.crypto_history.rsi['rsi'].iloc[-1]
            self.support                    = self.crypto_history.support
            self.resistance                 = self.crypto_history.resistance
            self.z_score                    = z_score
            self.min_gain                   = min_gain
            self.max_gain                   = max_gain
            self.buy_price_threshold        = buy_price_threshold
            self.sell_price_threshold       = sell_price_threshold
            self.var_24h                    = self.crypto_history.var_24h.iloc[-1]
            self.price_variance             = self.crypto_history.macd['price_variance'].iloc[-1]

        
        print(f"[UPDATE: {self.symbol}]:\t\tvariancia desde a ultima atualização: {self.price_variance:.2f} - \tscopre: {self.score} {'✅' if self.score > 0 else '❌'} ")
        if self.score >= 3:
            print(f"🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
            

    
    def should_buy(self) -> str:
        """
        Determines whether to buy based on various financial indicators.
        The method evaluates several factors including Z-score, RSI, MACD, 
        moving average, and trading volume to compute a score. Based on the 
        score, it returns a recommendation to buy, avoid buying, or wait.
        Returns:
            str: A recommendation string indicating whether to buy, avoid buying, 
                 or wait, along with the context of the decision.
        """
               
        context = []

        # Z-score factor
        if self.z_score > 0:
            self.score += 1            
            context.append(f"✅ Z-score positivo ({self.z_score:.2f}) indica tendência de alta")
        elif self.z_score < 0:
            self.score -= 1
            context.append(f"❌ Z-score negativo ({self.z_score:.2f}) indica tendência de baixa")

        # RSI factor
        if self.rsi < 30:
            self.score += 1
            context.append(f"✅ RSI baixo ({self.rsi:.2f}) indica sobrevendido")
        elif self.rsi > 70:
            self.score -= 1
            context.append(f"❌ RSI alto ({self.rsi:.2f}) indica sobrecomprado")

        # MACD factor
        if self.macd > self.macd_signal:
            self.score += 1
            context.append(f"✅ MACD ({self.macd:.2f}) acima do sinal ({self.macd_signal:.2f}) indica tendência de alta")
        elif self.macd < self.macd_signal:
            self.score -= 1
            context.append(f"❌ MACD ({self.macd:.2f}) abaixo do sinal ({self.macd_signal:.2f}) indica tendência de baixa")

        # Moving average factor
        if self.current_price < self.ma_50:
            self.score += 1
            context.append(f"✅ Preço atual ({self.current_price:.2f}) abaixo da MA de 50 dias ({self.ma_50:.2f})")
        elif self.current_price > self.ma_50:
            self.score -= 1
            context.append(f"❌ Preço atual ({self.current_price:.2f}) acima da MA de 50 dias ({self.ma_50:.2f})")

        # Volume factor
        if self.volume > 1000000:
            self.score += 1
            context.append(f"✅ Volume alto ({self.volume:.2f})")
        elif self.volume < 1000000:
            self.score -= 1
            context.append(f"❌ Volume baixo ({self.volume:.2f})")

        # Determine action based on score
        context_str = self.create_context_str(context)

        if self.score >= BUY_SCORE_THRESHOLD:
            return f"✅ Comprar Agora: Espera-se Aumento {context_str}"
        elif self.score < SELL_SCOPE_THRESHOLD:
            return f"❌ Evitar Compra: Espera-se Queda {context_str}"
        else:
            return f"⏳ Aguardar: Condições Neutras {context_str}"

    def create_context_str(self, context):
        """
        Creates a formatted string from a list of context items.

        Args:
            context (list): A list of context items to be formatted.

        Returns:
            str: A formatted string where each context item is prefixed with a newline and tab, 
                 and the entire string ends with a newline and tab.
        """
        context_str: str = ""
        for i in range(len(context)):
            context_str += f'\n\t\t|{context[i]}'
        context_str += '\n\t'
        return context_str

    def await_till_value(self) -> str:
        """
        Generates a string indicating the conditions that need to be met before taking action based on various
        financial indicators such as RSI, Z-score, price thresholds, MACD, and moving averages.
        Returns:
            str: A message indicating the conditions to wait for, or a message indicating no action is necessary.
        """
        context = []

        # RSI factor
        if self.rsi < 30:
            context.append(f"⏳ Aguardar Até RSI({self.rsi:.2f}) atingir um limite mínimo (30)")
        elif self.rsi > 70:
            context.append(f"⏳ Aguardar Até RSI({self.rsi:.2f}) retornar a um valor mais favorável")

        # Z-score factor
        if self.z_score > 0:
            context.append(f"⏳ Aguardar Até Z-score ({self.z_score:.2f}) retornar a um valor mais favorável")
        elif self.z_score < 0:
            context.append(f"⏳ Aguardar Até Z-score ({self.z_score:.2f}) atingir um limite mínimo")

        # Price thresholds
        if self.current_price <= self.buy_price_threshold:
            context.append(f"⏳ Aguardar até o preço atingir: ${self.buy_price_threshold:.2f}")
        elif self.current_price >= self.sell_price_threshold:
            context.append(f"⏳ Aguardar até o preço cair para: ${self.sell_price_threshold:.2f}")

        # MACD factor
        if self.macd > self.macd_signal:
            context.append(f"⏳ Aguardar até MACD ({self.macd:.2f}) cruzar abaixo do sinal ({self.macd_signal:.2f})")
        elif self.macd < self.macd_signal:
            context.append(f"⏳ Aguardar até MACD ({self.macd:.2f}) cruzar acima do sinal ({self.macd_signal:.2f})")

        # Moving average factor
        if self.current_price < self.ma_50:
            context.append(f"⏳ Aguardar até o preço subir acima da MA de 50 dias ({self.ma_50:.2f})")
        elif self.current_price > self.ma_50:
            context.append(f"⏳ Aguardar até o preço cair abaixo da MA de 50 dias ({self.ma_50:.2f})")

        if not context:
            return "❌ Nenhuma Ação Necessária"

        context_str = self.create_context_str(context)
        return f"⏳ Aguardar: {context_str}"

    def should_sell(self) -> str:
        """
        Determines whether to sell based on various financial indicators.
        The method evaluates the following factors:
        - Z-score: Indicates market trend (negative for bearish, positive for bullish).
        - RSI (Relative Strength Index): Indicates overbought (above 70) or oversold (below 30) conditions.
        - MACD (Moving Average Convergence Divergence): Indicates trend direction based on MACD and signal line comparison.
        - Moving Average (MA): Compares current price with the 50-day moving average.
        - Volume: Evaluates trading volume (high or low).
        Returns:
            str: A recommendation string indicating whether to sell, avoid selling, or wait, along with the context of the decision.
        """        
        context = []

        # Z-score factor
        if self.z_score < 0:
            self.score += 1            
            context.append(f"✅ Z-score negativo ({self.z_score:.2f}) indica tendência de baixa")
        elif self.z_score > 0:
            self.score -= 1
            context.append(f"❌ Z-score positivo ({self.z_score:.2f}) indica tendência de alta")

        # RSI factor
        if self.rsi > 70:
            self.score += 1
            context.append(f"✅ RSI alto ({self.rsi:.2f}) indica sobrecomprado")
        elif self.rsi < 30:
            self.score -= 1
            context.append(f"❌ RSI baixo ({self.rsi:.2f}) indica sobrevendido")

        # MACD factor
        if self.macd < self.macd_signal:
            self.score += 1
            context.append(f"✅ MACD ({self.macd:.2f}) abaixo do sinal ({self.macd_signal:.2f}) indica tendência de baixa")
        elif self.macd > self.macd_signal:
            self.score -= 1
            context.append(f"❌ MACD ({self.macd:.2f}) acima do sinal ({self.macd_signal:.2f}) indica tendência de alta")

        # Moving average factor
        if self.current_price > self.ma_50:
            self.score += 1
            context.append(f"✅ Preço atual ({self.current_price:.2f}) acima da MA de 50 dias ({self.ma_50:.2f})")
        elif self.current_price < self.ma_50:
            self.score -= 1
            context.append(f"❌ Preço atual ({self.current_price:.2f}) abaixo da MA de 50 dias ({self.ma_50:.2f})")

        # Volume factor
        if self.volume > 1000000:
            self.score += 1
            context.append(f"✅ Volume alto ({self.volume:.2f})")
        elif self.volume < 1000000:
            self.score -= 1
            context.append(f"❌ Volume baixo ({self.volume:.2f})")

        # Determine action based on score
        context_str = self.create_context_str(context)

        if self.score >= BUY_SCORE_THRESHOLD:
            return f"✅ Vender Agora: Espera-se Queda {context_str}"
        elif self.score < SELL_SCOPE_THRESHOLD:
            return f"❌ Evitar Venda: Espera-se Aumento {context_str}"
        else:
            return f"⏳ Aguardar: Condições Neutras {context_str}"


def get_24_variance(current_price: float, opening_price: float) -> float:
    """
    Calculate the 24-hour percentage change of a cryptocurrency.

    Parameters:
    - current_price (float): The current price of the cryptocurrency.
    - opening_price (float): The price of the cryptocurrency 24 hours ago.

    Returns:
    - float: The 24-hour percentage change, rounded to 2 decimal places.
    """
    if opening_price == 0:
        return 0

    percentage_change = ((current_price - opening_price) / opening_price) * 100
    return round(percentage_change, 2)

def get_crypto_history(symbol: str, interval: Literal['15m', '1h', '4h', '1d'] = '15m') -> CryptoCurrencyHistory | None:
    """
    Fetches historical cryptocurrency data for a given symbol and interval, and calculates various technical indicators.
    Args:
        symbol (str): The symbol of the cryptocurrency (e.g., 'BTCUSDT').
        interval (Literal['15m', '1h', '4h', '1d'], optional): The interval for the historical data. Defaults to '15m'.
    Returns:
        CryptoCurrencyHistory | None: An object containing the cryptocurrency's historical data and technical indicators, or None if no data is available.
    The returned CryptoCurrencyHistory object contains:
        - symbol (str): The symbol of the cryptocurrency.
        - name (str): The name of the cryptocurrency (derived from the symbol).
        - history (List[dict]): A list of dictionaries containing historical data points.
            Each dictionary contains:
                - timestamp (int): The timestamp of the data point.
                - open (float): The opening price.
                - high (float): The highest price.
                - low (float): The lowest price.
                - close (float): The closing price.
                - volume (float): The trading volume.
                - variance (float): The price variance between the current and previous data points.
                - 24h_var (float): The 24-hour price variance.
        - macd (dict): The MACD (Moving Average Convergence Divergence) data.
        - ma_50 (float): The 50-period moving average.
        - ma_200 (float): The 200-period moving average.
        - rsi (float): The Relative Strength Index.
        - support (float): The support level.
        - resistance (float): The resistance level.
        - price_variance (float): The price variance from the MACD data.
        - var_24h (float): The 24-hour price variance from the MACD data.
    """
    
    klines: List[list] = client.get_historical_klines(symbol=symbol, interval=interval, start_str="24 hours ago UTC", end_str="now UTC", limit=None)
    if not klines:
        return None
    history: List[dict] = []
    var_24h: float = get_24_variance(float(klines[-1][4]), float(klines[0][4]))

    for i in range(len(klines)):
        kline: list = klines[i]
        prevkline: list = klines[i - 1] if i > 0 else None
        # use price variance to get prev vs current variance
        if prevkline:
            price_variance = ((float(kline[4]) - float(prevkline[4])) / float(prevkline[4])) * 100
        else:
            price_variance = 0.0


        history.append({
            'timestamp':    kline[0],
            'open':         float(kline[1]),
            'high':         float(kline[2]),
            'low':          float(kline[3]),
            'close':        float(kline[4]),
            'volume':       float(kline[5]),
            'variance':     price_variance,
            '24h_var':      var_24h,
            'momentum':     float(kline[4]) - float(prevkline[4]) if prevkline else 0.0
        })
        
    
    
    macd_data = calculate_macd(history)

    return CryptoCurrencyHistory(
        symbol                  =   symbol,
        name                    =   symbol[:-4],
        history                 =   history,
        macd                    =   macd_data,
        ma_50                   =   calculate_moving_average(history),
        ma_200                  =   calculate_moving_average(history, 200),
        rsi                     =   calculate_rsi(history),
        support                 =   calculate_support_resistance(history)[0],
        resistance              =   calculate_support_resistance(history)[1],
        price_variance          =   macd_data['price_variance'],
        var_24h                 =   macd_data['24h_var'],
        momentum                =   macd_data['momentum']   
    )

predictions = []
def predict_price_movement(crypto: CryptoCurrency) -> str:
    """
    Predicts whether the price of the cryptocurrency will go up, down, or remain neutral based on a combination of calculated statistics.
    The prediction is based on the following factors:
    - Z-score: A positive Z-score increases the score, while a negative Z-score decreases it.
    - RSI (Relative Strength Index): An RSI below 30 increases the score, while an RSI above 70 decreases it.
    - MACD (Moving Average Convergence Divergence): If the MACD is above the MACD signal line, the score increases; if below, the score decreases.
    - Moving Average (50-day): If the current price is below the 50-day moving average, the score increases; if above, the score decreases.
    The final score determines the prediction:
    - Score > 1: High confidence in price increase ("acrescento")
    - Score < -1: High confidence in price decrease ("queda")
    - -1 <= Score <= 1: Low confidence, neutral movement expected ("neutral")
    Args:
        crypto (CryptoCurrency): An instance of the CryptoCurrency class containing the necessary attributes for prediction.
    Returns:
        str: A string indicating the predicted price movement ("acrescento", "queda", or "neutral").        
    """
    score = 0
    max_score = 4

    # Z-score factor
    if crypto.z_score > 0:
        score += 1
    elif crypto.z_score < 0:
        score -= 1

    # RSI factor
    if crypto.rsi < 30:
        score += 1
    elif crypto.rsi > 70:
        score -= 1

    # MACD factor
    if crypto.macd > crypto.macd_signal:
        score += 1
    elif crypto.macd < crypto.macd_signal:
        score -= 1

    # Moving average factor
    if crypto.current_price < crypto.ma_50:
        score += 1
    elif crypto.current_price > crypto.ma_50:
        score -= 1

    # Determine prediction based on score
    print(f"[{score = }] {'✅ significa que espera-se aumento' if score > 0 else '❌ significa que espera-se queda'}")
    print(f"Pontuação: {score}/{max_score} - {crypto.symbol} - {'✅ espera-se aumento' if score > 0 else '❌ espera-se queda'}") 
    
    if score > 1:
        print(f"✅ confiabilidade alta  - esperado aumento")
        return "up"
    elif score < -1:
        print(f"✅ confiabilidade alta  - esperado queda")
        return "down"
    else:
        print(f"❌ confiabilidade baixa - espera-se movimento neutro")
        return "neutral"

def check_prediction_accuracy(predictions: List[Tuple[str, str, float, float]]) -> float:
    """
    Calculate the accuracy of predictions.
    This function takes a list of tuples where each tuple contains a predicted value and the actual value.
    It calculates the percentage of correct predictions.
    Args:
        predictions (List[Tuple[str, str]]): A list of tuples, where each tuple contains a predicted value and the actual value.
    Returns:
        float: The accuracy of the predictions as a percentage. If there are no predictions, it returns 0.0.
    """

    correct_predictions = sum(1 for prediction, actual, _, _ in predictions if prediction == actual)
    total_predictions = len(predictions)
    if total_predictions == 0:
        return 0.0
    
    print(f"Predições:")
    for prediction, actual, current_price, acrescimo_percentual in predictions[-5:]:
        # Calculate the average percentage increase
        avg_acrescimo_percentual = sum(acrescimo_percentual for _, _, _, acrescimo_percentual in predictions) / len(predictions)
        predicted_price = current_price * (1 + avg_acrescimo_percentual / 100)
        print("Predição:")
        print(f"\tMédia do acréscimo percentual: {avg_acrescimo_percentual:.2f}%")
        print(f"\tAcertos: {correct_predictions} Erros: {total_predictions} - Taxa de acertos: {correct_predictions / total_predictions * 100:.2f}%")
        print(f"\tPreço previsto: {predicted_price:.2f} - Preço atual: {current_price:.2f}")
        print(f"\tPredição: {prediction} - Real: {actual} - {'✅' if prediction == actual else '❌'}")
    return (correct_predictions / total_predictions) * 100

periods: int = 15
current_period: int = 0

def update_predictions(crypto: CryptoCurrency):    
    """
    Updates the predictions for the given cryptocurrency and calculates the accuracy of the predictions.
    Args:
        crypto (CryptoCurrency): An instance of the CryptoCurrency class containing the current price and historical data.
    Updates:
        - Increments the global variable `current_period`.
        - Appends the predicted and actual price movement to the global `predictions` list.
        - Calculates the accuracy of the predictions.
    Prints:
        - The accuracy of the predictions if the current period equals the total number of periods.
        - The current period and accuracy of the predictions otherwise.
    """
    
    global periods, current_period
    current_period += 1
    prediction = predict_price_movement(crypto)
    actual_movement = "acrecento" if crypto.current_price > crypto.crypto_history.history[-2]['close'] else "queda"
    predictions.append((
        prediction                                                                                                                      # predição
        , actual_movement                                                                                                               # movimento real                                                    
        , crypto.current_price                                                                                                          # preço atual
        , crypto.current_price / crypto.crypto_history.history[-2]['close'] if crypto.crypto_history.history[-2]['close'] != 0 else 0.0 # acrescimo percentual
    ))
    
    accuracy = check_prediction_accuracy(predictions)

    if current_period == periods:
        print(f"Taxa de acertos: {accuracy:.2f}%")
        current_period = 0
    else:
        print(f"Período {current_period}/{periods} - Taxa de acertos: {accuracy:.2f}%")



def calculate_24_hour_change(current_price, opening_price):
    """
    This function takes the current price and the opening price of a cryptocurrency
    from 24 hours ago and calculates the percentage change over that period. The 
    result is rounded to two decimal places for precision.
    - float: The 24-hour percentage change, rounded to 2 decimal places. If the 
        opening price is zero, the function returns 0 to avoid division by zero.
    
    Calculate the 24-hour percentage change of a cryptocurrency.

    Parameters:
    - current_price (float): The current price of the cryptocurrency.
    - opening_price (float): The price of the cryptocurrency 24 hours ago.

    Returns:
    - float: The 24-hour percentage change, rounded to 2 decimal places.
    """
    if opening_price == 0:
        return 0

    percentage_change = ((current_price - opening_price) / opening_price) * 100
    return round(percentage_change, 2)

def calculate_macd(data: List[dict], short_period: int = 3, long_period: int = 10, signal_period: int = 15) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given dataset.
    Parameters:
    data (List[dict]): A list of dictionaries containing the dataset with keys 'timestamp', 'close', 'high', 'variance', and '24h_var'.
    short_period (int): The period for the short-term exponential moving average (EMA). Default is 3.
    long_period (int): The period for the long-term exponential moving average (EMA). Default is 10.
    signal_period (int): The period for the signal line EMA. Default is 15.
    Returns:
    pd.DataFrame: A DataFrame containing the 'timestamp', 'macd', 'signal', 'price_variance', and '24h_var' columns.
    The function performs the following steps:
    1. Converts the input data into a pandas DataFrame.
    2. Converts the 'close' and 'high' columns to float type.
    3. Calculates the short-term EMA and long-term EMA of the 'close' prices.
    4. Computes the MACD as the difference between the short-term EMA and long-term EMA.
    5. Calculates the signal line as the EMA of the MACD.
    6. Adds the 'variance' and '24h_var' columns to the DataFrame as 'price_variance' and 'var_24h', respectively.
    7. Returns a DataFrame with the 'timestamp', 'macd', 'signal', 'price_variance', and '24h_var' columns.
    """
    df: pd.DataFrame = pd.DataFrame(data)    
    df['close']             = df['close'].astype(float)
    df['high']              = df['high'].astype(float)
    df['ema_short']         = df['close'].ewm(span=short_period, min_periods=1, adjust=False).mean()
    df['ema_long']          = df['close'].ewm(span=long_period, min_periods=1, adjust=False).mean()
    df['macd']              = df['ema_short'] - df['ema_long']
    df['signal']            = df['macd'].ewm(span=signal_period, min_periods=1, adjust=False).mean()        
    df['price_variance']    = df['variance']
    df['var_24h']           = df['24h_var']
    

    return df[['timestamp', 'macd', 'signal', 'price_variance', '24h_var']]


def calculate_moving_average(data: List[dict], period: int = 10) -> pd.DataFrame:
    """
    Calculate the moving average of the 'close' prices over a specified period.

    Args:
        data (List[dict]): A list of dictionaries containing the data with 'timestamp' and 'close' keys.
        period (int, optional): The period over which to calculate the moving average. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing 'timestamp' and the calculated moving average 'ma'.
    """
    df: pd.DataFrame = pd.DataFrame(data)
    df['close'] = df['close'].astype(float)
    df['ma'] = df['close'].rolling(window=period).mean()
    return df[['timestamp', 'ma']]


def calculate_rsi(data: List[dict], period: int = 14) -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI) for a given dataset.

    Parameters:
    data (List[dict]): A list of dictionaries containing the data with 'timestamp' and 'close' keys.
    period (int): The period over which to calculate the RSI. Default is 14.

    Returns:
    pd.DataFrame: A DataFrame containing 'timestamp' and 'rsi' columns.
    """
    df: pd.DataFrame = pd.DataFrame(data)
    df['close'] = df['close'].astype(float)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df[['timestamp', 'rsi']]


def calculate_z_score(data: pd.DataFrame) -> float:
    """
    Calculate the z-score of the 'close' column in the given DataFrame.
    The z-score is a measure of how many standard deviations an element is from the mean.
    This function returns the z-score of the last element in the 'close' column.
    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing a 'close' column with numerical values.
    Returns:
    float: The z-score of the last element in the 'close' column, or None if the input data is empty,
           does not contain a 'close' column, or if the z-score calculation results in NaN.
    """
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
    """
    Calculate the support and resistance levels from a list of market data.

    Args:
        data (List[dict]): A list of dictionaries containing market data with keys 'high' and 'low'.

    Returns:
        Tuple[float, float]: A tuple containing the support (minimum low) and resistance (maximum high) levels.
    """
    df: pd.DataFrame = pd.DataFrame(data)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    support = df['low'].min()
    resistance = df['high'].max()
    return support, resistance


def calculate_volume(data: List[dict]) -> float:
    """
    Calculate the total volume from a list of dictionaries containing volume data.

    Args:
        data (List[dict]): A list of dictionaries where each dictionary contains a 'volume' key.

    Returns:
        float: The total volume calculated by summing the 'volume' values.
    """
    df: pd.DataFrame = pd.DataFrame(data)
    df['volume'] = df['volume'].astype(float)
    total_volume = df['volume'].sum()
    return total_volume

def calculate_gain_range(history: List[dict]) -> Tuple[float, float]:
    """
    Calculate the minimum and maximum gain range from a trading history.
    Args:
        history (List[dict]): A list of dictionaries containing trading history data. 
                              Each dictionary should have 'high' and 'low' keys with float values.
    Returns:
        Tuple[float, float]: A tuple containing the minimum gain percentage and the maximum gain percentage.
    """
    # Convert trading history to a DataFrame
    df = pd.DataFrame(history)
    
    # Ensure columns are in float format for calculations
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    # Find the lowest price and the highest price
    min_price = df['low'].min()
    max_price = df['high'].max()
    
    # Calculate the minimum gain as the smallest possible gain from low to high
    min_gain = ((df['high'].min() - min_price) / min_price) * 100  # Min gain percentage
    
    # Calculate the maximum gain as the largest possible gain from low to high
    max_gain = ((max_price - min_price) / min_price) * 100  # Max gain percentage
    
    return min_gain, max_gain

def worker(ticker: dict) -> CryptoCurrency:
    """
    Processes a cryptocurrency ticker and returns a CryptoCurrency object if the ticker meets certain criteria.
    Args:
        ticker (dict): A dictionary containing the cryptocurrency ticker information. 
                       Expected to have a 'symbol' key with the ticker symbol as its value.
    Returns:
        CryptoCurrency: An object containing detailed information about the cryptocurrency if it meets the criteria.
                        Returns None if the criteria are not met.
    Criteria:
        - The ticker symbol must end with 'USDT'.
        - The closing price of the latest entry in the cryptocurrency's history must be less than CRYPTO_MAX_PRICE.
        - The function calculates various metrics such as z-score, gain range, volume, MACD, moving averages, RSI, 
          support, resistance, and price variance.
        - Dynamic buy and sell thresholds are defined based on the current price.
    The returned CryptoCurrency object includes:
        - symbol: The ticker symbol.
        - name: The name of the cryptocurrency.
        - current_price: The latest closing price.
        - volume: The trading volume.
        - quote_volume: The quote volume (set to 0).
        - macd: The latest MACD value.
        - macd_signal: The latest MACD signal value.
        - ma_50: The latest 50-day moving average.
        - ma_200: The latest 200-day moving average.
        - rsi: The latest RSI value.
        - support: The support level.
        - resistance: The resistance level.
        - z_score: The calculated z-score.
        - min_gain: The minimum gain.
        - max_gain: The maximum gain.
        - buy_price_threshold: The calculated buy price threshold.
        - sell_price_threshold: The calculated sell price threshold.
        - price_variance: The price variance.
        - var_24h: The 24-hour variance.
        - crypto_history: The historical data of the cryptocurrency.
    """
    symbol: str = ticker['symbol']
    if symbol.endswith('USDT'):
        crypto_history: CryptoCurrencyHistory = get_crypto_history(symbol, interval=INTERVAL_TO_WATCH)
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
                sell_price_threshold=sell_price_threshold,
                price_variance=crypto_history.macd['price_variance'].iloc[-1],
                var_24h=crypto_history.macd['24h_var'].iloc[-1],
                crypto_history=crypto_history
            )
            return crypto
    return None


def check_coin_state(debug: bool = False, query: List[str] = None) -> List[CryptoCurrency]:
    """
    Checks the state of specified cryptocurrencies and returns a list of those with a maximum gain above a certain threshold.
    Args:
        debug (bool): If True, prints debug information. Defaults to False.
        query (List[str], optional): A list of cryptocurrency symbols to check. If None, checks all available cryptocurrencies. Defaults to None.
    Returns:
        List[CryptoCurrency]: A list of CryptoCurrency objects that have a maximum gain above the defined threshold.
    Raises:
        Any exceptions raised by the client.get_all_tickers() or client.get_ticker() methods.
    Example:
        >>> check_coin_state(debug=True, query=["BTCUSDT", "ETHUSDT"])
        Checking coins: ['BTCUSDT', 'ETHUSDT']
        Found 2 coins with a max gain above WATCH_VAR_MIN_TARGET
    Notes:
        - The function uses multiprocessing to parallelize the processing of tickers.
        - The results are filtered to include only those cryptocurrencies with a maximum gain above the GAIN_THRESHOLD.
        - The results are sorted in descending order based on their maximum gain.
    """
    if debug:
        print("Checking all coins..." if query is None else f"Checking coins: {query}")            

    if query is None:
        tickers: List[dict] = client.get_all_tickers()    
    else:
        tickers: List[dict] = [client.get_ticker(symbol=symbol) for symbol in query]
    

    with Pool(cpu_count()) as pool:        
        results: List[CryptoCurrency] = [result for result in pool.map(worker, tickers) if result is not None]

    results = [crypto for crypto in results if crypto is not None and crypto.max_gain > GAIN_THRESHOLD]
    results.sort(key=lambda crypto: crypto.max_gain, reverse=True)
    print(f"Found {len(results)} coins with a max gain above {GAIN_THRESHOLD}%")
    return results


def print_coin_details(file: TextIO, crypto: CryptoCurrency) -> None:
    """
    Prints detailed information about a cryptocurrency to the specified file.
    Args:
        file (TextIO): The file object where the details will be printed.
        crypto (CryptoCurrency): An instance of the CryptoCurrency class containing the cryptocurrency details.
    Prints:
        Detailed information about the cryptocurrency including:
        - Symbol
        - Name
        - Current Price
        - Resistance
        - Support
        - MACD
        - MACD Signal
        - 50-day Moving Average
        - 200-day Moving Average
        - RSI (Relative Strength Index)
        - Volume
        - Z-score
        - Minimum and Maximum Gain
        - 24-hour Price Variation
        - Price Variance
        - Alerts if price variance is greater than WATCH_VAR_MIN_TARGET
        - Recommended actions (Buy, Sell, Wait)
        - Explanation of indicators (RSI, Volume, MACD, Moving Averages, Z-score)
    Note:
        The function also provides recommendations based on the RSI, MACD, and moving averages.
    """

    print("------------------------------------------------------------------", file=file)
    print(f"📊 Símbolo: {crypto.symbol}", file=file)
    print(f"🔤 Nome: {crypto.name}", file=file)
    print(f"💲 Preço Atual: R${crypto.current_price}", file=file)
    print(f"📈 Resistência: R${crypto.resistance:.2f}", file=file)
    print(f"📉 Suporte: R${crypto.support:.2f}", file=file)
    print(f"📊 MACD: {crypto.macd:.2f}", file=file)
    print(f"📊 Sinal MACD: {crypto.macd_signal:.2f}", file=file)
    print(f"📊 Média Móvel de 50 dias: {crypto.ma_50:.2f}", file=file)
    print(f"📊 Média Móvel de 200 dias: {crypto.ma_200:.2f}", file=file)
    print(f"📊 RSI: {crypto.rsi:.2f}", file=file)
    print(f"📊 Volume: {crypto.volume:.2f}", file=file)
    print(f"📊 Z-score: {crypto.z_score:.2f}", file=file)
    print(f"📊 Ganho Mínimo: {crypto.min_gain:.2f}%, Ganho Máximo: {crypto.max_gain:.2f}%", file=file)
    print(f"📊 Variação de Preço (24h): {crypto.var_24h:.2f}%", file=file)
    print(f"📊 Variação de Preço: {crypto.price_variance:.2f}%", file=file)
    if crypto.price_variance > 5:  # Alerta se a variação de preço for maior que 10%
        print("🚨 Alerta: A variação de preço está muito alta! 🚨", file=file)
    print("------------------------------------------------------------------", file=file)
    print("🔍 Ação Recomendada: ", file=file)
    print(f"    📈 COMPRAR: [*] {crypto.should_buy()}", file=file)
    print(f"    📉 VENDER:  [*] {crypto.should_sell()}", file=file)
    print(f"    ⏳ AGUARDAR:[*] {crypto.await_till_value()}", file=file)
    print("------------------------------------------------------------------", file=file)
    print("📖 Explicação dos Indicadores:", file=file)
    print(f"📊 RSI (Índice de Força Relativa): {crypto.rsi:.2f} - ", end='', file=file)
    if crypto.rsi < 30:
        print("✅ Indica que a criptomoeda está sobrevendida, um ponto positivo para compra.", file=file)
    elif crypto.rsi > 70:
        print("❌ Indica que a criptomoeda está sobrecomprada, um ponto negativo para compra.", file=file)
    else:
        print("😐 Está em uma zona neutra.", file=file)
        

    print(f"📊 Volume: {crypto.volume:.2f} - Volume alto pode indicar um forte interesse na criptomoeda, enquanto volume baixo pode indicar o contrário.", file=file)

    print(f"📊 MACD: {crypto.macd:.2f}, Sinal MACD: {crypto.macd_signal:.2f} - ", end='', file=file)
    if crypto.macd > crypto.macd_signal:
        print("✅ MACD acima do sinal MACD indica uma tendência de alta, um ponto positivo para compra.", file=file)
    else:
        print("❌ MACD abaixo do sinal MACD indica uma tendência de baixa, um ponto negativo para compra.", file=file)

    print(f"📊 Média Móvel de 50 dias: {crypto.ma_50:.2f}, Média Móvel de 200 dias: {crypto.ma_200:.2f} - ", end='', file=file)
    if crypto.ma_50 > crypto.ma_200:
        print("✅ A média móvel de 50 dias acima da média móvel de 200 dias indica uma tendência de alta, um ponto positivo para compra.", file=file)
    else:
        print("❌ A média móvel de 50 dias abaixo da média móvel de 200 dias indica uma tendência de baixa, um ponto negativo para compra.", file=file)

    print(f"📊 Z-score: {crypto.z_score:.2f} - ", end='', file=file)
    if crypto.z_score > 0:
        print("✅ Um Z-score positivo indica que o preço está acima da média, o que pode ser um ponto positivo para compra.", file=file)
    else:
        print("❌ Um Z-score negativo indica que o preço está abaixo da média, o que pode ser um ponto negativo para compra.", file=file)
    if crypto.crypto_history:
        update_predictions(crypto)
        
        


def order_crypto_list(cryptos: List[CryptoCurrency], symbol: Literal["max_min", "max_max"]) -> List[CryptoCurrency]:
    """
    Orders a list of cryptocurrencies based on their gain values.
    Parameters:
    cryptos (List[CryptoCurrency]): A list of CryptoCurrency objects to be ordered.
    symbol (Literal["max_min", "max_max"]): A string indicating the ordering criteria.
        - "max_max": Orders the list by maximum gain in descending order.
        - "max_min": Orders the list by minimum gain in descending order.
    Returns:
    List[CryptoCurrency]: The ordered list of CryptoCurrency objects.
    Raises:
    AssertionError: If the symbol is not "max_min" or "max_max".
    """
    print(f"{symbol = }")
    if not symbol: 
        return cryptos

    if symbol == "max_max": 
        return sorted(cryptos, key=lambda crypto: crypto.max_gain, reverse=True)

    if symbol == "max_min": 
        return sorted(cryptos, key=lambda crypto: crypto.min_gain, reverse=True)

    assert False, "Unexpected symbol"
    
def filter_crypto_list_by_names(cryptos: List[CryptoCurrency], symbols: List[str]) -> List[CryptoCurrency]:
        """
        Filtra uma lista de criptomoedas com base em uma outra lista.

        :param other_list: Lista de instâncias de CryptoCurrency.
        :return: Lista de criptomoedas que estão em ambas as listas.
        """
        filtered_cryptos = [crypto for crypto in cryptos if crypto.symbol in symbols]
        return filtered_cryptos

class Watcher:
    """
    A class to monitor a list of cryptocurrencies and determine when to buy, sell, or hold.
    Attributes:
    -----------
    cryptos : list
        A list of instances of CryptoCurrency.
    Methods:
    --------
    __init__(crypto_list: list):
        Initializes the Watcher with a list of cryptocurrencies.
    main_loop(file: TextIO):
        The main loop that monitors the cryptocurrencies and prints whether to buy, sell, or hold.
    """
    def __init__(self, crypto_list: list):
        """
        Inicializa o Watcher com uma lista de criptomoedas.

        :param crypto_list: Lista de instâncias de CryptoCurrency.
        """
        self.cryptos = crypto_list

    def variance_main_loop(self, file: TextIO):
        """
        Loop principal que monitora as criptomoedas e imprime quando deve comprar, vender ou aguardar.
        """
        var_dict: dict[str, float] = defaultdict(float)
        while True:
            os.system("cls")
            print(LOGO)
            for crypto in self.cryptos:
                
                last_var = var_dict.get(crypto.symbol, 0)
                var_dict[crypto.symbol] = crypto.price_variance

                if last_var == 0:
                    continue
            
                # Para atualizar score dentro da função predict_price_movement
                crypto.should_buy()
                crypto.should_sell()
                crypto.await_till_value()

                if crypto.price_variance > WATCH_VAR_MIN_TARGET and crypto.price_variance > last_var:
                    tts.log(f"Pedro, a variação de preço de {crypto.symbol} está muito alta! variacao de {crypto.price_variance}%")
                    print(f"🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")                    
                    print(f"🚨 Alerta: A variação de preço está muito alta! 🚨", file=file)
                    print(f"🚨 Alerta: A variação de preço de {crypto.symbol} está muito alta! 🚨", file=file)
                    print_coin_details(stdout, crypto)
                    if crypto.price_variance > WATCH_VAR_MAX_TARGET:
                        tts.log(f"Pedro, a variação de preço de {crypto.symbol} está acima de {WATCH_VAR_MAX_TARGET}%! Talvez esteja no pico.")

                
                crypto.update()
                crypto.score = 0 # resetar score
                

            

    def main_loop(self, file: TextIO):
        """
        Loop principal que monitora as criptomoedas e imprime quando deve comprar, vender ou aguardar.
        """    
        while True:
            for crypto in self.cryptos:
                print("*" * 50)
                crypto.update()
                
                os.system("cls")
                print(LOGO)
                print_coin_details(file, crypto)
            time.sleep(10)





def main():
    """
    Main function to parse command-line arguments and execute the appropriate actions for cryptocurrency analysis and monitoring.
    Command-line arguments:
    --get: Accepts a comma-separated list of cryptocurrency symbols (e.g., "BTC,ETH,XRP") without spaces.
    --outfile: Specifies the name of the output file to save the results.
    --watch: Enables watch mode to monitor the activity of the specified cryptocurrency symbols.
    --start_bot: Initializes the trading process as programmed (currently commented out).
    --max_max: Filters only the symbols with maximum gain above the specified value.
    --max_min: Filters only the symbols with the highest gain (minimum or maximum) above the specified value.
    The function performs the following steps:
    1. Parses the command-line arguments.
    2. Initializes variables for output file handling, filter list, and watch mode.
    3. Processes the --get argument to create a filter list of symbols.
    4. Sets the watch mode if the --watch argument is provided.
    5. Opens the output file if the --outfile argument is provided.
    6. Retrieves the state of cryptocurrencies based on the filter list.
    7. Filters the cryptocurrency list by names if a filter list is provided.
    8. Orders the cryptocurrency list based on the --max_min or --max_max arguments.
    9. If watch mode is enabled, initializes and starts the watcher loop.
    10. Otherwise, prints the details of each cryptocurrency to the output file or stdout.
    11. Closes the output file if it was opened.
    """
    parser = ArgumentParser(description="Script para análise de criptoativos")
    # Adicionando as flags
    parser.add_argument("--get",        type=str,               help="Aceitar uma lista de strings como symbol1,sym2,sym3,...,sym3 SEM ESPAÇOS!")    
    parser.add_argument("--outfile",    type=str,               help="Nome do arquivo de saída para salvar os resultados")
    parser.add_argument("--watch",      action="store_true",    help="Modo vigilância, usado para monitorar a atividade dos símbolos de cripto")
    parser.add_argument("--start_bot",  action="store_true",    help="Inicializa o processo de trading conforme programado")
    parser.add_argument("--max_max",    action="store_true",     help="Filtrar somente os símbolos com ganho máximo acima do valor especificado")
    parser.add_argument("--max_min",    action="store_true",     help="Filtrar somente os símbolos com o maior ganho (mínimo ou máximo) acima do valor especificado")
    parser.add_argument("--watch_var", action="store_true", help="Modo vigilância, usado para monitorar a atividade dos símbolos de cripto com variante de preço acima de WATCH_VAR_MIN_TARGET em curto período")
    


    

    args                    = parser.parse_args()
    outFile:    TextIO      = stdout
    openedFile: bool        = False
    filterList: List[str]   = None 
    watch:      bool        = False
    watch_var:  bool        = False
    

    
    if args.get:
        tts.log("Carregando lista de criptomoedas filtradas")
        filterList = [symbol for symbol in args.get.split(",")]        
        
    if args.watch:
        tts.log("Modo vigilância ativado")
        watch = True
    if args.watch_var:
        tts.log(f"Modo vigilância ativado com variante de preço acima de {WATCH_VAR_MIN_TARGET}")
        watch_var = True
    
    if args.outfile:
        tts.log(f"Arquivo de saída: {args.outfile}")
        openedFile = True 
        try:
            outFile = open(args.outfile, "w", encoding="utf-8")

        except FileNotFoundError as err:
            parser.print_help()            
            print("[ERRO]: Arquivo nao encontrado")
            exit(1)

    cryptos: List[CryptoCurrency] = check_coin_state(debug=True, query=filterList)        
    if filterList:
        cryptos = filter_crypto_list_by_names(cryptos, filterList)

    
    if args.max_min or args.max_max:
        tts.log(f"Ordenando criptomoedas por {'maior ganho' if args.max_max else 'menor ganho'}")
        cryptos = order_crypto_list(cryptos, "max_min" if args.max_min else "max_max")


    if watch or watch_var:        
        watcher: Watcher = Watcher(cryptos)

        if watch:
            tts.log("Iniciando modo vigilância")
            watcher.main_loop(outFile)
        else:
            tts.log(f"Iniciando modo vigilância com variante de preço acima de {WATCH_VAR_MIN_TARGET}")
            watcher.variance_main_loop(outFile)


    else:
        for crypto in cryptos:
            print_coin_details(outFile, crypto)

    if openedFile:
        outFile.close()


if __name__ == "__main__":    
    print(LOGO)
    tts = textToSpeech.TTS()
    tts.log("Iniciando script de análise de criptoativos")    
    main()