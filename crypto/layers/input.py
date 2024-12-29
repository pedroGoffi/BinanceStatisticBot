## 1. Input Layer: Market Data Acquisition
# **Components:**
# - Market Data Feeds: APIs for real-time data from exchanges (price, volume, order book, etc.).
# - Data Aggregators: Aggregate data from multiple exchanges or sources.
# - Websockets: For low-latency streaming of market data.
# - Historical Data Storage: For backtesting and strategy development.

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
from binance.client import Client
from typing import List, Dict




def load_cryptos(client: Client, query: List[str] = None, currency: str = "USDT") -> List[Dict[str, str]]:
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

    if query is None:
        tickers: List[Dict[str, str]] = client.get_all_tickers()
    else:
        tickers: List[Dict[str, str]] = [client.get_ticker(symbol=symbol) for symbol in query]

    if currency:
        tickers = [ticker for ticker in tickers if ticker['symbol'].endswith(currency)]
        

    return tickers
    
    