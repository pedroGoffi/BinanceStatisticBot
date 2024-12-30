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


import pprint
from binance.client         import Client
from dotenv                 import load_dotenv
from crypto.layers.ai import CryptoChartPredictor
from crypto.layers.kernel import ITradeKernel, StrategyOne
from crypto.layers.logger   import Logger
from crypto.layers          import preprocess
from crypto.layers.executor import Executor
from crypto.layers.backtest import BacktestEngine, BacktestEngineResult, ISimulationFramework, SimulationFramework
from typing import List, Dict, Any
import os

from main import CryptoCurrency
load_dotenv()
LOGO = (
    "  _                           \n"
    " |_) o ._   __   _ _|_  _. _|_  _ \n"
    " |_) | | |      _>  |_ (_|  |_ _> \n"
    "==============================\n"
    "By: Pedro Henrique Goffi de Paulo."
)
#tts: textToSpeech.TTS
BUY_SCORE_THRESHOLD:    int     = 2
SELL_SCOPE_THRESHOLD:   int     = -2
GAIN_THRESHOLD:         float   = float(os.getenv("GAIN_THRESHOLD"))        or 0.0
CRYPTO_MAX_PRICE:       float   = float(os.getenv("CRYPTO_MAX_PRICE"))      or 0.0
WATCH_VAR_MIN_TARGET:   float   = float(os.getenv("WATCH_VAR_MIN_TARGET"))  or 5.0
INTERVAL_TO_WATCH:      str     = os.getenv("INTERVAL_TO_WATCH")            or '5m'
WATCH_VAR_MAX_TARGET:   float   = float(os.getenv("WATCH_VAR_MAX_TARGET"))  or 20.0
API_KEY:                str     = os.getenv("API_KEY")
API_SECRET:             str     = os.getenv("API_SECRET")

logger: Logger = Logger()
client: Client = Client(API_KEY, API_SECRET)
def main():        
    query: List[str] = ["DFUSDT", "PHAUSDT", "ORCAUSDT"]
        
    cryptos_as_dict: List[Dict[str, str]]   = preprocess.load_cryptos(client=client, query=query, currency="USDT")
    cryptos: List[CryptoCurrency] = preprocess.fast_preprocess_cryptos_dictionary_list(client, cryptos_as_dict, INTERVAL_TO_WATCH, logger)
    preprocess.save_cryptos(cryptos)

            
    logger.debug(f"Inicializando simulacao")
    INITIAL_CASH:   float = 500
    buy_threshold:  float = 0.05
    sell_threshold: float = 0.10
    sub_sets:       int   = 1 
    for crypto in cryptos:               
        kernel:     ITradeKernel        = StrategyOne(logger, crypto)               
        simulator:  SimulationFramework = SimulationFramework("simple simulator", kernel, logger)                
        engine: BacktestEngine = BacktestEngine(kernel, logger)
        engine.atach_simulator(simulator)

        results: BacktestEngineResult = engine.run_simulators(INITIAL_CASH, buy_threshold, sell_threshold, sub_sets)        
        
        logger.debug("*" * 50)
        pprint_fmt: str = pprint.pformat(results)
        logger.debug(pprint_fmt)
        chart: List[preprocess.ChartData] = kernel.crypto.history.chart
        
        # Initialize the model
        start, end = chart[0: len(chart) // 2], chart[len(chart) // 2:]
        
        crypto_predictor = CryptoChartPredictor(data=start)

        # Prepare data
        crypto_predictor.build_model()
        train_loop: int = 4
        for _ in range(train_loop):        
            crypto_predictor.train_model(epochs=50)

        # Predict on new data    
        predicted_price = crypto_predictor.predict(end)
        MIN = min(predicted_price)
        MAX = max(predicted_price)

        print(f"Predicted Price: {predicted_price} | max: {MAX} | min: {MIN}")

                                                         
        break
        

    

if __name__ == "__main__":        
    main()

