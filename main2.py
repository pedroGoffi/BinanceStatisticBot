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



from binance.client         import Client
from dotenv                 import load_dotenv
from crypto.layers.kernel import ITradeKernel, StrategyOne
from crypto.layers.logger   import Logger
from crypto.layers          import preprocess
from crypto.layers.executor import Executor
from crypto.layers.backtest import SimulationFramework
from typing import List, Dict
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
INTERVAL_TO_WATCH:      str     = os.getenv("INTERVAL_TO_WATCH")            or '15m'
WATCH_VAR_MAX_TARGET:   float   = float(os.getenv("WATCH_VAR_MAX_TARGET"))  or 20.0
API_KEY:                str     = os.getenv("API_KEY")
API_SECRET:             str     = os.getenv("API_SECRET")

logger: Logger = Logger()
client: Client = Client(API_KEY, API_SECRET)



#from crypto.layers import 
def main():
    
    #executor: Executor = Executor(client=client, slippage_percentage=0.0, isTest=True)
    cryptos: List[Dict[str, str]]  = preprocess.load_cryptos(client=client, query=["BTCUSDT"])
    assert len(cryptos) >= 1

    crypto: CryptoCurrency = preprocess.preprocess_crypto_dictionary(client, cryptos[0], INTERVAL_TO_WATCH)

    kernel: ITradeKernel = StrategyOne(logger, crypto)
    simulator: SimulationFramework = SimulationFramework(kernel, logger)
    simulator.run_simulation(        
        buy_threshold   = 0.05,
        sell_threshold  = 0.10
    )

    

if __name__ == "__main__":        
    main()