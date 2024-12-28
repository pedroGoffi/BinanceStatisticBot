from    binance.client  import Client
from    dataclasses     import dataclass
from    typing          import Literal, List, Dict
from    .input           import load_cryptos
from    .logger          import Logger
import  pandas          as pd

Intervals_ty =  Literal["1m", "3m", "5m", "15m", "30m", "1h", "6h", "12h", "1d", "3d"] 

@dataclass 
class ChartData:
    timestamp:  float
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float

@dataclass
class CryptoCurrencyHistory:
    chart:  List[ChartData]
    


@dataclass
class CryptoCurrency:
    symbol:         str
    name:           str
    current_price:  float
    volume:         float    
    history:        CryptoCurrencyHistory
    
    def update(self) -> None:    
        self.crypto_history = get_crypto_history(self.symbol)
        if self.crypto_history:        
            # TODO: fix here 
            self.volume                     = 0.0 # calculate_volume(self.crypto_history.history)                        
            current_price                   = self.crypto_history.history[-1]['close']
            self.current_price              = current_price                 
                        

def get_crypto_history(client: Client, symbol: str, interval: Intervals_ty = '15m') -> CryptoCurrencyHistory | None:    
    klines: List[list] = client.get_historical_klines(symbol=symbol, interval=interval, start_str="24 hours ago UTC", end_str="now UTC", limit=None)
    if not klines:
        return None
    
    chart: List[ChartData] = []    
    for i in range(len(klines)):        
        chart.append(
            ChartData(
                klines[i][0],
                float(klines[i][1]),
                float(klines[i][2]),
                float(klines[i][3]),
                float(klines[i][4]),
                float(klines[i][5])
            ))
        
    return CryptoCurrencyHistory(chart=chart)



#def calculate_volume(data: List[dict]) -> float:
#    """
#    Calculate the total volume from a list of dictionaries containing volume data.
#
#    Args:
#        data (List[dict]): A list of dictionaries where each dictionary contains a 'volume' key.
#
#    Returns:
#        float: The total volume calculated by summing the 'volume' values.
#    """
#    df: pd.DataFrame = pd.DataFrame(data)
#    df['volume'] = df['volume'].astype(float)
#    total_volume = df['volume'].sum()
#    return total_volume


def preprocess_crypto_dictionary(client: Client, ticker: dict, interval: Intervals_ty) -> CryptoCurrency:    
    symbol: str = ticker['symbol']
    crypto_history: CryptoCurrencyHistory = get_crypto_history(client, symbol, interval=interval)
    if crypto_history:
        # TODO: fix volume calculation
        volume = 0.0 # calculate_volume(crypto_history.history)
        
        current_price = crypto_history.chart[-1].close
        crypto: CryptoCurrency = CryptoCurrency(
            symbol=symbol,
            name=symbol[:-4],
            current_price=current_price,
            volume=volume,                
            history=crypto_history
        )
        
        return crypto
    return None



def teste(client: Client, symbol: str, logger: Logger):
    tickers: List[Dict[str, str]] = load_cryptos(client, query=[symbol])
    cryptos: List[CryptoCurrency] = [preprocess_crypto_dictionary(client=client, ticker=ticker, interval="15m") for ticker in tickers]
    for crypto in cryptos:
        logger.debug(f"Loaded crypto: {crypto.symbol} PRICE: {crypto.current_price}")
    

    
