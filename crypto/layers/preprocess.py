from    multiprocessing     import Pool, cpu_count
from    functools           import partial
from    binance.client      import Client
from    dataclasses         import dataclass, asdict, field
from    typing              import Literal, List, Dict
from    .input              import load_cryptos
from    .logger             import Logger
import  json
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
    chart:  List[ChartData] = field(default_factory=list)
    
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
                        

def get_crypto_history(client: Client, symbol: str, interval: Intervals_ty, logger: Logger) -> CryptoCurrencyHistory | None:    
    klines: List[list] = client.get_historical_klines(symbol=symbol, interval=interval, start_str="24 hours ago UTC", end_str="now UTC", limit=None)
    if not klines:
        logger.error(f"Failed to fetch {symbol} klines!")
        return None
    
    chart: List[ChartData] = [
        ChartData(
            kline[0],
            float(kline[1]),
            float(kline[2]),
            float(kline[3]),
            float(kline[4]),
            float(kline[5]))
            for kline in klines
        ]
        
    return CryptoCurrencyHistory(chart=chart)

def preprocess_crypto_dictionary(client: Client, ticker: dict, interval: Intervals_ty, logger: Logger) -> CryptoCurrency:    
    symbol: str = ticker['symbol']
    crypto_history: CryptoCurrencyHistory = get_crypto_history(client, symbol, interval=interval, logger=logger)
    
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

from concurrent.futures import ThreadPoolExecutor, as_completed

def fast_preprocess_cryptos_dictionary_list(
    client: Client, 
    cryptos: List[Dict[str, str]], 
    interval: Intervals_ty, 
    logger: Logger
) -> List[CryptoCurrency]:
    """
    Processes a list of cryptocurrency tickers by retrieving and processing their historical data 
    from the Binance API using multithreading.

    Args:
        client (Client): The Binance API client used to fetch historical data.
        cryptos (List[Dict[str, str]]): A list of dictionaries, where each dictionary contains 
                                        information about a cryptocurrency ticker (e.g., symbol).
        interval (Intervals_ty): The time interval for historical data, such as "1m", "15m", etc.
        logger (Logger): A logger instance for capturing debug information and status messages.

    Returns:
        List[CryptoCurrency]: A list of `CryptoCurrency` objects, each containing the symbol, 
                               current price, volume, and historical data for the respective cryptocurrency.
    """
    logger.debug("Starting to preprocess cryptocurrencies using multithreading.")
    
    # Initialize an empty list to store results
    results: List[CryptoCurrency] = []
    
    # Use ThreadPoolExecutor to process tickers in parallel
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        # Submit tasks for each ticker and collect futures
        futures = [executor.submit(preprocess_crypto_dictionary, client, ticker, interval, logger) for ticker in cryptos]
        
        # Wait for each future to complete and collect results
        for future in as_completed(futures):
            crypto = future.result()
            if crypto:
                results.append(crypto)
                logger.debug(f"Processed {crypto.symbol} with current price: {crypto.current_price}")
            
    
    logger.debug(f"Finished preprocessing. Processed {len(results)} cryptocurrencies.")
    return results

def teste(client: Client, symbol: str, logger: Logger):
    tickers: List[Dict[str, str]] = load_cryptos(client, query=[symbol])
    cryptos: List[CryptoCurrency] = [preprocess_crypto_dictionary(client=client, ticker=ticker, interval="15m") for ticker in tickers]
    for crypto in cryptos:
        logger.debug(f"Loaded crypto: {crypto.symbol} PRICE: {crypto.current_price}")
    

    
def save_cryptos(cryptos: List[CryptoCurrency]):
    """Save cryptocurrencies to individual files based on their symbols."""
    for crypto in cryptos:
        filename = f"crypto/layers/data/{crypto.symbol}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(crypto), f, indent=4)

def get_saved_cryptos(query: List[str]) -> List[CryptoCurrency]:
    """Retrieve cryptocurrencies from files based on their symbols."""
    results: List[CryptoCurrency] = []
    for symbol in query:
        filename = f"crypto/layers/data/{symbol}.json"
        try:
            with open(filename, "r") as file:
                data = json.load(file)
                crypto = CryptoCurrency(
                    symbol=data['symbol'],
                    name=data['name'],
                    current_price=data['current_price'],
                    volume=data['volume'],
                    history=CryptoCurrencyHistory(
                        chart=[ChartData(**chart_data) for chart_data in data['history']['chart']]
                    )
                )
                results.append(crypto)
        except FileNotFoundError:
            return None
    return results
