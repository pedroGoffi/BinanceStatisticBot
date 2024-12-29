from dataclasses import dataclass
from typing import List, Dict, Tuple
from crypto.layers.kernel import ITradeKernel, StrategyResult
from crypto.layers.logger import Logger
from .preprocess import ChartData, CryptoCurrency, CryptoCurrencyHistory

class MetricsCollector:
    """
    Records key performance indicators (e.g., Sharpe ratio, win rate).
    """
    def __init__(self):
        self.metrics = {}

    def record_metrics(self, profit: float, trades: List[str]):
        """
        Records metrics for a given backtest.
        :param profit: Total profit from the strategy.
        :param trades: List of trades executed during the backtest.
        """
        self.metrics["profit"] = profit
        self.metrics["trades"] = trades
        self.metrics["win_rate"] = sum(1 for t in trades if "Sold" in t) / len(trades) if trades else 0

    def get_metrics(self) -> Dict:
        """
        Retrieves recorded metrics.
        """
        return self.metrics

@dataclass 
class StrategyResultSubset:
    iteration:                  int 
    interval:                   int 
    subset_strategy_result:     StrategyResult
    
@dataclass
class SimulationResult:
    initial_cash:               float 
    final_cash:                 float 
    win_rate:                   float 
    strategy_result_in_subset:  List[StrategyResultSubset]

class ISimulationFramework:
    kernel: ITradeKernel
    logger: Logger
    name:   str

    @staticmethod
    def __init__(self, name: str, kernel: ITradeKernel, logger: Logger) -> None: ... 

    @staticmethod
    def run_simulation(self, initial_cash: float, buy_threshold: float, sell_threshold: float) -> SimulationResult: ...

class SimulationFramework(ISimulationFramework):
    def __init__(self, name: str, kernel: ITradeKernel, logger: Logger):                        
        self.name               = name 
        self.kernel             = kernel
        self.logger             = logger        

    def run_simulation(self, initial_cash: float, buy_threshold: float, sell_threshold: float) -> SimulationResult:
        """
        Analyzes the price trend for the cryptocurrency in self.kernel.crypto using SMA and RSI indicators.

        Args:
            sma_interval (int): Interval for the SMA calculation.
            rsi_interval (int): Interval for the RSI calculation.
        """

    
        
        sma_interval    = 15
        rsi_interval    = 15
        final_cash      = initial_cash
        win_rate        = 0

        intervals: int = 15
        strategy_result_in_subset: List[StrategyResult] = []
        for i in range(0, len(self.kernel.crypto.history.chart), intervals):
            chart_subset = self.kernel.crypto.history.chart[i:i + intervals]
            subset_strategy_result: StrategyResult = self.kernel.analyze_market(chart_subset=chart_subset)
            strategy_result_in_subset.append(StrategyResultSubset(iteration=i, interval=intervals, subset_strategy_result=subset_strategy_result))

        return SimulationResult(
            initial_cash=initial_cash, 
            final_cash=final_cash, 
            win_rate=win_rate,
            strategy_result_in_subset = strategy_result_in_subset
        )

#NOTE: created this more as a template for especific return that i'll change in the future
@dataclass
class BacktestEngineResult:
    simulators_results: List[SimulationResult]

class BacktestEngine:
    """
    Simulates strategy performance on historical data.
    """
    kernel: ITradeKernel
    simulators: List[ISimulationFramework]
    logger: Logger
    def __init__(self, kernel: ITradeKernel, logger: Logger):
        self.kernel     = kernel
        self.logger     = logger
        self.simulators = []
        

    def atach_simulator(self, simulator: ISimulationFramework) -> None:
        assert issubclass(simulator.__class__, ISimulationFramework), "Expected simulator to extend ISimulationFramework"
        if simulator.kernel.crypto.symbol != self.kernel.crypto.symbol:
            self.logger.critical(f"In BacktestEngine expected self.kernel [{simulator.kernel.crypto.symbol}] be the same simulator.kernel [{self.kernel.crypto.symbol}]")
            return 
        self.simulators.append(simulator)

    def run_simulators(self, initial_cash: float, buy_threshold: float, sell_threshold: float) -> BacktestEngineResult:
        if len(self.simulators) == 0:
            self.logger.warning(f"Expected a simulator to be atached in order to run simulations in BacktestEngine")
            return None


        simulators_results: List[SimulationResult] = []
        for simulator in self.simulators:
            self.logger.debug(f"RUNNING SIMULATOR: {simulator.name}")
            simulator_result: SimulationResult = simulator.run_simulation(initial_cash=initial_cash, buy_threshold=buy_threshold, sell_threshold=sell_threshold)            
            simulators_results.append(simulator_result)

        return BacktestEngineResult(
            simulators_results = simulators_results
        )
            