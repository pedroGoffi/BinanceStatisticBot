from abc import ABC
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Tuple
from crypto.layers.kernel import ITradeKernel, StrategyResult
from crypto.layers.logger import Logger
from .preprocess import ChartData, CryptoCurrency, CryptoCurrencyHistory


MetricActionTy = Literal["BUY", "SELL"] 
@dataclass
class Metric:    
    action:     MetricActionTy
    symbol:     str
    price:      float
    pnl:        float                   = field(default_factory=float)

@dataclass
class MetricAllocator:
    history: List[Metric]               = field(default_factory=list)
    metrics: Dict[str, List[Metric]]    = field(default_factory=dict)

    def get_last_metric_of_action(self, symbol: str, action: MetricActionTy) -> Metric | None:
        for metric in reversed(self.history):
            if metric.symbol == symbol and metric.action == action:
                return metric
        
        return None

    def buy_symbol(self, symbol: str, price: float):
        if self.metrics.get(symbol) is None:
            self.metrics[symbol] = []

        metric: Metric = Metric("BUY", symbol, price)
        self.metrics[symbol].append(metric)
        return metric

    def sell_symbol(self, symbol: str, price: float) -> Metric | None:
        buy_metric: Metric = self.get_last_metric_of_action(symbol=symbol, action="BUY")
        if buy_metric is None:
            # TODO: debug error 
            print("error: buy_metric is None")
            return None 

        bought_price: float = buy_metric.price
        pnl: float = price / bought_price
        return Metric("SELL", symbol, price, pnl)

    def action(self, symbol: str, action: MetricActionTy, price: float) -> Metric:
        if action == "SELL":
            return self.sell_symbol(symbol=symbol, price=price)
        elif action == "BUY":
            return self.buy_symbol(symbol=symbol, price=price)

class MetricsCollector:
    allocator: MetricAllocator
    
    def __init__(self):
        self.allocator = MetricAllocator()

    def record_metrics(self, action: MetricActionTy, symbol: str, quantity: float):
        metric: Metric = self.allocator.action(action, symbol, quantity)
        print(f"{metric = :!r}")

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

class ISimulationFramework(ABC):
    kernel: ITradeKernel
    logger: Logger
    name:   str
    metrics: MetricsCollector

    @staticmethod
    def __init__(self, name: str, kernel: ITradeKernel, logger: Logger) -> None: ... 

    @staticmethod
    def run_simulation(self, initial_cash: float, buy_threshold: float, sell_threshold: float, sub_sets: int) -> SimulationResult: ...

class SimulationFramework(ISimulationFramework):
    def __init__(self, name: str, kernel: ITradeKernel, logger: Logger):                        
        self.name               = name 
        self.kernel             = kernel        
        self.logger             = logger        
        self.metrics            = MetricsCollector()

    def run_simulation(self, initial_cash: float, buy_threshold: float, sell_threshold: float, sub_sets: int) -> SimulationResult:                    
        final_cash      = initial_cash
        wins:   int = 0
        losses: int = 0

        intervals: int = len(self.kernel.crypto.history.chart) // sub_sets
        strategy_result_in_subset: List[StrategyResult] = []
        for i in range(0, len(self.kernel.crypto.history.chart), intervals):
            chart_subset = self.kernel.crypto.history.chart[i:i + intervals]
            subset_strategy_result: StrategyResult = self.kernel.analyze_market(chart_subset=chart_subset)

            if subset_strategy_result.buy_signals:
                # TODO: backtest a bough based on final_cash
                pass 

            if subset_strategy_result.sell_signals:
                # TODO: backtest a sell action and calculate pnl
                pass 


            strategy_result_in_subset.append(StrategyResultSubset(iteration=i, interval=intervals, subset_strategy_result=subset_strategy_result))

        all_trades: int = wins + losses
        win_rate: float = wins / all_trades if all_trades != 0 else 0.0


        return SimulationResult(
            initial_cash                    = initial_cash, 
            final_cash                      = final_cash, 
            win_rate                        = win_rate,
            strategy_result_in_subset       = strategy_result_in_subset
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

    def run_simulators(self, initial_cash: float, buy_threshold: float, sell_threshold: float, sub_sets: int) -> BacktestEngineResult:
        if len(self.simulators) == 0:
            self.logger.warning(f"Expected a simulator to be atached in order to run simulations in BacktestEngine")
            return None


        simulators_results: List[SimulationResult] = []
        for simulator in self.simulators:
            self.logger.debug(f"RUNNING SIMULATOR: {simulator.name}")
            simulator_result: SimulationResult = simulator.run_simulation(initial_cash=initial_cash, buy_threshold=buy_threshold, sell_threshold=sell_threshold, sub_sets=sub_sets)
            simulators_results.append(simulator_result)

        return BacktestEngineResult(
            simulators_results = simulators_results
        )
            