import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import List, Dict, Optional
from crypto.layers.kernel import ITradeKernel
from .logger import Logger
import asyncio



class OrderRouter:
    """Responsible for sending orders to the exchange."""
    client: Client
    isTest: bool
    logger: Logger
    def __init__(self, client: Client, logger: Logger, isTest: bool = False):
        self.client = client
        self.isTest = isTest
        self.logger = logger
    
    async def send_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> dict:
        """Send an order to the exchange (buy or sell), or simulate the order if isTest=True"""
        if self.isTest:
            # Simulate the order in backtest mode
            self.logger.info(f"Simulated Order - {side} {quantity} {symbol} at price {price if price else 'Market Price'}")
            return {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price if price else 'Market Price',
                'status': 'Simulated'
            }
        
        # If isTest is False, execute the order on the exchange
        try:
            if price:
                # Limit Order
                order = await asyncio.to_thread(self.client.order_limit, symbol=symbol, side=side, quantity=quantity, price=str(price))
            else:
                # Market Order
                order = await asyncio.to_thread(self.client.order_market, symbol=symbol, side=side, quantity=quantity)
            self.logger.info(f"Order placed: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error placing order: {e}")
            return {}

class SlippageManager:
    """Handles slippage adjustments for price."""
    def __init__(self, slippage_percentage: float = 0.1):
        self.slippage_percentage = slippage_percentage
    
    def adjust_for_slippage(self, price: float, side: str) -> float:
        """Adjust price for slippage based on buy/sell side."""
        adjustment = price * (self.slippage_percentage / 100)
        if side == 'BUY':
            return price + adjustment
        elif side == 'SELL':
            return price - adjustment
        return price

class PositionTracker:
    """Tracks open positions and calculates profit and loss."""
    def __init__(self):
        self.positions = {}
        self.trades = []  # Track trades for backtesting
    
    def update_position(self, symbol: str, quantity: float) -> None:
        """Update the position with the current quantity of the symbol."""
        self.positions[symbol] = quantity
    
    def get_position(self, symbol: str) -> float:
        """Get the current position for the symbol."""
        return self.positions.get(symbol, 0.0)
    
    def calculate_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate profit and loss for the given symbol."""
        position = self.get_position(symbol)
        return position * current_price    

    def record_trade(self, symbol: str, side: str, quantity: float, price: float) -> None:
        """Record the trade for backtesting purposes."""
        self.trades.append({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'time': time.time()  # Record the timestamp of the trade
        })

    def get_trades(self) -> List[Dict]:
        """Get all recorded trades for backtesting results."""
        return self.trades

class ErrorHandler:
    """Handles errors and attempts retries."""
    logger: Logger
    def __init__(self, logger: Logger):
        self.retry_limit = 3
        self.logger = logger
    
    async def handle_error(self, error: BinanceAPIException, action: str) -> bool:
        """Handles errors by retrying a failed action."""
        self.logger.critical(f"{error} : {action}")
        return False

class Executor:
    """Main class for trade execution with backtesting option."""
    client:             Client
    isTest:             bool
    order_router:       OrderRouter
    slippage_manager:   SlippageManager
    position_tracker:   PositionTracker
    error_handler:      ErrorHandler
    logger:             Logger
    kernel:             ITradeKernel

    def __init__(self, client: Client, logger: Logger, kernel: ITradeKernel, slippage_percentage: float = 0.1, isTest: bool = False):
        self.logger             = logger
        self.client             = client
        self.isTest             = isTest
        self.order_router       = OrderRouter(client, logger, isTest)
        self.slippage_manager   = SlippageManager(slippage_percentage)
        self.position_tracker   = PositionTracker()
        self.error_handler      = ErrorHandler(logger)
        self.kernel             = kernel
        
    
    async def execute_trade(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> dict:
        """Execute a trade, adjusting for slippage and placing the order."""
        if price:
            adjusted_price = self.slippage_manager.adjust_for_slippage(price, side)
        else:
            adjusted_price = None

        # Send order to the exchange or simulate in test mode
        order = await self.order_router.send_order(symbol, side, quantity, adjusted_price)
        
        if order and order.get("status") != "Simulated":
            # Update position after a successful order
            current_position = self.position_tracker.get_position(symbol)
            if side == 'BUY':
                self.position_tracker.update_position(symbol, current_position + quantity)
            elif side == 'SELL':
                self.position_tracker.update_position(symbol, current_position - quantity)
        elif order:
            # Record the trade if it is a simulated backtest
            self.position_tracker.record_trade(symbol, side, quantity, adjusted_price if adjusted_price else 0)

        return order
    
    async def handle_error(self, error: BinanceAPIException, action: str) -> bool:
        """Handle errors that occur during trade execution."""
        return await self.error_handler.handle_error(error, action)
    
    def get_position(self, symbol: str) -> float:
        """Retrieve the current position for a symbol."""
        return self.position_tracker.get_position(symbol)
    
    def calculate_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate profit or loss for the given symbol."""
        return self.position_tracker.calculate_pnl(symbol, current_price)

    def get_trades(self) -> List[Dict]:
        """Retrieve all trades for backtesting."""
        return self.position_tracker.get_trades()
    

    async def evaluate_and_execute_trades(self) -> None:
        self.logger.error("TODO: evaluate_and_execute_trades")