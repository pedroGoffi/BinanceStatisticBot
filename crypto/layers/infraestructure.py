import os
import logging
import sqlite3
import psycopg2
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from pika import BlockingConnection, ConnectionParameters, PlainCredentials, Channel

# Abstract Database Manager class
class AbstractDatabaseManager(ABC):
    @abstractmethod
    def store_trade_data(self, trade_id: str, data: str) -> None:
        pass

    @abstractmethod
    def retrieve_trade_data(self, trade_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def close_connection(self) -> None:
        pass

# SQLite Database Manager
class SQLiteDatabaseManager(AbstractDatabaseManager):
    def __init__(self, db_file: str, logger: logging.Logger) -> None:
        self.db_file = db_file
        self.logger = logger
        self.conn = self._connect_to_db()
        self.cursor = self.conn.cursor()

    def _connect_to_db(self) -> sqlite3.Connection:
        """Connect to SQLite database."""
        if not os.path.exists(self.db_file):
            self.logger.info(f"Database file {self.db_file} not found. Creating new database.")
            conn = sqlite3.connect(self.db_file)
            self._initialize_db(conn)
        else:
            self.logger.info(f"Connecting to database: {self.db_file}.")
            conn = sqlite3.connect(self.db_file)
        return conn

    def _initialize_db(self, conn: sqlite3.Connection) -> None:
        """Initialize SQLite database."""
        self.logger.info("Initializing the database with tables.")
        conn.execute('''CREATE TABLE IF NOT EXISTS trade_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            trade_id TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            data TEXT
                          )''')
        self.logger.info("SQLite Database initialized.")

    def store_trade_data(self, trade_id: str, data: str) -> None:
        """Store trade logs in SQLite database."""
        self.cursor.execute("INSERT INTO trade_logs (trade_id, data) VALUES (?, ?)", (trade_id, data))
        self.conn.commit()
        self.logger.info(f"Stored trade data for {trade_id}.")

    def retrieve_trade_data(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trade data from SQLite database."""
        self.cursor.execute("SELECT * FROM trade_logs WHERE trade_id = ?", (trade_id,))
        row = self.cursor.fetchone()
        if row:
            self.logger.info(f"Retrieved data for {trade_id}.")
            return {"trade_id": row[1], "timestamp": row[2], "data": row[3]}
        return None

    def close_connection(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()
        self.logger.info("SQLite connection closed.")

# PostgreSQL Database Manager
class PostgreSQLDatabaseManager(AbstractDatabaseManager):
    def __init__(self, host: str, dbname: str, user: str, password: str, logger: logging.Logger) -> None:
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.logger = logger
        self.conn = self._connect_to_db()
        self.cursor = self.conn.cursor()

    def _connect_to_db(self) -> psycopg2.extensions.connection:
        """Connect to PostgreSQL database."""
        self.logger.info(f"Connecting to PostgreSQL database: {self.dbname}.")
        conn = psycopg2.connect(
            host=self.host,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )
        self._initialize_db(conn)
        return conn

    def _initialize_db(self, conn: psycopg2.extensions.connection) -> None:
        """Initialize PostgreSQL database."""
        self.logger.info("Initializing the database with tables.")
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS trade_logs (
                                id SERIAL PRIMARY KEY,
                                trade_id VARCHAR(255),
                                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                data TEXT
                              )''')
        conn.commit()
        self.logger.info("PostgreSQL Database initialized.")

    def store_trade_data(self, trade_id: str, data: str) -> None:
        """Store trade logs in PostgreSQL database."""
        self.cursor.execute("INSERT INTO trade_logs (trade_id, data) VALUES (%s, %s)", (trade_id, data))
        self.conn.commit()
        self.logger.info(f"Stored trade data for {trade_id}.")

    def retrieve_trade_data(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trade data from PostgreSQL database."""
        self.cursor.execute("SELECT * FROM trade_logs WHERE trade_id = %s", (trade_id,))
        row = self.cursor.fetchone()
        if row:
            self.logger.info(f"Retrieved data for {trade_id}.")
            return {"trade_id": row[1], "timestamp": row[2], "data": row[3]}
        return None

    def close_connection(self) -> None:
        """Close the PostgreSQL connection."""
        self.conn.close()
        self.logger.info("PostgreSQL connection closed.")

# Message Queue (RabbitMQ) Manager
class MessageQueueManager:
    def __init__(self, host: str, queue_name: str, logger: logging.Logger) -> None:
        self.host = host
        self.queue_name = queue_name
        self.logger = logger
        self.connection = self._connect_to_queue()
        self.channel = self.connection.channel()

    def _connect_to_queue(self) -> BlockingConnection:
        """Connect to the message queue."""
        credentials = PlainCredentials('guest', 'guest')
        parameters = ConnectionParameters(self.host, credentials=credentials)
        self.logger.info(f"Connecting to message queue: {self.host}.")
        return BlockingConnection(parameters)

    def send_message(self, message: str) -> None:
        """Send a message to the queue."""
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=message,
            properties=self._get_message_properties()
        )
        self.logger.info(f"Sent message: {message}")

    def _get_message_properties(self) -> Dict[str, Any]:
        """Return message properties."""
        return {
            'delivery_mode': 2,  # Make message persistent
        }

    def close_connection(self) -> None:
        """Close the message queue connection."""
        self.connection.close()
        self.logger.info("Message queue connection closed.")

# Infrastructure Layer
class InfrastructureLayer:
    """Handles the entire infrastructure, including database management and message queueing."""
    
    def __init__(self, db_manager: AbstractDatabaseManager, mq_manager: MessageQueueManager, logger: logging.Logger) -> None:
        self.db_manager = db_manager
        self.mq_manager = mq_manager
        self.logger = logger

    def store_trade_data(self, trade_id: str, data: str) -> None:
        """Store trade data in the database."""
        self.db_manager.store_trade_data(trade_id, data)

    def retrieve_trade_data(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trade data from the database."""
        return self.db_manager.retrieve_trade_data(trade_id)

    def send_trade_message(self, message: str) -> None:
        """Send a trade-related message to the message queue."""
        self.mq_manager.send_message(message)

# Example usage:
# def main() -> None:
#     # Initialize logger
#     logger = logging.getLogger("InfrastructureLayer")
#     logger.setLevel(logging.INFO)
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     logger.addHandler(console_handler)
# 
#     # Initialize components
#     db_manager = SQLiteDatabaseManager(db_file="trade_data.db", logger=logger)
#     mq_manager = MessageQueueManager(host="localhost", queue_name="trade_queue", logger=logger)
# 
#     # Initialize infrastructure layer
#     infrastructure_layer = InfrastructureLayer(db_manager, mq_manager, logger)
# 
#     # Example: Storing trade data
#     trade_id = "T12345"
#     trade_data = '{"symbol": "BTCUSD", "action": "buy", "price": 50000, "quantity": 1}'
#     infrastructure_layer.store_trade_data(trade_id, trade_data)
# 
#     # Example: Sending message to the queue
#     infrastructure_layer.send_trade_message("Trade executed: BTCUSD, buy, price: 50000")
# 
#     # Example: Retrieving trade data
#     retrieved_data = infrastructure_layer.retrieve_trade_data(trade_id)
#     if retrieved_data:
#         print(f"Retrieved Data: {retrieved_data}")
# 
# if __name__ == "__main__":
#     main()
