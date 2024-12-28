import logging
from colorama import Fore, Back, Style

class Logger:
    def __init__(self, name: str = 'default_logger', level: int = logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.hasHandlers():
            self._add_handler()

    def _add_handler(self):
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_formatter())
        self.logger.addHandler(handler)

    def _get_formatter(self):
        levels = {
            logging.DEBUG:    (Fore.BLUE,    Back.BLACK, '[DEBUG]'),
            logging.INFO:     (Fore.GREEN,   Back.BLACK, '[INFO]'),
            logging.WARNING:  (Fore.YELLOW,  Back.BLACK, '[WARNING]'),
            logging.ERROR:    (Fore.RED,     Back.BLACK, '[ERROR]'),
            logging.CRITICAL: (Fore.MAGENTA, Back.BLACK, '[CRITICAL]')
        }

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                color, bg_color, label = levels.get(record.levelno, (Fore.WHITE, Back.BLACK, '[LOG]'))
                formatter = logging.Formatter(f'%(asctime)s - {color}{bg_color}{Style.BRIGHT}{label} - %(message)s{Style.RESET_ALL}')
                return formatter.format(record)

        return CustomFormatter()

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)