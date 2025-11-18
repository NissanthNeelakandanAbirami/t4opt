import logging
import sys
from typing import Optional
from datetime import datetime
import os


class Logger:
    """Simple logger for T4-OPT."""
    
    def __init__(self, name: str = "T4-OPT", log_file: Optional[str] = None, level: str = "INFO"):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
        """
        self.name = name
        self.log_file = log_file
        

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        self.logger.handlers = []
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def log(self, message: str, level: str = "INFO"):
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)

