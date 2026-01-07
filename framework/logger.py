import logging
import sys

# ANSI Escape Codes for Colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors based on the log level and logger name.
    """
    def __init__(self, fmt):
        super().__init__(fmt)
        
    def format(self, record):
        # Colorize the Logger Name to distinguish components
        if record.name == "Worker":
            name_color = Colors.CYAN
        elif record.name == "Framework":
            name_color = Colors.MAGENTA
        elif "Agent" in record.name:
            name_color = Colors.BLUE
        else:
            name_color = Colors.WHITE
            
        record.name = f"{name_color}{Colors.BOLD}{record.name}{Colors.RESET}"
        
        # Colorize Level Name (INFO, ERROR, etc.)
        if record.levelno == logging.INFO:
            record.levelname = f"{Colors.GREEN}{record.levelname}{Colors.RESET}"
        elif record.levelno == logging.WARNING:
            record.levelname = f"{Colors.YELLOW}{record.levelname}{Colors.RESET}"
        elif record.levelno == logging.ERROR:
            record.levelname = f"{Colors.RED}{Colors.BOLD}{record.levelname}{Colors.RESET}"
        elif record.levelno == logging.DEBUG:
            record.levelname = f"{Colors.WHITE}{record.levelname}{Colors.RESET}"
            
        return super().format(record)

def get_logger(name, level=logging.INFO):
    """
    Returns a configured logger with the specified name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Format: [Time] [LoggerName] [Level] Message
        # The color codes are injected by our custom formatter
        formatter = ColoredFormatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        )
        formatter.datefmt = "%H:%M:%S"
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid double printing
        logger.propagate = False
        
    return logger