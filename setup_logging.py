# setup_logging.py
import logging
import os
from datetime import datetime

def setup_logs(log_directory='logs'):
    """Sets up timestamped file and console logging."""
    
    # Create the log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)
    
    # Generate a timestamped log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_directory, f"bot_log_{timestamp}.log")
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the lowest level to capture all messages
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Create a file handler to write to the log file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO) # Log INFO and higher to the file
    
    # Create a console handler to print to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Log INFO and higher to the console
    
    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("Logging configured. Output will be saved to %s", log_filename)