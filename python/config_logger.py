import logging

def config_logger(level: int, filename="execution.log"):
	""" Configures the logger with a specified level and log file name

    Provides a unified configuration for the logging module. Sets up both a console 
    and a file handler for logging, both of which use the same specified logging level 
    and format. The log messages will simultaneously be printed to the console and saved 
    to a log file

    Args:
        level (int): Logging level, e.g., logging.DEBUG, logging.INFO, etc
        filename (str, optional): Name of the log file to which logs will be saved. Defaults to "execution.log"
    """
	logger = logging.getLogger()
	logger.setLevel(level)

	console_handler = logging.StreamHandler()
	file_handler = logging.FileHandler(filename, mode='w')

	console_handler.setLevel(level)
	file_handler.setLevel(level)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	console_handler.setFormatter(formatter)
	file_handler.setFormatter(formatter)

	logger.addHandler(console_handler)
	logger.addHandler(file_handler)
