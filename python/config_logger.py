import logging

def config_logger(level: int, filename="execution.log"):
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
