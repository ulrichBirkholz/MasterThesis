import json

def get_config():
	# Load the configuration file
	with open('config.json') as config_file:
		config = json.load(config_file)
	return config