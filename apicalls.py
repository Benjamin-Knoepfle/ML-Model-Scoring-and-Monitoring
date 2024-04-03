import os
from pprint import pprint
import requests
import utils

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8080/"

config = utils.load_configuration('development')
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


# Call each API endpoint and store the responses
response1 = requests.post(f'{URL}prediction?data_path={test_data_path}', timeout=10).text
response2 = requests.get(f'{URL}scoring', timeout=10).text  # put an API call here
response3 = requests.get(f'{URL}summarystats', timeout=10).text
response4 = requests.get(f'{URL}diagnostics', timeout=10).text
response5 = requests.get(f'{URL}outdatedpackages', timeout=10).text

# combine all API responses
responses = {
    'prediction': response1,
    'f1_score': response2,
    'summary_stats': response3,
    'diagnostics': response4,
    'outdated_packages': response5
}

# write the responses to your workspace
with open(os.path.join(output_model_path, 'apireturns.txt'), 'w', encoding='utf-8') as fp:
    pprint(responses, width=70, stream=fp)
