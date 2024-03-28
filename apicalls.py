import requests
import diagnostics
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path']) 


#Call each API endpoint and store the responses
response1 = requests.post(f'{URL}prediction?data_path={test_data_path}').text #put an API call here
response2 = requests.get(f'{URL}scoring').text #put an API call here
response3 = requests.get(f'{URL}summarystats').text
response4 = requests.get(f'{URL}diagnostics').text

#combine all API responses
responses = "\n".join([response1, response2, response3, response4])

#write the responses to your workspace
with open(os.path.join(output_model_path,'apireturns.txt'), 'w') as fp:
    fp.write(responses)


