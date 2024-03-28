import os
import json

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
input_folder_path = os.path.join(config['input_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_name = config['model_name']

def run():
    ##################Check and read new data
    #first, read ingestedfiles.txt
    with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as fp:
        ingested = fp.read().split("\n")

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    new_data = [data for data in os.listdir(input_folder_path) if data not in ingested and data.endswith(".csv")] 


    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if len(new_data) == 0:
        return

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as fp:
        f1_model = float(fp.read())
        
    f1_new = scoring.score_model(test_data=new_data)

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if f1_new >= f1_model:
        return

    ingestion.merge_multiple_dataframe()
    training.train_model()
    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    deployment.store_model_into_pickle()
    
    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    reporting.score_model()
    #run diagnostics using the apicalls ;) do not forget to run the flask server first by starting app.py in another terminal
    import apicalls







