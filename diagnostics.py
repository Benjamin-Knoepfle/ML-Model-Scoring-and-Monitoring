
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle

import subprocess
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_name = config['model_name']

def read_input_data(data_path:str):
    input_data = pd.concat(
        [pd.read_csv(os.path.join(data_path,input_file), index_col='corporation') for input_file in os.listdir(data_path) if input_file.endswith(".csv")]
    )
    return input_data

##################Function to get model predictions
def model_predictions(input_data: pd.DataFrame):
    #read the deployed model and a test dataset, calculate predictions
    #read deployed model
    with open(os.path.join(prod_deployment_path, model_name), 'rb') as fp:
        model = pickle.load(fp)
    #drop target column if in data
    if 'exited' in input_data.columns:
        input_data = input_data.drop(['exited'], axis=1)
    #predict the test dataset
    prediction = model.predict(input_data)
     
    return list(prediction)#return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    data = read_input_data(test_data_path)
    #calculate summary statistics here
    summary_statistics = data.describe().loc[['mean','std','50%']]
    summary_statistics.index = ['mean', 'std', 'median']
    summary_list = [ {col:summary_statistics[col].to_dict()}  for col in summary_statistics.columns]
    return summary_list #return value should be a list containing all summary statistics

###################Function to get percentage of missing data
def dataframe_missing_values():
    data = read_input_data(test_data_path)
    #For ease of use I will leverage the count values of the describe method because it counts the non-null values ;)
    samples = len(data)
    non_null_counts = data.describe().loc[['count']]
    missing_percentages = (1- non_null_counts/samples) *100
    missing_list = [ {col:missing_percentages[col].values[0]}  for col in missing_percentages.columns]
    return missing_list
        
    
    
##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    import ingestion
    ingestion_duration = timeit.timeit(ingestion.merge_multiple_dataframe, number=1)
    
    import training
    training_duration = timeit.timeit(training.train_model, number=1)

    return [ingestion_duration,training_duration] #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    proc = subprocess.Popen(['python','-m', 'pip', 'list','-o'],stdout=subprocess.PIPE, text=True)
    packages = proc.stdout.read()
    return packages.replace('\n', ' ')
    
    
    
    
    
if __name__ == '__main__':
    #model_predictions(read_input_data(test_data_path))
    #dataframe_summary()
    #execution_time()
    print(outdated_packages_list())





    
