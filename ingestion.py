import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
final_data_name = config['final_data_name']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    #create a list of csv-files within the input folder
    input_files = [file for file in os.listdir(input_folder_path) if file.endswith(".csv")]
    #read and compile together the input_files
    data_set = pd.concat(
        [pd.read_csv(os.path.join(input_folder_path,input_file)) for input_file in os.listdir(input_folder_path) if input_file.endswith(".csv")]
    )
    data_set.drop_duplicates(inplace=True)
    #write the data_set to an output file
    data_set.to_csv(os.path.join(output_folder_path,final_data_name), index=False)
    #save a record of ingested files
    with open(os.path.join(output_folder_path,'ingestedfiles.txt'), 'a') as fp:
        _= [fp.write(file+"\n") for file in input_files]

if __name__ == '__main__':
    merge_multiple_dataframe()
