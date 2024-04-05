import os
import json
import pickle
import pandas as pd

def load_configuration(mode:str):
    # Load config.json and get input and output paths
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config[mode]

def read_input_data(mode:str='development'):
    config = load_configuration(mode)
    data_path = os.path.join(config['output_folder_path'])
    input_data = pd.concat(
        [pd.read_csv(os.path.join(data_path, input_file), index_col='corporation')
         for input_file in os.listdir(data_path) if input_file.endswith(".csv")
         ]
    )
    return input_data

def read_model(mode='development'):
    # load model
    config = load_configuration(mode)
    model_path = os.path.join(config['output_model_path'])
    model_name = config['model_name']
    model_file = os.path.join(model_path, model_name)
    with open(os.path.join(model_file), 'rb') as fp:
        model = pickle.load(fp)
    return model