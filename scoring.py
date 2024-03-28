from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
final_data_name = config['final_data_name']
model_path = os.path.join(config['output_model_path']) 
model_name = config['model_name']


#################Function for model scoring
def score_model(test_data=[]):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    #read the test data. Can read and concat mulitple csv files to one test_Data set if given 
    
    if len(test_data)==0:
        test_data = os.listdir(test_data_path)
    test_data = pd.concat(
        [pd.read_csv(os.path.join(test_data_path,input_file), index_col='corporation') for input_file in test_data if input_file.endswith(".csv")]
    )
    ground_truth = test_data.pop('exited')

    #load model
    with open(os.path.join(model_path, model_name), 'rb') as fp:
        model = pickle.load(fp)        
    #create prediction on test_data
    prediction = model.predict(test_data)
    #calculate the f1_score
    f1 = metrics.f1_score(ground_truth, prediction)
    #write f1 score to file
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as fp:
        fp.write(str(f1))
        
    return f1

if __name__ == '__main__':
    score_model()