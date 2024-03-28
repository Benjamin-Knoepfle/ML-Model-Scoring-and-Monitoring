from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics 
import scoring
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():       
    #call the prediction function you created in Step 3
    data_path = request.args.get('data_path')
    data = diagnostics.read_input_data(data_path)
    predictions = diagnostics.model_predictions(data)
    return json.dumps(str(predictions)) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    f1 = scoring.score_model()
    return json.dumps(f1) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarize_stats():        
    #check means, medians, and modes for each column
    summary_statistics = diagnostics.dataframe_summary()
    return summary_statistics #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    #check timing and percent NA values
    na_prcnt = diagnostics.dataframe_missing_values()
    timings = diagnostics.execution_time()
    return json.dumps(str([na_prcnt, timings])) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
