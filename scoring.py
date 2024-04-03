import pickle
import os
import pandas as pd
from sklearn import metrics
import utils

def read_model(mode='development'):
    # load model
    config = utils.load_configuration(mode)
    model_path = os.path.join(config['output_model_path'])
    model_name = config['model_name']
    model_file = os.path.join(model_path, model_name)
    with open(os.path.join(model_file), 'rb') as fp:
        model = pickle.load(fp)
    return model

# Function for model scoring


def score_model(model=None, test_data=[], write=False, mode='development'):
    # this function should take a trained model, load test data,
    # and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    # read the test data. Can read and concat mulitple csv files to one
    # test_Data set if given
    config = utils.load_configuration(mode)
    test_data_path = os.path.join(config['test_data_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])
    model_path = os.path.join(config['output_model_path'])
    model_name = config['model_name']

    if len(test_data) == 0:
        test_data = os.listdir(test_data_path)
    test_data = pd.concat(
        [
            pd.read_csv(
                os.path.join(
                    test_data_path,
                    input_file),
                index_col='corporation')
            for input_file in test_data if input_file.endswith(".csv")
        ]
    )
    ground_truth = test_data.pop('exited')

    # if no model is given, load the deployed model
    if model is None:
        model = read_model(os.path.join(prod_deployment_path, model_name))

    # create prediction on test_data
    prediction = model.predict(test_data)
    # calculate the f1_score
    f1 = metrics.f1_score(ground_truth, prediction)

    if write:
        # write f1 score to file
        with open(os.path.join(model_path, 'latestscore.txt'), 'w', encoding='utf-8') as fp:
            fp.write(str(f1))

    return f1


if __name__ == '__main__':
    score_model(
        read_model(),
        write=True
    )
