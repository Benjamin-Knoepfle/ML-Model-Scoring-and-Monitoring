import pickle
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import utils



# Function for training the model
def train_model(mode='development'):
    config = utils.load_configuration(mode)
    input_folder_path = config['output_folder_path']
    input_name = config['final_data_name']
    output_folder_path = config['output_model_path']
    model_name = config['model_name']

    # read in training data
    train_data = pd.read_csv(
        os.path.join(
            input_folder_path,
            input_name),
        index_col='corporation')
    target = train_data.pop('exited')

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # fit the logistic regression to your data
    model.fit(train_data, target)

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    with open(os.path.join(output_folder_path, model_name), 'wb') as fp:
        pickle.dump(model, fp)

    return model


if __name__ == '__main__':
    train_model()
