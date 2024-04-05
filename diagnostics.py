import timeit
import subprocess
import functools
import pandas as pd
import utils
import ingestion
import training


# Function to get model predictions


def model_predictions(input_data: pd.DataFrame, mode='development'):
    # read the deployed model and a test dataset, calculate predictions
    # read deployed model
    model = utils.read_model(mode)
    # drop target column if in data
    if 'exited' in input_data.columns:
        input_data = input_data.drop(['exited'], axis=1)
    # predict the test dataset
    prediction = model.predict(input_data)

    # return value should be a list containing all predictions
    return list(prediction)

# Function to get summary statistics


def dataframe_summary(mode='development'):
    data = utils.read_input_data(mode)
    # calculate summary statistics here
    summary_statistics = data.describe().loc[['mean', 'std', '50%']]
    summary_statistics.index = ['mean', 'std', 'median']
    summary_list = [{col: summary_statistics[col].to_dict()}
                    for col in summary_statistics.columns]
    return summary_list  # return value should be a list containing all summary statistics

# Function to get percentage of missing data


def dataframe_missing_values(mode='development'):
    data = utils.read_input_data(mode)
    # For ease of use I will leverage the count values of the describe method
    # because it counts the non-null values ;)
    samples = len(data)
    non_null_counts = data.describe().loc[['count']]
    missing_percentages = (1 - non_null_counts / samples) * 100
    missing_list = [{col: missing_percentages[col].values[0]}
                    for col in missing_percentages.columns]
    return missing_list


# Function to get timings
def execution_time(mode='development'):
    # calculate timing of training.py and ingestion.py
    ingestion_duration = timeit.timeit(
        functools.partial(
            ingestion.merge_multiple_dataframe,
            mode
            ),
        number=1
        )

    training_duration = timeit.timeit(
        functools.partial(
            training.train_model,
            mode
            ),
        number=1
        )

    # return a list of 2 timing values in seconds
    return [ingestion_duration, training_duration]

# Function to check dependencies


def outdated_packages_list():
    # get a list of
    with subprocess.Popen(
        ['python', '-m', 'pip', 'list', '-o'],
        stdout=subprocess.PIPE, text=True
        ) as proc:
        packages = proc.stdout.read()
    return packages.replace('\n', ' ')


if __name__ == '__main__':
    model_predictions(utils.read_input_data('development'))
    dataframe_summary()
    execution_time()
    print(outdated_packages_list())
