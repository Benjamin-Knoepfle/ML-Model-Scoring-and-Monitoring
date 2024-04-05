import os
import shutil  # useful module for the task at hand ;)
import utils

# function for deployment


def store_model_into_pickle(mode='development'):
    # copy the latest pickle file, the latestscore.txt value,
    # and the ingestfiles.txt file into the deployment directory
    # copy the latest pickle file
    config = utils.load_configuration(mode)
    data_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])
    model_name = config['model_name']
    deployment_path = os.path.join(config['prod_deployment_path'])
    latest_pickle_file = os.path.join(model_path, model_name)
    deploy_pickle_file = os.path.join(deployment_path, model_name)
    shutil.copyfile(latest_pickle_file, deploy_pickle_file)
    # copy the latestscore.txt
    latest_score_file = os.path.join(model_path, 'latestscore.txt')
    deploy_score_file = os.path.join(deployment_path, 'latestscore.txt')
    shutil.copyfile(latest_score_file, deploy_score_file)
    # copy ingestfiles.txt file
    ingested_file = os.path.join(data_path, 'ingestedfiles.txt')
    deploy_ingested_file = os.path.join(deployment_path, 'ingestedfiles.txt')
    shutil.copyfile(ingested_file, deploy_ingested_file)


if __name__ == '__main__':
    store_model_into_pickle()
    