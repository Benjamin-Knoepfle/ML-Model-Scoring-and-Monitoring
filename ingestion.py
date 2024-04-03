import os
import pandas as pd

import utils


    
# Function for data ingestion
def merge_multiple_dataframe(mode='development'):
    config = utils.load_configuration(mode)
    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    final_data_name = config['final_data_name']
    # check for datasets, compile them together, and write to an output file
    # create a list of csv-files within the input folder
    input_files = [file for file in os.listdir(
        input_folder_path) if file.endswith(".csv")]
    # read and compile together the input_files
    data_set = pd.concat([pd.read_csv(os.path.join(input_folder_path, input_file))
                          for input_file in os.listdir(input_folder_path)
                          if input_file.endswith(".csv")])
    data_set.drop_duplicates(inplace=True)
    # write the data_set to an output file
    data_set.to_csv(
        os.path.join(
            output_folder_path,
            final_data_name),
        index=False)
    # save a record of ingested files
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'a', encoding='utf-8') as fp:
        _ = [fp.write(file + "\n") for file in input_files]


if __name__ == '__main__':
    merge_multiple_dataframe()
