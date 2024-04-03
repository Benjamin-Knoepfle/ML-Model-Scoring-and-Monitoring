import os
from sklearn import metrics
import matplotlib.pyplot as plt
import utils
import diagnostics


##############Function for reporting
def score_model(mode='development'):
    ###############Load config.json and get path variables
    config = utils.load_configuration(mode)
    test_data_path = os.path.join(config['test_data_path'])
    output_model_path = os.path.join(config['output_model_path'])
    #calculate a confusion matrix using the test data and the deployed model
    test_data = diagnostics.read_input_data(test_data_path)
    test_data['prediction'] = diagnostics.model_predictions(test_data)
    confusion_matrix = metrics.confusion_matrix(test_data['exited'],test_data['prediction'])
    #write the confusion matrix to the workspace
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = confusion_matrix,
        display_labels = [False, True]
        )
    cm_display.plot()
    with open(os.path.join(output_model_path, 'confusionmatrix.png'), 'wb') as fp:
        plt.savefig(fp)


if __name__ == '__main__':
    score_model()
