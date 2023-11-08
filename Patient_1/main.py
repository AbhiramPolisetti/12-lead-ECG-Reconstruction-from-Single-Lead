# main.py
from prediction_utils import predict_and_save

if __name__ == '__main__':
    file_path = 's0001_re.csv'
    input_column = 'ii'
    sequence_length = 1000
    target_columns = ['i', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    model_folder = 'Models'
    output_csv = 'predictions.csv'

    predict_and_save(file_path, input_column, sequence_length, target_columns, model_folder, output_csv)
