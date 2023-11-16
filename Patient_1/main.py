# main.py
from prediction_utils import predict_and_save

if __name__ == '__main__':
    file_path = r"/content/test_patient2_1000.csv"
    input_column = 'ii'
    target_columns = ['i', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    model_folder = r"/content"
    scaler_folder = r"/content"
    output_csv = 'predictions.csv'


    predict_and_save(file_path, input_column, target_columns, model_folder, scaler_folder, output_csv)
