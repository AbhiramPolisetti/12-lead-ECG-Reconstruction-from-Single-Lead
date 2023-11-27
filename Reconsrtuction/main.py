#main.py
from prediction_utils import predict_and_save

if __name__ == '__main__':
    file_path = r"C:\Users\iabhi\Desktop\test.csv"
    input_column = 'ii'
    target_columns = ['i', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    model_folder = r"C:\Users\iabhi\Desktop\raw\models"
    output_csv = r"C:\Users\iabhi\Desktop\predictions.csv"
    sequence_length = 1000  

    predict_and_save(file_path, input_column, sequence_length, target_columns, model_folder, output_csv)
