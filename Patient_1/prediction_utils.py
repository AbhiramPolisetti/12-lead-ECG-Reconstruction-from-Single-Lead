# prediction_utils.py
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import time
def load(model_folder, scaler_folder, model_name, input_data, target_column):
    model_path = os.path.join(model_folder, "{}.h5".format(model_name))
    model = load_model(model_path)

    scaler_path = os.path.join(scaler_folder, "{}.pkl".format(model_name))
    scaler = joblib.load(scaler_path)

    # Reshape input_data to have two dimensions
    input_data = input_data.reshape(-1, 1)
    scaled_data = scaler.transform(input_data)

    predicted_values = model.predict(scaled_data)
    predicted_values = scaler.inverse_transform(predicted_values)

    return predicted_values, target_column

def calculate_independent_leads(i_data, ii_data):
    iii = ii_data - i_data
    avR = - (i_data + ii_data) / 2
    avL = i_data - (ii_data / 2)
    avF = ii_data - (i_data / 2)

    return iii, avR, avL, avF

def predict_and_save(file_path, input_column, target_columns, model_folder, scaler_folder, output_csv):
    input_data = pd.read_csv(file_path)[input_column].values

    predictions_df = pd.DataFrame()
    predictions_df[input_column] = input_data

    for target_column in target_columns:
        predicted_values, target_name = load(model_folder, scaler_folder, target_column, input_data, target_column)
        predictions_df[target_name] = predicted_values

    if 'i' in target_columns:
        i_data = predictions_df['i'].values
        ii_data = predictions_df[input_column].values

        iii, avR, avL, avF = calculate_independent_leads(i_data, ii_data)

        predictions_df['iii'] = iii
        predictions_df['avr'] = avR
        predictions_df['avl'] = avL
        predictions_df['avf'] = avF

    column_order = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    predictions_df = predictions_df[column_order]

    predictions_df.to_csv(output_csv, index=False)




if __name__ == '__main__':
    file_path = r"/content/test_patient2_1000.csv"
    input_column = 'ii'
    target_columns = ['i', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    model_folder = r"/content"
    scaler_folder = r"/content"  # Adjust this path
    output_csv = 'predictions.csv'


    predict_and_save(file_path, input_column, target_columns, model_folder, scaler_folder, output_csv)

