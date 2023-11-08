# prediction_utils.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


def load(model_folder, model_name, input_data, sequence_length):
    model_path = os.path.join(model_folder, f"{model_name}.h5")
    model = load_model(model_path)

    X = []
    for i in range(len(input_data) - sequence_length):
        X.append(input_data[i:i + sequence_length])
    X = np.array(X)

    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data.reshape(-1, 1))
    predicted_values = model.predict(X)
    predicted_values = scaler.inverse_transform(predicted_values)

    return predicted_values


def calculate_dependent_leads(i_data, ii_data):
    iii = ii_data - i_data
    avR = - (i_data + ii_data) / 2
    avL = i_data - (ii_data / 2)
    avF = ii_data - (i_data / 2)

    return iii, avR, avL, avF


def predict_and_save(input_column, sequence_length, target_columns, model_folder, output_csv, file_path=None):
    input_data = pd.read_csv(file_path)[input_column].values
    predictions_df = pd.DataFrame({'Time': range(len(input_data))})

    for target_column in target_columns:
        predicted_values = load(model_folder, target_column, input_data, sequence_length)
    predictions_df[target_column] = predicted_values

    # Calculate additional leads
    i_data = pd.read_csv(file_path)['i'].values
    ii_data = pd.read_csv(file_path)['ii'].values
    iii, avR, avL, avF = calculate_dependent_leads(i_data, ii_data)

    predictions_df['iii'] = iii
    predictions_df['avR'] = avR
    predictions_df['avL'] = avL
    predictions_df['avF'] = avF

    predictions_df.to_csv(output_csv, index=False)
