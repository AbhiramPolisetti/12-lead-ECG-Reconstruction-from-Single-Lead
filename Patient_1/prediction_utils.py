# prediction_utils.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load(model_folder, model_name, input_data, sequence_length):
    model_path = os.path.join(model_folder, "{}.h5".format(model_name))
    model = load_model(model_path)

    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data.reshape(-1, 1))
    
    X = []
    for i in range(len(input_data)):
        X.append(input_data[i:i + sequence_length])

    X = pad_sequences(X, padding='post', maxlen=sequence_length)
    X = np.array(X)

    predicted_values = model.predict(X)
    predicted_values = scaler.inverse_transform(predicted_values)

    return predicted_values


def calculate_independent_leads(i_data, ii_data):
    iii = ii_data - i_data
    avR = - (i_data + ii_data) / 2
    avL = i_data - (ii_data / 2)
    avF = ii_data - (i_data / 2)

    return iii, avR, avL, avF


def predict_and_save(file_path, input_column, sequence_length, target_columns, model_folder, output_csv):
    input_data = pd.read_csv(file_path)[input_column].values

    predictions_df = pd.DataFrame()
    predictions_df['ii'] = pd.read_csv(file_path)['ii'].values

    for target_column in target_columns:
        predicted_values = load(model_folder, target_column, input_data, sequence_length)
        predictions = predicted_values.flatten()

        predictions_df[target_column] = predictions

    i_data = predictions_df['i'].values
    ii_data = predictions_df['ii'].values

    iii, avR, avL, avF = calculate_independent_leads(i_data, ii_data)

    predictions_df['iii'] = iii
    predictions_df['avR'] = avR
    predictions_df['avL'] = avL
    predictions_df['avF'] = avF

    predictions_df.to_csv(output_csv, index=False)

