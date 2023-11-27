# prediction_utils.py
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def load(model_folder, model_name, input_data):
    model_path = os.path.join(model_folder, "{}.h5".format(model_name))
    model = load_model(model_path)

    predicted_values = model.predict(input_data)
    return predicted_values, model_name

def calculate_independent_leads(i_data, ii_data):
    iii = ii_data - i_data
    avR = - (i_data + ii_data) / 2
    avL = i_data - (ii_data / 2)
    avF = ii_data - (i_data / 2)

    return iii, avR, avL, avF

def predict_and_save(file_path, input_column, sequence_length, target_columns, model_folder, output_csv):
    input_data = pd.read_csv(file_path)[input_column].values.reshape(-1, 1)

    # Prepare sequences
    input_sequences = []
    for i in range(len(input_data) - sequence_length):
        input_sequences.append(input_data[i:i + sequence_length])

    input_sequences = np.array(input_sequences)

    predictions_df = pd.DataFrame()
    predictions_df[input_column] = input_data[sequence_length:].flatten()

    for target_column in target_columns:
        predicted_values, target_name = load(model_folder, target_column, input_sequences)
        predictions_df[target_name] = predicted_values.flatten()

    # If additional leads are needed
    if 'i' in target_columns:
        i_data = predictions_df['i'].values
        ii_data = predictions_df[input_column].values

        iii, avR, avL, avF = calculate_independent_leads(i_data, ii_data)

        predictions_df['iii'] = iii
        predictions_df['avr'] = avR
        predictions_df['avl'] = avL
        predictions_df['avf'] = avF

    predictions_df.to_csv(output_csv, index=False)




       
