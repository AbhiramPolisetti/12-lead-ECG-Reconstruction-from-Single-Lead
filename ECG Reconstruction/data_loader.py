# data_loader.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_data(file_path, input_column, target_column):
    data = pd.read_csv(file_path)
    return data[input_column].values, data[target_column].values


def preprocess_data(input_data, target_data, sequence_length):
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data.reshape(-1, 1))
    target_data = scaler.fit_transform(target_data.reshape(-1, 1))

    X, y = [], []
    for i in range(len(input_data) - sequence_length):
        X.append(input_data[i:i + sequence_length])
        y.append(target_data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler
