# data_loading.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data, target_column):
    """Preprocess the data."""
    X = data['ii'].values.reshape(-1, 1)
    y = data[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, f'scaler_{target_column}.pkl')

    input_shape = (X_train.shape[1], 1)

    return X_train, X_test, y_train, y_test, scaler,input_shape
