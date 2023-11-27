import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense, Dropout


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(lead_ii_data, lead_i_data, sequence_length):
    X, y = [], []

    for i in range(len(lead_ii_data) - sequence_length):
        X.append(lead_ii_data[i:i + sequence_length])
        y.append(lead_i_data[i + sequence_length])

    return np.array(X), np.array(y)


def build_model(sequence_length):
    model = Sequential()

    model.add(Conv1D(128, 5, activation='relu', input_shape=(sequence_length, 1)))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(LSTM(64))
    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    correlation_coefficient, _ = pearsonr(y_test.flatten(), y_pred.flatten())
    regression_coefficient = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R-squared : {r2:.4f}")
    print(f"Correlation Coefficient: {correlation_coefficient:.4f}")
    print(f"Regression Coefficient : {regression_coefficient:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return y_pred


def save_model(model, filename):
    model.save(filename)


def plot_results(time_series, y_test, y_pred, sample_size=5000):
    df = pd.DataFrame({'Time': time_series, 'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Time', y='Actual', data=df[:sample_size], label='Actual', linewidth=2, color='blue')
    sns.lineplot(x='Time', y='Predicted', data=df[:sample_size], label='Predicted', linewidth=2, color='red')
    plt.xlabel('Time')
    plt.ylabel('Lead')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show()
