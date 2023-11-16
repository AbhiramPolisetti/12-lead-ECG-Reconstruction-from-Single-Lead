# model.py
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model


def save_model(model, model_filename):
    model.save(model_filename)
