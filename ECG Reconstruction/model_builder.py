# model_builder.py
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense, Dropout


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
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
