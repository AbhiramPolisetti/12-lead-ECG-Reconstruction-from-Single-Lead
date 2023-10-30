#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system(' pip install tensorflow-gpu')


# In[ ]:


import tensorflow as tf

# Check for GPU availability
if tf.test.is_gpu_available():
    print("GPU is available and will be used for training.")
else:
    print("GPU is not available.")


# In[ ]:


# Explicitly specify GPU device
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Set memory growth to prevent GPU memory allocation from the beginning
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Set to use the first GPU (device 0)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense, Dropout


# In[ ]:


# Function to load and preprocess data
def load_and_preprocess_data(csv_file):
    data = pd.read_csv(csv_file)
    input_column = 'ii'
    target_column = 'v5'

    input_data = data[input_column].values
    target_data = data[target_column].values

    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data.reshape(-1, 1))
    target_data = scaler.fit_transform(target_data.reshape(-1, 1))

    sequence_length = 1000

    X, y = [], []
    for i in range(len(input_data) - sequence_length):
        X.append(input_data[i:i+sequence_length])
        y.append(target_data[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# In[ ]:


# Initialize the model
model = Sequential()
model.add(Conv1D(128, 5, activation='relu', input_shape=(1000, 1)))
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


# In[ ]:


folder_path = '/content/drive/MyDrive/ptb_csv/'


# In[ ]:


csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]


# In[ ]:


for csv_file in csv_files:
    X_train, X_test, y_train, y_test = load_and_preprocess_data(os.path.join(folder_path, csv_file))
    model.fit(X_train, y_train, epochs=200, batch_size=512)


# In[ ]:


model.save('lead_v5.h5')

