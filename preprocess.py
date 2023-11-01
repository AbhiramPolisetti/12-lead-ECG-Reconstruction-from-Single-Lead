import os
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import time

input_dir = r"D:\deepfacts\ecg_coversion\csv_output" 
output_dir = "D:\deepfacts\ecg_coversion\preprocessed_output"

os.makedirs(output_dir, exist_ok=True)

leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

# Measure execution time decorator
def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper

# Function to apply bandpass filter
def bandpass_filter(data, lowcut, highcut, fs):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# Function to apply baseline correction using median filter
def baseline_correction(data, kernel_size=2001):
    return data - np.median(data, axis=0)


# Process and save data
@measure_execution_time
def preprocess_and_save(file_path, leads):
    df = pd.read_csv(file_path)
    df = df[leads]
    fs = 1000 
    df = df.apply(lambda col: baseline_correction(bandpass_filter(col, 0.05, 40.0, fs)))
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_file, index=False)


for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_dir, file_name)
        preprocess_and_save(file_path, leads)
