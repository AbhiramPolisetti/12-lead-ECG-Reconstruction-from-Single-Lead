#preprocess.py
import os
import pandas as pd
from scipy.signal import butter, lfilter, medfilt
import time

input_dir = r"data"
output_dir = r"preprocessed"

os.makedirs(output_dir, exist_ok=True)

# Define the list of ECG leads
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

# Bandpass filter function
@measure_execution_time
def bandpass_filter(data, lowcut=0.05, highcut=40.0, fs=1000):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, data)

# Baseline correction function using median filter
@measure_execution_time
def baseline_correction(data, kernel_size=701):
    return data - medfilt(data, kernel_size)

# Process and save data
@measure_execution_time
def preprocess_and_save(file_path, leads):
    df = pd.read_csv(file_path)
    df = df[leads]
    df = df.apply(lambda col: baseline_correction(bandpass_filter(col)))
    output_file = os.path.join(output_dir, f'preprocessed_{os.path.basename(file_path)}')
    df.to_csv(output_file, index=False)


for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_dir, file_name)
        preprocess_and_save(file_path, leads)
