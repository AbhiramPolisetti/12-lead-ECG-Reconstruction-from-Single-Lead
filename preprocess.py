import os
import pandas as pd
from scipy.signal import butter, lfilter, medfilt


input_dir = "input_folder"
output_dir = "output_folder"

os.makedirs(output_dir, exist_ok=True)

leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

# Function to apply preprocessing and save a csv file
def preprocess_and_save(file_path, leads):
       df = pd.read_csv(file_path)

    
    df = df[leads]

    # Apply bandpass filter
    lowcut = 0.05  # Low cutoff frequency in Hz
    highcut = 40.0  # High cutoff frequency in Hz
    fs = 1000  # Sampling frequency in Hz
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    for lead in leads:
        df[lead] = bandpass_filter(df[lead], lowcut, highcut, fs)

    #baseline wander correction using median filter
    for lead in leads:
        df[lead] = baseline_correction(df[lead])

    output_file = os.path.join(output_dir, os.path.basename(file_path))

    df.to_csv(output_file, index=False)

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# Baseline correction function using median filter
def baseline_correction(data):
    return data - medfilt(data, kernel_size=2001)


for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_dir, file_name)
        preprocess_and_save(file_path, leads)
