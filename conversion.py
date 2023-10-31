import wfdb
import pandas as pd
import os
import time

#To measure execution time 
def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper


@measure_execution_time
#to convert to csv
def convert_to_csv(file_name):
    record_name = os.path.splitext(file_name)[0]
    record = wfdb.rdsamp(os.path.join(folder_path, record_name))
    df = pd.DataFrame(record[0], columns=record[1]['sig_name'])
    csv_file_name = os.path.join(output_path, f'{record_name}.csv')
    df.to_csv(csv_file_name, index=False)


folder_path = "input"
output_path = "output" 
if not os.path.exists(output_path):
    os.makedirs(output_path)

file_list = [f for f in os.listdir(folder_path) if f.endswith('.dat')]


list(map(convert_to_csv, file_list))

print("Conversion completed.")
