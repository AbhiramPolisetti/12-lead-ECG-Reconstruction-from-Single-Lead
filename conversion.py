import wfdb
import pandas as pd
import os


folder_path = "input_folder"
output_path = "input_folder"
if not os.path.exists(output_path):
    os.makedirs(output_path)

file_list = [f for f in os.listdir(folder_path) if f.endswith('.dat')]  


for file_name in file_list:
    record_name = os.path.splitext(file_name)[0]  
    record = wfdb.rdsamp(os.path.join(folder_path, record_name))  
    df = pd.DataFrame(record[0], columns=record[1]['sig_name'])  
    csv_file_name = os.path.join(output_path, f'{record_name}.csv') 
    df.to_csv(csv_file_name, index=False) 

print("Conversion completed for all files.")
