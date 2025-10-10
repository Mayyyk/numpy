import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.ndimage import uniform_filter1d
import os
import glob



print("Folder contet")
for root, dirs, files in os.walk("motionsense_dataset"):
    print(f"\n{root}:")
    for file in files[:5]:
        print(f"  {file}")


def load_activity_data(activity="wlk", trial=1, user=1):
    path_pattern = (
        f"motionsense_data/A_DeviceMotion_data/A_DeviceMotion_data/{activity}_*"
    )
    folders = glob.glob(path_pattern)
    

    if not folders:
        print(f"No folders for {activity}")
        return None
    print(folders)
    
    activity_all_data = []
    
    for folder in folders:
        trial_num = folder.split('_')[-1]
        csv_files = glob.glob(f"{folder}/*.csv")
        print(f"Folder {os.path.basename(folder)} - found {len(csv_files)} CSV files.")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                df['trial'] = int(trial_num)
                df['activity'] = activity
                
                activity_all_data.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}:\n{e}") 
                
    if not activity_all_data:
        print(f"No data found for {activity}.")
        return None
    
    print(f"Found {len(folders)} folders for {activity}.")

    combined_df = pd.concat(activity_all_data, ignore_index=True)
    
    print(f"Read {len(combined_df)} rows from {len(activity_all_data)} 'csv' files.")

    return combined_df



walking_data = load_activity_data("wlk")
print(walking_data.shape if walking_data is not None else "No data")
