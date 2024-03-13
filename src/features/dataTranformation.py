
import sys
sys.path.append('/opt/homebrew/lib/python3.11/site-packages')

import os
import mne
import pandas as pd
import numpy as np

folder_path = '/Users/trinityvermeire/DreamCraftAI/data/sleep-cassette-raw'
output_path_HYP = '/Users/trinityvermeire/DreamCraftAI/data/interim/HYP'
output_path_PSG = '/Users/trinityvermeire/DreamCraftAI/data/interim/PSG'

#sort the list of files
sorted_files = sorted(os.listdir(folder_path))

#0.01 interval for given PSG data
sampling_frequency = 100

# Iterate through each PSG file in folder
def create_csv():
    for file in sorted_files:
        if file.endswith('PSG.edf'):
            file_path = os.path.join(folder_path, file)
            # Load PSG file
            raw_PSG = mne.io.read_raw_edf(file_path)
            # Extract PSG data and channel names
            data_PSG, times_PSG = raw_PSG.get_data(return_times=True)
            #select only first column and remove first row
            data_PSG = data_PSG[0]
            # Reshape the PSG data into 1-second intervals
            data_PSG_1s = data_PSG.reshape(-1, sampling_frequency).mean(axis=1)
            # Create dataframe for PSG data
            df_PSG = pd.DataFrame(data_PSG_1s)
            #remame column of data
            df_PSG.rename(columns = {0:'EEG Fpz-Cz'}, inplace = True)
            # Filter dataframe to keep only frontal cortex column and transpose to get correct columns and rows
            #df_PSG = df_PSG.T
            #create csv file with PSG data
            df_PSG.to_csv(os.path.join(output_path_PSG, f'{file[:8]}-PSG.csv'), index=False)
            print('PSG DONE')
        elif file.endswith('Hypnogram.edf'):
            file_path = os.path.join(folder_path, file)
            # Load HYP file
            raw_HYP = mne.read_annotations(file_path)
            # Extract HYP data and channel names
            df_HYP = pd.DataFrame({'Onset': raw_HYP.onset, 'Duration': raw_HYP.duration, 'Description': raw_HYP.description})
            #drop last row of dataframe
            df_HYP = df_HYP.drop(df_HYP.index[-1])
            #print csv of hypnogram from dataframe
            df_HYP.to_csv(os.path.join(output_path_HYP, f'{file[:8]}-HYP.csv'), index=False)
            print('HYP DONE')
    print("Succesfully Completed!!!")

input_HYP = '/Users/trinityvermeire/DreamCraftAI/data/interim/HYP'
input_PSG = '/Users/trinityvermeire/DreamCraftAI/data/interim/PSG'
output_dir = '/Users/trinityvermeire/DreamCraftAI/data/interim/HYP-PSG'

sorted_PSG = sorted(os.listdir(input_PSG))
sorted_HYP = sorted(os.listdir(input_HYP))

def merge_csv():
    for i in range(len(sorted_PSG)):
        #create new list to store data that will become dataframe
        data_PSG = []
        #join paths and load csvs
        print(sorted_PSG[i])
        print(sorted_HYP[i + 1])
        path_PSG = os.path.join(input_PSG, sorted_PSG[i])
        path_HYP = os.path.join(input_HYP, sorted_HYP[i + 1])
        PSG = pd.read_csv(path_PSG, header=None)
        HYP = pd.read_csv(path_HYP, encoding='latin1')
        #iterate through rows of HYP dataframe
        for row in range (len(HYP)):
            #loop that runs as long as the duration of the sleep cycle to add sleep cycle and eeg signal to another dataframe
            for duration in range(int(HYP.iloc[row, 1])):
                eeg = PSG.iloc[duration + int(HYP.iloc[row, 0]), 0]
                hyp = HYP.iloc[row, 2]
                data_PSG.append([eeg, hyp])
        data_PSG_df = pd.DataFrame(data_PSG)
        #remove first row of the dataframe
        data_PSG_drop = data_PSG_df.drop(data_PSG_df.index[0])
        data_PSG_drop.to_csv(os.path.join(output_dir, f'{sorted_PSG[i][:8]}-EEG.csv'), index=False)
        print('Merging done!')
    print('Succesfully Completed!!!!')

