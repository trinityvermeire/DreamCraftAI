
import sys
sys.path.append('/opt/homebrew/lib/python3.11/site-packages')

import os
import eeglib
import mne
import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.fft import fft

folder_path = '/Users/trinityvermeire/DreamCraftAI/data'
output_path_HYP = '/Users/trinityvermeire/DreamCraftAI/data/interim/HYP'
output_path_PSG = '/Users/trinityvermeire/DreamCraftAI/data/interim/PSG'

#sort the list of files
sorted_files = sorted(os.listdir(folder_path))

#0.01 interval for given PSG data
sampling_frequency = 100

#define band frequency
BAND_RANGES = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 100)
    }

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
input_PSG = '/Users/trinityvermeire/DreamCraftAI/data/interim/Frequency'
output_dir = '/Users/trinityvermeire/DreamCraftAI/data/processed-freq'

sorted_PSG = sorted(os.listdir(input_PSG))
sorted_HYP = sorted(os.listdir(input_HYP))

def merge_csv():
    for i in range(len(sorted_PSG) - 1):
        #create new list to store data that will become dataframe
        data_PSG = []
        #join paths and load csvs
        print(sorted_PSG[i + 1])
        print(sorted_HYP[i + 1])
        path_PSG = os.path.join(input_PSG, sorted_PSG[i + 1])
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

eeg_path = '/Users/trinityvermeire/DreamCraftAI/data/interim/PSG-No-Reshape'
eeg_output = '/Users/trinityvermeire/DreamCraftAI/data/pickle'
sorted_EEG = sorted(os.listdir(eeg_path))
files = os.listdir(eeg_path)

def calculate_frequency(amplitudes, sample_rate=100):
    peak_indices = []
    #find peaks
    for i in range(2, len(amplitudes) - 2):
        if amplitudes[i] == max(amplitudes[i - 2: i + 3]):
            peak_indices.append(i)
    #calculate time differences between adjacent peaks
    time_differences = []
    for i in range(1, len(peak_indices)):
        time_diff = (peak_indices[i] - peak_indices[i - 1]) / sample_rate
        time_differences.append(time_diff)
    #check if time differences list is empty
    if len(time_differences) == 0:
        return 0.0
    #calculate average time difference
    total = sum(time_differences)
    avg_time_diff = total / len(time_differences)
    #calculate frequency from average time difference
    frequency = 1.0 / avg_time_diff
    return frequency

def calculate_band_powers(eeg_data, sample_rate=256):
    # Perform FFT to obtain spectrum
    spectrum = fft(eeg_data)
    # Set frequency resolution
    freq_resolution = 1 / sample_rate
    # Calculate the power of each band
    band_powers = {}
    for band, band_range in BAND_RANGES.items():
        # Convert frequency range to indices
        start_index = int(band_range[0] / freq_resolution)
        end_index = int(band_range[1] / freq_resolution)
        # Calculate power using integration (trapezoidal rule)
        band_power = np.trapz(spectrum[start_index:end_index], dx=freq_resolution)
        # Normalize power
        band_power /= np.sum(spectrum)
        # Store band power
        band_powers[band] = band_power
    return band_powers


def approximate_frequency(band_powers):
    weighted_sum = total_power = 0
    # Calculate weighted sum of frequencies
    for band, power in band_powers.items():
        # Calculate midpoint frequency of each band
        band_range = BAND_RANGES[band]
        band_length = band_range[1] - band_range[0]
        midpoint_freq = (band_range[0] + band_range[1]) / 2
        power /= band_length
        # Update weighted sum with the midpoint frequency multiplied by the power
        weighted_sum += midpoint_freq * power
        # Update total power
        total_power += power
    # Calculate average frequency
    return np.round(weighted_sum / total_power, 2) # Rounds the output frequency to two decimal places

def compute_frequency():
    for file in files:
        if file.endswith('Store'):
            continue
        print(file)
        dfVolt = pd.read_csv(os.path.join(eeg_path, file))
        #create array to store results
        result_data = []
        #loop through rows
        for i in range(0, len(dfVolt), 100):
            #extract voltage
            microvoltages_batch = dfVolt.iloc[i:i+100, 0].values
            # Calculate frequency using custom function
            bands = calculate_band_powers(microvoltages_batch, 100)
            frequency = approximate_frequency(bands)
            # Format peak_frequency to increase significant digits
            frequency_formatted = '{:.8f}'.format(frequency)
            #append new row to result data list
            result_data.append({'Frequency': frequency_formatted})
        #create dataframe from result list
        result_df = pd.DataFrame(result_data)
        result_df.to_pickle(os.path.join(eeg_output, file[:8] + '-FREQ.pkl'))

    print('DONEEEEE')

compute_frequency()