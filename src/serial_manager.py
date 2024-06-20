import numpy as np
import eeglib
from scipy.fft import fft
import serial
import csv
from datetime import datetime
from time import sleep
import os

SERIAL_PORT = 'COM7'    # Update with your serial port
BAUD_RATE = 115200      
SAMPLE_RATE = 256
TIME_INTERVAL = 5       # Time in seconds between each measurement

OUTPUT_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) + "/sleep_data"

BAND_RANGES = {
        'delta': (0, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }

def calculate_band_powers(eeg_data, sample_rate=SAMPLE_RATE):
    # Perform FFT to obtain spectrum
    spectrum = fft(eeg_data)
    
    # Set frequency resolution
    freq_resolution = 1 / sample_rate

    # Calculate the power of each band
    band_powers = eeglib.features.bandPower(spectrum, BAND_RANGES, freq_resolution, normalize=True)
    band_powers = {band: power/(BAND_RANGES[band][1]-BAND_RANGES[band][0]) for band, power in band_powers.items()} # Optional Adjustment

    adjustment = len(band_powers.values()) / sum({abs(power) for band, power in band_powers.items()}) # Normalizer for the band powers so that the sum is equal to the length

    band_powers = {band: round(abs(power*adjustment), 5) for band, power in band_powers.items()}
    
    return band_powers

def approximate_frequency(band_powers):
    weighted_sum = total_power = 0

    # Calculate weighted sum of frequencies
    for band, power in band_powers.items():
        # Calculate midpoint frequency of each band
        band_range = BAND_RANGES[band]
        band_length = band_range[1] - band_range[0]
        midpoint_freq = (band_range[0] + band_range[1]) / 2
        # power /= band_length # Optional Adjustment

        # Update weighted sum with the midpoint frequency multiplied by the power
        weighted_sum += midpoint_freq * power
        
        # Update total power
        total_power += power

    # Calculate average frequency
    return round(weighted_sum / total_power, 2) # Rounds the output frequency to two decimal places

def read_arduino(UserID):
    rows = []
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        # Create instance of serial connection
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout = 0.1)
        print("Serial connection established")

        rows.append(("UserID", *BAND_RANGES.keys(), "Frequency", "Timestamp"))  # Write column headers
        data_array = []

        while True:
            # Read bytes from Arduino
            data_bytes = arduino.readline().strip().decode('latin-1')
            
            if data_bytes:
                data = float(''.join(char for char in data_bytes if char.isdigit() or char in '.-')) # Filter out non-numeric characters
                data_array.append(data)
                if len(data_array) >= SAMPLE_RATE*TIME_INTERVAL:
                    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    bands = calculate_band_powers(data_array)
                    freq = approximate_frequency(bands)
                    print(datetime.now().strftime('%H:%M:%S'), f'{freq}hz', bands)
                    rows.append((UserID, *bands.values(), freq, time)) # Write data rows
                    data_array = []

    except serial.SerialException as e:
        print("Error:", e)
    finally:
        arduino.close()
        with open(f"{OUTPUT_DIRECTORY}/{current_date}_sleep_data.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(rows)

if __name__ == "__main__":
    #ensures app is only ran when executed in terminal
    read_arduino("Bradley")
