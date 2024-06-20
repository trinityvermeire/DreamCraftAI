import sys
sys.path.append('/opt/homebrew/lib/python3.11/site-packages')

import numpy as np
import pandas as pd
from keras.models import model_from_json, Model
from keras.layers import Input, Dense, Concatenate, LSTM, Dropout
from keras.losses import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

input_dir = '/Users/trinityvermeire/DreamCraftAI/data/LSTM'
eeg_input_shape = (1000, 1)
sleep_input_shape = (7,)
sleep_stages_input = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R', 'Sleep stage W', 'Movement time']

#function to load json file
def load_model(input_dir):
    json_file_path = os.path.join(input_dir, "LSTM_DreamCraftAI.json")
    weights_file_path = os.path.join(input_dir, "LSTM_DreamCraftAI_Weights.h5")
    #load model architecture from JSON file
    with open(json_file_path, "r") as json_file:
        lstm_model_json = json_file.read()
    lstm_model = model_from_json(lstm_model_json)
    #load model weights from h5 file
    lstm_model.load_weights(weights_file_path)
    return lstm_model

#function to create personalized LSTM model
def create_personalized_lstm_model(base_model):
    #define LSTM input layer
    lstm_input = Input(shape=eeg_input_shape, name='LSTM_Input')
    #freeze layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
    #connect LSTM input to base LSTM layers
    lstm_output = base_model(lstm_input)
    #cdd additional layers for personalization
    personalized_output = Dense(units=300, activation='relu')(lstm_output)
    personalized_output = Dense(units=len(sleep_stages_input), activation='softmax')(personalized_output)
    # Create personalized model
    personalized_model = Model(inputs=lstm_input, outputs=personalized_output)
    # Compile model
    optimizerF = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    personalized_model.compile(optimizer=optimizerF, loss='categorical_crossentropy', metrics=['accuracy'])
    return personalized_model

#function to predict sleep stages from LSTM model
def predicted_sleep_stage(base_model, eeg_data):
    # Predict sleep stage using the base model
    return base_model.predict(eeg_data)

#function to predict sleep stages from LSTM model
def predicted_future_sleep_stage(base_model, eeg_data):
    return base_model.predict(eeg_data[-1000:,:,:])

# Function to normalize EEG data
def normalize_eeg_data(eeg_data):
    # Reshape the EEG data to (-1, 1) shape
    eeg_data = eeg_data.reshape(-1, 1)
    # Initialize StandardScaler
    scaler = StandardScaler()
    # Fit the scaler to the data and transform it
    normalized_data = scaler.fit_transform(eeg_data)
    return normalized_data

#funciton to iterate through the window of data points to be processed by the neural network
def sliding_window(csv_file, window_size, step_size):
    data = pd.read_csv(csv_file)
    # Iterate through data with step size 
    for i in range(0, len(data) - window_size, step_size):
        window_data = data.iloc[i : i + window_size]
        eeg_values = window_data.iloc[:, 0].values
        # Normalize EEG values
        normalized_eeg_values = normalize_eeg_data(eeg_values)
        # Reshape the EEG values into a 3D array
        normalized_eeg_values = np.expand_dims(normalized_eeg_values, axis=1)  # Add a new axis for features
        normalized_eeg_values = np.expand_dims(normalized_eeg_values, axis=0)  # Add a new axis for batch size
        yield normalized_eeg_values

# Load the LSTM model for predicting current sleep stage
base_lstm_model = load_model(input_dir)

# Create personalized LSTM model
personalized_lstm_model = create_personalized_lstm_model(base_lstm_model)

for eeg_values in sliding_window('/Users/trinityvermeire/DreamCraftAI/data/processed-freq/SC4001E0-EEG.csv', window_size=1000, step_size=1000):
    # Perform testing on each window of EEG data
    current_sleep_stage = predicted_sleep_stage(base_lstm_model, eeg_values)
    future_sleep_stage = predicted_future_sleep_stage(personalized_lstm_model, eeg_values)
    print("Current Sleep Stage:", sleep_stages_input[np.argmax(current_sleep_stage)])
    print("Future Sleep Stage:", sleep_stages_input[np.argmax(future_sleep_stage)])
