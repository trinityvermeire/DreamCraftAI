import sys
sys.path.append('/opt/homebrew/lib/python3.11/site-packages')

import numpy as np
import pandas as pd
from keras.models import model_from_json, Model
from keras.layers import Input, Dense, Concatenate, LSTM, Dropout
from keras.losses import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os

input_dir = '/Users/trinityvermeire/DreamCraftAI/data/processed'
eeg_input_shape = (2000, 1)
sleep_input_shape = (7,)
sleep_stages_input = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R', 'Sleep stage W', 'Movement time']

#function to load json file
def load_model(input_dir, eeg_input_shape, sleep_input_shape):
    json_file_path = os.path.join(input_dir, "LSTM_DreamCraftAI.json")
    weights_file_path = os.path.join(input_dir, "LSTM_DreamCraftAI_Weights.h5")
    #load model architecture from JSON file
    with open(json_file_path, "r") as json_file:
        lstm_model_json = json_file.read()
    lstm_model = model_from_json(lstm_model_json)
    #load model weights from h5 file
    lstm_model.load_weights(weights_file_path)
    #create combined model with LSTM branch and the new EEg + sleep stage branch
    combined_model = create_combined_model(eeg_input_shape, sleep_input_shape, lstm_model)
    return combined_model

#function to predict sleep stages from float values
def predicted_sleep_stage(EEG_input, lstm_model):
    predicted_stage = lstm_model.predict(EEG_input)
    predicted_stage_index = np.argmax(predicted_stage)
    return sleep_stages_input[predicted_stage_index]

def create_combined_model(eeg_input_shape, sleep_input_shape, lstm_model):
    #define EEG input layer
    EEG_input = Input(shape=eeg_input_shape, name='EEG_Input')
    #define LSTM branch
    lstm_output = LSTM(units=330, return_sequences=True)(EEG_input)
    lstm_output = Dropout(0.2)(lstm_output)
    lstm_output = LSTM(units=210, return_sequences=True)(lstm_output)
    lstm_output = LSTM(units=120)(lstm_output)
    #define sleep stage input layer
    sleep_stage_input = Input(shape=sleep_input_shape, name='Sleep_Stage_Input')
    #concatenate LSTM output and sleep stage input
    concat = Concatenate()([lstm_output, sleep_stage_input])
    #add dense layers 
    combined_output = Dense(units=128, activation='relu')(concat)
    combined_output = Dense(units=64, activation='relu')(combined_output)
    #define final output layer
    final_output = Dense(units=1, activation=None)(combined_output) 
    #create combined model
    combined_model = Model(inputs=[EEG_input, sleep_stage_input], outputs=final_output)
    combined_model.compile(optimizer='adam', loss=mean_squared_error)
    return combined_model


#function to calculate standard deviation
def calculate_std_dev(eeg_input, lstm_model):
    #predict sleep stage from EEG input
    predicted_stage = predicted_sleep_stage(eeg_input, lstm_model)
    #encode sleep stage input
    encoded_sleep_input, _ = encode_sleep_stages([predicted_stage])
    #predict EEG values using the loaded model
    predicted_eeg = lstm_model.predict([eeg_input, encoded_sleep_input])
    #calculate standard deviation between predicted and input EEG values
    return np.std(predicted_eeg - eeg_input)

def encode_sleep_stages(sleep_stages):
    label_encoder = LabelEncoder()
    encoded_sleep_stages = label_encoder.fit_transform(sleep_stages)
    return encoded_sleep_stages, label_encoder.classes_

loaded_model = load_model(input_dir, eeg_input_shape, sleep_input_shape)
serial_input = [np.random.rand(1, 2000, 1), np.random.rand(1, 7)]
print("Standard Deviation:", calculate_std_dev(serial_input[0], loaded_model))