import sys
sys.path.append('/opt/homebrew/lib/python3.11/site-packages')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
import os

#sampling rate: every 0.01 seconds
input_EEG_dir = '/Users/trinityvermeire/DreamCraftAI/data/interim/HYP-PSG'
output_dir = '/Users/trinityvermeire/DreamCraftAI/data/processed'
epochs = 4
batch_size = 64
classes = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R', 'Sleep stage W', 'Movement time']

def lstm_model():
    #define and train LSTM model
        model = Sequential()
        model.add(LSTM(units = 330, input_shape = (2000, 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=210, return_sequences=True))
        model.add(LSTM(units=120, return_sequences=True))
        model.add(LSTM(units=120))
        model.add(Dense(units = 7, activation = 'softmax'))
        #compile model with adam optimizer with lower learning rate
        optimizer = RMSprop(learning_rate = 0.001)
        model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model

def lstm(input_EEG_dir, output_dir, epochs, batch_size):
    #initialize model
    model = lstm_model()
    count = 0
    #loop through files in directory
    for file in sorted(os.listdir(input_EEG_dir)):
        #skip ds store file
        if file.endswith('Store'):
             continue
        #print file name
        print(file)
        #access file path
        data_EEG_path = os.path.join(input_EEG_dir, file)
        #load data
        data_EEG = pd.read_csv(data_EEG_path, encoding='latin1')
        #seperate data
        X = data_EEG.iloc[:,0].values
        y = data_EEG.iloc[:,1].values

        #noramlize PSG data
        scaler = StandardScaler()
        data_EEG_N = scaler.fit_transform(X.reshape(-1, 1))

        #split data into test and train
        X_train, X_test, y_train, y_test = train_test_split(data_EEG_N, y, test_size=0.2, random_state=22)

        #convert string labels to numerical representation w one hot
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        #find missing classes
        missing_classes = set(classes) - set(label_encoder.classes_)
        #get training and test data
        y_train_one_hot = pd.get_dummies(y_train_encoded, columns=np.arange(len(missing_classes)))
        y_test_one_hot = pd.get_dummies(y_test_encoded, columns=np.arange(len(missing_classes)))
        #pad one hot encoding labels if classes arent equivalent
        if missing_classes:
             padding_train = np.zeros((len(y_train_one_hot), len(missing_classes)))
             padding_test = np.zeros((len(y_test_one_hot), len(missing_classes)))
             #create dtaaframe with zeroes and appropriate column names
             missing_columns = list(missing_classes)
             padding_train_df = pd.DataFrame(padding_train, columns=missing_columns)
             padding_test_df = pd.DataFrame(padding_test, columns=missing_columns)
             #concarenare padding with existing one hot labels
             y_train_one_hot = np.concatenate((y_train_one_hot, padding_train_df), axis = 1)
             y_test_one_hot = np.concatenate((y_test_one_hot, padding_test_df), axis = 1)
        print('xTrain', len(X_train))
        print('xTest', len(X_test))
        print('yTrain', y_train_one_hot.shape)
        print('yTest', y_test_one_hot.shape)

        #train model
        model.fit(X_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_one_hot))
        count += 1
        print(count)

    # Export model to JSON file
    model_json = model.to_json()
    json_file_path = os.path.join(output_dir, "LSTM_DreamCraftAI.json")
    with open(json_file_path, "w") as json_file:
        json_file.write(model_json)

    # Save model weights to file
    model_weights_path = os.path.join(output_dir, "LSTM_DreamCraftAI_Weights.h5")
    model.save_weights(model_weights_path)

    print('DONEEEEEEE')

lstm(input_EEG_dir, output_dir, epochs, batch_size)