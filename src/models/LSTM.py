import sys
sys.path.append('/opt/homebrew/lib/python3.11/site-packages')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras import optimizers, losses, activations, Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Dropout, Activation, Bidirectional, BatchNormalization, Conv2D, MaxPool1D, Flatten, Concatenate, MaxPooling2D, Convolution2D, SpatialDropout1D, GlobalMaxPool1D, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
import os
from keras.regularizers import l2
import pickle
from keras.utils import plot_model, to_categorical

#import matplotlib.pyplot as plt

#sampling rate: every 0.01 seconds
input_EEG_dir = '/Users/trinityvermeire/DreamCraftAI/data/pickle'
output_dir = '/Users/trinityvermeire/DreamCraftAI/data/LSTM'
output_pred = '/Users/trinityvermeire/DreamCraftAI/data/predictions'
epochs = 16
batch_size = 32
classes = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R', 'Sleep stage W', 'Movement time']

shared_conv1 = Conv2D(filters=8, kernel_size=3, padding='same')
shared_bn1 = BatchNormalization()
shared_relu1 = Activation('relu')
shared_mp1 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1),padding='same')
shared_do1 = Dropout(0.2)

shared_conv2 = Conv2D(filters=16, kernel_size=3, padding='same')
shared_bn2 = BatchNormalization()
shared_relu2 = Activation('relu')
shared_mp2 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1),padding='same')
shared_do2 = Dropout(0.2)

shared_flatten = Flatten()

shared_dense1 = Dense(units=7)
shared_bn4 = BatchNormalization()
shared_relu4 = Activation('relu')

def lstm_model():
    # Define the input layer for one channel
    input1 = Input(shape=(1000, 1))
    # Build CNN layers for one channel
    cnn_output1 = build_cnn(input1)
    # Reshape the output of CNN to fit into LSTM
    cnn_output_reshaped = Reshape((1, 7))(cnn_output1)
    # Define LSTM layer
    lstm_output = Bidirectional(LSTM(15, return_sequences=True, kernel_regularizer=l2(0.01)))(cnn_output_reshaped)
    # Flatten the output of LSTM layer
    lstm_output_flat = Flatten()(lstm_output)
    # Output layer
    output = Dense(7, activation='softmax')(lstm_output_flat) 
    # Define the model with one input and output
    model = Model(inputs=input1, outputs=output)
    #compile model with adam optimizer with lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.001)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Print model summary
    print(model.summary())
    # Save model architecture plot
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

def load_data(data_group):
    data_raw = pickle.load(open(data_group, 'rb'))
    seven_class_labels = []
    #seperate data into labels and values
    data = data_raw.iloc[:,0].values
    labels = data_raw.iloc[:,1].values
    # Artifact/Unlabeled: N/A as any such labels were already removed in spectrogram generation
    label_mapping = {
        'Sleep stage 1': 0,
        'Sleep stage 2': 1,
        'Sleep stage 3': 2,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
        'Sleep stage W': 5,
        'Movement time': 6
    }
    # Map the original labels to their corresponding indices
    mapped_labels = [label_mapping[label] for label in labels]
    
    # Convert labels to one-hot encoding
    y = to_categorical(mapped_labels, num_classes=7)
    
    # Initialize lists to store spectrogram data
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []
    y = []

    for i in range(len(data)):
        X1_tmp = data[i][0]
        X2_tmp = data[i][1]
        X3_tmp = data[i][2]
        X4_tmp = data[i][3]
        X5_tmp = data[i][4]

        # Make sure the size of the spectrogram is correct before adding it to the array
        # For some reason the third dimension of the spectrogram output is not right
        if (np.shape(X1_tmp)[2] != 48) or (np.shape(X2_tmp)[2] != 48) or (np.shape(X3_tmp)[2] != 48) or (np.shape(X4_tmp)[2] != 48) or (np.shape(X5_tmp)[2] != 48):
            continue

        X1.append(X1_tmp)
        X2.append(X2_tmp)
        X3.append(X3_tmp)
        X4.append(X4_tmp)
        X5.append(X5_tmp)
        y.append(seven_class_labels[i])

    # Reshape and convert to numpy arrays
    X1 = np.array(X1).reshape(len(X1), 64, 48, 2)
    X2 = np.array(X2).reshape(len(X2), 64, 48, 2)
    X3 = np.array(X3).reshape(len(X3), 64, 48, 2)
    X4 = np.array(X4).reshape(len(X4), 64, 48, 2)
    X5 = np.array(X5).reshape(len(X5), 64, 48, 2)
    y = to_categorical(y)

    return X1, X2, X3, X4, X5, y

def performance_metrics(true, predictions, labels):
    predictions_class = []
    for prediction in predictions:
        for row in prediction:
            predictions_class.append(np.argmax(row))

    true_class = []
    for true_row in true:
        for row in true_row:
            true_class.append(np.argmax(row))

    conf_mat = confusion_matrix(true_class, predictions_class)
    conf_mat = pd.DataFrame(conf_mat, columns=labels, index=labels)

    f1 = f1_score(true_class, predictions_class, average=None)
    f1 = pd.Series(f1, index=labels)

    return conf_mat, f1, predictions_class, true_class

def build_cnn(input):
    cnn_output = build_cnn_per_channel(input)
    cnn_output = Reshape((1, 7))(cnn_output)
    return cnn_output

def build_cnn_per_channel(input):
    cnn_output = shared_conv1(input[..., None])
    cnn_output = shared_bn1(cnn_output)
    cnn_output = shared_relu1(cnn_output)
    cnn_output = shared_mp1(cnn_output)
    cnn_output = shared_do1(cnn_output)
    cnn_output = shared_conv2(cnn_output)
    cnn_output = shared_bn2(cnn_output)
    cnn_output = shared_relu2(cnn_output)
    cnn_output = shared_mp2(cnn_output)
    cnn_output = shared_do2(cnn_output)
    cnn_output = shared_flatten(cnn_output)
    cnn_output = shared_dense1(cnn_output)
    cnn_output = shared_bn4(cnn_output)
    cnn_output = shared_relu4(cnn_output)
    cnn_output = Reshape((1, 7))(cnn_output)
    return cnn_output

def model_arch():
    input1 = Input(shape=(64, 48, 2))
    input2 = Input(shape=(64, 48, 2))
    input3 = Input(shape=(64, 48, 2))
    input4 = Input(shape=(64, 48, 2))
    input5 = Input(shape=(64, 48, 2))

    cnn_output1 = build_cnn(input1)
    cnn_output2 = build_cnn(input2)
    cnn_output3 = build_cnn(input3)
    cnn_output4 = build_cnn(input4)
    cnn_output5 = build_cnn(input5)

    #combine all five outputs into a single input for the LSTM
    cnn_concat = Concatenate(axis=1)([cnn_output1, cnn_output2, cnn_output3, cnn_output4, cnn_output5])

    lstm_output = Bidirectional(LSTM(15, return_sequences=True, kernel_regularizer=l2(0.01)))(cnn_concat)
    lstm_output = Dense(7, activation='softmax')(lstm_output)

    model = Model(inputs=[input1, input2, input3, input4, input5], outputs=lstm_output)
    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')

    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


def lstm():
    #X1_train, X2_train, X3_train, X4_train, X5_train, y_train = load_data('train')
    #X1_val, X2_val, X3_val, X4_val, X5_val, y_val = load_data('val')

    #change this to true to train the model again
    if True:
        checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='rmsprop')
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        #history = model.fit([X1_train, X2_train, X3_train, X4_train, X5_train],y_train,batch_size=BATCH_SIZE,epochs=20,validation_data=([X1_val, X2_val, X3_val, X4_val, X5_val], y_val),callbacks=[earlystop, checkpoint, CSVLogger('history.csv')],verbose=1)


    
'''
def lstm_model():
    #define and train LSTM model
        model = Sequential()

        model.add(LSTM(units = 330, input_shape = (1000, 1), return_sequences=True))
        model.add(Bidirectional(LSTM(512, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Dense(500))
        model.add(BatchNormalization())
        model.add(LSTM(units=210, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(units=120))
        model.add(Dense(units = 7, activation = 'softmax'))
        #compile model with adam optimizer with lower learning rate
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.01)
        model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model

def lstm(input_EEG_dir, output_dir, epochs, batch_size):
    #initialize model
    model = lstm_model()
    count = 0
    #loop through files in directory
    for file in sorted(os.listdir(input_EEG_dir)):
        #skip ds store file
        if file.startswith('.DS'):
             continue
        #print file name
        print(file)
        #access file path
        data_EEG_path = os.path.join(input_EEG_dir, file)
        #load data
        data_EEG = pd.read_csv(data_EEG_path, encoding='latin1')
        #exclude first and last 20,000 rows
        data_EEG = data_EEG.iloc[20000:-20000]
        #seperate data
        X = data_EEG.iloc[:,0].values
        y = data_EEG.iloc[:,1].values

        #noramlize PSG data
        scaler = StandardScaler()
        data_EEG_N = scaler.fit_transform(X.reshape(-1, 1))

        #split data into test and train
        X_train, X_test, y_train, y_test = train_test_split(data_EEG_N , y, test_size=0.2, random_state=1333)
        
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

        #checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_accuracy", mode="max", patience=20, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=5, verbose=2)
        callbacks_list = [early, redonplat]  # early
        #train model
        #model.fit(X_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test_one_hot))
        count += 1
        X_train_array = np.array(X_train)
        X_test_array = np.array(X_test)
        print('xtrain', X_train_array.shape)
        print('xtest', X_test_array.shape)
        print('ytrainhot', y_train_one_hot.shape)
        print('ytesthot', y_test_one_hot.shape)
        model.fit(X_train, y_train_one_hot, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_test, y_test_one_hot), callbacks=callbacks_list)
        print(count)

        #predict sleep stage for current file
        predictions = model.predict(X_test)
        print(X_test[:5])
        print(len(X_test))
        #convert predictions to class labels and save them
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_classes = label_encoder.inverse_transform(predicted_classes)
        prediction_file_path = os.path.join(output_pred, file[:-4] + '-PRED-X.csv')
        pd.DataFrame({'True Labels': y_test, 'Predicted Labels': predicted_classes}).to_csv(prediction_file_path, index=False)

    # Export model to JSON file
    model_json = model.to_json()
    json_file_path = os.path.join(output_dir, "LSTM_DreamCraftAI.json")
    with open(json_file_path, "w") as json_file:
        json_file.write(model_json)

    # Save model weights to file
    model_weights_path = os.path.join(output_dir, "LSTM_DreamCraftAI_Weights.h5")
    model.save_weights(model_weights_path)

    print('DONEEEEEEE')
    '''

#lstm(input_EEG_dir, output_dir, epochs, batch_size)
