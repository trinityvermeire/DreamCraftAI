import sys
#sys.path.append('/opt/homebrew/lib/python3.11/site-packages')

import numpy as np
import logging
import pandas as pd
#from utils import threadsafe_generator

class DataGenerator(object):
    def __init__(self, file_path, BATCH_SIZE=32, SEQ_LEN=64, n_classes=4, split_criterion=0.33, shuffle_opt=False, FEATURES_DIM=2):
        self.file_path = file_path
        self.BATCH_SIZE = BATCH_SIZE
        self.SEQ_LEN = SEQ_LEN
        self.n_classes = n_classes
        self.shuffle = shuffle_opt
        self.split_criterion = split_criterion
        self.FEATURES_DIM = FEATURES_DIM
        self.logger = logging.getLogger(__name__)

    #@threadsafe_generator
    def generate_training_sequences(self, epochs=4):
        while True:
            data = pd.read_csv(self.file_path, delimiter='\t', header=None)
            eeg_data = data.iloc[:, 0].values
            sleep_stage_data = data.iloc[:, 1].values

            for i in range(0, len(eeg_data) - self.SEQ_LEN, self.BATCH_SIZE):
                batch_eeg_data = []
                batch_sleep_stage_data = []
                for j in range(self.BATCH_SIZE):
                    batch_eeg_data.append(eeg_data[i + j:i + j + self.SEQ_LEN])
                    batch_sleep_stage_data.append(sleep_stage_data[i + j + self.SEQ_LEN - 1])
                yield np.array(batch_eeg_data), np.array(batch_sleep_stage_data)

    #@threadsafe_generator
    def generate_validation_sequences(self, epochs=4):
        while True:
            data = pd.read_csv(self.file_path, delimiter='\t', header=None)
            eeg_data = data.iloc[:, 0].values
            sleep_stage_data = data.iloc[:, 1].values

            for i in range(0, len(eeg_data) - self.SEQ_LEN, self.BATCH_SIZE):
                batch_eeg_data = []
                batch_sleep_stage_data = []
                for j in range(self.BATCH_SIZE):
                    batch_eeg_data.append(eeg_data[i + j:i + j + self.SEQ_LEN])
                    batch_sleep_stage_data.append(sleep_stage_data[i + j + self.SEQ_LEN - 1])
                yield np.array(batch_eeg_data), np.array(batch_sleep_stage_data)

    def generate_test_sequences(self):
        result_feat = []
        result_lab = []

        self.logger.info("Loading Test Data")
        for subject in self.validation_subject_keys:
            subjectGroupHandle = self.sleep_data[subject]
            subjectGroupKeys = subjectGroupHandle.keys()

            dateValue = subjectGroupKeys[0]
            dateGroupHandle = subjectGroupHandle[dateValue]

            filteredGroupHandle = dateGroupHandle['FILTERED']
            filteredGroupKeys = filteredGroupHandle.keys()

            self.logger.info("Total epochs : {}".format(len(filteredGroupKeys)))

            for epoch_count, filteredValue in enumerate(filteredGroupKeys):
                self.logger.debug("Filtered values : {}".format(filteredValue))
                dataSetHandle = filteredGroupHandle[filteredValue]

                single_signal = []
                labels_list = []

                for dataSet in range(len(dataSetHandle)):
                    single_signal.append([float(dataSetHandle[dataSet][1]), float(dataSetHandle[dataSet][2])])
                    labels_list.append(dataSetHandle.attrs['label'])

                for index in range(len(single_signal) - self.SEQ_LEN):
                    result_feat.append(single_signal[index: index + self.SEQ_LEN])
                    result_lab.append(labels_list[index])

        self.logger.debug("Test X shape : {}, Test Y Shape : {}".format(np.array(result_feat).shape, np.array(result_lab).shape))
        return np.array(result_feat), np.array(result_lab)
    
    def get_exploration_order(self, filteredSignal):
        # Generate an array of indices
        indexes = np.arange(len(filteredSignal))

        # Shuffle the indices if shuffle option is enabled
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes
    
    def create_subsequence(self, filteredSignal, labels):
        result_feat = []
        result_lab = []

        if self.FEATURES_DIM > 2:
            for index in range(filteredSignal.shape[0] - self.SEQ_LEN):
                result_feat.append(filteredSignal[index: index + self.SEQ_LEN])
                result_lab.append(labels[index])
        else:
            for index in range(len(filteredSignal) - self.SEQ_LEN):
                result_feat.append(filteredSignal[index: index + self.SEQ_LEN])
                result_lab.append(labels[index])
        yield np.array(result_feat), np.array(result_lab)

    def categorical(self, labels):
        # As labels start from 1, we use j+1
        return np.array([[1 if labels[i] == j+1 else 0 for j in range(self.n_classes)] for i in range(labels.shape[0])])

    def training_sample_count(self):
        data = pd.read_csv(self.file_path, delimiter='\t', header=None)
        return len(data) - self.SEQ_LEN

    def validation_sample_count(self):
        data = pd.read_csv(self.file_path, delimiter='\t', header=None)
        return len(data) - self.SEQ_LEN

    def get_total_sample_count(self):
        return self.training_sample_count() + self.validation_sample_count()
