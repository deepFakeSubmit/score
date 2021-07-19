import tensorflow
import numpy as np
from feature_extraction import FeatureExtraction


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, files, labels, batch_size=32, shuffle=True, random_state=42):
        'Initialization'
        self.files = files
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        files_batch = [self.files[k] for k in indexes]
        y = np.array([self.labels[k] for k in indexes])

        # Generate data
        x = self.__data_generation(files_batch)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __data_generation(self, files):

        features = []
        for audio_file in files:
            feature = FeatureExtraction(path=audio_file).vector_199_40()
            features.append(feature)
        features = np.stack(features, axis=0)
        features = features.reshape((features.shape[0], features.shape[1], features.shape[2], 1))

        return features
