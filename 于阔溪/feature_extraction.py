import librosa
import numpy as np
import librosa.display


class FeatureExtraction:
    def __init__(self, path=r"D:\Users\11201\Downloads\data_aishell\data_aishell\wav\train\S0002\BAC009S0002W0123.wav",
                 sr=16000, n_mfcc=40):
        self.y, self.sr = librosa.load(path, sr=sr)

        self.n_mfcc = n_mfcc
        self.mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=self.n_mfcc)
        self.normalized_mfccs = (self.mfccs - np.repeat(np.mean(self.mfccs, axis=1).reshape(self.n_mfcc, 1),
                                                        self.mfccs.shape[1], axis=1)) / \
                                np.repeat(np.std(self.mfccs, axis=1).reshape(self.n_mfcc, 1),self.mfccs.shape[1], axis=1)
        self.normalized_mfccs = self.normalized_mfccs.T

    def vector_199_40(self):
        if self.normalized_mfccs.shape[0] >= 199:
            return self.normalized_mfccs[:199, :]
        return np.pad(self.normalized_mfccs, ((0, 199 - self.normalized_mfccs.shape[0]), (0, 0)), 'constant',
                      constant_values=(0, 0))


if __name__ == "__main__":
    print(FeatureExtraction().vector_199_40().shape)



