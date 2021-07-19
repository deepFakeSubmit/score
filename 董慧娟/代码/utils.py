from scipy.io import wavfile

from python_speech_features import mfcc
import sys

#得到fs(采样率)、sinnal(呢容)
def read_wav(fname):
    fs, signal = wavfile.read(fname)
    if len(signal.shape) != 1:
        print("convert stereo to mono")
        signal = signal[:, 0]
    return fs, signal

#提取MFCC特征
def get_feature(fs, signal):
    mfcc_feature = mfcc(signal, fs)
    if len(mfcc_feature) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(signal)
    return mfcc_feature
