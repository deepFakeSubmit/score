# print("hello")
import librosa
import scipy.io.wavfile
import numpy as np
import soundfile as sf

from spafe.utils import vis
from spafe.features.mfcc import mfcc, imfcc
from python_speech_features import delta
from sklearn import preprocessing
from spafe.features.mfcc import mfcc
# print("1")
# init input vars
def get_Mfcc(path):
    num_ceps = 13
    low_freq = 0
    high_freq = 2000
    nfilts = 24
    nfft = 512
    dct_type = 2
    use_energy = False
    lifter = 5
    normalize = False
    sig, fs = librosa.load(path, sr=None)
    
    mfcc_feature = mfcc(sig=sig,
                 fs=fs,
                 num_ceps=num_ceps,
                 nfilts=nfilts,
                 nfft=nfft,
                 low_freq=low_freq,
                 high_freq=high_freq,
                 dct_type=dct_type,
                 use_energy=use_energy,
                 lifter=lifter,
                 normalize=normalize)
    mfcc_feature = preprocessing.scale(mfcc_feature) # cms
    deltas = delta(mfcc_feature,2)
    double_deltas = delta(deltas,2)
    combined = np.hstack((mfcc_feature,deltas,double_deltas)) #cd
    return combined
#     print(mfcc1)
#     # mfccs = mfcc(sig=sig,
#     #              fs=fs,
#     #              num_ceps=num_ceps,
#     #              nfilts=nfilts,
#     #              nfft=nfft,
#     #              low_freq=low_freq,
#     #              high_freq=high_freq,
#     #              dct_type=dct_type,
#     #              use_energy=use_energy,
#     #              lifter=lifter,
#     #              normalize=normalize)
#     # print(mfccs)
#     # visualize spectogram
#     vis.spectogram(sig, fs)
#     # visualize features
#     vis.visualize_features(mfcc1, 'MFCC Index', 'Frame Index')