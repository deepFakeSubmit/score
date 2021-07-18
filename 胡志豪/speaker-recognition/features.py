import sys

#from python_speech_features import mfcc
import numpy as np
# 换成spafe.features.mfc 库里的 mfcc
from spafe.features.mfcc import mfcc

def get_feature(fs, signal):
    num_ceps = 13;
    low_freq = 0
    high_freq = 2000
    nfilts = 24
    nfft = 512
    dct_type = 2;
    use_energy = False;
    lifter = 5
    normalize = False
    mfcc_feature = mfcc(sig=signal,fs= fs, num_ceps=num_ceps, nfilts=nfilts, nfft=nfft, low_freq=low_freq, high_freq=high_freq,
         dct_type=dct_type, use_energy=use_energy, lifter=lifter, normalize=normalize)
   # mfcc_feature = mfcc(signal, fs)
    if len(mfcc_feature) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(signal)
    return mfcc_feature
