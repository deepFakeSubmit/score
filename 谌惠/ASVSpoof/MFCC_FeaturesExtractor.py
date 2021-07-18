import soundfile as sf
from spafe.features.mfcc import mfcc
from sklearn import preprocessing
from python_speech_features import delta
import numpy as np

class MFCC_FeaturesExtractor:
    def __init__(self):
        # init input vars
        self.num_ceps = 13
        self.low_freq = 0
        self.high_freq = 2000
        self.nfilts = 24
        self.nfft = 512
        self.dct_type = 2,
        self.use_energy = False,
        self.lifter = 5
        self.normalize = False
        # pass

    def extract_features(self,audio_path):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double
        deltas.
        使用spafe库从音频中提取包括Mel频率倒谱系数(MFCC)在内的语音特征，
        执行倒谱均值归一化(CMS)，并将其与MFCC delta和MFCC double del相结合

        :param audio_path(str): path to flac file without silent moments.
        :return: (array) : Extracted features matrix. 提取的特征矩阵
        """
        sig,fs = sf.read(audio_path)
        mfcc_feature = mfcc(sig=sig,
                fs=fs,
                num_ceps=self.num_ceps,
                nfilts=self.nfilts,
                nfft=self.nfft,
                low_freq=self.low_freq,
                high_freq=self.high_freq,
                dct_type=self.dct_type,
                use_energy=self.use_energy,
                lifter=self.lifter,
                normalize=self.normalize)
        mfcc_feature = preprocessing.scale(mfcc_feature) # cms
        deltas = delta(mfcc_feature,2)
        double_deltas = delta(deltas,2)
        combined = np.hstack((mfcc_feature,deltas,double_deltas)) #cd
        return combined
