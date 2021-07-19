import pickle
from collections import defaultdict
from skgmm import GMMSet
import time

from utils import get_feature

#接口文件，对GMM模型进行了一定的封装
class ModelInterface:

    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()

    #得到MFCC特征参数
    def enroll(self, name, fs, signal):
        feat = get_feature(fs, signal)
        self.features[name].extend(feat)

    #训练出GMM模型
    def train(self):
        self.gmmset = GMMSet()
        start_time = time.time()
        for name, feats in self.features.items():
            try:
                self.gmmset.fit_new(feats, name)
            except Exception as e :
                print ("%s failed"%(name))
        print (time.time() - start_time, " seconds")


    def dump(self, fname):
        """ dump all models to file"""
        self.gmmset.before_pickle()
        with open(fname, 'wb') as f:
            pickle.dump(self, f, -1)
        self.gmmset.after_pickle()

    #利用GMM模型进行预测
    def predict(self, fs, signal):
        """
        return a label (name)
        """
        try:
            feat = get_feature(fs, signal)
            #print(feat)
        except Exception as e:
            print(e)
        return self.gmmset.predict_one(feat)

    @staticmethod
    def load(fname):
        """ load from a dumped model file"""
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            R.gmmset.after_pickle()
            return R