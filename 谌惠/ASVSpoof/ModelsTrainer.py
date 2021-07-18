import os
import numpy as np
from MFCC_FeaturesExtractor import MFCC_FeaturesExtractor
from LFCC_FeaturesExtractor import LFCC_FeaturesExtractor
from IMFCC_FeaturesExtractor import IMFCC_FeaturesExtractor
from CQCC_FeaturesExtractor import CQCC_FeaturesExtractor
from sklearn.mixture import GaussianMixture as GMM
import pickle

class ModelsTrainer:
    def __init__(self,result_file_path):
        self.result_file_path = result_file_path
        self.features_extractor = MFCC_FeaturesExtractor() # 提取MFCC特征向量
        # self.features_extractor = LFCC_FeaturesExtractor() # 提取LFCC特征向量
        # self.features_extractor = IMFCC_FeaturesExtractor() # 提取IMFCC特征向量
        # self.features_extractor = CQCC_FeaturesExtractor() # 提取CQCC特征向量

    def process(self): # 模型训练
        spoofs, bonafides = self.get_files_paths(self.result_file_path) #分别获取真实语音和合成语音的文件集合

        # collect voice features
        print("# collect voice features")
        spoofs_voice_features = self.collect_features(spoofs)
        bonafides_voice_features = self.collect_features(bonafides)

        # generate gaussian mixture models
        spoofs_gmm = GMM(n_components=8, max_iter=200, covariance_type='diag', n_init=3)
        bonafides_gmm = GMM(n_components=8, max_iter=200, covariance_type='diag', n_init=3)

        # fit features to models 分别给spoof数据集和bonafide数据集训练模型
        spoofs_gmm.fit(spoofs_voice_features)
        bonafides_gmm.fit(bonafides_voice_features)

        # save models
        print("# save models")
        self.save_gmm(spoofs_gmm, "train_spoofs_cd")
        self.save_gmm(bonafides_gmm, "train_bonafides_cd")

    def get_files_paths(self,file):
        path = 'zju_deepfake/train/flac'  # train_set path
        spoofs = []
        bonafides = []
        #get file paths
        for line in open(file):
            value = line.split()
            if value[1] == "spoof":
                spoofs.append(value[0])
            elif value[1] == "bonafide":
                bonafides.append(value[0])
        spoofs_path = [os.path.join(path,s + ".flac") for s in spoofs]
        bonafides_path = [os.path.join(path,b + ".flac") for b in bonafides]
        return spoofs_path, bonafides_path

    def collect_features(self,files):
        """
        Collect voice features of true and false speech

        :param files (list) :List of voice file paths.
        :return:(array) : Extracted features matrix.
        """
        features = np.asarray(())
        # extract features for each
        for file in files:
            print("%5s %10s" % ("PROCESSNG ", file))
            # extract MFCC & delta MFCC features from audio
            vector = self.features_extractor.extract_features(file)
            # stack the features
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
        return features

    def save_gmm(self,gmm,name):
        """
        Save Gaussian mixture model using pickle.
        :param gmm: Gaussian mixture model.
        :param name: 模型存储路径
        :return:
        """
        filename = name + ".gmm"
        with open(filename, "wb") as  gmm_file:
            pickle.dump(gmm, gmm_file)

if __name__=="__main__":
    models_trainer = ModelsTrainer("zju_deepfake/train.txt") # train_set label file
    models_trainer.process()