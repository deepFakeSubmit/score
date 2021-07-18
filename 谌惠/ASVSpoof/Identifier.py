from MFCC_FeaturesExtractor import MFCC_FeaturesExtractor
from LFCC_FeaturesExtractor import LFCC_FeaturesExtractor
from IMFCC_FeaturesExtractor import IMFCC_FeaturesExtractor
from CQCC_FeaturesExtractor import CQCC_FeaturesExtractor
import pickle
import os
import numpy as np

class Identifier:
    def __init__(self,files_path,spoofs_model_path,bonafides_model_path):
        self.testing_path = files_path
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.right = 0
        self.total_sample = 0
        self.features_extractor = MFCC_FeaturesExtractor() # 提取MFCC特征向量
        # self.features_extractor = IMFCC_FeaturesExtractor() # 提取IMFCC特征向量
        # self.features_extractor = LFCC_FeaturesExtractor() # 提取LFCC特征向量
        # self.features_extractor = CQCC_FeaturesExtractor() # 提取CQCC特征向量

        # load models
        self.spoofs_gmm = pickle.load(open(spoofs_model_path, 'rb'))
        self.bonafides_gmm = pickle.load(open(bonafides_model_path, 'rb'))

    def process(self):
        # 注释部分用于在dev数据集验证模型
        tags = []
        num = 0
        for line in open("zju_deepfake/dev.txt"):
            value = line.split()
            if value[1] == "bonafide":
                num += 1
            tags.append(value[1])
        i = 0

        # result =  open(r'mission1_7.txt',mode='w') # Path to save the evaluation result
        files = self.get_file_paths(self.testing_path)
        # read the test directory and get the list of test audio files

        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))
            vector = self.features_extractor.extract_features(file)
            winner = self.identify_Authenticity(vector)

            # result.write(os.path.basename(file).split(".")[0])
            # result.write(' ')
            # result.write(winner)
            # result.write('\n')

            if winner == tags[i]: # 分类正确
                self.right += 1
                if winner == "bonafide":
                    self.tp += 1
                else:
                    self.tn += 1
            else:  # 分类错误
                if winner == "bonafide":
                    self.fp += 1
                else:
                    self.fn += 1
            i += 1

            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))
            print("----------------------------------------------------")

        # result.close()

        # 训练结果
        print("真实语音一共有："+str(num)+" 被正确判定的真实语音一共有：" + str(self.tp))
        accuracy = ( float(self.right) / float(self.total_sample) )*100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)

        recall = ( float(self.tp) / float(self.tp + self.fn) )*100
        recall_msg = "*** Recall = " + str(round(recall, 3)) + "% ***"
        print(recall_msg)

        precision = ( float(self.tp) / float(self.tp + self.fp) )*100
        precision_msg = "*** Precision = " + str(round(precision, 3)) + "% ***"
        print(precision_msg)

    def get_file_paths(self,files_path):
        # get file paths
        files = [os.path.join(files_path, f) for f in os.listdir(files_path)]
        return files

    def identify_Authenticity(self,vector):
        # spoof
        spoof_scores = np.array(self.spoofs_gmm.score(vector))
            # spoof_scores = self.spoofs_gmm.score_samples(vector)
        is_spoof_log_likehood = spoof_scores.sum()
            # is_spoof_log_likehood = self.spoofs_gmm.score(vector)
            # print("score1: " + str(spoof_scores.sum()) + "score2:"+ str(is_spoof_log_likehood))
        # bonafides
        bonafides_scores = np.array(self.bonafides_gmm.score(vector))
            # bonafides_scores = self.bonafides_gmm.score_samples(vector)
        is_bonafide_log_likehood = bonafides_scores.sum()
            # is_bonafide_log_likehood = self.bonafides_gmm.score(vector)
            # print("score1："+str(bonafides_scores.sum()) + "score2："+str(is_bonafide_log_likehood))

        print("%10s %7s %1s" % ("+ SPOOF SCORE", ":", str(round(is_spoof_log_likehood, 3))))
        print("%10s %5s %1s" % ("+ BONAFIDE SCORE", ":", str(round(is_bonafide_log_likehood, 3))))

        if is_spoof_log_likehood > is_bonafide_log_likehood:
            winner = "spoof"
        else:
            winner = "bonafide"
        return winner

if __name__== "__main__":
    identifier = Identifier("zju_deepfake/dev/flac","train_spoofs.gmm","train_bonafides.gmm") # test_set directory, spoof_gmm, bonafide_gmm
    identifier.process()

