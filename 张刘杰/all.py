import librosa
import scipy
import pickle
import os
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
from python_speech_features import delta
from spafe.features.mfcc import mfcc
warnings.filterwarnings("ignore")

#读取音频 提取MFCC特征并进行处理 函数
def get_Mfcc(path):
    # 读取音频
    sig, fs = librosa.load(path, sr=None)
    mfcc_feature = mfcc(sig=sig,
                fs=fs,
                num_ceps= 13,
                nfilts=24,
                nfft=512,
                low_freq=0,
                high_freq=2000,
                dct_type=2,
                use_energy=False,
                lifter=5,
                normalize=False)
    mfcc_feature = preprocessing.scale(mfcc_feature)  #对特征进行预处理 标准化
    deltas = delta(mfcc_feature, 2)
    double_deltas = delta(deltas, 2)
    combined = np.hstack((mfcc_feature, deltas, double_deltas))
    return combined

#保存模型函数
def save_gmm(gmm,name):
    """ 
    Save Gaussian mixture model using pickle.
    Args:
        gmm        : Gaussian mixture model.
        name (str) : File name.
    """
    filename = name + ".gmm"
    with open(filename, "wb") as  gmm_file:
        pickle.dump(gmm, gmm_file)

#判别函数
def identify(vector):
        # female hypothesis scoring
        is_spoof_scores         = np.array(spoof_gmm.score(vector))
        is_spoof_log_likelihood = is_spoof_scores.sum()
        # male hypothesis scoring
        is_bonafide_scores         = np.array(bonafides_gmm.score(vector))
        is_bonafide_log_likelihood = is_bonafide_scores.sum()

        #print("%10s %5s %1s" % ("+ SPOOF SCORE",":", str(round(is_spoof_log_likelihood, 3))))
        #print("%10s %7s %1s" % ("+ BONAFIDE SCORE", ":", str(round(is_bonafide_log_likelihood,3))))

        if is_spoof_log_likelihood > is_bonafide_log_likelihood:
            winner = "spoof"
        else:
            winner = "bonafide"
        return winner

#####################训练部分#########################
#读取train.txt 从中分别提取出自然语音文件的文件名 和合成语音的文件名
fpath = 'train.txt'
spoofs = []
bonafides = []
with open(fpath, 'r') as f:
    for line in f.readlines():
        x = line.strip().split(' ', 1)
        if line.strip().endswith('spoof'):
            #print(line.strip())
            #print(x)
            #print(x[0])
            spoofs.append(x[0])
        else:
            bonafides.append(x[0])

#分别对自然语音和合成语音进行特征提取和整合
features_spoof=np.asarray(())
for f in spoofs: 
    # compute features
    vector = get_Mfcc(source + f + '.flac')
    if features_spoof.size == 0:
        features_spoof = vector
    else:
        features_spoof = np.vstack((features_spoof, vector))
#自然声 
features_bonafide=np.asarray(())
for f in bonafides:
    vector = get_Mfcc(source + f + '.flac')
    if features_bonafide.size == 0:
        features_bonafide = vector
    else:
        features_bonafide = np.vstack((features_bonafide, vector))

#利用特征训练处自然声音和环境声的模型 并存储为文件
# generate gaussian mixture models
spoof_gmm = GMM(n_components = 8, max_iter = 200, covariance_type='diag', n_init = 3)
bonafides_gmm = GMM(n_components = 8, max_iter = 200, covariance_type='diag', n_init = 3)
# fit features to models 训练模型
spoof_gmm.fit(features_spoof)
bonafides_gmm.fit(features_bonafide)
# save models
save_gmm(spoof_gmm, "spoof")
save_gmm(spoof_gmm,   "bonafide")

######################dev测试阶段##########################

#利用dev 测试准确率
#利用dict 键值对及对应期望答案 便于后续验证准确率
files={}
for line in open("dev.txt"):
    value = line.split()
    files[value[0]] = value[1]
            
dev_path="dev/flac/"
sum = 0
right = 0
# read the test directory and get the list of test audio files
for file in files:
    sum += 1
    print("%10s %8s %1s" % ("--> TESTING", ":", dev_path + file + ".flac"))
    vector = get_Mfcc(dev_path + file + '.flac')
    winner = identify(vector)
    expected_result = files[file]

    print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_result))
    print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

    if winner == expected_result: 
        right += 1
    print("----------------------------------------------------")
print("sum: " + str(sum))
print("right: " + str(right))
print("accuracy: " + str(float(right) / float(sum) * 100) + "%")

#######################eval1判别阶段############################
#根据模型 对eval 1中音频进行判断并输出结果
eval1_path = "eval1/flac/"
eval1_files = [os.path.join(eval1_path, f) for f in os.listdir(eval1_path)]
eval1_out = open("eval1.txt", "w")
for file in eval1_files:
    vector = get_Mfcc(file)
    predict = identify(vector)
    eval1_out.write(file[11:23] + " " + predict + "\n")  #输出结果到文件中

#######################eval1判别阶段############################
#根据模型 对eval 2中音频进行判断并输出结果
eval2_path = "eval2/flac/"
eval2_files = [os.path.join(eval2_path, f) for f in os.listdir(eval2_path)]
eval2_out = open("eval2.txt", "w")
for file in eval2_files:
    vector = get_Mfcc(file)
    predict = identify(vector)
    eval2_out.write(file[11:23] + " " + predict + "\n")