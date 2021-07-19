from feature_extraction import FeatureExtraction
import os
import numpy as np


def predictModel(model, output_model_file):
    # load the weights of model
    model.load_weights(output_model_file)
    features = []
    test_dir = r'C:\Users\11201\Downloads\zju_deepfake\eval1\flac'
    # test_dir = r'D:\Users\11201\Downloads\data_aishell\data_aishell\wav\train\S0002'
    res = []
    # with open("result.txt", "w") as f:
    num = 0
    fname = []
    for root, path, files in os.walk(test_dir):
        for file in files:
            num += 1
            audio_path = os.path.join(root, file)
            # feature = FeatureExtraction(path=audio_path).vector_199_40().reshape(199, 40, 1)
            # # print(feature.shape)
            # pre = model.predict(feature)
            # if pre[0][0] > pre[0][1]:
            #     res.append((audio_path.split("\\")[-1].split(".")[0], "spoof"))
            #     print(audio_path, ' is classified as Spoof.')
            # else:
            #     res.append((audio_path.split("\\")[-1].split(".")[0], "bonafide"))
            #     print(audio_path, ' is classified as Bonafide.')
            fname.append(audio_path.split("\\")[-1].split(".")[0])
            feature = FeatureExtraction(path=audio_path).vector_199_40().reshape(199, 40, 1)
            features.append(feature)
    features = np.stack(features, axis=0)
    pre = model.predict(features)
    pre = np.argmax(pre, axis=1)
    spoof = 0
    with open("result0716.txt", "w") as f:
        for i in range(len(pre)):
            f.writelines(fname[i] + " " + ("spoof" + "\n" if pre[i] == 0 else "bonafide" + "\n"))
            spoof += pre[i] == 0
    print(spoof)


if __name__ == "__main__":
    predictModel("", "")
    # from numpy import array
    # from numpy import hstack
    # from numpy import insert
    # from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    #
    # # 给定数据
    # in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
    # out_seq = array([25, 45, 65, 85, 105, 125, 145, 165, 185, 205])
    # in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    # in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    # out_seq = out_seq.reshape((len(out_seq), 1))
    # # 按行的方向进行堆叠
    # dataset = hstack((in_seq1, in_seq2))
    # print(dataset)
