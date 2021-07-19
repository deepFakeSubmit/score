# *coding:utf-8 *
import os

#运用ffmpeg多媒体处理工具，将音频文件从.flac格式变为.wav格式
def flac_to_wav(filepath, savedir):
    filename = filepath.replace('.flac', '.wav')
    savefilename = filename.split('\\')
    print(savefilename)
    save_dir = savedir + '\\' + savefilename[-1]
    print(save_dir)
    cmd = r"D:\ffmpeg\ffmpeg-4.4-full_build\bin\ffmpeg.exe -i " + filepath + ' ' + save_dir
   # print(cmd)
    os.system(cmd)

def class_wav(filepath, savedir):
    filename = filepath.replace('.wav', '.wav')
    savefilename = filename.split('\\')
    print(savefilename)
    save_dir = savedir + '\\' + savefilename[-1]
    print(save_dir)
    cmd = r"D:\ffmpeg\ffmpeg-4.4-full_build\bin\ffmpeg.exe -i " + filepath + ' ' + save_dir
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    # 批量处理，将音频文件从.flac格式变为.wav格式
    savedir = r"D:\zju_deepfake\train\dev"    #用户可以改变，为输出文件的位置
    path = r'D:\zju_deepfake\res'      #用户可以改变，为输入文件的位置
    for root, dirs, files in os.walk(path):
        for name in files:
            filepath = root + "\\" + name
            print("filepath"+filepath)
            if filepath.split('.')[-1] == "flac":
                flac_to_wav(filepath, savedir)

    #将train文件夹下的数据分成两个文件夹，一个文件夹存放真实语音，另一个文件夹存放伪造语音，分别在这两个数据集上训练GMM模型
    savedir1 = r"D:\zju_deepfake\true"
    savedir2 = r"D:\zju_deepfake\false"
    path = r'D:\zju_deepfake\res'
    for root, dirs, files in os.walk(path):
        for name in files:
            filepath = root + "\\" + name
            print("filepath"+filepath)
            file = open(r'D:\zju_deepfake\train.txt')
            for line in file.readlines():
                curLine = line.strip().split(" ")
                if name.split('.')[0] == curLine[0]:
                    if(curLine[1]=="spoof"):
                        flac_to_wav(filepath, savedir1)
                    else:
                        flac_to_wav(filepath, savedir2)


