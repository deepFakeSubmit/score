import os
import sys
import itertools
import glob
import argparse
from utils import read_wav
from interface import ModelInterface

#设置命令行参数
def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """ """
    parser = argparse.ArgumentParser(description=desc,
                                     epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "enroll" or "predict" or "test"',
                       required=True)

    parser.add_argument('-i', '--input',
                       help='Input Files(to predict) or Directories(to enroll)',
                       required=True)

    parser.add_argument('-m', '--model',
                       help='Model file to save(in enroll) or use(in predict)',
                       required=False)

    ret = parser.parse_args()
    return ret

#用户注册，训练GMM模型时使用
def task_enroll(input_dirs, output_model):
    m = ModelInterface()
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]

    files = []
    if len(dirs) == 0:
        print("No valid directory found!")
        sys.exit(1)

    for d in dirs:
        label = os.path.basename(d.rstrip('/'))
        wavs = glob.glob(d + '/*.wav')

        if len(wavs) == 0:
            print("No wav file found in %s"%(d))
            continue
        for wav in wavs:
            try:
                fs, signal = read_wav(wav)
                m.enroll(label, fs, signal)
                print("wav %s has been enrolled"%(wav))
            except Exception as e:
                print(wav + " error %s"%(e))

    m.train()
    m.dump(output_model)

#对测试数据进行验证计算
def task_test1(input_dirs):
        m1 = ModelInterface.load("T.out")
        m2 = ModelInterface.load("F.out")
        input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
        dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
        dirs = [d for d in dirs if os.path.isdir(d)]

        if len(dirs) == 0:
            print("No valid directory found!")
            sys.exit(1)

        c1 = 0
        c2 = 0
        for d in dirs:
            wavs = glob.glob(d + '/*.wav')
            for wav in wavs:
                c1 += 1
                print(wav)
                fs, signal = read_wav(wav)
                label1, score1 = m1.predict(fs, signal)
                print(wav, '->', label1, ", score->", score1)
                label2, score2 = m2.predict(fs, signal)
                print(wav, '->', label2, ", score->", score2)

                if (score1 < score2):
                    judge = "bonafide"
                else:
                    judge = "spoof"
                print(judge)
                name = wav.split('\\')[-1].split('.')[0]

                print("name" + name)
                file = open(r'./dev.txt')
                for line in file.readlines():
                    curLine = line.strip().split(" ")
                    print("每一行")
                    print(curLine)
                    if name == curLine[0]:
                        if (curLine[1] == judge):
                            print("c2c2c2c2c2c")
                            c2+=1
        print(c1)
        print(c2)
        acc=c2/c1
        print("模型准确率")
        print(acc)

#对测试数据进行预测
def task_predict(input_dirs):
    m1= ModelInterface.load("T.out")
    m2= ModelInterface.load("F.out")
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]

    if len(dirs) == 0:
        print("No valid directory found!")
        sys.exit(1)
    for d in dirs:
        wavs = glob.glob(d + '/*.wav')
        print(wavs)
        file_handle = open('1.txt', mode='w')
        for wav in wavs:
            print(wav)
            fs, signal = read_wav(wav)
            label1, score1 = m1.predict(fs, signal)
            print(wav, '->', label1, ", score->", score1)
            label2, score2= m2.predict(fs, signal)
            print(wav, '->', label2, ", score->", score2)

            if(score1<score2):
               judge="bonafide"
            else:
                judge="spoof"
            print(judge)
            name=wav.split('\\')[-1].split('.')[0]

            print("name"+name)
            file_handle.write(wav.split('\\')[-1].split('.')[0] + " " + judge+"\n")



if __name__ == "__main__":
    global args
    args = get_args()

    task = args.task
    # 首先，训练出两个GMM模型，需要在命令行进行两次输入
    # 第一次输入：python speaker-recognition.py -t enroll -i "./true" -m "T.out"
    # 第二次输入：python speaker-recognition.py -t enroll -i "./false" -m "F.out
    if task == 'enroll':
        task_enroll(args.input, args.model)
    #用给定的数据集dev，测试训练出的模型的准确率，在命令行输入:
    #python speaker-recognition.py -t predict -i "./testall"
    elif task == 'test':
        task_test1(args.input)
    #用训练出的模型，进行预测，，在命令行输入:
    #python speaker-recognition.py -t predict -i "./testall"
    elif task == 'predict':
        task_predict(args.input)

