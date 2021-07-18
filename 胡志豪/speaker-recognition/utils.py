from scipy.io import wavfile
import soundfile as sf
#import librosa
def read_wav(fname):
    signal, fs = sf.read(fname)  # 这两个参数换了个位置， 第二个返回值fs代表了 音频的rate
#    fs, signal = wavfile.read(fname);
    print(fname)
    if len(signal.shape) != 1:
        print("convert stereo to mono")
        signal = signal[:,0]
    return fs, signal
