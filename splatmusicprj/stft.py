import os
import gc
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np

dpath="D:/Users/poly_Z/Music/splat10sV2/"
cpath="D:/Users/poly_Z/Documents/splatmusicprj/stftV2/"
def file_check(path):
    i=0
    updlist=[]
    for pathname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            i=i+1
            # print(str(filename)+" "+str(i)+"/"+str(len(filenames)))
            if(os.path.exists(cpath+filename+"img.png")):
                print(str(filename)+"->skip "+str(i)+"/"+str(len(filenames)))
                continue
            else:
                # print(filename)
                x, fs = librosa.load(path+filename)
                S = np.abs(librosa.stft(x))
                # out = librosa.feature.mfcc(x, sr=fs)
                # out=librosa.feature.melspectrogram(x, sr=fs)
                fig = plt.figure()
                # librosa.display.specshow(out, sr=fs)#, x_axis='time')
                librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max))
                # plt.colorbar()
                fig.savefig(cpath+filename+"img.png")
                plt.close()
                # plt.show()
                print(str(filename)+"->output "+str(i)+"/"+str(len(filenames)))
                del x,fs,S,fig
                gc.collect()
                updlist.append(filename)
            # break
        # break
    return updlist



news=file_check(dpath)
print("complete")
for newone in news:
    print(newone)
print("add:"+str(len(news)))
# print(news)