import os
import gc
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment

def file_check(inputPath,outputPath):
    i=0
    updlist=[]
    for pathname, dirnames, filenames in os.walk(inputPath):
        for filename in filenames:
            i=i+1
            # print(str(filename)+" "+str(i)+"/"+str(len(filenames)))
            if(os.path.exists(outputPath+filename+"img.png")):
                print(str(filename)+"->skip "+str(i)+"/"+str(len(filenames)))
                continue
            else:
                # print(filename)
                x, fs = librosa.load(inputPath+filename)
                # out = librosa.feature.mfcc(x, sr=fs)
                out=librosa.feature.melspectrogram(x, sr=fs)
                fig = plt.figure()
                librosa.display.specshow(out, sr=fs)#, x_axis='time')
                # plt.colorbar()
                fig.savefig(outputPath+filename+"img.png")
                plt.close()
                # plt.show()
                print(str(filename)+"->output "+str(i)+"/"+str(len(filenames)))
                del x,fs,out,fig
                gc.collect()
                updlist.append(filename)
            # break
        # break
    return updlist


if(__name__ == '__main__'):
    inpath="D:/Users/poly_Z/Music/splat10sV2/"
    outpath="D:/Users/poly_Z/Documents/splatmusicprj/fig_v2_logmel/"
    news=file_check(inpath,outpath)
    print("complete")
    for newone in news:
        print(newone)
    print("add:"+str(len(news)))
    # print(news)