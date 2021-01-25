
#単体 wav入力をpngに変換し，曲名を判別する
#py -3.7 .\record2classifier.py
import pyaudio
import wave
import time

import os
# import gc
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D 
from keras.datasets import mnist
# import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import load_model
import random



label = ["batteryfull","chanponchant","chipdamage","dontslip","easyqueasy","endolphinsurge","entropical","finsandfiddles","inkoming","prettytactics","ripentry","seaformshanty","seasick","shipwreckin","suddentheory","undertow"]
# randomAcc=random.choice(label)
# inputFilePath="D:/Users/poly_Z/Documents/splatmusicprj/wavNewInput/"
# outputFilePath="D:/Users/poly_Z/Documents/splatmusicprj/wavNewOutput/"
sampleDataFileName="sample"
audio_output_path = "D:/Users/poly_Z/Documents/splatmusicprj/wavNewInput/"+sampleDataFileName+".wav"
image_output_path="D:/Users/poly_Z/Documents/splatmusicprj/wavNewOutput/"+sampleDataFileName+".png"
model_path="D:/Users/poly_Z/Documents/splatmusicprj/modelSample.h5"
# fileName=inputFilePath+randomAcc+"_01.wav"
# imgfile=outputFilePath+randomAcc+"_01.wavimg.png"

DEVICE_INDEX = 5
CHUNK = 1024
FORMAT = pyaudio.paInt16 # 32bit
CHANNELS = 2             # monaural
RATE = 44100             # sampling frequency [Hz]

time = 10 # record time [s]

while(1):
    frames = []
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index = DEVICE_INDEX,
        frames_per_buffer=CHUNK
    )
    print("standby")
    input_data = input()
    print("recording ...")


    for i in range(0, int(RATE / CHUNK * time)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("done.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(audio_output_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    x, fs = librosa.load(audio_output_path)
    out=librosa.feature.melspectrogram(x, sr=fs)
    fig = plt.figure()
    librosa.display.specshow(out, sr=fs)
    fig.savefig(image_output_path)
    plt.close()

    model=load_model(model_path)
    image_size = 200

    print("hello CUDA")

    X = []
    Y = []
    image = Image.open(image_output_path)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X.append(data)
    X = np.array(X)
    X = X.astype('float32')
    X = X / 255.0
    pred = model.predict(X)
    score = np.max(pred)
    pred_label = label[np.argmax(pred[0])]
    # print("accName:",randomAcc)
    print('choiceName:',pred_label)
    print('score:',score)


