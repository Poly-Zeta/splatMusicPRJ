#py -3.7 .\record2classifier.py
import cv2
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
import datetime
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
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from scipy.signal import argrelmax

#GPUセットアップ RAM消費を抑えるやつ
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

#曲名のラベル
label = ["batteryfull","chanponchant","chipdamage","dontslip","easyqueasy","endolphinsurge","entropical","finsandfiddles","inkoming","prettytactics","ripentry","seafoamshanty","seasick","shipwreckin","suddentheory","undertow"]

#各種パス
model_path="D:/Users/poly_Z/Documents/splatmusicprj/modelSampleV2.h5"#楽曲判別モデル
sampleDataFileName="admsample"#レコード，画像ファイル用仮置きファイル名
# audio_output_path = "D:/Users/poly_Z/Documents/splatmusicprj/wavNewInput/"+sampleDataFileName+".wav"#レコードファイル
# image_output_path="D:/Users/poly_Z/Documents/splatmusicprj/wavNewOutput/"+sampleDataFileName+".png"#メルスペクトログラム画像
# audio_output_path = "D:/Users/poly_Z/Documents/splatmusicprj/samplingDatas/"+sampleDataFileName#+".wav"#レコードファイル
audio_output_path = "D:/Users/poly_Z/Music/splat10sAutoLog/"+sampleDataFileName#+".wav"#レコードファイル
image_output_path="D:/Users/poly_Z/Documents/splatmusicprj/wavNewOutput/"+sampleDataFileName#+".png"#メルスペクトログラム画像

#学習モデルのロード
model=load_model(model_path)

samplingCounter=0

#録音デバイスの設定
recDevName0='デジタル オーディオ インターフェイス (ezcap U3 c'#候補1
recDevName1='CABLE Output (VB-Audio Virtual '#候補2
recHostApi=0#MMEで録音
recSampleRate=44100.0#サンプリングレート
devHit=False#検知フラグ
pa = pyaudio.PyAudio()#設定用に開く
for recDevNum in range(pa.get_device_count()):
    currdevice=pa.get_device_info_by_index(recDevNum)
    print(currdevice)
    if(currdevice["name"]==recDevName0 and currdevice["hostApi"]==recHostApi and currdevice["defaultSampleRate"]==recSampleRate):
        print("rec device ready")
        devHit=True
        break
if(not devHit):
    for recDevNum in range(pa.get_device_count()):
        currdevice=pa.get_device_info_by_index(recDevNum)
        print(currdevice)
        if(currdevice["name"]==recDevName1 and currdevice["hostApi"]==recHostApi and currdevice["defaultSampleRate"]==recSampleRate):
            print("rec device ready")
            devHit=True
            break
    if(not devHit):
        print("cant found input")
# pa.terminate()

#録音用固定値
DEVICE_INDEX = recDevNum
CHUNK = 1024
FORMAT = pyaudio.paInt16# 16bit
CHANNELS = 2
RATE = int(recSampleRate)# sampling frequency [Hz]
recTime = 10# record time [s]
dt = 1/RATE
freq = np.linspace(0,1.0/dt,CHUNK)

print("setup ready")

#カメラ設定
cap = cv2.VideoCapture(1)#キャプボ
print(cap.isOpened())
width=728#リサイズ設定
height=410#リサイズ設定
threshold=220#2値化閾値
img = cv2.imread("D:/Users/poly_Z/Documents/splatmusicprj/threshPicture.jpg", 0)#検知対象画像

#検知用に成形しておく
img = cv2.resize(img,(width,height))
ret,img=cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
# print(img.shape)
cv2.rectangle(img, (0, 0), (215, height), (0,0,0),thickness=-1)
cv2.rectangle(img, (500, 0), (width, height), (0,0,0),thickness=-1)
cv2.rectangle(img, (216, 0), (499, 75), (0,0,0),thickness=-1)
cv2.rectangle(img, (216, 201), (499, height), (0,0,0),thickness=-1)

# captureX, captureY = 0, 0
# captureW, captureH = 513, 384
image_size = 200

print("stream ready")

# cv2.imshow('frame',img)#->ただの画像でもキー入力受けてくれるので，画面表示多分こっちのがいい
while 1:
    ret,frame = cap.read()
    # print(frame.shape)
    # frame = frame[captureY:captureY+captureH,captureX:captureX+captureW]
    # print("###",frame.shape)
    frame = cv2.resize(frame,(width, height))
    # cv2.imshow('frame',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,gray=cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    cv2.rectangle(gray, (0, 0), (215, height), (0,0,0),thickness=-1)
    cv2.rectangle(gray, (500, 0), (width, height), (0,0,0),thickness=-1)
    cv2.rectangle(gray, (216, 0), (499, 75), (0,0,0),thickness=-1)
    cv2.rectangle(gray, (216, 201), (499, height), (0,0,0),thickness=-1)
    # print(gray.shape)
    # print(np.count_nonzero(gray==img))
    if(np.count_nonzero(gray==img)>295000):
        # cap.release()
        # cv2.destroyAllWindows()
        print("stream ready")
        print("game start##################################################")
        # time.sleep(0.1)
        frames = []
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            # output_device_index = DEVICE_INDEX_O,
            input=True, 
            # output=True,
            input_device_index = DEVICE_INDEX,
            frames_per_buffer=CHUNK
        )
        print("recording ...")
        for i in range(0, int(RATE / CHUNK * recTime)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("done.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(audio_output_path+str(samplingCounter)+".wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        x, fs = librosa.load(audio_output_path+str(samplingCounter)+".wav")
        out=librosa.feature.melspectrogram(x, sr=fs)
        fig = plt.figure()
        librosa.display.specshow(out, sr=fs)
        fig.savefig(image_output_path+".png")
        plt.close()
        
        # print("hello CUDA")
        X = []
        Y = []
        image = Image.open(image_output_path+".png")
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        X = np.array(X)
        X = X.astype('float32')
        X = X / 255.0
        pred = model.predict(X)
        ret,frame = cap.read()
        score = np.max(pred)
        pred_label = label[np.argmax(pred[0])]
        # print("accName:",randomAcc)
        print('choiceName:',pred_label)
        print('score:',score)
        os.rename(audio_output_path+str(samplingCounter)+".wav",audio_output_path+str(datetime.datetime.now().strftime('%y%m%d%H%M%S'))+str(samplingCounter)+pred_label+".wav")
        samplingCounter=samplingCounter+1
        frames = []
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            # output_device_index = DEVICE_INDEX_O,
            input=True, 
            # output=True,
            input_device_index = DEVICE_INDEX,
            frames_per_buffer=CHUNK
        )
        print("Press q Key...")
        while 1:
            # data = stream.read(CHUNK)
            # stream.write(data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # cap = cv2.VideoCapture(1)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print(cap.isOpened())
        print("next")
        ret,frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break



cap.release()
cv2.destroyAllWindows()

stream.stop_stream()
stream.close()
p.terminate()