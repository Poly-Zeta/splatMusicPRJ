# 自動分類&戦績表示のプロト版
# python3.7,tensorflow,kerasで実行
# キャプボ，グラボ，学習済みモデル，各種入出力ファイルやディレクトリがある前提


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
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras import optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import ImageFont, ImageDraw, Image
import glob
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from scipy.signal import argrelmax

# myModule
import urlKnocker
import widgetWindow


# img_blank = cv2.imread(
#     "D:/Users/poly_Z/Documents/splatmusicprj/cv2Pictures/blank.png", 0)
blankLogtxt = []
widgetWidth, widgetHeight, widgetSep = widgetWindow.windowSize()
baseImg = np.zeros([widgetHeight, widgetWidth, 3])

blankLogtxt, baseImg = widgetWindow.print4imgBlank(
    "GPU SETUP", blankLogtxt, baseImg)
# GPUセットアップ RAM消費を抑えるやつ
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(
            device, tf.config.experimental.get_memory_growth(device)))
    blankLogtxt, baseImg = widgetWindow.print4imgBlank(
        "OK", blankLogtxt, baseImg)
else:
    print("Not enough GPU hardware devices available")

blankLogtxt, baseImg = widgetWindow.print4imgBlank(
    "SETUP", blankLogtxt, baseImg)
# 曲名のラベル
label = ["batteryfull", "chanponchant", "chipdamage", "dontslip", "easyqueasy", "endolphinsurge", "entropical", "finsandfiddles",
         "inkoming", "prettytactics", "ripentry", "seafoamshanty", "seasick", "shipwreckin", "suddentheory", "undertow"]

# 各種パス
model_path = "D:/Users/poly_Z/Documents/splatmusicprj/modelSampleV3.h5"  # 楽曲判別モデル
sampleDataFileName = "admsample"  # レコード，画像ファイル用仮置きファイル名
# audio_output_path = "D:/Users/poly_Z/Documents/splatmusicprj/wavNewInput/"+sampleDataFileName+".wav"#レコードファイル
# image_output_path="D:/Users/poly_Z/Documents/splatmusicprj/wavNewOutput/"+sampleDataFileName+".png"#メルスペクトログラム画像
# audio_output_path = "D:/Users/poly_Z/Documents/splatmusicprj/samplingDatas/"+sampleDataFileName#+".wav"#レコードファイル
audio_output_path = "D:/Users/poly_Z/Music/splat10sAutoLog/" + \
    sampleDataFileName  # +".wav"#レコードファイル
image_output_path = "D:/Users/poly_Z/Documents/splatmusicprj/wavNewOutput/" + \
    sampleDataFileName  # +".png"#メルスペクトログラム画像

# 学習モデルのロード
model = load_model(model_path)

samplingCounter = 0
blankLogtxt, baseImg = widgetWindow.print4imgBlank("OK", blankLogtxt, baseImg)
blankLogtxt, baseImg = widgetWindow.print4imgBlank(
    "AUDIO SETUP", blankLogtxt, baseImg)
# 録音デバイスの設定
recDevName0 = 'デジタル オーディオ インターフェイス (ezcap U3 c'  # 候補1
recDevName1 = 'CABLE Output (VB-Audio Virtual '  # 候補2
recHostApi = 0  # MMEで録音
recSampleRate = 44100.0  # サンプリングレート
recDevHit = False  # 検知フラグ
pa = pyaudio.PyAudio()  # 設定用に開く
for recDevNum in range(pa.get_device_count()):
    currdevice = pa.get_device_info_by_index(recDevNum)
    print(currdevice)
    if(currdevice["name"] == recDevName0 and currdevice["hostApi"] == recHostApi and currdevice["defaultSampleRate"] == recSampleRate):
        print("rec device ready")
        recDevHit = True
        break
if(not recDevHit):
    for recDevNum in range(pa.get_device_count()):
        currdevice = pa.get_device_info_by_index(recDevNum)
        print(currdevice)
        if(currdevice["name"] == recDevName1 and currdevice["hostApi"] == recHostApi and currdevice["defaultSampleRate"] == recSampleRate):
            print("rec device ready")
            devHit = True
            break
    if(not recDevHit):
        print("cant found input")
# pa.terminate()

# 録音用固定値
DEVICE_INDEX = recDevNum
CHUNK = 1024
FORMAT = pyaudio.paInt16  # 16bit
CHANNELS = 2
RATE = int(recSampleRate)  # sampling frequency [Hz]
recTime = 10  # record time [s]
dt = 1/RATE
freq = np.linspace(0, 1.0/dt, CHUNK)
blankLogtxt, baseImg = widgetWindow.print4imgBlank("OK", blankLogtxt, baseImg)
print("setup ready")

blankLogtxt, baseImg = widgetWindow.print4imgBlank(
    "CAM SETUP", blankLogtxt, baseImg)
# カメラ設定
cap = cv2.VideoCapture(1)  # キャプボ
print(cap.isOpened())
width = 728  # リサイズ設定
height = 410  # リサイズ設定
threshold = 220  # 2値化閾値
img_start = cv2.imread(
    "D:/Users/poly_Z/Documents/splatmusicprj/cv2Pictures/startThreshPicture.jpg", 0)  # 開始点検知対象画像
img_end = cv2.imread(
    "D:/Users/poly_Z/Documents/splatmusicprj/cv2Pictures/finishThreshPicture.jpg", 0)
# endth=11

# 検知用に成形しておく
img_start = cv2.resize(img_start, (width, height))
# ret,img_start=cv2.threshold(img_start, threshold, 255, cv2.THRESH_BINARY)
img_end = cv2.resize(img_end, (width, height))
# cv2.imshow('frame',img_end)
# cv2.waitKey()
# cv2.destroyAllWindows()
# print(img.shape)
# cv2.rectangle(img_start, (0, 0), (215, height), (0,0,0),thickness=-1)
# cv2.rectangle(img_start, (500, 0), (width, height), (0,0,0),thickness=-1)
# cv2.rectangle(img_start, (216, 0), (499, 75), (0,0,0),thickness=-1)
# cv2.rectangle(img_start, (216, 201), (499, height), (0,0,0),thickness=-1)
# captureX, captureY = 0, 0
# captureW, captureH = 513, 384
image_size = 200
blankLogtxt, baseImg = widgetWindow.print4imgBlank("OK", blankLogtxt, baseImg)
print("stream ready")

blankLogtxt, baseImg = widgetWindow.print4imgBlank(
    "READY", blankLogtxt, baseImg)
blankLogtxt, baseImg = widgetWindow.print4imgBlank(
    "SEARCHING 'GO!'", blankLogtxt, baseImg)
blankLogtxt, baseImg = widgetWindow.print4imgBlank(
    "MANUAL START:S", blankLogtxt, baseImg)
blankLogtxt, baseImg = widgetWindow.print4imgBlank(
    "MANUAL ITRPT:Ctrl+C", blankLogtxt, baseImg)
# cv2.imshow('frame',img_blank_c)#->ただの画像でもキー入力受けてくれるので，画面表示多分こっちのがいい
mainLoopCounter = 0
getJsonOptionCode = 200
while 1:
    try:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        colchk = np.count_nonzero(gray == img_start)
        print(colchk)  # printでログが荒れるのはわかってるけどなぜか無いと安定しない
        if((colchk > 280000) or (cv2.waitKey(1) & 0xFF == ord('s'))):
            # if((np.count_nonzero(gray==img_start)>285000) or (cv2.waitKey(1) & 0xFF == ord('s'))):
            # blankLogtxt=print4imgBlank("GAME START",blankLogtxt)
            frames = []
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                # output_device_index = DEVICE_INDEX_O,
                input=True,
                # output=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK
            )
            # cv2.destroyAllWindows()
            # cv2.imshow('frame',img_start)
            # cap.release()
            # cv2.destroyAllWindows()
            print("stream ready")
            print("game start")
            print("recording ...")
            for i in range(0, int(RATE / CHUNK * recTime)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("done.")
            stream.stop_stream()
            stream.close()
            p.terminate()
            blankLogtxt, baseImg = widgetWindow.print4imgBlank(
                "DONE", blankLogtxt, baseImg)
            wf = wave.open(audio_output_path+str(samplingCounter)+".wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            x, fs = librosa.load(audio_output_path+str(samplingCounter)+".wav")
            out = librosa.feature.melspectrogram(x, sr=fs)
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
            ret, frame = cap.read()
            score = np.max(pred)
            pred_label = label[np.argmax(pred[0])]
            # print("accName:",randomAcc)
            blankLogtxt, baseImg = widgetWindow.print4imgBlank(
                'LABEL:'+pred_label, blankLogtxt, baseImg)
            blankLogtxt, baseImg = widgetWindow.print4imgBlank(
                'SCORE:'+str(score), blankLogtxt, baseImg)
            print('choiceName:', pred_label)
            print('score:', score)
            os.rename(audio_output_path+str(samplingCounter)+".wav", audio_output_path+str(
                datetime.datetime.now().strftime('%y%m%d%H%M%S'))+str(samplingCounter)+pred_label+".wav")
            samplingCounter = samplingCounter+1
            # frames = []
            # p = pyaudio.PyAudio()
            # stream = p.open(
            #     format=FORMAT,
            #     channels=CHANNELS,
            #     rate=RATE,
            #     # output_device_index = DEVICE_INDEX_O,
            #     input=True,
            #     # output=True,
            #     input_device_index = DEVICE_INDEX,
            #     frames_per_buffer=CHUNK
            # )
            blankLogtxt, baseImg = widgetWindow.print4imgBlank(
                "SEARCHING 'FINISH'", blankLogtxt, baseImg)
            blankLogtxt, baseImg = widgetWindow.print4imgBlank(
                "MANUAL END:E", blankLogtxt, baseImg)
            print("Press e Key...")
            while 1:
                ret, frame = cap.read()
                frame = cv2.resize(frame, (width, height))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                compareGrey = np.count_nonzero(gray == img_end)
                print(colchk)
                if ((compareGrey > 266500) or (cv2.waitKey(1) & 0xFF == ord('e'))):
                    blankLogtxt, baseImg = widgetWindow.print4imgBlank(
                        "GAME END", blankLogtxt, baseImg)
                    print("game end")
                    break
            # cv2.destroyAllWindows()
            # cv2.imshow('frame',img_end)
            # stream.stop_stream()
            # stream.close()
            # p.terminate()
            blankLogtxt, baseImg = widgetWindow.print4imgBlank(
                "SEARCHING 'GO!'", blankLogtxt, baseImg)
            blankLogtxt, baseImg = widgetWindow.print4imgBlank(
                "MANUAL START:S", blankLogtxt, baseImg)
            blankLogtxt, baseImg = widgetWindow.print4imgBlank(
                "MANUAL ITRPT:Ctrl+C", blankLogtxt, baseImg)

            ###############################################################################
            # 戦績表示

            # 試合が終了してからAPIが応答してくれる時間まで待つ
            for i in range(16):
                time.sleep(1)
                print("waitnow "+str(i))
            # 直近50戦の簡易データを取得
            jsonData, getJsonOptionCode = urlKnocker.getJson(
                "https://app.splatoon2.nintendo.net/api/results", getJsonOptionCode
            )
            # 内容から最新の試合番号を返す
            buttleNumber, getJsonOptionCode = urlKnocker.getNewestButtleNumber(
                jsonData,
                getJsonOptionCode
            )
            # 最新の試合の詳細データを取得
            jsonData, getJsonOptionCode = urlKnocker.getJson(
                "https://app.splatoon2.nintendo.net/api/results/" +
                str(buttleNumber),
                getJsonOptionCode
            )
            # データの成形
            buttleResult, buttleRule = urlKnocker.getNewestButtleResult(
                jsonData,
                getJsonOptionCode
            )
            # データの表示
            baseImg = widgetWindow.printButtleResult(
                buttleResult,
                buttleRule,
                baseImg
            )
            ###############################################################################

            print("next")
    # elif (cv2.waitKey(1) & 0xFF == ord('i')):
    #     break
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        stream.stop_stream()
        stream.close()
        p.terminate()
        pa.terminate()
        print("end")
        break
