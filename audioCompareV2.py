
#再生位置推定のデモ

#既知音源Aとストリームから入力したBを保持しておく
#A,Bは同一のタイミングでトリガされて録音を開始した，等幅のデータとする
#(Audacityで両方読み込んで片方に位相反転フィルタかけてみて，BGMが消えるデータならok)

#Bの特定の再生地点tから，幅dtのデータ=1CHUNKをリストに保持->B'
#Aはdtの中央から前後に1CHUNK程度のデータを読み込んでおく->A'
#A'の各次元とB'を比較し，もっとも距離(各項目の差分)の近いもののタイミングを算出

#そのタイミングに基づいてシフトした(A-B)音源を生成し，再生する

#          |    t---dt---|   |
# -----------------------------------------------------------
#   |      |####|###     |   |各行ごとにスタートをずらし
# A |      |    |########|   |B'と比較していく
#   |      |    |    ####|###|
# -----------------------------------------------------------
#   |           |########|
# B |           |########|
#   |           |########|
# -----------------------------------------------------------


import pyaudio
import wave
import time
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import math
import pprint

#オートトリガで取得された同BGMを読ませる
# input_path_A="D:/Users/poly_Z/Music/splat10sAutoV1/chipdamage_AC.wav"
# input_path_B="D:/Users/poly_Z/Music/splat10sAutoV1/chipdamage_A01.wav"
# input_path_A="D:/Users/poly_Z/Music/splat10sAutoV1/suddentheory_AC.wav"
# input_path_B="D:/Users/poly_Z/Music/splat10sAutoV1/suddentheory_A01.wav"
input_path_A="D:/Users/poly_Z/Music/splat10sAutoV1/shipwreckin_AC.wav"
input_path_B="D:/Users/poly_Z/Music/splat10sAutoV1/shipwreckin_A02.wav"


#読み込み形式 channel2で読み込んでいる(実際の使用環境に合わせた)
FORMAT = pyaudio.paInt16# 16bit
CHANNELS = 2
RATE = 44100

#wavファイル長 秒で指定
fileLength=10

#開始時刻　ここを起点にBから1CHUNK切り出してB'を作る
startT=random.randint(1,9)#sec

#1CHUNKのデータ幅
CHUNK =8192

#必要なリストを作っておく
listA=[]

listB=[]

#音源Aの読み取り
wr=wave.open(input_path_A,'r')
data=wr.readframes(wr.getnframes())
wr.close()
listA=np.frombuffer(data,dtype=np.int16)
listA=np.reshape(listA,(len(listA)//2,2)).T
print(listA[0],len(listA[0]))
print(listA[1],len(listA[1]))

#音源Bの読み取り
wr=wave.open(input_path_B,'r')
data=wr.readframes(wr.getnframes())
wr.close()
listB=np.frombuffer(data,dtype=np.int16)
listB=np.reshape(listB,(len(listB)//2,2)).T
print(listB[0],len(listB[0]))

# target_sig = listA[0] * 1.0
# delay = 800
# sig1 = np.random.normal(size=2000) * 0.2
# sig1[delay:delay+1000] += target_sig
# sig2 = np.random.normal(size=2000) * 0.2
# sig2[:1000] += target_sig
########################################################################

startNum=int(startT/10*len(listA[0]))
sig1L = listA[0][startNum:startNum+CHUNK]#+listA[1][startNum:startNum+CHUNK]
sig1L = sig1L - sig1L.mean()
sig2L = listB[0][startNum:startNum+CHUNK]#+listB[1][startNum:startNum+CHUNK]
sig2L = sig2L - sig2L.mean()
sig1R = listA[1][startNum:startNum+CHUNK]#+listA[1][startNum:startNum+CHUNK]
sig1R = sig1R - sig1R.mean()
sig2R = listB[1][startNum:startNum+CHUNK]#+listB[1][startNum:startNum+CHUNK]
sig2R = sig2R - sig2R.mean()

sig1L=np.array(sig1L)
sig1R=np.array(sig1R)
sig2L=np.array(sig2L)
sig2R=np.array(sig2R)

########################################################################

sig1Lfft=np.fft.fft(sig1L)
sig1Rfft=np.fft.fft(sig1R)
sig2Lfft=np.fft.fft(sig2L)
sig2Rfft=np.fft.fft(sig2R)

sig1Lfft=np.array(sig1Lfft)
sig1Rfft=np.array(sig1Rfft)
sig2Lfft=np.array(sig2Lfft)
sig2Rfft=np.array(sig2Rfft)

########################################################################

sig1Lamp=np.abs(sig1Lfft)/len(listA[0])*2
sig1Ramp=np.abs(sig1Rfft)/len(listA[0])*2
sig2Lamp=np.abs(sig2Lfft)/len(listA[0])*2
sig2Ramp=np.abs(sig2Rfft)/len(listA[0])*2

sig1Lamp=np.array(sig1Lamp)
sig1Ramp=np.array(sig1Ramp)
sig2Lamp=np.array(sig2Lamp)
sig2Ramp=np.array(sig2Ramp)

sig1Lamp[0]=sig1Lamp[0]/2
sig1Ramp[0]=sig1Ramp[0]/2
sig2Lamp[0]=sig2Lamp[0]/2
sig2Ramp[0]=sig2Ramp[0]/2

########################################################################

# 1        ____________
#          |          |
#          |          |
#0________________________________
#          fl         fh


# fh=300
# fl=50
# fq=np.linspace(0,RATE,len(sig1L))
# # print(np.shape(sig1Lfft))
# sig1Lfft[(fq>fh)]=0
# sig1Rfft[(fq>fh)]=0
# sig2Lfft[(fq>fh)]=0
# sig2Rfft[(fq>fh)]=0

# sig1Lfft[(fq<fl)]=0
# sig1Rfft[(fq<fl)]=0
# sig2Lfft[(fq<fl)]=0
# sig2Rfft[(fq<fl)]=0

# al=0.5
# sig1Lfft[(sig1Lamp<al)]=0
# sig1Rfft[(sig1Ramp<al)]=0
# sig2Lfft[(sig2Lamp<al)]=0
# sig2Rfft[(sig2Ramp<al)]=0

########################################################################

sig1fft=sig1Rfft+sig1Lfft
sig2fft=sig2Rfft+sig2Lfft
# sig1fft[()]

sig1=np.fft.ifft(sig1fft)
sig2=np.fft.ifft(sig2fft)
sig1=sig1.real
sig2=sig2.real

########################################################################

corr = np.correlate(sig1, sig2, "full")
estimated_delay = corr.argmax() - (len(sig2) - 1)
print("estimated delay is " + str(estimated_delay))

plt.subplot(4, 1, 1)
plt.ylabel("sig1")
plt.plot(sig1)

plt.subplot(4, 1, 2)
plt.ylabel("sig2")
plt.plot(sig2, color="g")

plt.subplot(4, 1, 3)
plt.ylabel("fit")
plt.plot(np.arange(len(sig1)), sig1)
plt.plot(np.arange(len(sig2)) + estimated_delay, sig2 )
plt.xlim([0, len(sig1)])

plt.subplot(4, 1, 4)
plt.ylabel("corr")
plt.plot(np.arange(len(corr)) - len(sig2) + 1, corr, color="r")
plt.xlim([0, len(sig1)])

plt.show()

diffT=estimated_delay
if(diffT>0):
    listA_0=np.array(listA[0][:-diffT])
    listA_1=np.array(listA[1][:-diffT])
    listB_0=np.array(listB[0][diffT:])
    listB_1=np.array(listB[1][diffT:])
elif(diffT<0):
    listA_0=np.array(listA[0][:diffT])
    listA_1=np.array(listA[1][:diffT])
    listB_0=np.array(listB[0][-diffT:])
    listB_1=np.array(listB[1][-diffT:])
else:
    listA_0=np.array(listA[0][:])
    listA_1=np.array(listA[1][:])
    listB_0=np.array(listB[0][:])
    listB_1=np.array(listB[1][:])

# listO=np.array([listB_0-listA_0,listB_1-listA_1])
listO=np.array([listB_0-listA_0,listB_1-listA_1])

audio_output_path="D:/Users/poly_Z/Documents/splatmusicprj/wavNewOutput/sampledft.wav"

p = pyaudio.PyAudio()
wf = wave.open(audio_output_path, 'wb')
wf.setnchannels(2)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
listO=np.reshape(listO.T,(len(listO[0])*2))
wf.writeframes(b''.join(listO))
wf.close()


# plt.show()
