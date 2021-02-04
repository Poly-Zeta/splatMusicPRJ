
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

#オートトリガで取得された同BGMを読ませる
input_path_A="D:/Users/poly_Z/Music/splat10sAutoV1/chipdamage_AC.wav"
input_path_B="D:/Users/poly_Z/Music/splat10sAutoV1/chipdamage_A01.wav"


#読み込み形式 channel2で読み込んでいる(実際の使用環境に合わせた)が，右のみ使う
FORMAT = pyaudio.paInt16# 16bit
CHANNELS = 2
RATE = 44100

#wavファイル長 秒で指定
fileLength=10

#開始時刻　ここを起点にBから1CHUNK切り出してB'を作る
startT=1#random.randint(1,9)#sec
dftList=[]
#1CHUNKのデータ幅
CHUNK =1024

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

for startT in range(1,9):
    listAP=[[],[]]
    freqA=[[],[]]
    ampA=[[],[]]
    listBP=[[],[]]
    freqB=[[],[]]
    ampB=[[],[]]
    # listAの切り出し
    listAPStart=int(len(listA[0])*(startT/fileLength))
    print(listAPStart)
    # print(listA[0][len(listA[0])-1])

    for i in range(0,CHUNK):
        # print(listA[0][listA_pStart-int(CHUNK/4)+i],listA[0][listA_pStart-int(CHUNK/4)+CHUNK*2+i])
        # print((listA[1][listAPStart-int(CHUNK/4)+i:listAPStart-int(CHUNK/4)+int(CHUNK/2)+i]))
        listAP[0].append(listA[0][listAPStart-int(CHUNK/4)+i:listAPStart-int(CHUNK/4)+int(CHUNK/2)+i])
        listAP[1].append(listA[1][listAPStart-int(CHUNK/4)+i:listAPStart-int(CHUNK/4)+int(CHUNK/2)+i])
        # freqA.append(np.fft.fft(listA[0][listA_pStart-int(CHUNK/4)+i:listA_pStart-int(CHUNK/4)+int(CHUNK/2)+i]))
        # ampA.append(np.abs(np.fft.fft(listA[0][listA_pStart-int(CHUNK/4)+i:listA_pStart-int(CHUNK/4)+int(CHUNK/2)+i])))
    listAP=np.array(listAP)
    print(np.shape(listAP))
    print(np.shape(listAP[0][0]))
    for i in range(0,CHUNK):
        freqA[0].append(np.fft.fft(listAP[0][i]))
        freqA[1].append(np.fft.fft(listAP[1][i]))
    freqA=np.array(freqA)
    print("frshape:",np.shape(freqA))
    for i in range(0,CHUNK):
        ampA[0].append(np.abs(freqA[0][i]/len(listAP[0][i])))#*2))
        ampA[1].append(np.abs(freqA[1][i]/len(listAP[1][i])))#*2))
    ampA=np.array(ampA)
    print("ampshape:",np.shape(ampA))
    # freqA=np.array(freqA)
    # ampA=np.array(ampA)
    # print(listA_p[0])
    # print(listA_p[1])
    # print(listA_p[2])
    # print(len(listA_p),len(listA_p[0]))

    # listBの切り出し
    listBPStart=int(len(listB[0])*(startT/fileLength))
    # print(listB_pStart)
    listBP[0]=np.array(listB[0][listBPStart:listBPStart+int(CHUNK/2)])
    listBP[1]=np.array(listB[1][listBPStart:listBPStart+int(CHUNK/2)])
    listBP=np.array(listBP)
    freqB[0]=np.fft.fft(listBP[0])
    freqB[1]=np.fft.fft(listBP[1])
    freqB=np.array(freqB)
    ampB[0]=np.abs(freqB[0]/len(listBP[0]))#*2)
    ampB[1]=np.abs(freqB[1]/len(listBP[1]))#*2)
    ampB=np.array(ampB)
    # freqB=np.array(np.fft.fft(listB_p))
    # ampB=np.array(np.abs(freqB))
    # print(len(listB_p))

    # listA_p[256],listA_p[257]=listA_p[257],listA_p[256]
    # listA_p[256],listA_p[255]=listA_p[255],listA_p[256]

    compare=[]
    graphDataX=[]
    graphDataY=[]
    print(np.shape(ampA))
    # print(listA_p[0])
    # print(listB_p)
    # print(listB_p-listA_p[i])
    # print(abs(listB_p-listA_p[i]))
    for t in range(0,CHUNK):
        diff=0
        graphDataX.append(t)
        for i in range(int(CHUNK/2)):
            # print("t:",t," i:",i)
            clear=abs(ampB[0][i]-ampB[1][i])
            # minA=min(ampA[0][t][i],ampA[1][t][i])
            # minB=min(ampB[0][i]-clear,ampB[1][i]-clear)
            # diff=diff+abs(minA-minB)
            diff=diff+abs(ampA[0][t][i]-(ampB[0][i]-clear))+abs(ampA[1][t][i]-(ampB[1][i]-clear))
        graphDataY.append(diff)
        compare.append(diff)
    print("end")

    # print(compare[254:258])
    diffT=compare.index(min(compare))#-55#start=4のとき-55の補正，diffT=723
    print(min(compare),compare.index(min(compare)),compare[diffT-4:diffT+4],diffT)

    print(listA[0],len(listA[0]))
    print(listB[0],len(listB[0]))


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

    print(listA_0,len(listA_0))
    print(listB_0,len(listB_0))

    # listO_0=listA_0-listB_0
    # listO_1=listA_1-listB_1
    listO=np.array([listB_0-listA_0,listB_1-listA_1])
    # listO=np.array([listB_0*2-listA_0*2,listB_1*2-listA_1*2])
    # listO=np.append(listO,listA_0-listB_0)
    # listO=np.append(listO,listA_1-listB_1)
    print(listO)
    # print(listO.T)

    audio_output_path="D:/Users/poly_Z/Documents/splatmusicprj/wavNewOutput/sample"+startT+".wav"

    p = pyaudio.PyAudio()
    wf = wave.open(audio_output_path, 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    # wf.writeframes(b''.join(np.concatenate([listO_0,listO_1],0)))
    listO=np.reshape(listO.T,(len(listO[0])*2))
    # listO=listO.astype(np.int16).tostring()
    wf.writeframes(b''.join(listO))
    # wf.writeframes(b''.join(np.reshape(listO.T.tolist, (len(listO[0])*2))))
    wf.close()

    plt.plot(graphDataX,graphDataY)
    plt.plot(723,compare[723],marker='.', markersize=10)
    plt.plot(diffT,min(compare),marker='.', markersize=10)
    plt.show()
    dftList.append(diffT)
print(dftList)