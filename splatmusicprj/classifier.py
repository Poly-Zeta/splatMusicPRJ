#!python3.7

#単体 作成済みの判別器をつかって画像入力の曲名を判別する
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D 
from keras.datasets import mnist
import matplotlib.pyplot as plt
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
model=load_model("D:/Users/poly_Z/Documents/splatmusicprj/modelSample.h5")
randomAcc=random.choice(label)
imgfile="D:/Users/poly_Z/Documents/splatmusicprj/fig_v2_logmel/"+randomAcc+"_01.wavimg.png"
image_size = 200

print("hello CUDA")

X = []
Y = []
image = Image.open(imgfile)
image = image.convert("RGB")
image = image.resize((image_size, image_size))
data = np.asarray(image)
X.append(data)
# Y.append(index)

X = np.array(X)
# Y = np.array(Y)
X = X.astype('float32')
X = X / 255.0
# Y = np_utils.to_categorical(Y, 16)
pred = model.predict(X)
score = np.max(pred)
pred_label = label[np.argmax(pred[0])]
print("accName:",randomAcc)
print('choiceName:',pred_label)
print('score:',score)

