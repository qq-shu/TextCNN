#!python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.layers import Embedding, LSTM, GRU, Dropout, Dense, Input
from keras.models import Model, Sequential, load_model
from keras.preprocessing import sequence
from keras.datasets import imdb
import gensim
from gensim.models.word2vec import Word2Vec
import re
from collections import *
from keras.utils.np_utils import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file=open('train-ops.txt')
ops=[]
for line in file.readlines():
#     print(line)
    curLine=line.strip().split(" ")
#     floatLine=list(map(float,curLine))
    ops.append(curLine[:])
print(len(ops))


def train_W2V(sentenList, embedSize=300, epoch_num=1):
    w2vModel = Word2Vec(sentences=sentenList, hs=0, negative=5, min_count=5, window=5, iter=epoch_num, size=embedSize)
    return w2vModel

def build_word2idx_embedMatrix(w2vModel):
    word2idx = {"_stopWord": 0}
    vocab_list = [(w, w2vModel.wv[w]) for w, v in w2vModel.wv.vocab.items()]
    embedMatrix = np.zeros((len(w2vModel.wv.vocab.items()) + 1, w2vModel.vector_size))
    for i in range(0, len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embedMatrix[i + 1] = vocab_list[i][1]
    return word2idx, embedMatrix

def make_deepLearn_data(sentenList, word2idx):
    X_train_idx = [[word2idx.get(w, 0) for w in sen] for sen in sentenList]
    X_train_idx = np.array(sequence.pad_sequences(X_train_idx, maxlen=3397))
    return X_train_idx

def Lstm_model(embedMatrix):
    input_layer = Input(shape=(3397,), dtype='int32')
    embedding_layer = Embedding(input_dim=len(embedMatrix), output_dim=len(embedMatrix[0]),
                                weights=[embedMatrix],
                                trainable=True)(input_layer)
    Lstm_layer = LSTM(units=20, return_sequences=False)(embedding_layer)
    drop_layer = Dropout(0.5)(Lstm_layer)
    dense_layer = Dense(units=9, activation="relu")(drop_layer)
    model = Model(inputs=[input_layer], outputs=[dense_layer])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
#basepath="C:/Users/12418/Documents/WeChat Files/wxid_f8v2we60kk4a22/FileStorage/File/2020-07/恶意代码/wuyunxunlianshuju/"

# basepath = "C:/Users/12418/Documents/WeChat Files/wxid_f8v2we60kk4a22/FileStorage/File/2020-07/恶意代码/Kaggle_Microsoft/Kaggle_Microsoft/dataSample/"
#subtrain = pd.read_csv('subtrainLabels.csv', encoding='latin-1', engine='python')
#ops = []
#y_labels = []

#for side in subtrain.Class:
    #y_labels.append(side)
# print(y_labels)

#from keras.utils.np_utils import to_categorical

#y_all = to_categorical(y_labels)
# print(y_all)

#for sid in subtrain.Id:
    #filename = basepath + sid + ".asm"
    #print(filename)
    #ops1 = getOpcodeSequence(filename)
    #ops.append(ops1)

subtrain = pd.read_csv('trainLabels.csv', encoding='latin-1', engine='python')

from keras.utils.np_utils import to_categorical
y_labels = []
for side in subtrain.Class:
    y_labels.append(side)
# print(y_labels)
y_labels[:] = [x - 1 for x in y_labels]
# print(y_labels)
y_all = to_categorical(y_labels)

# print(y_all)



w2vModel = train_W2V(ops, embedSize=300, epoch_num=2)
word2idx, embedMatrix = build_word2idx_embedMatrix(w2vModel)

X_all_idx = make_deepLearn_data(ops, word2idx)
y_all_idx = np.array(y_all)
X_tra_idx, X_val_idx, y_tra_idx, y_val_idx = train_test_split(X_all_idx, y_all_idx, test_size=0.2, random_state=0, stratify=y_all_idx)

model = Lstm_model(embedMatrix)
model.fit(X_tra_idx, y_tra_idx, validation_data=(X_val_idx, y_val_idx),
          epochs=1, batch_size=16, verbose=1)
# print(h.history)
y_pred = model.predict(X_val_idx)
print(y_pred)
# y_pred_idx = [1 if prob[0] > 0.5 else 0 for prob in y_pred]

# print(f1_score(y_val_idx, y_pred_idx))
# print(confusion_matrix(y_val_idx, y_pred_idx))