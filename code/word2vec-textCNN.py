#!python
from keras.layers import Embedding, LSTM, GRU, Dropout, Dense, Input
# from keras.models import Model, Sequential, load_model
from keras.preprocessing import sequence
from keras import Input
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
# from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
from keras.layers import Conv1D, Dense, Flatten, concatenate, Embedding
from keras.layers import MaxPooling1D
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


file=open('del-major-trian-ops.txt')
ops=[]
for line in file.readlines():
#     print(line)
    curLine=line.strip().split(" ")
#     floatLine=list(map(float,curLine))
    ops.append(curLine[:])
print(len(ops))

def train_W2V(sentenList, embedSize=300, epoch_num=50):
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
    X_train_idx = np.array(sequence.pad_sequences(X_train_idx, maxlen=800))
    return X_train_idx

# w2v_model=Word2Vec.load('sentiment_analysis/w2v_model.pkl')
# embedding_matrix = np.zeros((len(vocab) + 1, 300))
# for word, i in vocab.items():
#     try:
#         embedding_vector = w2v_model[str(word)]
#         embedding_matrix[i] = embedding_vector
#     except KeyError:
#         continue

def TextCNN_model_2(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, embedding_matrix):
    main_input = Input(shape=(800,), dtype='float64')
    #     embedder = Embedding(len(word2idx) + 1, 300, input_length=19422, weights=[embedding_matrix], trainable=False)
    embedder = Embedding(len(word2idx), 300, input_length=800, weights=[embedding_matrix], trainable=True)
    # embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    #     print(embedding_matrix[1])
    embed = embedder(main_input)
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=38)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=37)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=36)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(9, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)

    model.fit(x_train_padded_seqs, y_train, batch_size=16, epochs=50)

    # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)

    result = model.predict(x_test_padded_seqs)
    result_labels = np.argmax(result, axis=1)

    y_predict = result_labels
    y_tests = np.argmax(y_test, axis=1)

    #     y_predict = list(map(str, result_labels))
    print('accuracy', metrics.accuracy_score(y_tests, y_predict))
    print('average f1-score:', metrics.f1_score(y_tests, y_predict, average='weighted'))
    print(confusion_matrix(y_tests, y_predict))
    print(classification_report(y_tests, y_predict,digits=6))
    

#     model.summary(line_length=200,positions=[0.30,0.60,0.7,1.0])

#basepath="C:/Users/shuju/"

subtrain = pd.read_csv('del-major-train-label.csv', encoding='latin-1', engine='python')

from keras.utils.np_utils import to_categorical
y_labels = []
for side in subtrain.Class:
    y_labels.append(side)
# print(y_labels)
y_labels[:] = [x - 1 for x in y_labels]
# print(y_labels)
y_all = to_categorical(y_labels)

# print(y_all)

w2vModel = train_W2V(ops, embedSize=300, epoch_num=50)
word2idx, embedMatrix = build_word2idx_embedMatrix(w2vModel)

X_all_idx = make_deepLearn_data(ops, word2idx)
y_all_idx = np.array(y_all)
X_tra_idx, X_val_idx, y_tra_idx, y_val_idx = train_test_split(X_all_idx, y_all_idx, test_size=0.3, random_state=0, stratify=y_all_idx)

TextCNN_model_2(X_tra_idx, y_tra_idx, X_val_idx, y_val_idx, embedMatrix)

# model = (X_tra_idx, y_tra_idx, X_val_idx, y_val_idx, embedMatrix)

# model.fit(X_tra_idx, y_tra_idx, validation_data=(X_val_idx, y_val_idx),
#           epochs=1, batch_size=10, verbose=1)
# print(h.history)
# y_pred = model.predict(X_val_idx)
# print(y_pred)
# y_pred_idx = [1 if prob[0] > 0.5 else 0 for prob in y_pred]

# print(f1_score(y_val_idx, y_pred_idx))
# print(confusion_matrix(y_val_idx, y_pred_idx))


