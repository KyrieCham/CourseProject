from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
import numpy as np
import svm
import glove
from knn import *
import pandas as pd

max_features = 50

train_rate_path = './data/train.rating.txt'
train_rate = np.zeros([21000])
with open(train_rate_path, 'r') as f:
    i = 0
    for line in f:
        train_rate[i] = int(line) - 1
        i += 1

train_rate_path = './data/test.rating.txt'
test_rate = np.zeros([6813])
with open(train_rate_path, 'r') as f:
    i = 0
    for line in f:
        test_rate[i] = int(line) - 1
        i += 1

def getPaddingGlove(messages):
    numRows = len(messages)
    resArray = np.zeros((numRows, 100, 50))
    gloveDict = glove.loadGloveToDict()
    print('The length of glove dict is', len(gloveDict))
    for i in range(len(messages)):
        message = messages[i]
        wordvector = np.zeros([100, 50])
        n = len(message)
        # if the length is greater than 100, just ignore
        for i in range(n):
            if i >= 100:
                break
            word = message[i]
            if word in gloveDict:
                wordvector[i] += gloveDict[word]

    return resArray

def buildModel(X_train,y_train,X_test,y_test,batch_size):
    print('Build model...')
    model = Sequential()
    # X_train = np.array(X_train).reshape([21000,1,4223])
    # X_test = np.array(X_test).reshape([6813, 1, 4223])
    # model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, input_shape=(1, 4223)))  # try using a GRU instead, for fun
    model.add(LSTM(128, input_shape=(100, 50)))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    print(X_train.shape, X_test.shape)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

def loadTrainTestDataSet():
    train_review_path = './data/train.review.txt'
    train_rate_path = './data/train.rating.txt'
    dev_review_path = './data/dev.review.txt'
    dev_rate_path = './data/dev.rating.txt'
    test_review_path = './data/test.review.txt'
    test_rate_path = './data/test.rating.txt'
    review_path = './data/review_full.txt'
    rate_path = './data/rate_full.txt'

    train_messages = load_review_dataset(train_review_path)
    train_resArray = getPaddingGlove(train_messages)
    dev_messages = load_review_dataset(dev_review_path)
    dev_resArray = getPaddingGlove(dev_messages)
    test_messages = load_review_dataset(test_review_path)
    test_resArray = getPaddingGlove(test_messages)
    trainingDataX = train_resArray[:, ]
    devDataX = dev_resArray[:, ]
    testDataX = test_resArray[:, ]
    traRateY = pd.read_csv(train_rate_path, header=None)
    devRateY = pd.read_csv(dev_rate_path, header=None)
    tesRateY = pd.read_csv(test_rate_path, header=None)

    return (trainingDataX,traRateY,testDataX,tesRateY)

def main():
    # X_train, y_train, X_test, y_test = svm.loadTrainTestDataSet()
    X_train,y_train,X_test,y_test = loadTrainTestDataSet()
    print(X_train.shape, y_train.shape, y_test.shape)
    batch_size = 32
    buildModel(X_train, train_rate, X_test, test_rate, batch_size)




if __name__ == "__main__":
    main()
