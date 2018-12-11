from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras import optimizers
import numpy as np
import svm
import glove
from knn import *
import pandas as pd
import traincv

max_features = 50

train_rate_path = './data/train.ra.txt'
train_rate = np.zeros([33005,5])
with open(train_rate_path, 'r') as f:
    i = 0
    for line in f:
        train_rate[i, int(line)-1] = 1
        i += 1

test_rate_path = './data/test.rating.txt'
test_rate = np.zeros([6813,5])
with open(test_rate_path, 'r') as f:
    i = 0
    for line in f:
        test_rate[i, int(line)-1] = 1
        i += 1

dev_rate_path = './data/dev.rating.txt'
dev_rate = np.zeros([6814,5])
with open(dev_rate_path, 'r') as f:
    i = 0
    for line in f:
        dev_rate[i, int(line)-1] = 1
        i += 1

def getPaddingGlove(messages):
    numRows = len(messages)
    resArray = np.zeros((numRows, 100, 50))
    gloveDict = glove.loadGloveToDict()
    # print('The length of glove dict is', len(gloveDict))
    for i in range(len(messages)):
        message = messages[i]
        wordvector = np.zeros([100, 50])
        n = len(message)
        # if the length is greater than 100, just ignore
        for j in range(n):
            if j >= 100:
                break
            word = message[j]
            if word in gloveDict:
                wordvector[j] += gloveDict[word]
        resArray[i, :, :] = wordvector
    # print(resArray)
    return resArray

def buildModel(X_train,y_train,devDataX, devRateY,X_test,y_test,batch_size):
    print('Build model...')
    model = Sequential()
    # X_train = np.array(X_train).reshape([33005,1,4223])
    # devDataX = np.array(devDataX).reshape([6814,1,4223])
    # X_test = np.array(X_test).reshape([6813, 1, 4223])
    # model.add(LSTM(1, dropout_W=0.2, dropout_U=0.2, input_shape=(1, 4223)))  # try using a GRU instead, for fun
    # model.add(Dense(32, input_dim=50))
    # model.add(Embedding(4223, 32, input_length=100))
    model.add(LSTM(256, input_shape=(100, 50), return_sequences=(32, 128, 25)))
    model.add(LSTM(64))
    model.add(Dense(10))
    # model.add(Dense(32))
    model.add(Dense(5, activation='softmax'))
    # model.add(Activation('softmax'))
    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    print(X_train.shape, X_test.shape)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_data=(devDataX, devRateY))
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

def create_dictionary(messages):
    tempDict = collections.defaultdict(int)
    for message in messages:
        setWords = set(message)
        for word in setWords:
            tempDict[word] += 1
    resDict = collections.defaultdict(int)
    i = 0
    for key,val in tempDict.items():
        if(val>=6):
            resDict[key] = i
            i +=1
    print('The length of the dictionary is', len(resDict))
    return resDict

def getMessageMatrix(message, dict):
    l = len(message)
    m = len(dict)
    resA = np.zeros([l, 100])
    for i in range(l):
        n = len(message[i])
        for j in range(n):
            if j>=100:
                break
            resA[i, j] = dict[message[i][j]]

    return resA


def loadTrainTestDataSet():
    train_review_path = './data/train.re.txt'
    train_rate_path = './data/train.ra.txt'
    dev_review_path = './data/dev.review.txt'
    dev_rate_path = './data/dev.rating.txt'
    test_review_path = './data/test.review.txt'
    test_rate_path = './data/test.rating.txt'
    review_path = './data/review_full.txt'
    rate_path = './data/rate_full.txt'
    messages = load_review_dataset(review_path)
    # word_dict = create_dictionary(messages)
    #
    # train_messages = load_review_dataset(train_review_path)
    # train_resArray = transform_text(train_messages, word_dict)
    # dev_messages = load_review_dataset(dev_review_path)
    # dev_resArray = transform_text(dev_messages, word_dict)
    # test_messages = load_review_dataset(test_review_path)
    # test_resArray = transform_text(test_messages, word_dict)

    messages = load_review_dataset(review_path)
    word_dict = create_dictionary(messages)
    train_messages = load_review_dataset(train_review_path)
    train_resArray = getPaddingGlove(train_messages)
    dev_messages = load_review_dataset(dev_review_path)
    dev_resArray = getPaddingGlove(dev_messages)
    test_messages = load_review_dataset(test_review_path)
    test_resArray = getPaddingGlove(test_messages)
    # train_resArray = getMessageMatrix(train_messages, word_dict)
    # dev_resArray = getMessageMatrix(dev_messages, word_dict)
    # test_resArray = getMessageMatrix(test_messages, word_dict)
    trainingDataX = train_resArray[:, ]
    devDataX = dev_resArray[:, ]
    testDataX = test_resArray[:, ]
    traRateY = pd.read_csv(train_rate_path, header=None)
    devRateY = pd.read_csv(dev_rate_path, header=None)
    tesRateY = pd.read_csv(test_rate_path, header=None)

    return (trainingDataX,traRateY,devDataX, devRateY, testDataX,tesRateY)

def main():
    # X_train, y_train, X_test, y_test = svm.loadTrainTestDataSet()
    X_train,y_train,devDataX, devRateY, X_test,y_test = loadTrainTestDataSet()
    print(X_train.shape)
    batch_size = 32
    buildModel(X_train, train_rate,devDataX, dev_rate, X_test, test_rate, batch_size)




if __name__ == "__main__":
    main()
