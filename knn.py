import numpy as np
import collections
import pandas as pd
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

train_review_path = './data/train.review.txt'
train_rate_path = './data/train.rating.txt'
dev_review_path = './data/dev.review.txt'
dev_rate_path = './data/dev.rating.txt'
test_review_path = './data/test.review.txt'
test_rate_path = './data/test.rating.txt'
review_path = './data/review_full.txt'

def load_review_dataset(path):
    with open(path,'r',encoding='utf8') as f:
        content = f.readlines()
    res = []
    for line in content:
        a = line.strip('.').lower().split()
        b = []
        exclude = set(string.punctuation)
        for word in a:
            word = ''.join(ch for ch in word if ch not in exclude)
            b.append(word)
        res.append(b)
    # res = [line.strip('.').lower().split() for line in content]
    return res

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
    return resDict

def transform_text(messages, word_dictionary):
    numRows = len(messages)
    print(numRows)
    numCols = max(word_dictionary.values())+1
    print(numCols)
    resArray = np.zeros((numRows,numCols))
    for i in range(len(messages)):
        message = messages[i]
        for word in message:
            if(word in word_dictionary):
                col = word_dictionary[word]
                resArray[i][col] +=1
    return resArray

def loadTrainTestDataSet():
    train_review_path = './data/train.review.txt'
    train_rate_path = './data/train.rating.txt'
    dev_review_path = './data/dev.review.txt'
    dev_rate_path = './data/dev.rating.txt'
    test_review_path = './data/test.review.txt'
    test_rate_path = './data/test.rating.txt'
    review_path = './data/review_full.txt'
    rate_path = './data/rate_full.txt'
    messages = load_review_dataset(review_path)
    word_dict = create_dictionary(messages)

    train_messages = load_review_dataset(train_review_path)
    train_resArray = transform_text(train_messages, word_dict)
    dev_messages = load_review_dataset(dev_review_path)
    dev_resArray = transform_text(dev_messages, word_dict)
    test_messages = load_review_dataset(test_review_path)
    test_resArray = transform_text(test_messages, word_dict)

    trainingDataX = train_resArray[:, ]
    devDataX = dev_resArray[:, ]
    testDataX = test_resArray[:, ]
    traRateY = pd.read_csv(train_rate_path, header=None)
    devRateY = pd.read_csv(dev_rate_path, header=None)
    tesRateY = pd.read_csv(test_rate_path, header=None)
    # print(len(traRateY))
    trainingDataY = traRateY[:][0]
    testDataY = tesRateY[:][0]
    return (trainingDataX,traRateY, devDataX, devRateY, testDataX,tesRateY)

def getCorrectness(predict, y):
    length = len(predict)
    count = 0
    for j in range(length):
        if predict[j] == y[j]:
            count += 1
    return count / length

def main():
    review_path = './data/review_small.txt'
    rate_path = './data/rate_small.txt'
    messages = load_review_dataset(review_path)
    word_dict = create_dictionary(messages)
    resArray = transform_text(messages,word_dict)
    #To construct the training dataset
    trainingDataX, trainingRateY, devDataX, devRateY, testDataX, tesRateY = loadTrainTestDataSet()

    diffK = []

    diffK.append(KNeighborsClassifier(n_neighbors=4))
    diffK.append(KNeighborsClassifier(n_neighbors=5))
    diffK.append(KNeighborsClassifier(n_neighbors=6))

    for n in range(len(diffK)):
        classifier = diffK[n]
        classifier.fit(trainingDataX, trainingRateY)
        predict = classifier.predict(devDataX)
        correct = getCorrectness(predict, devRateY)
        print('The correctness is', correct)




if __name__ == "__main__":
    main()



