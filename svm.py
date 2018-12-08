import numpy as np
import collections
import pandas as pd
import string
from sklearn import svm
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

    # resArray = transform_text(messages, word_dict)
    # print('res array shape ',resArray.shape)
    # glove = getAverageGlove(messages)
    # print('glove', glove.shape)
    # To construct the training dataset
    # numOfDataPoints = len(resArray)
    # numOfTraining = int(numOfDataPoints * 0.8)
    trainingDataX = train_resArray[:, ]
    testDataX = test_resArray[:, ]
    traRateY = pd.read_csv(train_rate_path, header=None)
    tesRateY = pd.read_csv(test_rate_path, header=None)
    # print(len(traRateY))
    trainingDataY = traRateY[:][0]
    testDataY = tesRateY[:][0]
    return (trainingDataX,traRateY,testDataX,tesRateY)

def main():
    review_path = './data/review_small.txt'
    rate_path = './data/rate_small.txt'
    messages = load_review_dataset(review_path)
    word_dict = create_dictionary(messages)
    resArray = transform_text(messages,word_dict)
    #To construct the training dataset
    numOfDataPoints = len(resArray)
    numOfTraining = int(numOfDataPoints*0.8)
    trainingDataX = resArray[:numOfTraining,]
    rateY = pd.read_csv(rate_path,header = None)
    trainingDataY = rateY[:numOfTraining][0]
    #Once we have all the training data ready
    lin_clf = svm.LinearSVC()
    print(cross_val_score(lin_clf, trainingDataX, trainingDataY, cv=5))

    #To use radial kernel in SVM
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(trainingDataX, trainingDataY)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    #With the best parameter range found, we can do some close check
    C_2d_range = [1e-2, 1, 1e2]
    gamma_2d_range = [1e-1, 1, 1e1]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(trainingDataX, trainingDataY)
            #This part we need to change to get the cross validation accuracy.
            classifiers.append((C, gamma, clf))
    #lin_clf.fit(trainingDataX,trainingDataY)

if __name__ == "__main__":
    main()



