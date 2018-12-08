import numpy as np
import collections
import pandas as pd
import glove
import string
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


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
    print('The length of the dictionary is', len(resDict))
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


def getAverageGlove(messages):
    numRows = len(messages)
    print(numRows)
    resArray = np.zeros((numRows, 50))
    for i in range(len(messages)):
        message = messages[i]
        # a = glove.loadWordVectors('the')
        # print(a, 'aaa')
        wordvector = np.zeros(50)
        zero = np.zeros(50)
        j = 0
        nullExample = 0
        for word in message:
            a = glove.loadWordVectors(word)

            if not (a == zero).all():
                # print(word, wordvector)
                j += 1
                wordvector += a
        if j != 0:
            resArray[i, :] = wordvector/j
        else:
            resArray[i, :] = wordvector
            nullExample += 1
            print('This example is null', message)
    print(resArray,'rrr')
    return resArray

def getAverageGloveFromDict(messages):
    """
    Build a dictionary and use that to get the glove vector
    :param messages: All the comments. Each row represent one comment
    :return: the average glove vectors for each comment
    """
    numRows = len(messages)
    resArrays = np.zeros([numRows,50])
    #Build the word dict from glove first
    gloveDict = glove.loadGloveToDict()
    print('The length of glove dict is', len(gloveDict))
    for i in range(numRows):
        message = messages[i]
        wordVector = np.zeros(50)
        for word in message:
            if(word in gloveDict):
                wordVector = np.stack([wordVector,gloveDict[word]])
        wordVector = np.mean(wordVector)
        print(wordVector.shape)
        resArrays[i,:] = wordVector
    return resArrays



def main():
    review_path = './data/review_small.txt'
    rate_path = './data/rate_small.txt'
    messages = load_review_dataset(review_path)
    # word_dict = create_dictionary(messages)
    # resArray = transform_text(messages, word_dict)
    # print('res array shape',resArray.shape)
    glove = getAverageGloveFromDict(messages)
    print('glove', glove.shape)
    # To construct the training dataset
    # numOfDataPoints = len(resArray)
    # numOfTraining = int(numOfDataPoints * 0.8)
    trainingDataX = glove[:, ]
    rateY = pd.read_csv(rate_path, header=None)
    print(len(rateY))
    trainingDataY = rateY[:][0]
    # print(trainingDataX.shape, trainingDataY.shape)
    # devDataX = resArray[numOfTraining:, ]
    # devDataY = rateY[numOfTraining:][0]
    # print(devDataY.shape, type(devDataY))
    # Once we have all the training data ready
    print('Method: SVM')
    # lin_clf = svm.LinearSVC()
    # lin_clf.fit(trainingDataX, trainingDataY)

    print('Method: KNN')
    classifier = KNeighborsClassifier(n_neighbors=5)
    # classifier.fit(trainingDataX, trainingDataY)
    # predicted_labels = classifier.predict(devDataX)
    print(cross_val_score(classifier, trainingDataX, trainingDataY, cv=5))


if __name__ == "__main__":
    main()
