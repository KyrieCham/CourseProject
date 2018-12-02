import numpy as np
import collections
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score

def load_review_dataset(path):
    with open(path,'r',encoding='utf8') as f:
        content = f.readlines()
    res = [line.strip('.').lower().split() for line in content]
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
    #lin_clf.fit(trainingDataX,trainingDataY)

if __name__ == "__main__":
    main()



