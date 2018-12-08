import numpy as np
import collections

DEFAULT_FILE_PATH = "./data/glove.6B.50d.txt"

def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors"""
    # wordVectors = np.zeros((len(tokens), dimensions))
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token == tokens:
                data = [float(x) for x in row[1:]]
                wordVectors = np.array(data)
                return wordVectors
        return np.zeros(50)
            # # print(token,' toktok')
            # if token not in tokens:
            #     continue
            # data = [float(x) for x in row[1:]]
            # if len(data) != dimensions:
            #     raise RuntimeError("wrong number of dimensions")
            # wordVectors[tokens[token]] = np.asarray(data)
    # return wordVectors

def loadGloveToDict(filePath = DEFAULT_FILE_PATH):
    """
    Build the dictionary for the GLove.
    :return: a dicitonary with word as the key and the glove vector
    as the value(an array).
    """
    res = {}
    with open(filePath) as f:
        for line in f:
            line = line.strip()
            if(not line):
                continue
            row = line.split()
            token = row[0]
            data = [float(x) for x in row[1:]]
            data = np.array(data)
            res[token] = data
    return res
