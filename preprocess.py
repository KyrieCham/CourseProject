import csv
import numpy as np

def load_csv(csv_path):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    # Load headers
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
    # print (headers)

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i] == 'reviews.text']
    l_cols = [i for i in range(len(headers)) if headers[i] == 'reviews.rating']

    # inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    # labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    with open('review_full.txt', 'w') as f:
        with open(csv_path, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                f.write(row[x_cols[0]])
                f.write('\n')

    with open('rate_full.txt', 'w') as f:
        with open(csv_path, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                f.write(row[l_cols[0]])
                f.write('\n')
    # csvFile.close()
    # with open(csv_path, 'r', newline='') as csv_fh:
    #     j = 0
    #     content = []
    #     for row in csv_fh:
    #         print (row)
    #         j+=1
    #         content.append(row[i] for i in x_cols)
    #         if j == 50:
    #             break
    # print (content)

    # np.savetxt('small.txt', inputs)
    #
    # if inputs.ndim == 1:
    #     inputs = np.expand_dims(inputs, -1)

    # return inputs, labels

path = './data/1429_1.csv'
load_csv(path)