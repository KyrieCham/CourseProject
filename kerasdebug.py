from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
import numpy as np
import svm

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

def buildModel(X_train,y_train,X_test,y_test,batch_size):
    print('Build model...')
    model = Sequential()
    X_train = np.array(X_train).reshape([21000,1,4223])
    X_test = np.array(X_test).reshape([6813, 1, 4223])
    model.add(LSTM(1, dropout_W=0.2, dropout_U=0.2, input_shape=(1,4223)))  # try using a GRU instead, for fun
    model.add(Dense(5))
    # model.add(Dense(1))
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

def main():
    X_train,y_train,X_test,y_test = svm.loadTrainTestDataSet()
    print(y_train.shape, y_test.shape)
    batch_size = 32
    buildModel(X_train, train_rate, X_test, test_rate, batch_size)




if __name__ == "__main__":
    main()
