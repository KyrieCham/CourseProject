from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
import svm

max_features = 50

def buildModel(X_train,y_train,X_test,y_test,batch_size):
    print('Build model...')
    model = Sequential()
    #model.add(Embedding(max_features, 128, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='categorical-crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

def main():
    X_train,y_train,X_test,y_test = svm.loadTrainTestDataSet()
    batch_size = 32
    buildModel(X_train, y_train, X_test, y_test, batch_size)




if __name__ == "__main__":
    main()
