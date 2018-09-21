import pickle
from os import path
from numpy import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

from tabulate import tabulate


def plot(h, metric):
    fig, ax = plt.subplots()
    epochs = range(len(h[metric]))
    ax.plot(epochs, h[metric], c='blue', label='train')
    ax.plot(epochs, h['val_'+metric], c='green', label='test')
    ax.legend()
    plt.title(metric)
    ax.set_xlabel('epochs')
    plt.show()

def create_baseline_model():
     model = Sequential()
     model.add(Dense(10, input_dim=10, activation='relu'))
     model.add(Dense(20, activation='relu'))
     model.add(Dense(1, activation='linear'))
     model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

     return model

def load_hist():
    exists = path.exists('trainHistoryDict.p')

    if not exists:
        return None

    return pickle.load(open('trainHistoryDict.p', 'rb'))


if __name__ == '__main__':

    class hist: pass
    hist.history = load_hist()

    if not hist.history:
        train_file = '../datasets/train.csv'
        test_file = '../datasets/test.csv'

        X_test = loadtxt(test_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
        Y_test = loadtxt(test_file, unpack=True, usecols=(11), delimiter=',')

        X = loadtxt(train_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
        Y = loadtxt(train_file, unpack=True, usecols=(11), delimiter=',')

        print('Data loaded!')

        model = create_baseline_model()
        hist = model.fit(X, Y, validation_data=(X_test, Y_test), epochs=7)
        preds = model.predict(X_test)

        #pickle.dump(hist.history, open('trainHistoryDict.p', 'wb'))

        pred = preds.reshape(len(preds))
        real = Y_test
        t = tabulate(array([real, pred]).T[:50], headers=['Real', 'Predict'], tablefmt='orgtbl')
        print(t)


    plot(hist.history, 'mean_squared_error')
    plot(hist.history, 'cosine_proximity')
