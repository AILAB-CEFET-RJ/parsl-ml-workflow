import matplotlib as mpl
mpl.use('Agg')

from numpy import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

from tabulate import tabulate


def plot(h, metric):
    fig, ax = plt.subplots()
    epochs = range(len(h[metric]))
    ax.plot(epochs, h[metric], c='blue', label='train')
    ax.plot(epochs, h['val_'+metric], c='green', label='test')
    ax.legend()
    plt.title(metric)
    ax.set_xlabel('epochs')
    plt.savefig('../plot/'+metric+'.pdf')

def plot_scatter(X, Y, X_test, Y_test, preds):
    #plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1)
    plt.ylabel("Redshift")
    plt.scatter(X[:, 3], Y)
    plt.ylabel("Treino")

    plt.subplot(3, 1, 2)
    plt.scatter(X_test[:, 3], Y_test)
    plt.ylabel("Teste")

    plt.subplot(3, 1, 3)
    plt.scatter(X_test[:, 3], preds)
    plt.ylabel("Predito")
    plt.savefig('../plot/redshift.png')


def create_baseline_model():
     model = Sequential()
     model.add(Dense(100, input_dim=10, activation='relu'))
     model.add(Dense(100, activation='relu'))
     model.add(Dense(1, activation='linear'))
     model.compile(loss='mse', optimizer='adam', metrics=['mse'])

     return model


if __name__ == '__main__':

    train_file = '../datasets/train.csv'

    X = loadtxt(train_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
    Y = loadtxt(train_file, unpack=True, usecols=(11), delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    print('Data loaded!')

    model = create_baseline_model()
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
    preds = model.predict(X_test)

    pred = preds.reshape(len(preds))
    real = y_test
    t = tabulate(array([real, pred]).T[:50], headers=['Real', 'Predict'], tablefmt='orgtbl')
    print(t)

    plot_scatter(X_train, y_train, X_test, y_test, preds)

    plot(hist.history, 'mean_squared_error')

