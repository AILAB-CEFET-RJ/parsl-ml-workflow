import matplotlib as mpl
mpl.use('Agg')

from numpy import *
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from modules.plotting.plot_service import *


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
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000)
    preds = model.predict(X_test)

    pred = preds.reshape(len(preds))
    real = y_test

    plot_table(real, pred)
    plot_scatter(X_train, y_train, X_test, y_test, preds)
    plot(hist.history, 'mean_squared_error')
    plot_hm(real, preds.T[0])

