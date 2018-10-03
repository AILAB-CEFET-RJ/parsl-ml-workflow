import matplotlib as mpl
mpl.use('Agg')

from numpy import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from modules.plotting.plot_service import *


if __name__ == '__main__':

    train_file = '../datasets/train.csv'

    X = loadtxt(train_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
    Y = loadtxt(train_file, unpack=True, usecols=(11), delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    print('Data loaded!')

    model = RandomForestRegressor(n_estimators = 1000, random_state = 42, verbose=2, n_jobs=-1)
    hist = model.fit(X_train, y_train)
    preds = model.predict(X_test)

    pred = preds.reshape(len(preds))
    real = y_test

    plot_table(real, pred)
    plot_scatter(X_train, y_train, X_test, y_test, preds)

    errors = abs(pred - real)
    print('Mean Absolute Error:', round(mean(errors), 2), 'degrees.')

    mape = 100 * (errors / real)
    accuracy = 100 - mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    plot_hm(real, pred)

