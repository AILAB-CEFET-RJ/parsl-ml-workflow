import matplotlib as mpl
mpl.use('Agg')

from numpy import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from modules.plotting.plot_service import *


if __name__ == '__main__':

    train_file = '../datasets/train.csv'

    X = loadtxt(train_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
    Y = loadtxt(train_file, unpack=True, usecols=(11), delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    print('Data loaded!')

    model = LinearRegression()

    hist = model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = model.score(X_val, y_val)

    pred = preds.reshape(len(preds))
    real = y_test

    plot_table(real, pred)
    plot_table_cf('Coeficientes', model.coef_)

    print("Score: ", score)
    print("Mean squared error: %.2f" % mean_squared_error(real, pred))
    print('Variance score: %.2f' % r2_score(real, pred))

    plot_scatter(X_train, y_train, X_val, y_val, X_test, y_test, preds)
    plot_scatter_lr(X_train, y_train, real, pred)
    plot_hm(real, pred)
