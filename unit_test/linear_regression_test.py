import matplotlib as mpl
mpl.use('Agg')

from numpy import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import RandomizedSearchCV

from modules.plotting.plot_service import *


def find_best_params(X_train, y_train):
    eta0 = random.uniform(low=0.00001, high=0.001, size=10)
    alpha = random.uniform(low=0.001, high=0.01, size=5)


    # create random grid
    param_grid = {
        'learning_rate': ['optimal', 'invscaling'],
        'eta0': eta0,
        'alpha': alpha
    }

    # Random search of parameters
    search = RandomizedSearchCV(estimator=SGDRegressor(), param_distributions=param_grid, scoring='neg_mean_squared_error', n_iter=60, cv=3, verbose=1, random_state=42, n_jobs=-1)
    # Fit the model
    search.fit(X_train, y_train)

    # print results
    print('Best Params:', search.best_params_)

    return search.best_params_


if __name__ == '__main__':

    train_file = '../datasets/redshifts.csv'

    X = loadtxt(train_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
    Y = loadtxt(train_file, unpack=True, usecols=(11), delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print('Data loaded!')
    plot_simple_table(X_train.T[:, :30])

    best_params = find_best_params(X_train, y_train)

    model = SGDRegressor(
        learning_rate=best_params['learning_rate'],
        eta0=best_params['eta0'],
        alpha=best_params['alpha'],
        max_iter=700,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = model.score(X_val, y_val)

    pred = preds.reshape(len(preds))
    real = y_test

    plot_table(real, pred)
    plot_table_cf('Coeficientes', model.coef_)

    print("Score: ", score)
    print("Mean squared error: ", mean_squared_error(real, pred))
    print("Variance score: ", r2_score(real, pred))

    plot_scatter(X_train, y_train, X_val, y_val, X_test, y_test, preds)
    plot_scatter_lr(X_train, y_train, real, pred)
    plot_hm(real, pred)
