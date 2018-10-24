import matplotlib as mpl
mpl.use('Agg')

from numpy import *
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV

from modules.plotting.plot_service import *


def find_best_params(X_train, y_train):
    n_neighbors = random.randint(low=3, high=300, size=30)

    # create random grid
    param_grid = {
        'n_neighbors': n_neighbors
    }

    # Random search of parameters
    search = RandomizedSearchCV(estimator=KNeighborsRegressor(), param_distributions=param_grid, scoring='neg_mean_squared_error', n_iter=30, cv=3, verbose=1, random_state=42, n_jobs=-1)
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

    model = KNeighborsRegressor(n_neighbors = best_params['n_neighbors'])
    model.fit(X_train, y_train)

    score = model.score(X_val, y_val)
    preds = model.predict(X_test)

    pred = preds.reshape(len(preds))
    real = y_test

    plot_table(real, pred)
    plot_scatter(X_train, y_train, X_val, y_val, X_test, y_test, preds)

    # Compute the mean squared error of our predictions.
    mse = (((pred - real) ** 2).sum()) / len(pred)

    print('Cross-Val Score:', score)
    print('Mean Squared Error:', mse)

    plot_hm(real, pred)

