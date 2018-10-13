import matplotlib as mpl
mpl.use('Agg')

from numpy import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from modules.plotting.plot_service import *


def find_best_params(X_train, y_train):
    # number of trees in random forest
    n_estimators = [int(x) for x in linspace(start=200, stop=2500, num=10)]

    # max depth
    max_depth = [int(x) for x in linspace(100, 500, num=11)]
    max_depth.append(None)

    # create random grid
    random_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }

    # Random search of parameters
    search = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
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

    model = RandomForestRegressor(
        n_estimators = best_params['n_estimators'],
        max_depth = best_params['max_depth'],
        random_state = 42,
        verbose=2,
        n_jobs=-1
    )

    hist = model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    preds = model.predict(X_test)

    pred = preds.reshape(len(preds))
    real = y_test

    plot_scatter(X_train, y_train, X_val, y_val, X_test, y_test, preds)

    errors = abs(pred - real)
    print('Mean Absolute Error:', round(mean(errors), 2), 'degrees.')

    print('Cross-Val Score:', score)

    mape = 100 * (errors / real)
    accuracy = 100 - mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    plot_hm(real, pred)

