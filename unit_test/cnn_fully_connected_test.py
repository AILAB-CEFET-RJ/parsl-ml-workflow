import matplotlib as mpl
mpl.use('Agg')

from numpy import *
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


from modules.plotting.plot_service import *


def create_baseline_model():
     model = Sequential()
     model.add(Dense(250, input_dim=10, activation='relu'))
     model.add(Dense(125, activation='relu'))
     model.add(Dense(1, activation='linear'))
     model.compile(loss='mse', optimizer='adam', metrics=['mse'])

     return model


def find_best_params(model, X_train, y_train):

    batch_size = [int(x) for x in linspace(start=10, stop=100, num=10)]
    epochs = [int(x) for x in linspace(start=50, stop=300, num=50)]
    random_grid = {
        'batch_size': batch_size,
        'epochs': epochs
    }

    # Random search of parameters
    n_iter = len(batch_size) * len(epochs)
    search = RandomizedSearchCV(estimator=model, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=n_iter, cv=3, verbose=2, random_state=42, n_jobs=-1)

    # Fit the model
    search.fit(X_train, y_train)

    # print results
    print('Best Params:', search.best_params_)

    return search.best_params_


if __name__ == '__main__':

    train_file = '../datasets/train.csv'

    X = loadtxt(train_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
    Y = loadtxt(train_file, unpack=True, usecols=(11), delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    print('Data loaded!')

    model = KerasRegressor(build_fn = create_baseline_model())
    best_params = find_best_params(model, X_train, y_train)

    hist = model.fit(
        X_train,
        y_train,
        validation_data = (X_val, y_val),
        epochs = best_params['epochs'],
        batch_size = best_params['batch_size']
    )

    preds = model.predict(X_test)

    pred = preds.reshape(len(preds))
    real = y_test

    plot_table(real, pred)
    plot_scatter(X_train, y_train, X_val, y_val, X_test, y_test, preds)
    plot(hist.history, 'mean_squared_error')
    plot_hm(real, preds.T[0])

