import matplotlib as mpl
mpl.use('Agg')

from numpy import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from modules.plotting.plot_service import *


def create_baseline_model(l1_units=300, l1_dp=0.1, l2_units=150, l2_dp=0.05):
     model = Sequential()
     model.add(Dense(l1_units, input_dim=10, kernel_initializer='normal', activation='relu'))
     model.add(Dropout(l1_dp))
     model.add(Dense(l2_units, kernel_initializer='normal', activation='relu'))
     model.add(Dropout(l2_dp))
     model.add(Dense(1, kernel_initializer='normal', activation='linear'))
     model.compile(loss='mse', optimizer='adam', metrics=['mse'])

     return model


def create_model(l1_units=250, l1_dp=0.2, l2_units=125, l2_dp=0.2):
    return KerasRegressor(build_fn=create_baseline_model, l1_units=l1_units, l1_dp=l1_dp, l2_units=l2_units, l2_dp=l2_dp, verbose=1)


def find_best_params(X_train, y_train):
    batch_size = [int(x) for x in linspace(start=10, stop=100, num=10)]
    epochs = [int(x) for x in linspace(start=50, stop=300, num=6)]
    l1_units = [int(x) for x in linspace(start=200, stop=500, num=12)]
    l1_dp = [0.1, 0.2, 0.3, 0,4, 0.6]
    l2_units = [int(x) for x in linspace(start=100, stop=500, num=15)]
    l2_dp = [0.1, 0.2, 0.3, 0,4, 0.6]

    params = dict(
        batch_size = batch_size,
        epochs = epochs,
        l1_units = l1_units,
        l1_dp = l1_dp,
        l2_units = l2_units,
        l2_dp = l2_dp
    )

    # Random search of parameters
    search = RandomizedSearchCV(estimator=create_model(), param_distributions=params, scoring='neg_mean_squared_error', n_iter=100, cv=3, verbose=1, random_state=42, n_jobs=-1)

    # Fit the model
    search.fit(X_train, y_train)

    # print results
    print('Best Params:', search.best_params_)
    print('Best Score:', search.best_score_)

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
    model = create_model(best_params['l1_units'], best_params['l1_dp'], best_params['l2_units'], best_params['l2_dp'])

    plot_losses = ann_plot_losses_callback()
    tensorboard = ann_tensorboard_callback()

    hist = model.fit(
        X_train,
        y_train,
        verbose=1,
        validation_data = (X_val, y_val),
        epochs = best_params['epochs'],
        batch_size = best_params['batch_size'],
        callbacks=[tensorboard, plot_losses]
    )

    score = model.score(X_val, y_val)
    preds = model.predict(X_test)

    pred = preds.reshape(len(preds))
    real = y_test

    plot_table(real, pred)
    print('Cross-Val Score:', score)
    plot_scatter(X_train, y_train, X_val, y_val, X_test, y_test, preds)
    plot(hist.history, 'mean_squared_error')
    plot_hm(real, pred)

