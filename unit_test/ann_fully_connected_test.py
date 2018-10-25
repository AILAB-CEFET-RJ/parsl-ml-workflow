import matplotlib as mpl
mpl.use('Agg')

from numpy import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve

from modules.plotting.plot_service import *


def create_baseline_model(l1_units=300, l1_dp=0.1, l2_units=150, l2_dp=0.05, lr=0.001):
     model = Sequential()
     model.add(Dense(l1_units, input_dim=10, kernel_initializer='normal', activation='relu'))
     model.add(Dropout(l1_dp))
     model.add(Dense(l2_units, kernel_initializer='normal', activation='relu'))
     model.add(Dropout(l2_dp))
     model.add(Dense(1, kernel_initializer='normal', activation='linear'))

     adam = Adam(lr=lr)
     model.compile(loss='mse', optimizer=adam, metrics=['mse'])

     return model


def create_model(l1_units=300, l1_dp=0.1, l2_units=150, l2_dp=0.05, lr=0.001 ):
    return KerasRegressor(build_fn=create_baseline_model, l1_units=l1_units, l1_dp=l1_dp, l2_units=l2_units, l2_dp=l2_dp, lr=lr, verbose=1)


def find_best_params(X_train, y_train):
    l1_units = random.randint(low=50, high=500, size=10)
    l1_dp = [0.0, 0.1, 0.3, 0.6]

    l2_units = random.randint(low=100, high=500, size=15)
    l2_dp = [0.0, 0.1, 0.3, 0.6]

    lr = [0.1, 0.001, 0.0001]

    params = dict(
        l1_units = l1_units,
        l1_dp = l1_dp,
        l2_units = l2_units,
        l2_dp = l2_dp,
        lr = lr
    )

    # Random search of parameters
    search = RandomizedSearchCV(estimator=create_model(), param_distributions=params, scoring='neg_mean_squared_error', n_iter=30, cv=3, verbose=1, random_state=42, n_jobs=-1)

    # Fit the model
    search.fit(X_train, y_train)

    # print results
    print('Best Params:', search.best_params_)
    print('Best Score:', search.best_score_)

    return search.best_params_


def build_learning_data(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=3, scoring='neg_mean_squared_error', train_sizes=linspace(0.01, 1.0, 50))

    # Create means and standard deviations of training set scores
    train_mean = mean(train_scores, axis=1)
    train_std = std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = mean(test_scores, axis=1)
    test_std = std(test_scores, axis=1)

    return train_sizes, train_mean, train_std, test_mean, test_std


if __name__ == '__main__':

    train_file = '../datasets/redshifts.csv'

    X = loadtxt(train_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
    Y = loadtxt(train_file, unpack=True, usecols=(11), delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print('Data loaded!')
    plot_simple_table(X_train.T[:, :30])

    best_params = find_best_params(X_train, y_train)
    model = create_model(best_params['l1_units'], best_params['l1_dp'], best_params['l2_units'], best_params['l2_dp'], best_params['lr'])

    tensorboard = ann_tensorboard_callback()

    hist = model.fit(
        X_train,
        y_train,
        verbose=1,
        validation_data = (X_val, y_val),
        epochs = 300,
        callbacks=[tensorboard]
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

    train_sizes, train_mean, train_std, test_mean, test_std = build_learning_data(model, X_train, y_train)
    plot_curves(train_sizes, train_mean, train_std, test_mean, test_std)

