import datetime
import socket
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from modules.sampling.data_util import build_dataset


def runKNN(inputs, ks):
    data = pickle.loads(inputs)
    outputs = []

    ts = datetime.datetime.now()

    for k in ks:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(data["x_train"], data["y_train"])

        score = model.score(data["x_val"], data["y_val"])
        preds = model.predict(data["x_test"])

        pred = preds.reshape(len(preds))
        real = data["y_test"]

        # Compute the mean squared error of our predictions.
        mse = mean_squared_error(real, pred)

        output = {}
        output['cross_val_score'] = score
        output['mse'] = mse
        output['k'] = k

        outputs.append(output)

    ts = datetime.datetime.now() - ts
    h = socket.gethostname()

    return {
        "msg": "Run KNN! [" + h + "] > elapsed time: " + str(ts),
        "data": outputs,
    }


def runLinearRegretion(inputs, lrs):
    if inputs:
        data = pickle.loads(inputs)
    else:
        data = build_dataset()

    outputs = []

    ts = datetime.datetime.now()

    for lr in lrs:
        model = SGDRegressor(
            eta0=lr,
            max_iter=700,
            random_state=42
        )
        model.fit(data["x_train"], data["y_train"])

        preds = model.predict(data["x_test"])
        score = model.score(data["x_val"], data["y_val"])

        pred = preds.reshape(len(preds))
        real = data["y_test"]

        mse = mean_squared_error(real, pred)

        output = {}
        output['score'] = score
        output['mse'] = mse
        output['lr'] = lr

        outputs.append(output)

    ts = datetime.datetime.now() - ts
    h = socket.gethostname()

    return {
        "msg": "Run LReg! [" + h + "] > elapsed time: " + str(ts),
        "data": outputs,
    }


def runRForest(inputs, hp):
    if inputs:
        data = pickle.loads(inputs)
    else:
        data = build_dataset()

    ts = datetime.datetime.now()

    outputs = []
    for n_estimator, max_depth in hp:
        model = RandomForestRegressor(
            n_estimators=n_estimator,
            max_depth=max_depth,
            random_state=42,
            verbose=1,
            n_jobs=1
        )

        model.fit(data["x_train"], data["y_train"])

        score = model.score(data["x_val"], data["y_val"])
        preds = model.predict(data["x_test"])

        pred = preds.reshape(len(preds))
        real = data["y_test"]

        mse = mean_squared_error(real, pred)

        output = {}
        output['score'] = score
        output['mse'] = mse
        output['hp'] = [n_estimator, max_depth]

        outputs.append(output)

    ts = datetime.datetime.now() - ts
    h = socket.gethostname()

    return {
        "msg": "Run LReg! [" + h + "] > elapsed time: " + str(ts),
        "data": outputs,
    }

