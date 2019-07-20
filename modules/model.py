import datetime
import socket
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


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