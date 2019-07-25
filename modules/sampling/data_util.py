import numpy as np
from numpy import loadtxt, savetxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_dataset(train_file):
    X = loadtxt(train_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
    Y = loadtxt(train_file, unpack=True, usecols=(11), delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    print('Data loaded!')

    return X_train, y_train, X_test, y_test, X_val, y_val

def build_dataset(train_file='/home/rfialho/shared/datasets/redshifts.csv'):

    x = np.loadtxt(train_file, usecols=(range(1, 6)), unpack=True, delimiter=',', dtype='float32').T
    y = np.loadtxt(train_file, unpack=True, usecols=(11), delimiter=',', dtype='float32')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_val = x_val.astype(np.float32)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    x_val = scaler.fit_transform(x_val)

    return {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "x_val": x_val, "y_val": y_val
    }

if __name__ == '__main__':
    train_file = '../../datasets/redshifts.csv'

    X_train, y_train, X_test, y_test, X_val, y_val = split_dataset(train_file)

    savetxt("../../datasets/X_train.csv", X_train, delimiter=",")
    savetxt("../../datasets/y_train.csv", y_train, delimiter=",")

    savetxt("../../datasets/X_test.csv", X_test, delimiter=",")
    savetxt("../../datasets/y_test.csv", y_test, delimiter=",")

    savetxt("../../datasets/X_val.csv", X_val, delimiter=",")
    savetxt("../../datasets/y_val.csv", y_val, delimiter=",")

