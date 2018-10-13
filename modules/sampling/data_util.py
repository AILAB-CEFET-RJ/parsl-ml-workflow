from numpy import loadtxt, savetxt
from sklearn.model_selection import train_test_split


def split_dataset(train_file):
    X = loadtxt(train_file, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), unpack=True, delimiter=',').T
    Y = loadtxt(train_file, unpack=True, usecols=(11), delimiter=',')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    print('Data loaded!')

    return X_train, y_train, X_test, y_test, X_val, y_val



if __name__ == '__main__':
    train_file = '../../datasets/redshifts.csv'

    X_train, y_train, X_test, y_test, X_val, y_val = split_dataset(train_file)

    savetxt("../../datasets/X_train.csv", X_train, delimiter=",")
    savetxt("../../datasets/y_train.csv", y_train, delimiter=",")

    savetxt("../../datasets/X_test.csv", X_test, delimiter=",")
    savetxt("../../datasets/y_test.csv", y_test, delimiter=",")

    savetxt("../../datasets/X_val.csv", X_val, delimiter=",")
    savetxt("../../datasets/y_val.csv", y_val, delimiter=",")

