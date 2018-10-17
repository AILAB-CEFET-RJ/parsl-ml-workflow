import keras

from matplotlib import pyplot as plt

from numpy import mean
from numpy import std
from numpy import shape
from numpy import array


class PlotLossesCallback(keras.callbacks.Callback):

    def set_output_file_dir(self, dir):
        self.__dir = dir

    def set_show_only(self, val):
        self.__show_only = val

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.stds = []
        self.val_stds = []

        self.fig = plt
        self.fig.figure()
        self.fig.grid()
        self.fig.title("Learning Curves")

        self.fig.xlabel("Training examples")
        self.fig.ylabel("Score")

        self.logs = []

    def on_train_end(self, logs={}):
        self.stds = array(self.stds)
        self.val_stds = array(self.val_stds)

        self.fig.fill_between(self.stds[:, 0], self.stds[:, 1], self.stds[:, 2], alpha=0.1, color="r")
        self.fig.fill_between(self.val_stds[:, 0], self.val_stds[:, 1], self.val_stds[:, 2], alpha=0.1, color="g")

        self.fig.plot(self.stds[:, 0], self.stds[:, 3], 'o-', color="r", label="Training score")
        self.fig.plot(self.val_stds[:, 0], self.val_stds[:, 3], 'o-', color="g", label="Cross-validation score")

        self.fig.legend(loc="best")

        if self.__show_only:
            self.fig.show()
        else:
            self.fig.savefig(self.__dir + 'learn_curves.png')


    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        if epoch > 0:
            train_scores_mean = mean(self.losses)
            train_scores_std = std(self.losses)
            test_scores_mean = mean(self.val_losses)
            test_scores_std = std(self.val_losses)

            shap = shape(self.losses)[0]
            self.stds.append([shap, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, train_scores_mean])

            val_shap = shape(self.val_losses)[0]
            self.val_stds.append([val_shap, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, test_scores_mean])


