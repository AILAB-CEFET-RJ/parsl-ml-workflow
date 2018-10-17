import keras

from IPython.display import clear_output
from matplotlib import pyplot as plt


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

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()

        if self.__show_only:
            plt.show()
        else:
            plt.savefig(self.__dir + 'losses.png')
