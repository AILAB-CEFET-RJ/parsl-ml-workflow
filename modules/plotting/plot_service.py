import matplotlib.pyplot as plt
from numpy import histogram2d
from numpy import array
from tabulate import tabulate

from matplotlib.ticker import NullFormatter
from modules.plotting.plot_losses_callback import PlotLossesCallback


def plot(h, metric, dir='../plot/', show_only=False):
    fig, ax = plt.subplots()
    epochs = range(len(h[metric]))
    ax.plot(epochs, h[metric], c='blue', label='train')
    ax.plot(epochs, h['val_'+metric], c='green', label='val')
    ax.legend()
    plt.title(metric)
    ax.set_xlabel('epochs')

    if show_only:
        plt.show()
    else:
        plt.savefig(dir + metric + '.pdf')


def plot_hm(x, y, dir='../plot/', show_only=False):
    heatmap, xedges, yedges = histogram2d(x, y, bins=500)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')

    if show_only:
        plt.show()
    else:
        plt.savefig(dir + 'heatmap.pdf')


def plot_scatter(X, Y, X_val, y_val, X_test, Y_test, preds, dir='../plot/', show_only=False):
    plt.clf()
    plt.suptitle("Infrared X Redshift", color='red')

    plt.subplot(221)
    plt.scatter(X[:, 3], Y)
    plt.title("Treino")

    plt.subplot(222)
    plt.scatter(X_val[:, 3], y_val)
    plt.title("Validacao")

    plt.subplot(223)
    plt.scatter(X_test[:, 3], Y_test)
    plt.title("Teste")

    plt.subplot(224)
    plt.scatter(X_test[:, 3], preds)
    plt.title("Predito")

    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

    if show_only:
        plt.show()
    else:
        plt.savefig(dir + 'redshift.png')


def plot_scatter_lr(X, Y, real, pred, dir='../plot/', show_only=False):
    plt.clf()
    plt.suptitle("Infrared X Redshift", color='red')

    plt.scatter(X[:, 3], Y, color='black')
    plt.plot(real, pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    if show_only:
        plt.show()
    else:
        plt.savefig(dir + 'linear_reg.png')


def plot_table(x, y):
    t = tabulate(array([x, y]).T[:50], headers=['Real', 'Predict'], tablefmt='orgtbl')
    print(t)


def plot_table_cf(header, cfs):
    t = tabulate(array([cfs]).T, headers=[header], tablefmt='orgtbl')
    print(t)

def plot_simple_table(data):
    t = tabulate(array([data]).T, tablefmt='orgtbl')
    print(t)

def ann_plot_losses_callback(dir='../plot/', show_only=False):
    cb = PlotLossesCallback()
    cb.set_show_only(show_only)
    cb.set_output_file_dir(dir)

    return cb
