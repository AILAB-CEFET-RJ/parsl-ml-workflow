import matplotlib.pyplot as plt
from numpy import histogram2d
from numpy import array
from tabulate import tabulate


def plot(h, metric, dir='../plot/'):
    fig, ax = plt.subplots()
    epochs = range(len(h[metric]))
    ax.plot(epochs, h[metric], c='blue', label='train')
    ax.plot(epochs, h['val_'+metric], c='green', label='test')
    ax.legend()
    plt.title(metric)
    ax.set_xlabel('epochs')
    plt.savefig(dir + metric + '.pdf')

def plot_hm(x, y, dir='../plot/'):
    heatmap, xedges, yedges = histogram2d(x, y, bins=500)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.savefig(dir + 'heatmap.pdf')

def plot_scatter(X, Y, X_test, Y_test, preds, dir='../plot/'):
    plt.subplot(3, 1, 1)
    plt.ylabel("Redshift")
    plt.scatter(X[:, 3], Y)
    plt.ylabel("Treino")

    plt.subplot(3, 1, 2)
    plt.scatter(X_test[:, 3], Y_test)
    plt.ylabel("Teste")

    plt.subplot(3, 1, 3)
    plt.scatter(X_test[:, 3], preds)
    plt.ylabel("Predito")
    plt.savefig(dir + 'redshift.png')

def plot_table(x, y):
    t = tabulate(array([x, y]).T[:50], headers=['Real', 'Predict'], tablefmt='orgtbl')
    print(t)