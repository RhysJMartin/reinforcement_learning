from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pickle

ACTION_SPACE = [0, 1, 2]
K = 4
SPEED_RANGE = [-0.07, 0.07]
POSITION_RANGE = [-1.20, 0.60]
RESOLUTION = 10
NUM_ACTIONS = 3

def plot_value_function(value_function, POSITION_RANGE, SPEED_RANGE, RESOLUTION):
    X = np.linspace(POSITION_RANGE[0], POSITION_RANGE[1], RESOLUTION+1)[:-1]
    Y = np.linspace(SPEED_RANGE[0], SPEED_RANGE[1], RESOLUTION+1)[:-1]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    Z = value_function
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('Position')
    ax.set_ylabel('Speed')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == '__main__':
    i = 19900
    file_name = 'results/weights_{}_test.p'.format(i)
    plotatble_weights = pickle.load(open(file_name, 'rb'))
    plot_value_function(plotatble_weights, POSITION_RANGE, SPEED_RANGE, RESOLUTION)
    file_name = 'results/scores_{}_test.p'.format(i)
    plotatble_scores = pickle.load(open(file_name, 'rb'))
    plt.plot(plotatble_scores)
    plt.show()