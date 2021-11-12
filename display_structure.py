import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

def displayStructure(shape):
    x = []
    y = []
    z = []
    for k in range(0, len(shape), 3):
        x.append(shape[k:k+3][0])
        y.append(shape[k:k+3][1])
        z.append(shape[k:k+3][2])

    fig = plt.figure()
    ax = Axes3D(fig)
    cset = ax.plot(x, y, z, zdir='z')
    ax.clabel(cset, fontsize=9, inline=1)
    plt.show()

shape = [0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.021, 0.002, 0.068, -0.008, -0.026, 0.036, -0.03, 0.026, 0.089, -0.029, 0.012, 0.048, 0.004, 0.022, 0.04]
displayStructure(shape)

