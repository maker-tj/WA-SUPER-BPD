import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def draw_direction(image, direction, b_heigh, b_width, M, N):



    heigh = M #image.shape[0]
    width = N #image.shape[1]

    plt.figure(1)
    x0 = np.arange(0, width, 1)
    x1 = np.arange(0, heigh, 1)
    X, Y = np.meshgrid(x0, x1)
    X = X.flatten()
    Y = Y.flatten()

    plt.xticks(x0)
    plt.yticks(x1)

    ax = plt.gca()

    grad1 = direction[:, b_heigh:b_heigh+M, b_width:b_width+N]
    grad = grad1.transpose(1, 2, 0)
    grad = list(reversed(grad.tolist()))
    grad = np.array(grad)
    grad = grad.transpose(2, 0, 1)
    grad_0 = grad[0].flatten()
    grad_1 = grad[1].flatten()

    plt.quiver(X, Y, grad_0, grad_1, angles="xy", color="#666666")

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, width+1, 1))
    ax.set_yticklabels(np.arange(1, heigh+1, 1))
    # Major ticks
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, heigh, 1))
    # Minor ticks
    ax.set_xticks(np.arange(-.5, width-0.5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, heigh-0.5, 1), minor=True)

    image_crop = image[b_heigh:b_heigh+M,b_width:b_width+N,:]

    plt.grid(linewidth=0.15, which='minor', axis='both')
    c = list(reversed(image_crop.tolist()))
    plt.imshow(c, origin='lower',aspect='equal', alpha = 1)
    plt.draw()
    plt.savefig('images/my_images/1.png')
    plt.show()

    ###########################################
    plt.figure(2)
    x0 = np.arange(0, width, 1)
    x1 = np.arange(0, heigh, 1)
    X, Y = np.meshgrid(x0, x1)
    X = X.flatten()
    Y = Y.flatten()

    plt.xticks(x0)
    plt.yticks(x1)

    ax = plt.gca()

    grad1 = direction[:, b_heigh:b_heigh+M, b_width:b_width+N]
    grad = grad1.transpose(1, 2, 0)
    grad = list(reversed(grad.tolist()))
    grad = np.array(grad)
    grad = grad.transpose(2, 0, 1)
    grad_0 = grad[0].flatten()
    grad_1 = grad[1].flatten()
    grad_norm = np.sqrt(grad_0 ** 2 + grad_1 ** 2) + 0.000001
    aa = grad_0 / grad_norm
    plt.quiver(X, Y, grad_0 / grad_norm, grad_1 / grad_norm, angles="xy", color="#666666")

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, width+1, 1))
    ax.set_yticklabels(np.arange(1, heigh+1, 1))
    # Major ticks
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, heigh, 1))
    # Minor ticks
    ax.set_xticks(np.arange(-.5, width-0.5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, heigh-0.5, 1), minor=True)

    image_crop = image[b_heigh:b_heigh+M, b_width:b_width+N, :]

    plt.grid(linewidth=0.15, which='minor', axis='both')
    c = list(reversed(image_crop.tolist()))
    plt.imshow(c, origin='lower',aspect='equal', alpha = 1)
    plt.draw()
    plt.savefig('images/my_images/2.png')
    plt.show()

