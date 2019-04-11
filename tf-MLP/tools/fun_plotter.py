import sys
import numpy as np
import matplotlib.pyplot as plt
from mnist_loss import *

if __name__ == '__main__':
    n = 5000
    x = [1]
    for i in range(int(n / 100) - 1):
        x.append(x[i] + 100)

    plt.plot(x, sigmoid, label='sigmoid')
    plt.plot(x, tanh, label='tanh')
    plt.plot(x, relu, label='relu')
    plt.plot(x, leaky_relu, label='leaky_relu')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('MNIST-5000 Loss Chart')
    plt.grid(True)
    plt.legend()
    plt.show()
