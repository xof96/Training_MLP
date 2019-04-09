import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # loss_file = sys.argv[1]

    plt.plot([1, 2], [14, 3])
    plt.plot([0, 1], [3, 5])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
