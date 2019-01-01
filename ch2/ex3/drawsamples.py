import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def main(n_points, n_samples):
    grid = np.linspace(0, 1, num=n_points)

    mean = np.zeros_like(grid)
    cov = np.minimum(grid[:,None], grid[None,:]) - np.outer(grid, grid)
    samples = scipy.stats.multivariate_normal.rvs(
        mean=mean, cov=cov, size=n_samples)

    for s in samples:
        plt.plot(grid, s)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 3:
        _, n_points_str, n_samples_str = sys.argv
        main(int(n_points_str), int(n_samples_str))
    else:
        print('Usage: python drawsamples.py n_points n_samples')
