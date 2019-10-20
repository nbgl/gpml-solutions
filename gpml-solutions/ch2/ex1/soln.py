import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

TRAIN_X = np.array([-4, -3, -1, 0, 2], dtype=float)
TRAIN_Y = np.array([-2, 0, 1, 2, -1], dtype=float)

EPSILON = 1e-8

np.random.seed(3141592)


def squared_exponential_kernel(x, x_):
    # Normally we'd use sklearn.gaussian_process.kernels.RBF but let's
    # implement it ourselves here.
    retval = x[...,np.newaxis] - x_[np.newaxis]

    # Inplace operations speed things up
    np.square(retval, out=retval)
    retval *= -.5
    np.exp(retval, out=retval)

    return retval


def draw_multivariate_normal(mean, covariance, size=1):
    # This exists in scipy.stats.multivariate_normal, but let's
    # implement it here as an exercise.
    assert len(mean.shape) == 1
    assert len(covariance.shape) == 2

    n = mean.shape[0]
    assert n == covariance.shape[0]
    assert n == covariance.shape[1]

    # Add a small multiple of the identity matrix to the covariance to
    # improve numerical stability before Cholesky decomposition. See
    # section A.2 of the book.
    covariance = covariance + np.eye(n) * EPSILON
    L = scipy.linalg.cholesky(covariance, lower=True, overwrite_a=True)

    # See section A.2 of the book.
    scalar_samples = np.random.normal(size=size * n).reshape(size, n)
    return mean + scalar_samples @ L.T


def draw_sampled_lines(mean, covariance,
                       x_samples,
                       n_lines=1,
                       plot_variance=True, dotted=False):
    y_samples = draw_multivariate_normal(mean, covariance, size=n_lines)

    if plot_variance:
        std = np.sqrt(np.diag(covariance))
        upper = mean + std * 2
        lower = mean - std * 2
        plt.fill_between(x_samples, upper, lower, color=(.7,.7,.7))

    for y_line in y_samples:
        plt.plot(x_samples, y_line, 'k.' if dotted else 'k-', ms=2)


def panel_a(start=-5, stop=5,
            n_dotted_lines=1, n_solid_lines=2,
            n_grid_points_dotted=51, n_grid_points_solid=501):
    dotted_x_samples = np.linspace(start, stop, n_grid_points_dotted)
    dotted_y_mean = np.zeros(n_grid_points_dotted)
    dotted_y_covariance = squared_exponential_kernel(dotted_x_samples,
                                                     dotted_x_samples)
    draw_sampled_lines(dotted_y_mean, dotted_y_covariance,
                       dotted_x_samples,
                       n_lines=n_dotted_lines,
                       plot_variance=False, dotted=True)

    solid_x_samples = np.linspace(start, stop, n_grid_points_solid)
    solid_y_mean = np.zeros(n_grid_points_solid)
    solid_y_covariance = squared_exponential_kernel(solid_x_samples,
                                                     solid_x_samples)
    draw_sampled_lines(solid_y_mean, solid_y_covariance,
                       solid_x_samples,
                       n_lines=n_solid_lines)
    
    plt.xlabel('input, x')
    plt.ylabel('output, f(x)')
    plt.show()


def panel_b(start=-5, stop=5, n_lines=3, n_grid_points=501):
    x_samples = np.linspace(start, stop, n_grid_points)
    y_mean = np.zeros(n_grid_points)

    K__ = squared_exponential_kernel(x_samples, x_samples)
    K_ = squared_exponential_kernel(TRAIN_X, x_samples)
    K = squared_exponential_kernel(TRAIN_X, TRAIN_X)

    # Assume no noise in training set
    KinvK_ = scipy.linalg.solve(K, K_, assume_a='pos')
    y_mean = KinvK_.T @ TRAIN_Y
    y_covariance = K__ - K_.T @ KinvK_

    draw_sampled_lines(y_mean, y_covariance, x_samples, n_lines=n_lines)
    
    plt.xlabel('input, x')
    plt.ylabel('output, f(x)')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python soln.py (a|b)')
    else:
        _, panel_number = sys.argv
        if panel_number == 'a':
            panel_a()
        elif panel_number == 'b':
            panel_b()
        else:
            print('Usage: python soln.py (a|b)')