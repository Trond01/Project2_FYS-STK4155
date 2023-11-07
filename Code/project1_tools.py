import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4  # + np.random.normal(scale=0.1,size=x.shape)


def feature_matrix_2d(x, y, num_features):
    """
    Generates a feature matrix based on x, y and the number of features.

    The function constructs features by taking combinations of powers of x and y,
    starting with the highest power of x.

    x: A 2D array with a single column of input values.
    y: A 2D array with a single column of input values.
    num_features: The number of feature columns.

    X: The feature matrix,a 2D numpy array with a column for each feature
    """
    X = np.zeros((len(x), num_features))

    deg = 0
    n = 0
    while True:
        for i in range(0, deg + 1):
            # Add feature row
            X[:, n] = x[:, 0] ** (deg - i) * y[:, 0] ** i

            # Increment number of points done, and end if needed
            n += 1
            if n == num_features:
                return X

        deg += 1


def scale_feature_matrix(X):
    """Scales feature matrix using mean/variance scaling.

    Args:
        X (Matrix): Feature Matrix

    Returns:
        Matrix: Scaled feature matrix
    """
    means = np.mean(X, axis=0).reshape((1, X.shape[1]))
    means[0, 0] = 0
    var = np.var(X, axis=0).reshape((1, X.shape[1]))
    var[0, 0] = 1
    X_copy = np.copy(X - means)
    X_copy = X_copy / np.sqrt(var)
    return X_copy, means, var


def r2_sampling(num_points, sigma2=0):
    """
    To add noise, input sigma2 > 0.
    """

    x = np.random.random((num_points, 1))
    y = np.random.random((num_points, 1))

    z = FrankeFunction(x, y) + np.random.normal(
        0, np.sqrt(sigma2), size=(num_points, 1)
    )

    return {"x": x, "y": y, "z": z}


def plot_surface(x, y, z, filename=None):
    # Init figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if filename:
        plt.savefig(filename)

    plt.show()
