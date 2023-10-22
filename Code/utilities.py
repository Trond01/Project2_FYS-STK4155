import jax.numpy as jnp
from jax import grad, nn
import numpy as np
from matplotlib import pyplot as plt

def feature_matrix(x, num_features):
    """
    x: array with x values
    num_features: the degree of polynomial 1+x+...+x^p

    returns:
    X: The feature matrix,a 2D numpy array with a column for each feature
    """

    return jnp.array([x**i for i in range(num_features)]).T[0]


def random_partition(X, y, batch_size):
    batches = []
    n = y.shape[0]
    m = int(n / batch_size)
    for i in range(m):
        index = list(range(i*batch_size, (i+1)*batch_size))
        batches.append((X[index, :], y[index]))

    return batches


def train_test_split(X, Y, percentage, test_index=None):
    """
    X: Feature matrix
    Y: Label vector(size=(n, 1))
    Percentage: How much of the dataset should be used as a test set.
    """

    n = X.shape[0]
    if test_index is None:
        test_index = np.random.choice(n, round(n * percentage), replace=False)
    test_X = X[test_index]
    test_Y = Y[test_index]

    train_X = np.delete(X, test_index, axis=0)
    train_Y = np.delete(Y, test_index, axis=0)

    return train_X, train_Y, test_X, test_Y, test_index


def jax_loss_grad(loss_func):
    return grad(loss_func)


def Ridge_loss_method(lam, model):
    return (lambda beta, X, y : Ridge_Loss(beta, X, y, model, lam))


def Ridge_Loss(beta, X, y, model, lam=0.01):
    return MSELoss(model(beta, X), y) + jnp.sum(jnp.power(beta, 2))*(lam/(2*jnp.size(y)))


def MSELoss_method(model):
    return (lambda beta, X, y : MSELoss(model(beta, X), y))


def MSELoss(y, y_pred):
    """MSE loss of prediction array.

    Args:
        y (ndarray): Target values
        y_pred (ndarray): Predicted values

    Returns:
        float: MSE loss
    """
    return jnp.sum(jnp.power(y - y_pred, 2)) / y.shape[0]

def OLS_grad(beta, X, y, model):
    n = y.shape[0]
    return 2*(np.dot(X.T, ( model(beta, X) - y))) / n

# def RIDGE_grad(beta, X, y, model):
#     n = y.shape[0]
#     return 2*(np.dot(X.T, ( model(beta, X) - y))) / n

def MSE_grad(model):
    return (lambda beta, X, y : OLS_grad(beta, X, y, model))


def plot_test_results(test_loss_list, train_loss_list, m):
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))  # 1 row, 2 columns

    # Subplot 1
    axs[0].plot(test_loss_list, label="test")
    axs[0].plot(train_loss_list, label="train")
    axs[0].set_xlabel("Training step")
    axs[0].set_ylabel("MSE")
    axs[0].set_title("Over all sub-epochs")
    axs[0].legend()

    # Subplot 2
    axs[1].plot(test_loss_list[::m], label="test")
    axs[1].plot(train_loss_list[::m], label="train")
    axs[1].set_xlabel("Training step")
    axs[1].set_title("End of epoch error")
    axs[1].legend()

    plt.tight_layout()
    plt.show()