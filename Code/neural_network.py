from descent_methods import *


# TODO


def train_nn(n_layers = 3)

    """
    ta inn shape?

    """


def jax_loss_grad(loss_func):
    return grad(loss_func)


def model_2(beta, X):
    X_1 = jnp.dot(X, beta["W1"])
    X_1 = nn.sigmoid(X_1)
    X_2 = jnp.dot(X_1, beta["W2"]) + beta["b2"]
    X_2 = nn.sigmoid(X_2)
    return X_2


OLS_loss_2 = MSELoss_method(model=model_2)

MSE_grad = jax_loss_grad(OLS_loss_2)

Ridge_grad_jax = jax_loss_grad(loss_func=Ridge_loss_method(0.1, model=model))
OLS_grad_jax = jax_loss_grad(loss_func=MSELoss_method(model))


num_params = 5
middle_layer  = 10
num_points = 100

beta_try = {"W1":np.random.random((num_params, middle_layer)), "W2":np.random.random((middle_layer, 1)), "b2":np.random.random((1, 1))}



beta0 = np.random.random((num_params, 1))
x = np.random.random((num_points, 1))
y = f(x)


