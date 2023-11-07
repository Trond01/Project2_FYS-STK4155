from Code.activation_functions import sigmoid
from Code.utilities import MSELoss_method, jax_loss_grad
from Code.descent_methods import SGD_adam

import numpy as np
import jax.numpy as jnp


def _beta_init(layer_list):
    """
    layer list, eg [2, 10, 1] for 2 input, 10 hidden neurons and 1 output
    """

    beta0 = {}

    # Add random initialisation
    for i in range(1, len(layer_list)):
        # Weight matrix
        beta0[f"W{i}"] = np.random.random((layer_list[i - 1], layer_list[i]))

        # Bias vector
        beta0[f"b{i}"] = np.random.random(layer_list[i])

    return beta0


def neural_network_model(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    """
    Function to evaluate the neural network prediction for feature matrix X
    """
    # First layer = input
    out = X.copy()

    # For each remaining layer we propagate forward
    print(beta.keys())
    print(len(beta.keys()) // 2)
    for i in range(1, len(beta.keys()) // 2):  # for each layer
        print("Hidden")
        print(out)
        # Dot with weights, add biases, apply activation function
        out = activation(jnp.add(jnp.dot(out, beta[f"W{i}"]), beta[f"b{i}"]))


    out_final = output_activation(jnp.add(
        jnp.dot(out, beta[f"W{len(beta.keys())//2}"]), beta[f"b{len(beta.keys())//2}"]
    ))

    return out_final


def neural_network_train(
    X_train,
    y_train,
    X_test,
    y_test,
    n_epochs=5,
    lr=0.01,
    gamma=0,
    batch_size=10,
    hidden_layer_list=[10],
    hidden_activation=sigmoid,
    output_activation=None,  ### TODO, actually implement this??
    descent_method=SGD_adam,
):
    
    # Dictionary for storing result
    result = {}  

    # Find layer structure and initialise beta0
    layer_list = [X_train.shape[1]] + hidden_layer_list + [y_train.shape[1]]
    beta0 = _beta_init(layer_list)

    # Construct the loss and gradient functions
    _neural_network_loss_func = MSELoss_method(
        lambda beta, X: neural_network_model(beta, X, activation=hidden_activation)
    )
    _neural_network_loss_grad = jax_loss_grad(_neural_network_loss_func)

    # Perform training
    result.update(
        descent_method(
            X_train,
            y_train,
            X_test,
            y_test,
            n_epochs=n_epochs,
            batch_size=batch_size,
            beta0=beta0,
            lr=lr,
            gamma=gamma,
            grad_method=_neural_network_loss_grad,
            test_loss_func=_neural_network_loss_func,
        )
    )

    return result
