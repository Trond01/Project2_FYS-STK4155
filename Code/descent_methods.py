from Code.utilities import random_partition, feature_matrix

import jax.numpy as jnp
import numpy as np
from jax import device_put, jit

########################################################################################################################################
########################################################################################################################################
### General descent interface, specialised with various step methods in functions below
########################################################################################################################################
########################################################################################################################################


def _SGD_general(
    X_train,
    y_train,
    X_test,
    y_test,
    grad_method,
    init_func,
    step_func,
    beta0: dict,
    n_epochs=50,
    batch_size=10,
    test_loss_func=None,
    gamma=0.0,
):
    # Get parameter keys
    keys = beta0.keys()

    # Initialise result storage
    result = {}
    if test_loss_func is not None:
        result["train_loss_list"] = [test_loss_func(beta0, X_train, y_train)]
        result["test_loss_list"] = [test_loss_func(beta0, X_test, y_test)]

    # Initialise step
    v = {}
    for key in keys:
        v[key] = jnp.zeros_like(beta0[key])

    # Partition and get number of batches
    m = int(y_train.shape[0] / batch_size)
    batches = random_partition(X_train, y_train, batch_size)

    # Store current beta
    beta_current = beta0.copy()

    # Perform training
    for epoch in range(n_epochs):
        # Accumulation variables
        tools = init_func(epoch, gamma, v)

        for i in range(m):
            # Draw a batch and compute gradients for this sub-epoch
            X_b, y_b = batches[np.random.randint(m)]

            # Divide by batch_size to get avg contribution from training samples
            gradients = grad_method(beta_current, X_b, y_b)
            for key in gradients.keys():
                gradients[key] = gradients[key] / batch_size

            # Perform a step with desired method
            beta_current, tools = step_func(beta_current, tools, gradients)

            # Store sub epoch errors
            if test_loss_func is not None:
                result["train_loss_list"].append(
                    test_loss_func(beta_current, X_train, y_train)
                )
                result["test_loss_list"].append(
                    test_loss_func(beta_current, X_test, y_test)
                )

        gamma = tools["gamma"]
        v = tools["v"]

    # Add betas to result
    result["beta_final"] = beta_current

    return result


############################
####### SGD
############################
def init_SGD(lr):
    tools = {
        "lr": lr,
    }

    return lambda epoch, gamma, v: tools | {"epoch": epoch, "gamma": gamma, "v": v}


def step_SGD(beta_prev, variables, gradients):
    new_beta = {}

    for key in beta_prev.keys():
        # SGD step
        update = variables["lr"] * gradients[key]

        # Perform step, if gamma != 0 it is done with momentum...
        variables["v"][key] = variables["gamma"] * variables["v"][key] + update
        new_beta[key] = beta_prev[key] - variables["v"][key]

    return new_beta, variables


def SGD(
    X_train,
    y_train,
    X_test,
    y_test,
    grad_method,
    beta0: dict,
    n_epochs: int = 50,
    batch_size: int = 10,
    test_loss_func=None,
    lr: float = 0.01,  # learning rate
    gamma: float = 0.0,  # momentum
):
    init_func = init_SGD(lr)

    return _SGD_general(
        X_train,
        y_train,
        X_test,
        y_test,
        grad_method,
        init_func,
        step_SGD,
        beta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        test_loss_func=test_loss_func,
        gamma=gamma,
    )


############################
####### Plain Gradient descent
############################
def GD(
    X_train,
    y_train,
    X_test,
    y_test,
    grad_method,
    beta0: dict,
    n_epochs: int = 50,
    test_loss_func=None,
    lr: float = 0.01,  # learning rate
    gamma: float = 0.0,  # momentum
):
    # USE SGD with full batch
    batch_size = y_train.shape[0]

    return SGD(
        X_train,
        y_train,
        X_test,
        y_test,
        grad_method,
        beta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        test_loss_func=test_loss_func,
        lr=lr,  # learning rate
        gamma=gamma,  # momentum
    )


############################
####### Adagrad
############################
def init_adagrad(lr, weights, delta):
    tools = {
        "eta": lr,
        "r": {},
        "delta": delta,
    }

    # Reset accumulation variables
    for key in weights.keys():
        tools["r"][key] = jnp.zeros_like(weights[key])

    return lambda epoch, gamma, v: tools | {"epoch": epoch, "gamma": gamma, "v": v}


def step_adagrad(beta_prev, variables, gradients):
    new_beta = {}

    for key in beta_prev.keys():
        # Accumulate to total gradient
        variables["r"][key] += gradients[key] * gradients[key]

        # Adagrad scaling, multiply gradient by downscaled learning rate
        lr_times_grad = (
            gradients[key]
            * variables["eta"]
            / (variables["delta"] + jnp.sqrt(variables["r"][key]))
        )

        # Perform step, if gamma != 0 it is done with momentum...
        variables["v"][key] = variables["gamma"] * variables["v"][key] + lr_times_grad
        new_beta[key] = beta_prev[key] - variables["v"][key]

    return new_beta, variables


def SGD_adagrad(
    X_train,
    y_train,
    X_test,
    y_test,
    grad_method,
    beta0: dict,
    n_epochs: int = 50,
    batch_size: int = 10,
    test_loss_func=None,
    lr: float = 0.01,  # learning rate
    gamma: float = 0.0,  # momentum
    delta: float = 1e-8,  # safe div
):
    init_func = init_adagrad(lr, beta0, delta)

    return _SGD_general(
        X_train,
        y_train,
        X_test,
        y_test,
        grad_method,
        init_func,
        step_adagrad,
        beta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        test_loss_func=test_loss_func,
        gamma=gamma,
    )


############################
####### RMS_prop
############################
def init_RMS_prop(lr, weights, delta, rho):
    tools = {
        "eta": lr,
        "Giter": {},
        "delta": delta,
        "rho": rho,
    }

    # Reset accumulation variables
    for key in weights.keys():
        tools["Giter"][key] = 0.0

    return lambda epoch, gamma, v: tools | {"epoch": epoch, "gamma": gamma, "v": v}


def step_RMS_prop(beta_prev, variables, gradients):
    new_beta = {}

    for key in beta_prev.keys():
        # Accumulate
        variables["Giter"][key] = (
            variables["rho"] * variables["Giter"][key]
            + (1 - variables["rho"]) * gradients[key] * gradients[key]
        )

        # RMS prop scaling
        update = (
            gradients[key]
            * variables["eta"]
            / (variables["delta"] + jnp.sqrt(variables["Giter"][key]))
        )

        # Perform step, if gamma != 0 it is done with momentum...
        variables["v"][key] = variables["gamma"] * variables["v"][key] + update
        new_beta[key] = beta_prev[key] - variables["v"][key]

    return new_beta, variables


def SGD_RMS_prop(
    X_train,
    y_train,
    X_test,
    y_test,
    grad_method,
    beta0: dict,
    n_epochs: int = 50,
    batch_size: int = 10,
    test_loss_func=None,
    lr: float = 0.01,  # learning rate
    gamma: float = 0.0,  # momentum
    delta: float = 1e-8,  # safe div
    rho: float = 0.99,
):
    init_func = init_RMS_prop(lr, beta0, delta, rho)

    return _SGD_general(
        X_train,
        y_train,
        X_test,
        y_test,
        grad_method,
        init_func,
        step_RMS_prop,
        beta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        test_loss_func=test_loss_func,
        gamma=gamma,
    )


############################
####### Adam
############################
def init_adam(lr, weights, beta1, beta2, delta):
    tools = {
        "eta": lr,
        "s": {},
        "r": {},
        "beta1": beta1,
        "beta2": beta2,
        "delta": delta,
    }

    # Reset accumulation variables
    for key in weights.keys():
        tools["s"][key] = 0
        tools["r"][key] = 0

    return lambda epoch, gamma, v: tools | {"epoch": epoch, "gamma": gamma, "v": v}


def step_adam(beta_prev, adam_variables, gradients):
    new_beta = {}

    for key in beta_prev.keys():
        # Accumulate and compute firsr and second term
        adam_variables["s"][key] = (
            adam_variables["beta1"] * adam_variables["s"][key]
            + (1 - adam_variables["beta1"]) * gradients[key]
        )
        adam_variables["r"][key] = (
            adam_variables["beta2"] * adam_variables["r"][key]
            + (1 - adam_variables["beta2"]) * gradients[key] * gradients[key]
        )

        first_term = adam_variables["s"][key] / (
            1 - adam_variables["beta1"] ** (adam_variables["epoch"] + 1)
        )
        second_term = adam_variables["r"][key] / (
            1 - adam_variables["beta2"] ** (adam_variables["epoch"] + 1)
        )

        # Adam scaling
        update = (
            adam_variables["eta"]
            * first_term
            / (jnp.sqrt(second_term) + adam_variables["delta"])
        )  # safe division with delta

        # Perform step, if gamma != 0 it is done with momentum...
        adam_variables["v"][key] = (
            adam_variables["gamma"] * adam_variables["v"][key] + update
        )
        new_beta[key] = beta_prev[key] - adam_variables["v"][key]

    return new_beta, adam_variables


def SGD_adam(
    X_train,
    y_train,
    X_test,
    y_test,
    grad_method,
    beta0: dict,
    n_epochs: int = 50,
    batch_size: int = 10,
    test_loss_func=None,
    lr: float = 0.01,
    gamma: float = 0.0,
    delta: float = 1e-8,
    beta1: float = 0.9,
    beta2: float = 0.99,
):
    init_func = init_adam(lr, beta0, beta1, beta2, delta)

    return _SGD_general(
        X_train,
        y_train,
        X_test,
        y_test,
        grad_method,
        init_func,
        step_adam,
        beta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        test_loss_func=test_loss_func,
        gamma=gamma,
    )
