import jax.numpy as jnp
from jax import grad, jit
import pandas as pd


def log_loss(y_pred, y_i):
    y_pred = jnp.clip(jnp.reshape(y_pred, (-1, 1)), 1e-15, 1 - 1e-15)
    y_i = jnp.reshape(y_i, (-1, 1))
    
    return (
        jnp.mean(-y_i * jnp.log(y_pred) - (1 - y_i) * jnp.log(1 - y_pred))
    )


def logistic_loss_func(model):
    return lambda beta, X, y: log_loss(model(beta, X), y)


def logistic_grad(model):
    return grad(logistic_loss_func(model))


def import_breast_cancer(filename="../Code/Data/breast-cancer-wisconsin.data"):
    """
    default filename assumes file is run from one folder deep from one folder outside Code ...
    """


    header = [
        "id",
        "thickness",
        "uni_cell_s",
        "uni_cell_sh",
        "marg_adh",
        "single_epithel",
        "bare_nuc",
        "bland_chromatin",
        "normal_nuc",
        "mitoses",
        "target",
    ]
    data = pd.read_csv(filename, names=header)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    data["target"] = data["target"] == 4
    y = jnp.array(data["target"], dtype=int)
    X = jnp.array(data.to_numpy()[:, 1:-1], dtype=jnp.float64)
    return X, y


def true_positive_threshold(y_pred, y_i, th):
    y_pred_bool = jnp.squeeze(y_pred >= th)
    y_i_squeezed = jnp.squeeze(y_i)
    return jnp.sum(y_i_squeezed*(y_pred_bool)) / jnp.count_nonzero(y_i_squeezed)


def false_positive_threshold(y_pred, y_i, th):
    y_pred_bool = jnp.squeeze(y_pred >= th)
    y_i_squeezed = jnp.squeeze(y_i)
    return jnp.sum((1 - y_i_squeezed)*y_pred_bool) / (jnp.count_nonzero(1 - y_i_squeezed))


def accuracy(y_pred, y_i):
    y_pred_bool = jnp.squeeze(y_pred >= 0.5)
    y_i_squeezed = jnp.squeeze(y_i)
    return jnp.sum(1 - jnp.abs(y_i_squeezed - y_pred_bool)) / y_i.shape[0]


def true_pos_variable_func(model, th):
    return lambda beta, X, y: true_positive_threshold(model(beta, X), y, th)


def false_pos_variable_func(model, th):
    return lambda beta, X, y: false_positive_threshold(model(beta, X), y, th)


def accuracy_func(model):
    return lambda beta, X, y: accuracy(model(beta, X), y)


def loss_func_creator(model, loss_compute):
    return lambda beta, X, y: loss_compute(model(beta, X), y)


def ridge_term(beta):
    s = 0.0
    for key in beta.keys():
        s += jnp.sum(jnp.power(beta[key], 2))
    return s
    


def log_loss_ridge(model, lam):
    log_loss_func = logistic_loss_func(model=model)
    return (lambda beta, X, y: jnp.add(log_loss_func(beta, X, y), lam*ridge_term(beta)))