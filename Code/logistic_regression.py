from utilities import jax_loss_grad
import jax.numpy as jnp
import pandas as pd

def log_loss(y_pred, y_i):
    y_pred = jnp.reshape(y_pred, (-1, 1))
    y_i = jnp.reshape(y_i, (-1, 1))
    return jnp.sum(-y_i*jnp.log(y_pred) - (1 - y_i)*jnp.log(1 - y_pred))/y_i.shape[0]


def logistic_loss_func(model):
    return (lambda beta, X, y: log_loss(model(beta, X), y))


def logistic_grad(model):
    return jax_loss_grad(logistic_loss_func(model))


def import_breast_cancer():
    header = ["id", "thickness", "uni_cell_s", "uni_cell_sh", "marg_adh", "single_epithel", "bare_nuc", "bland_chromatin", "normal_nuc", "mitoses", "target"]
    data = pd.read_csv("breast-cancer-wisconsin.data", names=header)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    data["target"] = data["target"] == 4
    y = jnp.array(data["target"], dtype=int)
    X = jnp.array(data.to_numpy()[:, 1:-1], dtype=jnp.float64)
    return X, y


def accuracy(y_pred, y_i):
    y_pred_bool = jnp.squeeze(y_pred >= 0.5)
    y_i_squeezed = jnp.squeeze(y_i)
    return jnp.sum(1 - jnp.abs(y_i_squeezed - y_pred_bool)) / y_i.shape[0]


def accuracy_func(model):
    return (lambda beta, X, y: accuracy(model(beta, X), y))


def loss_func_creator(model, loss_compute):
    return (lambda beta, X, y: loss_compute(model(beta, X), y))