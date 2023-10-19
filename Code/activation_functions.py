import numpy as np
import jax.numpy as jnp

def sigmoid(x):
    return 1 / (1+jnp.exp(-x))

def relu(x):
    return jnp.maximum(0, x)

def leaky_relu(x, alpha = 0.5):
    return jnp.maximum(0, x) + alpha*jnp.minimum(x, 0)

def leaky_relu_stochastic(x, alpha_min=0, alpha_max=1):
    alpha = np.random.uniform(alpha_min, alpha_max)
    return leaky_relu(x, alpha=alpha)