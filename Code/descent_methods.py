import jax.numpy as jnp
from utilities import random_partition, feature_matrix
import numpy as np

def SGD(grad_method, X, y, beta0:dict, n_epochs,  batch_size, test_loss_func = None, lr=0.01, 
        gamma=0.0):

    """

    gamma = 0 to avoid momentum...
        
    """

    # Get parameter keys
    keys = beta0.keys()

    # Initialise result storage
    result = {}
    if test_loss_func is not None:
        result["loss_list"]  = [test_loss_func(beta0, X, y)]
    betas = [beta0]

    # Initialise step
    v = {}
    for key in keys:
        v[key] = jnp.zeros_like(beta0[key])
    
    # Partition and get number of batches
    m = int(y.shape[0] / batch_size)
    batches = random_partition(X, y, batch_size)   

    # Perform training
    for epoch in range(n_epochs):

        # Draw from partition m times
        for j in range(m):

            # Storage of new params
            new_beta = {}

            # Draw a batch and compute gradients for this sub-epoch
            X_b, y_b = batches[np.random.randint(m)]
            gradients = grad_method(betas[-1], X_b, y_b)

            # Update at each key
            for key in keys:
                v[key] = gamma*v[key] + lr*gradients[key]
                new_beta[key] = betas[-1][key] - v[key] 

            # Store result of sub epoch
            betas.append(new_beta)
            if test_loss_func:
                result["loss_list"].append(test_loss_func(betas[-1], X, y))

    # Add final beta to results
    result["beta_list"] = betas

    return result


def GD(grad_method, X, y, beta0:dict, n_epochs, test_loss_func = None, lr=0.01, gamma=0.0):
    
    # USE SGD with full batch
    batch_size = y.shape[0]

    return SGD(grad_method, X, y, beta0, n_epochs, batch_size=batch_size, 
               test_loss_func = test_loss_func, lr=lr, gamma=gamma)


def SGD_adagrad(grad_method, X, y, beta0, lr,  n_epochs, batch_size,test_loss_func = None, gamma=0, delta = 1e-8):

    # Get parameter keys
    keys = beta0.keys()

    # Initialise result storage
    result = {}
    if test_loss_func is not None:
        result["loss_list"]  = [test_loss_func(beta0, X, y)]
    betas = [beta0]

    # Initialise step
    v = {}
    for key in keys:
        v[key] = jnp.zeros_like(beta0[key])


    # Partition and get number of batches
    m = int(y.shape[0] / batch_size)
    batches = random_partition(X, y, batch_size)   

    # Initial value for the tuneable learning rate
    eta = lr

    # Initialise accumulation variables
    r = {}

    # Perform training
    for epoch in range(n_epochs):

        # reset accumulation variable
        for key in keys:
            r[key] = jnp.zeros_like(beta0[key])

        for i in range(m):

            # Storage of new params
            new_beta = {}

            # Draw a batch and compute gradients for this sub-epoch
            X_b, y_b = batches[np.random.randint(m)]
            gradients = grad_method(betas[-1], X_b, y_b)

            for key in keys:
                # Add to total gradient                
                r[key] += gradients[key]*gradients[key]

                # Adagrad scaling, learning rate is scaled down, append new result
                lr_times_grad = eta/(delta+jnp.sqrt(r[key])) * gradients[key]

                # Perform step
                v[key] = gamma * v[key] + lr_times_grad

                new_beta[key] = betas[-1][key] - v[key]

            # Store result of sub epoch
            betas.append(new_beta)
            if test_loss_func:
                result["loss_list"].append(test_loss_func(betas[-1], X, y))

    
    # Add betas to result
    result["beta_list"] = betas
    
    return result

def SGD_RMS_prop(grad_method, X, y, beta0, lr,  n_epochs, batch_size,test_loss_func = None, gamma=0, delta = 1e-8, rho = 0.99):

    # Get parameter keys
    keys = beta0.keys()

    # Initialise result storage
    result = {}
    if test_loss_func is not None:
        result["loss_list"]  = [test_loss_func(beta0, X, y)]
    betas = [beta0]

    # Initialise step
    v = {}
    for key in keys:
        v[key] = jnp.zeros_like(beta0[key])
   
    # Partition and get number of batches
    m = int(y.shape[0] / batch_size)
    batches = random_partition(X, y, batch_size)   

    # Initial value for the tuneable learning rate
    eta = lr

    # Accumulation variables
    v = {}
    for key in keys:
        v[key] = jnp.zeros_like(beta0[key])
    Giter = {}

    # Perform training
    for i in range(n_epochs):

        # Reset giter
        for key in keys:
            Giter[key] = 0

        # Iterate over the batches
        for j in range(m):
            
            # Storage of new params
            new_beta = {}

            # Draw a batch and compute gradients for this sub-epoch
            X_b, y_b = batches[np.random.randint(m)]
            gradients = grad_method(betas[-1], X_b, y_b)
            
            for key in keys:

                # Evaluate gradient at previous x
                grad = (1.0/batch_size)*gradients[key]

                Giter[key] = (rho*Giter[key]+(1-rho)*grad*grad)

                update = grad*eta/(delta+jnp.sqrt(Giter[key]))
                # Perform step
                v[key] = gamma * v[key] + update

                new_beta[key] = betas[-1][key] - v[key]
            
            # Store result of sub epoch
            betas.append(new_beta)
            if test_loss_func:
                result["loss_list"].append(test_loss_func(betas[-1], X, y))

    # Add betas to result
    result["beta_list"] = betas

    return result


def SGD_adam(grad_method, X, y, beta0:dict, n_epochs,  batch_size, test_loss_func=None, lr=0.01, gamma=0.0, delta = 1e-8, beta1=0.9, beta2=0.99):

    # Get parameter keys
    keys = beta0.keys()

    # Initialise result storage
    result = {}
    if test_loss_func is not None:
        result["loss_list"]  = [test_loss_func(beta0, X, y)]
    betas = [beta0]
    
    # Initialise step
    v = {}
    for key in keys:
        v[key] = jnp.zeros_like(beta0[key])
    
    # Partition and get number of batches
    m = int(y.shape[0] / batch_size)
    batches = random_partition(X, y, batch_size)

    # Initial value for learning rate
    eta = lr

    # Initialise accumulation variables
    s = {}
    r = {}

    # Perform training
    for epoch in range(n_epochs):

        # Accumulation variables
        for key in keys:
            s[key] = 0
            r[key] = 0

        for i in range(m):
            
            new_beta = {}

            # Draw a batch and compute gradients for this sub-epoch
            X_b, y_b = batches[np.random.randint(m)]
            gradients = grad_method(betas[-1], X_b, y_b)

            for key in keys:
                # Accumulate
                s[key] = beta1*s[key] + (1-beta1)*gradients[key]
                r[key] = beta2*r[key] + (1-beta2)*gradients[key]*gradients[key]

                first_term = s[key]/(1-beta1**(epoch+1))            
                second_term = r[key]/(1-beta2**(epoch+1))

                # Adam scaling
                update = eta*first_term / (jnp.sqrt(second_term) + delta) # safe division with delta

                v[key] = gamma * v[key] + update
                # Perform step
                new_beta[key] = betas[-1][key] - v[key]
   
            # Store result of sub epoch
            betas.append(new_beta)
            if test_loss_func:
                result["loss_list"].append(test_loss_func(betas[-1], X, y))
        
    # Add betas to result
    result["beta_list"] = betas

    return result