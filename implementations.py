import matplotlib.pyplot as plt
import numpy as np
import time
from implementations_helpers import *

### Required Project Functions

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm.
        y: data array;
        tx: transposed x data;
        initial_w : initial weight array 
        max_iters : convergence criterion
        gamma : step size"""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w -gamma*grad
    return w, mse_loss(y, tx, w)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm.
        y: data array;
        tx: transposed x data;
        initial_w : initial weight array 
        max_iters : convergence criterion
        gamma : step size"""
    batchsize = 1 # if != 1 : mini-batch stochastic gradient descent
    
    # not to modify input arrays
    
    w = np.copy(initial_w)
    y1 = np.copy(y)
    tx1 = np.copy(tx)
    
    #####
    
    for n_iter in range(max_iters):
        concat = np.vstack((y1,tx1.T)) # prepare to shuffle y and tx together
        np.random.shuffle(concat.T) 
        y1, tx1 = concat[0], concat[1:].T # decouple y and tx after shuffle is completed 
        grad = compute_stochastic_gradient(y1, tx1, w, batchsize) #differs from compute_gradient because it chooses a batch 
        w = w - gamma*grad
    return w, mse_loss(y, tx, w)


def least_squares(y, tx):
    """calculate the least squares solution.
        y: data array;
        tx: transposed x data;
        Returns mse, and optimal weights
        Also prints execution time for comparison with GD"""
    start_time = time.time()
    #try to invert the matrix, if singular calculate pseudo-inverse instead
    try:
        w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError as err:
        A = np.dot(np.transpose(tx),tx)
        inverse = np.linalg.pinv(A)
        w = np.dot(np.dot(inverse,np.transpose(tx)),y)
        
    print("Execution time=%s seconds" % (time.time() - start_time))
    return w, mse_loss(y, tx, w)

def ridge_regression(y, tx, lambda_):
    '''y: output data
        tx: transposed input data vector
        lambda_: ridge parameter multiplying the L-2 norm
        Returns mse, and optimal weights'''
    #try to invert the matrix, if singular calculate pseudo-inverse instead
    try:
        w = np.linalg.solve(tx.T.dot(tx) + lambda_*(2*tx.shape[0])*np.identity(tx.shape[1]), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError as err:
        A = np.dot(np.transpose(tx),tx) + lambda_*(2*tx.shape[0])*np.identity(tx.shape[1])
        inverse = np.linalg.pinv(A)
        w = np.dot(np.dot(inverse,np.transpose(tx)),y)
        
    return w, mse_loss(y,tx,w)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression by gradient descent
        y: data array;
        tx: transposed x data;
        Returns mse, and optimal weights
    """
    
    w = np.copy(initial_w)
    
    loss = calculate_loss_logistic(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient_logistic(y, tx, w)
    # compute the gradient
    
    #hess = calculate_hessian_logistic(y, tx, w)
    # compute the hessian, for generalizations
    
    threshold = 1e-8
    
     
    previous_loss = 0
    
    for iter_ in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if iter_ > 1 and np.abs(loss - previousloss) < threshold:
            break
        previousloss = loss
    return w, mse_loss(y,tx,w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression by gradient descent
        y: data array;
        tx: transposed x data;
        Penalized w/ parameter lambda_
        Returns mse, and optimal weights"""
    
    w = np.copy(initial_w)
    
    loss = calculate_loss_logistic(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient_logistic(y, tx, w)
    # compute the gradient
    
    #hess = calculate_hessian_logistic(y, tx, w)
    # compute the hessian
    
    threshold = 1e-8
    
    previousloss = 0
    
    for iter_ in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if iter_ > 1 and np.abs(loss - previousloss) < threshold:
            break
        previousloss = loss
        
    return w, mse_loss(y,tx,w)



### Extra

def bayes_regression(y, tx, lambda_, q):
    '''y: output data
        tx: transposed input data vector
        lambda_: ridge parameter multiplying the L-2 norm
        q: L-q norm penalty
        Returns mse, and optimal weights'''
    #try to invert the matrix, if singular calculate pseudo-inverse instead
    w_estimate = ridge_regression(y, tx, lambda_)[0] #estimates w by ridge
    wL2 = np.sqrt(w_estimate**2)
    
    try:
        w = np.linalg.solve(tx.T.dot(tx) + lambda_ * tx.shape[0] * q *( wL2**(q-2) )*np.identity(tx.shape[1]), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError as err:
        A = tx.T.dot(tx) + lambda_ * tx.shape[0] * q *( wL2**(q-2) )*np.identity(tx.shape[1]), tx.T.dot(y)
        inverse = np.linalg.pinv(A)
        w = np.dot(np.dot(inverse,np.transpose(tx)),y)
    
    return w, mse_loss(y,tx,w)
