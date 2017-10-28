import matplotlib.pyplot as plt
import numpy as np

### Helper Functions for implementations


#Cost functions

def mse_loss(y, tx, w):
    """Calculates the MSE loss function
        y: data
        tx : x^T
        w: array of weights
        """
    e = y - np.dot(tx, w)
    return np.dot(np.transpose(e), e)/2/np.size(y)


#Gradient descent

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    #print (tx.T.shape, y.shape, w.shape)
    return -np.dot(tx.T, y - np.dot(tx, w))/np.size(y)


#Stochastic Gradient Descent

def compute_stochastic_gradient(y, tx, w, batchsize):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    #print (tx.T[:,:batchsize+1].shape, y[:batchsize+1].shape, tx[:batchsize+1,:].shape, w.shape )
    return -np.dot(tx.T[:,:batchsize+1], y[:batchsize+1] - np.dot(tx[:batchsize+1,:], w))/np.size(y[:batchsize+1])

#Logistic Regression

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/ (1 + np.exp(-t) ) #np.exp(t)/(1+np.exp(t))

def calculate_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    return sum(np.log(1+np.exp(np.dot(tx,w))) - y*np.dot(tx,w))

def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T, sigmoid(np.dot(tx,w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
    
    loss = calculate_loss_logistic(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient_logistic(y, tx, w)
    # compute the gradient
    
    # update w
    w = w-gamma*grad
    
    return loss, w

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
    
    loss = calculate_loss_logistic(y, tx, w) + lambda_/2*np.dot(w,w)
    # compute the cost
    
    grad = calculate_gradient_logistic(y, tx, w) + lambda_*w
    # compute the gradient
    
    # update w
    w = w-gamma*grad
    
    return loss, w

def calculate_hessian_logistic(y, tx, w):
    """return the hessian of the loss function."""
    diagS = (sigmoid(np.dot(tx, w))*(1- sigmoid(np.dot(tx, w))))
    S = np.diagflat([diagS])
    return np.dot(np.dot(tx.T, S), tx)

def learning_by_newton_method(y, tx, w, gamma):
    """
        Do one step on Newton's method.
        return the loss and updated w.
        """
    # ***************************************************
    # return loss, gradient, hessian
    loss,grad,hess = logistic_regression(y, tx, w)
    
    # update w
    w = w - gamma*np.dot(np.linalg.inv(hess), grad)
    
    return loss, w
