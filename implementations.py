import matplotlib.pyplot as plt
import numpy as np

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
    batchsize = 1 # if == 1 : true stochastic gradient descent
    
    # not to modify input arrays
    
    w = np.copy(initial_w)
    y1 = np.copy(y)
    tx1 = np.copy(tx)
    
    #####
    
    for n_iter in range(max_iters):
        concat = np.vstack((y1,tx1.T)) #to allow use of shuffle
        np.random.shuffle(concat.T) 
        y1, tx1 = concat[0], concat[1:].T # decouple after shuffle is completed 
        grad = compute_stochastic_gradient(y1, tx1, w, batchsize) 
        w = w - gamma*grad
    return w, mse_loss(y, tx, w) #batchsize+1 guarantees slicing at right point


def least_squares(y, tx):
    """calculate the least squares solution.
        y: data array;
        tx: transposed x data;
        Returns mse, and optimal weights"""
    w = np.dot(np.linalg.inv(np.dot(tx.T, tx)), np.dot(tx.T,y))
    e = y - np.dot(tx, w)
    mse = np.dot(e.T,e)/2/np.size(y)
    return w, mse

def ridge_regression(y, tx, lambda_):
    '''y: output data
        tx: transposed input data vector
        lambda_: ridge parameter multiplying the L-2 norm
        Returns loss and weights'''
    w = np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_/2/np.size(y)*np.identity(np.size(tx[0,:]))), np.dot(tx.T,y))
    e = y - np.dot(tx, w)
    loss = np.dot(e.T,e)/2/np.size(y)
    return w, loss



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression by gradient descent
        y: data array;
        tx: transposed x data;
        Returns mse, and optimal weights
    """
    
    w = np.copy(initial_w)
    
    loss = calculate_loss(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient(y, tx, w)
    # compute the gradient
    
    #hess = calculate_hessian(y, tx, w)
    # compute the hessian, for generalizations
    
    threshold = 1e-8
    
     
    previous_loss = 0
    
    for iter_ in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if iter_ > 1 and np.abs(loss - previousloss) < threshold:
            break
        previousloss = loss
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression by gradient descent
        y: data array;
        tx: transposed x data;
        Penalized w/ parameter lambda_
        Returns mse, and optimal weights"""
    
    w = np.copy(initial_w)
    
    loss = calculate_loss(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient(y, tx, w)
    # compute the gradient
    
    hess = calculate_hessian(y, tx, w)
    # compute the hessian
    
    threshold = 1e-8
    
    previousloss = 0
    
    for iter_ in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if iter_ > 1 and np.abs(loss - previousloss) < threshold:
            break
        previousloss = loss
        
    return w, loss



### Helper Functions for the main project files


def standardize_by_column(x):
    '''Standardizes each column of a matrix'''
    for col in range(x.shape[1]):
        x[:,col] = (x[:,col]-np.mean(x[:,col])) / np.std(x[:,col])
    
    return x


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
    return -np.dot(tx.T, y - np.dot(tx, w))/np.size(y)


#Stochastic Gradient Descent

def compute_stochastic_gradient(y, tx, w, batchsize):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return -np.dot(tx.T[:,:batchsize+1], y[:batchsize+1] - np.dot(tx[:batchsize+1], w[:batchsize+1]))/np.size(y[:batchsize+1])


#Build a polynomial basis Phi(x)

def build_poly_col_ones(x, degree):
    '''Polynomial basis functions for input data x, for j=0 up to j=degree.
        Returns the matrix formed by applying the polynomial basis to the input data
        Expects a col of ones and a col of data'''
    for k in range(degree-1):
        x = np.hstack((x,x[:,1:]**degree))
    return x
    

    
#Split data to do train vs test

def split_data(x, y, ratio, seed=1):
    '''
        split the dataset based on the split ratio. If ratio is 0.8
        you will have 80% of your data set dedicated to training
        and the rest dedicated to testing
        '''
    # set seed
    np.random.seed(seed)
    # ***************************************************
    x_y = np.c_[(x,y)]
    #print(x_y)
    Ntrain = int(np.size(y)*ratio)
    np.random.shuffle(x_y)
    #print(x_y)
    return x_y.T[0][:Ntrain], x_y.T[1][:Ntrain], x_y.T[0][Ntrain:], x_y.T[1][Ntrain:]



#Logistic Regression

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/ (1 + np.exp(-t) ) #np.exp(t)/(1+np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return sum(np.log(1+np.exp(np.dot(tx,w))) - y*np.dot(tx,w))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T, sigmoid(np.dot(tx,w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
    
    loss = calculate_loss(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient(y, tx, w)
    # compute the gradient
    
    # update w
    w = w-gamma*grad
    
    return loss, w

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
    
    loss = calculate_loss(y, tx, w) + lambda_/2*np.dot(w,w)
    # compute the cost
    
    grad = calculate_gradient(y, tx, w) + lambda_*w
    # compute the gradient
    
    # update w
    w = w-gamma*grad
    
    return loss, w

def calculate_hessian(y, tx, w):
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


def ridge_with_simple_splitting(y,x, degree, ratio, lambdas, seed = 1):
    '''performe ridge regression with simple splitting of the dataset.
    ratio is #train/#test
    print percentage of correct answers for each lambda
    plot loss vs lambda
    return the loss vector'''
    
    loss = []
    y_train, x_train, y_test, x_test = split_data(y, x, ratio, seed)

    phi_test = build_poly(x_test, degree)
    phi_train = build_poly(x_train, degree)

    for lambda_ in lambdas:

        w = ridge_regression(y_train, phi_train, lambda_)
        rmse_test = cost_function(y_test, phi_test, w)

        loss.append(rmse_test)

        print("Correct answers: ",predict(y_test,phi_test,w), '%', "for lambda = %f" %lambda_)
    semilog_loss_lambda_plot(loss, lambdas, seed, degree)
    return w

def cross_validation_ridge(y, x, k_fold, degree, lambdas, seed = 1):
    '''perform cross validation on ridge regression
    lambdas: array, better if log spaced
    plot in semilog scale rmse as function of lambda'''
    
    #create empty loss array
    loss = np.zeros((len(lambdas))) 
    #add poly values to the features
    phi = build_poly(x, degree)
    #build indices for cross validation
    k_indices = build_k_indices(y, k_fold, seed)

    for k in range(k_fold):
        #split data according to kth fold
        y_train, phi_train, y_test, phi_test = split_data_cross(y, phi, k, k_indices, degree)

        loss_temp = [] #empty list to store losses for a given k
        
        for lambda_ in lambdas:
            w = ridge_regression(y_train, phi_train, lambda_)
            
            rmse_test = cost_function(y_test, phi_test, w)/k_fold #divide by k_fold in order to mean over them
            loss_temp.append(rmse_test)

        loss += loss_temp #add together losses for each k
    semilog_loss_lambda_plot(loss, lambdas, seed, degree)
    
def create_submission(x_submission, degree, ids_submission):
    '''creates the y_predicted file for the submission on kaggle'''
    #build the polynomial from x
    phi_submission = build_poly(x_submission, degree) 
    #apply the found w
    y_predicted = np.sign(phi_submission.dot(w))
    #create the file
    create_csv_submission(ids_submission, y_predicted, "predictions.csv")





