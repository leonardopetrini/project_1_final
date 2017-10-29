import matplotlib.pyplot as plt
import numpy as np
import time#For execution times
from implementations import *

### Helper Functions for the main project files

#Standardize a column of X

def standardize_by_column(x):
    '''Standardizes each column of a matrix'''
    for col in range(x.shape[1]):
        x[:,col] = (x[:,col]-np.mean(x[:,col])) / np.std(x[:,col])
    
    return x

#Build a polynomial basis Phi(x)

def build_poly(x, degree, crossed = True):
    '''Polynomial basis functions for input data x, for j=0 up to j=degree.
        If crossed == False, it returns polynomials of order degree of the feature data,
        Else, it return crossed terms containing the features up to order 2:
        for example x1.x2, x2.x4, x23.x28 etc.
        Returns the matrix formed by applying the polynomial basis to the input data'''
    num_samples = x.shape[0]
    ones = np.ones((num_samples,))
    pol = np.asarray([x**power for power in range(1,degree+1)])
    pol = pol.transpose(1,2,0).reshape(-1,30*degree)
    if crossed == False:
        return np.c_[pol,ones]
    else:
        cross = np.asarray([np.multiply(x[:,i],x[:,j]) for i in range(x.shape[1]) for j in range(i) if i != j])
        return np.c_[pol,ones,cross.T]
    
#Function to get len(w) for GD,SGD (- Redundant!)

def get_length_w(x, degree, crossed = True):
    '''Similar to the above, just rturns number needed for len(w)'''
    num_samples = x.shape[0]
    ones = np.ones((num_samples,))
    pol = np.asarray([x**power for power in range(1,degree+1)])
    pol = pol.transpose(1,2,0).reshape(-1,30*degree)
    if crossed == False:
        return np.c_[pol,ones].shape[1]
    else:
        cross = np.asarray([np.multiply(x[:,i],x[:,j]) for i in range(x.shape[1]) for j in range(i) if i != j])
        return np.c_[pol,ones,cross.T].shape[1]
    
    
#Split data to do train vs test

def split_data(y,x, ratio, seed=1):
    '''
        split the dataset based on the split ratio. If ratio is 0.8
        you will have 80% of your data set dedicated to training
        and the rest dedicated to testing
        '''
    
    np.random.seed(seed)
    indices = list(range(x.shape[0]))
    np.random.shuffle(indices)
    
    indices_train = indices[:round(x.shape[0]*ratio)]
    indices_test = indices[round(x.shape[0]*ratio):]
    
    return y[indices_train],x[indices_train,:],y[indices_test],x[indices_test,:]

def non_values_to_random_normally_dist(x):
    '''substitute the -999 values in x with normally distributed values with mean and sigma of each column'''
    #loop over all columns
    for col in range(x.shape[1]):
        x1 = 1*x #support variable
        x1[x1==-999] = 0 #set non-values to zero
        #keep track of the non-values indices
        indices = [i for i,item in enumerate(x[:,col] == -999) if item]
        length = len(x1[:,col])
        nonzeros = sum(x1[:,col] != 0)
        mean = np.mean(x1[:,col])*length/nonzeros
        norm = np.linalg.norm(x1[:,col])/nonzeros
        for i in indices:
            #for each non-value replace it with random noise distributed according to the others belonging to the same feature
            x[i,col] = np.random.normal(mean, norm)
    return x

def compare(x_normalized, x_0):
    '''This funciton compares the two "cleaning" methods by subraction'''
    diff  = x_normalized - x_0
    indices = np.where(diff != 0)
    i1, i2 = indices[0], indices[1]
    print ("Number of indices which are different is",len(i1))
    return 0
    

def GD_with_simple_splitting(y,x, degree, ratio, initial_w, max_iters, gamma, seed = 1):
    '''performe LS regression with simple splitting of the dataset.
    ratio is #train/#test
    print percentage of correct answers and execution time
    plot loss vs lambda
    return the loss vector'''
    start_time = time.time()
    
    losses = []
    y_train, x_train, y_test, x_test = split_data(y, x, ratio, seed)
    phi_test = build_poly(x_test, degree)
    phi_train = build_poly(x_train, degree)

    

    w, loss = least_squares_GD(y_train, phi_train, initial_w, max_iters, gamma)
    rmse_test = np.sqrt(2*loss)
    
    losses.append(rmse_test)

    print("Correct answers: ",predict(y_test,phi_test,w), '%',"Execution time=%s seconds" % (time.time() - start_time))
    #semilog_loss_lambda_plot(losses, lambdas, seed, degree)
    return w

def SGD_with_simple_splitting(y,x, degree, ratio, initial_w, max_iters, gamma, seed = 1):
    '''performe LS regression with simple splitting of the dataset.
    ratio is #train/#test
    print percentage of correct answers and execution time
    plot loss vs lambda
    return the loss vector'''
    start_time = time.time()
    
    losses = []
    y_train, x_train, y_test, x_test = split_data(y, x, ratio, seed)
    phi_test = build_poly(x_test, degree)
    phi_train = build_poly(x_train, degree)

    

    w, loss = least_squares_SGD(y_train, phi_train, initial_w, max_iters, gamma)
    rmse_test = np.sqrt(2*loss)
    
    losses.append(rmse_test)

    print("Correct answers: ",predict(y_test,phi_test,w), '%',"Execution time=%s seconds" % (time.time() - start_time))
    #semilog_loss_lambda_plot(losses, lambdas, seed, degree)
    return w



def LS_with_simple_splitting(y,x, degree, ratio, seed = 1):
    '''performe LS regression with simple splitting of the dataset.
    ratio is #train/#test
    print percentage of correct answers for each lambda
    plot loss vs lambda
    return the loss vector'''
    
    losses = []
    y_train, x_train, y_test, x_test = split_data(y, x, ratio, seed)
    phi_test = build_poly(x_test, degree)
    phi_train = build_poly(x_train, degree)

    

    w, loss = least_squares(y_train, phi_train)
    rmse_test = np.sqrt(2*loss)

    losses.append(rmse_test)

    print("Correct answers: ",predict(y_test,phi_test,w))
    #semilog_loss_lambda_plot(losses, lambdas, seed, degree)
    return w

def ridge_with_simple_splitting(y,x, degree, ratio, lambdas, seed = 1):
    '''performe ridge regression with simple splitting of the dataset.
    ratio is #train/#test
    print percentage of correct answers for each lambda
    plot loss vs lambda
    return the loss vector'''
    
    losses = []
    y_train, x_train, y_test, x_test = split_data(y, x, ratio, seed)
    phi_test = build_poly(x_test, degree)
    phi_train = build_poly(x_train, degree)

    for lambda_ in lambdas:

        w, loss = ridge_regression(y_train, phi_train, lambda_)
        rmse_test = np.sqrt(2*loss)

        losses.append(rmse_test)

        print("Correct answers: ",predict(y_test,phi_test,w), '%', "for lambda = %f" %lambda_)
    semilog_loss_lambda_plot(losses, lambdas, seed, degree)
    return w

def bayes_with_simple_splitting(y,x, degree, ratio, lambdas, q, seed = 1):
    '''performe bayes regression or order q with simple splitting of the dataset.
    ratio is #train/#test
    print percentage of correct answers for each lambda
    plot loss vs lambda
    return the loss vector'''
    
    losses = []
    y_train, x_train, y_test, x_test = split_data(y, x, ratio, seed)
    phi_test = build_poly(x_test, degree)
    phi_train = build_poly(x_train, degree)

    for lambda_ in lambdas:

        w, loss = bayes_regression(y_train, phi_train, lambda_, q)
        rmse_test = np.sqrt(2*loss)

        losses.append(rmse_test)

        print("Correct answers: ",predict(y_test,phi_test,w), '%', "for lambda = %f" %lambda_)
    semilog_loss_lambda_plot(losses, lambdas, seed, degree)
    return w

def logistic_with_simple_splitting(y,x, degree, ratio, initial_w, max_iters, gamma, seed = 1):
    '''performe logistic regression or order q with simple splitting of the dataset.
    ratio is #train/#test
    print percentage of correct answers for each lambda
    plot loss vs lambda
    return the loss vector'''
    
    losses = []
    y_train, x_train, y_test, x_test = split_data(y, x, ratio, seed)
    phi_test = build_poly(x_test, degree)
    phi_train = build_poly(x_train, degree)

    

    w, loss = logistic_regression(y_train, phi_train, initial_w, max_iters, gamma)#Note that the loss here is an MSE loss
    rmse_test = np.sqrt(2*loss)

    losses.append(rmse_test)

    print("Correct answers: ",pred_log(y_test,phi_test,w), '%')
    return w



def reg_logistic_with_simple_splitting(y,x, degree, ratio, initial_w, max_iters, gamma, lambdas, seed = 1):
    '''performe logistic regression or order q with simple splitting of the dataset.
    ratio is #train/#test
    print percentage of correct answers for each lambda
    plot loss vs lambda
    return the loss vector'''
    
    losses = []
    y_train, x_train, y_test, x_test = split_data(y, x, ratio, seed)
    phi_test = build_poly(x_test, degree)
    phi_train = build_poly(x_train, degree)

    for lambda_ in lambdas:

        w, loss = reg_logistic_regression(y_train, phi_train, lambda_, initial_w, max_iters, gamma)#Note that the loss here is an MSE loss
        rmse_test = np.sqrt(2*loss)

        losses.append(rmse_test)

        print("Correct answers: ",pred_log(y_test,phi_test,w), '%', "for lambda = %f" %lambda_)
    semilog_loss_lambda_plot(losses, lambdas, seed, degree)
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
            w, _ = ridge_regression(y_train, phi_train, lambda_)
            
            rmse_test = mse_loss(y_test, phi_test, w)/k_fold #divide by k_fold in order to mean over them
            loss_temp.append(rmse_test)
            print("Correct answers: ",predict(y_test,phi_test,w), '%', "for lambda = %f" %lambda_)
        loss += loss_temp #add together losses for each k
    semilog_loss_lambda_plot(loss, lambdas, seed, degree)
    
    return w

def cross_validation_logistic(y, x, initial_w, max_iters, gamma, k_fold, degree, lambdas, seed = 1):
    '''perform cross validation on logistic regression
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
            w, _ = reg_logistic_regression(y_train, phi_train, lambda_, initial_w, max_iters, gamma)
            
            rmse_test = mse_loss(y_test, phi_test, w)/k_fold #divide by k_fold in order to mean over them
            loss_temp.append(rmse_test)
            print("Correct answers: ",predict(y_test,phi_test,w), '%', "for lambda = %f" %lambda_)
        loss += loss_temp #add together losses for each k
    semilog_loss_lambda_plot(loss, lambdas, seed, degree)
    
    return w

def predict(y, x, w):
    '''The function that creates the prediciton during the cross validation'''
    y_pred = x.dot(w)
    correct = sum(np.sign(y_pred) == y)/len(y)
    return correct*100

def pred_log(y, x, w):
    '''Create predictions for logistic regression'''
    pred = sigmoid(np.dot(x, w))
    for p in range(np.size(pred)):
        if pred[p]  > 0.5:
            pred[p] = 1
        else:
            pred[p] = -1
    correct = sum(np.sign(pred) == y)/len(y)
    return correct*100

def create_submission(x_submission, degree, ids_submission):
    '''creates the y_predicted file for the submission on kaggle'''
    #build the polynomial from x
    phi_submission = build_poly(x_submission, degree) 
    #apply the found w
    y_predicted = np.sign(phi_submission.dot(w))
    #create the file
    create_csv_submission(ids_submission, y_predicted, "predictions.csv")

'''Plotting functions'''

def semilog_loss_lambda_plot(loss, lambdas, seed, degree):
    plt.title("lambda vs loss for seed = %i and degree = %i" %(seed, degree))
    plt.xlabel("lambda")
    plt.ylabel("loss")
    plt.semilogx(lambdas, loss, 'r')
    plt.savefig("lambda_vs_loss_simple_splitting_ridge_seed%i.png" %seed)
    
'others'

def split_data_cross(y, phi, k, k_indices, degree, seed=1):
  
    y_test, phi_test = (y[k_indices[k]],phi[k_indices[k],:])
    
    not_k = [i for i,item in enumerate(y) if i not in k_indices[k]]
    y_train = y[not_k]
    phi_train = phi[not_k,:]
    
    return y_train, phi_train, y_test, phi_test

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


### A generalization of Ridge for the L-q norm: Bayes

## TO DO
def bayes_regression(y, tx, lambda_, q):
    '''y: output data
        tx: transposed input data vector
        lambda_: ridge parameter multiplying the L-2 norm
        q: L-q norm penalty
        Returns mse, and optimal weights'''
    #try to invert the matrix, if singular calculate pseudo-inverse instead
    w_estimate = ridge_regression(y, tx, lambda_)[0] #estimates w by ridge
    wL2 = w_estimate**2
    
    try:
        w = np.linalg.solve(tx.T.dot(tx) + lambda_ * tx.shape[0] * q *( wL2**(q/2-1) )*np.identity(tx.shape[1]), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError as err:
        A = tx.T.dot(tx) + lambda_/2*tx.shape[0]*q*(w_estimate**(q/2-1))*np.identity(tx.shape[1]), tx.T.dot(y)
        inverse = np.linalg.pinv(A)
        w = np.dot(np.dot(inverse,np.transpose(tx)),y)
    
    return w, mse_loss(y,tx,w)

###


