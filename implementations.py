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
    #try to invert the matrix, if singular calculate pseudo-inverse instead
    try:
        w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError as err:
        A = np.dot(np.transpose(tx),tx)
        inverse = np.linalg.pinv(A)
        w = np.dot(np.dot(inverse,np.transpose(tx)),y)
        
    return w, mse_loss(y, tx, w)

def ridge_regression(y, tx, lambda_):
    '''y: output data
        tx: transposed input data vector
        lambda_: ridge parameter multiplying the L-2 norm
        Returns mse, and optimal weights'''
    #try to invert the matrix, if singular calculate pseudo-inverse instead
    try:
        w = np.linalg.solve(tx.T.dot(tx) + lambda_/(2*tx.shape[0])*np.identity(tx.shape[1]), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError as err:
        A = np.dot(np.transpose(tx),tx) + lambda_/(2*tx.shape[0])*np.identity(tx.shape[1])
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
    
    hess = calculate_hessian_logistic(y, tx, w)
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

def build_poly(x, degree):
    '''Polynomial basis functions for input data x, for j=0 up to j=degree.
        Returns the matrix formed by applying the polynomial basis to the input data'''
    num_samples = x.shape[0]
    ones = np.ones((num_samples,))
    pol = np.asarray([x**power for power in range(1,degree+1)])
    pol = pol.transpose(1,2,0).reshape(-1,30*degree)
    return np.c_[pol,ones]
    

    
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

def predict_y_logistic(x,w):
    '''binary prediction for logistic.
    returns y vector of {-1,1}'''
    for xi in x:
        y_pred = [-1 if sigmoid(xi.dot(w)) <= 0.5 else 1 ]
    return y_pred

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

        w, _ = ridge_regression(y_train, phi_train, lambda_)
        rmse_test = mse_loss(y_test, phi_test, w)

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
            w, _ = ridge_regression(y_train, phi_train, lambda_)
            
            rmse_test = mse_loss(y_test, phi_test, w)/k_fold #divide by k_fold in order to mean over them
            loss_temp.append(rmse_test)

        loss += loss_temp #add together losses for each k
    semilog_loss_lambda_plot(loss, lambdas, seed, degree)
    
    return 0

def predict(y, x, w):
    y_pred = x.dot(w)
    correct = sum(np.sign(y_pred) == y)/len(y)
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





