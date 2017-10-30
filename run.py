'''run.py file giving our kaggle best prediction
uses ridge regression with polynomial up to degree 11 for each feature
degree 2 and some terms of degree 3 for products between different features
sqrt exp and log of each feature
lambda to use is obtained with cross validation on the subsample of the dataset
considering the computational time would have been too long using the whole'''

import numpy as np
from implementations import *
from general_helpers import *
from proj1_helpers import *
import matplotlib.pyplot as plt

#Load data
y, x, ids = load_csv_data("train.csv", sub_sample=False)
_, x_submission, ids_submission = load_csv_data("test.csv", sub_sample=False)

#Put -999 values to randomly distributed values according to mean and norm of the column
x = non_values_to_random_normally_dist(x)
x = standardize_by_column(x)

x_submission = non_values_to_random_normally_dist(x_submission)
x_submission = standardize_by_column(x_submission)

#-------------------------------#

#definition of parameters
degree = 11
ratio = .9995  #train/test examples, still keeping to have one last check
lambdas = [8e-5]
seed = 56

#perform ridge
w = ridge_with_simple_splitting_enriched(y,x, degree, ratio, lambdas, seed)

#build phi for the submission data
phi_submission = build_poly_enriched(x_submission, degree)
#apply the found w
y_predicted = np.sign(phi_submission.dot(w))

#print percentage of -1 just as a check
print((sum(y_predicted == -1))/len(y_predicted))

create_csv_submission(ids_submission, y_predicted, "predictions.csv")
