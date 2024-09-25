import numpy as np
import pandas as pd
import time 
from tqdm import tqdm 

import homotopy_conformal_prediction.approx_conformal_prediction as acp

filename = "../misc/riboflavin.csv"
#load the data, replace missing valuees with 0 and create the train and test sets
data = pd.read_csv(filename)
print(data)
data = data.fillna(0)
X = data.drop("y", axis=1).values
y = data["y"].values

X = np.array(X)
y = np.array(y)

# normalize the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = (y - np.mean(y)) / np.std(y)

n = len(X)
n_train = 50

Xtrain = X[:n_train]
ytrain = y[:n_train]
Xtest = X[n_train:]
ytest = y[n_train:]

# To change depending on the folder = model used
lambda_ = 0.25
method  = "lasso" 

print("method: ", method)
print("lambda: ", lambda_)

coverage = 0.9
y_range = np.min(y[:-1]), np.max(y[:-1])

# store the lower and upper bound of prediction intervals
intervals = np.zeros((len(Xtest), 2))
times = []

print("Total number of test samples: ", len(Xtest))

for i, x in tqdm(enumerate(Xtest)):
    X_augmented = np.vstack((X, x))
    begin = time.time()
    pred_set = acp.conf_pred(X_augmented, y, lambda_, y_range, alpha=1.0 - coverage, method=method)
    end = time.time()
    times.append(end - begin)

    intervals[i, 0] = pred_set.lower
    intervals[i, 1] = pred_set.upper

# compute the coverage
coverage = np.mean((intervals[:, 0] <= ytest) & (ytest <= intervals[:, 1]))
print("Coverage: ", coverage)
# compute the average length¨
length = np.mean(intervals[:, 1] - intervals[:, 0])
print("Average length: ", length)
# print the average computation time 
print("Mean time is ", np.mean(times), " seconds.")
