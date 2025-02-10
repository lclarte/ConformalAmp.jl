import numpy as np
import pandas as pd
import time 
from tqdm import tqdm 
import sklearn.model_selection as model_selection

import homotopy_conformal_prediction.approx_conformal_prediction as acp
import data

dataset = "riboflavin"
method  = "lasso" 

if dataset == "boston":
    X, y = data.load_boston()
    lambda_ = 1.0
    test_size = 0.2
elif dataset == "riboflavin":
    X, y = data.load_riboflavin()
    lambda_ = 0.25
    test_size = len(X) - 50
elif dataset == "wine":
    X, y = data.load_wine()
    lambda_ = 1.0
    test_size = 0.1

def run_seed(X, y, seed):
    n = len(X)
    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
    n_train = len(Xtrain)

    coverage = 0.9
    y_range = np.min(ytrain[:-1]), np.max(ytrain[:-1])

    # store the lower and upper bound of prediction intervals
    intervals = np.zeros((len(Xtest), 2))
    times = []

    for i, x in (enumerate(Xtest)):
        X_augmented = np.vstack((Xtrain, x))
        begin = time.time()
        pred_set = acp.conf_pred(X_augmented, ytrain, lambda_, y_range, alpha=1.0 - coverage, method=method)
        end = time.time()
        times.append(end - begin)

        intervals[i, 0] = pred_set.lower
        intervals[i, 1] = pred_set.upper

    return intervals, times, ytest

def run_experiment(X, y, nseeds):
    mean_coverage = []
    mean_length = []
    mean_time = []

    for seed in tqdm(range(nseeds)):
        intervals, times, ytest = run_seed(X, y, seed)
        coverage = np.mean((intervals[:, 0] <= ytest) & (ytest <= intervals[:, 1]))
        length = np.mean(intervals[:, 1] - intervals[:, 0])
        time = np.mean(times)

        mean_coverage.append(coverage)
        mean_length.append(length)
        mean_time.append(time)

    return mean_coverage, mean_length, mean_time

mean_coverage, mean_length, mean_time = run_experiment(X, y, 20)
print(f"Coverage : {np.mean(mean_coverage)} \pm {np.std(mean_coverage)}")
print(f"Length : {np.mean(mean_length)} \pm {np.std(mean_length)}")
print(f"Time : {np.mean(mean_time)} \pm {np.std(mean_time)}")