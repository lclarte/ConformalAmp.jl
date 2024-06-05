import numpy as np
import time 
from tqdm import tqdm 

import homotopy_conformal_prediction.approx_conformal_prediction as acp

# below, take the folder name as a command line argument 

import sys
# example : folder = "Lasso(Δ = 1.0, Δ̂ = 1.0, λ = 1.0, α = 0.5)_150"
folder = sys.argv[1]


X     = np.load(f"{folder}/X.npy")
y     = np.load(f"{folder}/y.npy")
Xtest = np.load(f"{folder}/Xtest.npy")
ytest = np.load(f"{folder}/ytest.npy")
w     = np.load(f"{folder}/w.npy")

# To change depending on the folder = model used
lambda_ = 1.0
method  = "lasso" 

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

# save the prediction intervals
np.save(f"{folder}/homotopy_intervals.npy", intervals)
print("Mean time is ", np.mean(times), " seconds.")

# append the mean time to the file homotopy_times.csv
# 1st col. of the csv is the folder name, 2nd col. is the mean time
with open("homotopy_times.csv", "a") as f:
    f.write(f"\"{folder}\",{np.mean(times)}\n")