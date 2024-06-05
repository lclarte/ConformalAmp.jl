# Implement here quantile conformal prediction which is split conformal prediction using quantile regression

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split

def get_estimators_qcp(X, y, coverage, lambda_, train_val_split=0.8):
    """
    lambda_
    """
    alpha = 1.0 - coverage
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_val_split, random_state=0)

    n, d  = X_train.shape


    qr_upper = QuantileRegressor(quantile = 1.0 - alpha / 2.0, alpha = lambda_ / n, solver="highs")
    qr_lower = QuantileRegressor(quantile = alpha / 2.0, alpha = lambda_ / n, solver="highs")

    qr_upper.fit(X_train, y_train)
    qr_lower.fit(X_train, y_train)

    # compute the residuals 
    residuals_upper = y_val - qr_upper.predict(X_val)
    residuals_lower = qr_lower.predict(X_val) - y_val

    scores_val = [max( a, b ) for a,b in zip(residuals_upper, residuals_lower)]
    # get the coverage * (1.0 + 1 / n_val ) quantile of the scores
    threshold = np.quantile(scores_val, coverage * (1.0 + 1.0 / len(y_val)))

    return qr_upper, qr_lower, threshold

def get_intervals_qcp(X_train, y_train, X_test, coverage, lambda_):
    qr_upper, qr_lower, threshold = get_estimators_qcp(X_train, y_train, coverage, lambda_)
    upper = qr_upper.predict(X_test)
    lower = qr_lower.predict(X_test)

    return lower - threshold, upper + threshold

if __name__ == "__main__":
    # generate data from a Gaussian teacher 
    d = 250
    n = 125
    ntest = 10000
    
    
    # for a teacher from the Gaussian distribution
    # wstar = np.random.normal(0.0, 1.0, size=(d, ))
    # for a teacher from the Laplace distribution
    wstar = np.random.laplace(0.0, 1.0, size=(d, ))
    
    X = np.random.normal(0.0, 1.0, size=(n, d)) / np.sqrt(d)
    y = X @ wstar + np.random.normal(0.0, 1.0, size=(n, ))
    X_test = np.random.normal(0.0, 1.0, size=(ntest, d)) / np.sqrt(d)
    y_test = X_test @ wstar + np.random.normal(0.0, 1.0, size=(ntest, ))

    lambda_ = 0.1
    coverage = 0.9

    lower, upper = get_intervals_qcp(X, y, X_test, coverage, lambda_)
    # compute the true coverage 
    coverage = np.mean( (y_test >= lower) & (y_test <= upper) )
    # print the mean size of the intervals
    print("Coverage: ", coverage)
    print("Mean size of the intervals: ", np.mean(upper - lower))