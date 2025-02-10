
import pandas as pd
import numpy as np

def load_boston():
    filename = "../data/boston.csv"
    #load the data, replace missing valuees with 0 and create the train and test sets
    data = pd.read_csv(filename)
    data = data.fillna(0)
    X = data.drop("MEDV", axis=1).values
    y = data["MEDV"].values

    X = np.array(X)
    y = np.array(y)

    # normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)
    return X, y

def load_riboflavin():
    filename = "../data/riboflavin.csv"
    #load the data, replace missing valuees with 0 and create the train and test sets
    data = pd.read_csv(filename)
    data = data.fillna(0)
    X = data.drop("y", axis=1).values
    y = data["y"].values

    X = np.array(X)
    y = np.array(y)

    # normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)
    return X, y

def load_wine():
    filename = "../data/WineQT.csv"
    data = pd.read_csv(filename)
    data = data.fillna(0)
    X = data.drop(["quality", "Id"], axis=1).values
    y = data["quality"].values

    X = np.array(X)
    y = np.array(y)

    n, d = X.shape
    # normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X /= np.sqrt(d)
    y = (y - np.mean(y)) / np.std(y)
    return X, y
