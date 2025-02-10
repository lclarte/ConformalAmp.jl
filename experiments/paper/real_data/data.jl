using CSV
using DataFrames
using Random
using Statistics

function load_wine()
    data = CSV.read("experiments/data/WineQT.csv", DataFrame, missingstring=["NA"])
    # remove the last column
    x = data[:, 1:end-2]
    y = data[:, end-1]

    # normalize the data
    x = Matrix(x)
    y = Vector(y)

    n, d = size(x)
    x = coalesce.(x, 0)
    x = (x .- mean(x, dims = 1)) ./ std(x, dims = 1)
    x ./= sqrt(d)
    y = (y .- mean(y)) / std(y)
    
    return x, y 

end

function load_boston()
    data = CSV.read("experiments/data/boston.csv", DataFrame, missingstring=["NA"])

    x = data[:, 1:end-1]
    y = data[:, end]

    # normalize the data
    # replace missing values by the mean
    x = Matrix(x)
    y = Vector(y)

    n, d = size(x)
    x = coalesce.(x, 0)
    x = (x .- mean(x, dims = 1)) ./ std(x, dims = 1)
    x ./= sqrt(d)
    y = (y .- mean(y)) / std(y)

    return x, y
end

function load_riboflavin()
    data = CSV.read("experiments/data/riboflavin.csv", DataFrame)

    X = data[:, 2:end]
    y = data[:, 1]

    # normalize the data
    X = Matrix(X)
    y = Vector(y)

    X = (X .- mean(X, dims = 1)) ./ std(X, dims = 1)
    X ./= sqrt(size(X, 2))
    y = (y .- mean(y)) / std(y)

    return X, y
end

function train_test_split(x::AbstractMatrix, y::AbstractVector, n_train::Integer, seed::Integer)
    Random.seed!(seed)
    idx = shuffle(1:size(x, 1))
    x_ = x[idx, :]
    y_ = y[idx]

    x_train, x_test = x_[1:n_train, :], x_[n_train+1:end, :]
    y_train, y_test = y_[1:n_train], y_[n_train+1:end]

    return x_train, x_test, y_train, y_test
end