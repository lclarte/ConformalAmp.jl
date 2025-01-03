# import the file wineQT.csv

using CSV, DataFrames, Statistics, LinearAlgebra, Random, Plots
using ConformalAmp

# Load the data
data = CSV.read("experiments/paper/wineQT.csv", DataFrame)
y = data[:, "quality"]
# remove the last two columns of data to generate the features
X = data[:, 1:end-2]
print(size(X))

# normalize the covariates

function normalize(df::DataFrame)
    normalized_df = DataFrame()
    for col in names(df)
        col_mean = mean(df[!, col])
        col_std = std(df[!, col])
        normalized_df[!, col] = (df[!, col] .- col_mean) ./ col_std
    end
    return normalized_df
end

n, d = size(X)

X_normalized = normalize(X)
X_normalized = Matrix(X_normalized)
y_normalized = (y .- mean(y)) ./ std(y)

problem = ConformalAmp.Ridge(Δ = 1.0, Δ̂ = 1.0, λ = 1.0, α = n / d)

ŵ_erm = ConformalAmp.fit(problem, X_normalized, y_normalized, ConformalAmp.ERM())
ŵ_gamp= ConformalAmp.fit(problem, X_normalized, y_normalized, ConformalAmp.GAMP(max_iter = 10, rtol = 1e-2),  ŵ_erm)

y_pred_erm = X_normalized * ŵ_erm
y_pred_gamp = X_normalized * ŵ_gamp

scatter(y_pred_erm, y_pred_gamp, label = "ERM")