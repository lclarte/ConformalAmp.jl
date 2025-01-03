using ConformalAmp

using Distributions: normpdf, quantile, Normal
using LinearAlgebra: diag, norm
using Plots
using StableRNGs: StableRNG
using Statistics: mean
using ProgressBars: ProgressBar

rng = StableRNG(0)

# sample a dataset and compute the leave-one-out residuals for all the points
α = 2.0
Δ = 1.0
λ = 1.0
d = 500

sampling_problem = ConformalAmp.BayesOptimalLasso(Δ = Δ, Δ̂ = 1.0, λ = 1.0, α = α)
(; X, w, y) = ConformalAmp.sample_all(rng, sampling_problem, d)

problem = ConformalAmp.Lasso(Δ = Δ, Δ̂ = 1.0, λ = λ, α = α)

# sample a dataset

method = ConformalAmp.GAMP(rtol = 1e-3, max_iter = 100)
ŵ      = ConformalAmp.fit(problem, X, y, method)
W_loo  = ConformalAmp.fit_leave_one_out(problem, X, y, method)

ŷ      = diag(ConformalAmp.predict(problem, W_loo, X))

stephist(abs.(y - ŷ), bins = 100, label = "residuals", normed = true)

ρ = norm(w)^2 / d
m = w' * ŵ / d
q = norm(ŵ)^2 / d

# observation : the residuals follow a Gaussian distribution (normal) with mean 0 
# and variance Δ + ρ - 2m + q
plot!(x -> 2 * abs(normpdf(0, sqrt(Δ + ρ - 2 * m + q), x)), 0.0, 5, label = "Gaussian fit")

κ = 0.1 # target coverage is 1 - κ
# compute the κ / 2 and 1 - κ / 2 quantiles of the Gaussian distribution to predict 
# the quantiles of the residuals
q₁ = quantile(Normal(0, sqrt(Δ + ρ - 2 * m + q)), 1 - κ / 2)
q₂ = quantile(Normal(0, sqrt(Δ + ρ - 2 * m + q)), κ / 2)
print("Predicted size of prediction interval for coverage $(1 - κ) : $(q₁ - q₂)\n")

# get a test set and compute the size of the intervals at 1 - κ
ntest = 100

X_test = ConformalAmp.sample_data_any_n(rng, d, ntest)
y_test = ConformalAmp.sample_labels(rng, sampling_problem, X_test, w)


fcp = ConformalAmp.FullConformal(coverage = 1 - κ, δy_range = 0.0:0.1:5.0)
sizes = []
for i in ProgressBar(1:ntest)
    xtest = X_test[i, :]
    ci = ConformalAmp.get_confidence_interval(problem, X, y, xtest, fcp, method)
    push!(sizes, maximum(ci) - minimum(ci))
end
println("Mean size of prediction intervals is $(mean(sizes))")

### state evolution 